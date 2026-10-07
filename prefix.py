from dataclasses import dataclass

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter, Sequential, Tanh
from transformers import DynamicCache, PreTrainedModel, set_seed

from primitives.dataset import load_metamathqa
from primitives.hub import save_prefix
from primitives.model import load_model
from primitives.train import TrainConfig, train


@dataclass
class Config(TrainConfig):
    organization: str = "meta-llama"
    model_name: str = "Llama-3.2-3B"
    prefix_len: int = 16
    mlp_dim: int = 512
    num_samples: int = 32768
    max_sequence_len: int = 2048
    seed: int = 0


class PrefixMLP(Module):
    # prefix를 직접 학습하면 각 파라미터가 독립적으로 업데이트됨.
    # 반면 mlp를 거치면 모든 prefix가 같은 Linear로부터 생성됨.
    # 이 방식이 empirically 안정적.
    def __init__(
        self,
        prefix: Tensor,
        mlp_dim: int,
    ) -> None:
        super().__init__()
        num_layers, _, _, num_kv_heads, prefix_len, head_dim = prefix.shape
        self.prefix = prefix
        self.embedding = Parameter(torch.randn(prefix_len, mlp_dim))
        self.mlp = Sequential(
            Linear(mlp_dim, mlp_dim),
            Tanh(),
            Linear(mlp_dim, num_layers * 2 * 1 * num_kv_heads * head_dim, bias=False),
        )
        # Linear는 기본적으로 무작위 초기화.
        # 그대로 두면 첫 스텝에 출력이 0이 아니어서 prefix가 초기값에서 벗어남.
        # 따라서 lora 때처럼 마지막 층을 0으로 둬 학습 시작 시 prefix가 초기값 유지할 수 있도록.
        torch.nn.init.zeros_(self.mlp[-1].weight)

    def forward(self) -> Tensor:
        num_layers, _, _, num_kv_heads, prefix_len, head_dim = self.prefix.shape
        delta = self.mlp(self.embedding).view(prefix_len, num_layers, 2, 1, num_kv_heads, head_dim)
        return self.prefix + delta.movedim(0, 4)


@torch.no_grad()
def initialize_prefix(
    model: PreTrainedModel,
    prefix_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    cache = DynamicCache()
    model(input_ids=prefix_ids.unsqueeze(0).to(model.device), past_key_values=cache)
    key_value = torch.stack([torch.stack([layer.keys, layer.values]) for layer in cache.layers])
    sink = key_value[..., :1, :].clone()
    prefix = key_value[..., 1:, :].float()
    return sink, prefix


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    model, tokenizer = load_model(config.organization, config.model_name)

    # prefix 초기화에 쓸 prefix_len개를 여유분으로 추가 로드.
    # 질문들을 이을 때 첫 질문의 bos만 남김.
    samples = load_metamathqa(tokenizer, config.prefix_len + config.num_samples, config.max_sequence_len)
    question_ids = [sample_question_ids if i == 0 else sample_question_ids[1:] for i, (sample_question_ids, _) in enumerate(samples[: config.prefix_len])]
    prefix_ids = torch.cat(question_ids)[: config.prefix_len + 1]
    samples = samples[config.prefix_len :]

    sink, prefix = initialize_prefix(model, prefix_ids)
    prefix_mlp = PrefixMLP(prefix, config.mlp_dim).to(model.device)
    train(model, samples, list(prefix_mlp.parameters()), config, sink=sink, prefix_mlp=prefix_mlp)

    path = f"prefix/{config.model_name}/prefix_len={config.prefix_len},learning_rate={config.learning_rate:g},seed={config.seed}"
    save_prefix(path, sink, prefix_mlp())


if __name__ == "__main__":
    main()
