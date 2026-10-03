from dataclasses import dataclass
import math

from simple_parsing import parse
import torch
from torch.nn import Parameter
from transformers import PreTrainedModel, set_seed

from primitives.data import load_metamathqa
from primitives.hub import get_repo_name, save_lora
from primitives.model import hook_lora, load_model
from primitives.train import TrainConfig, train


@dataclass
class Config(TrainConfig):
    model_name: str = "meta-llama/Llama-3.2-3B"
    rank: int = 4
    scale: float = 2.0
    num_samples: int = 32768
    max_seq_len: int = 2048
    seed: int = 0


def init_lora(
    model: PreTrainedModel,
    rank: int,
) -> dict[str, Parameter]:
    lora = {}
    for module_name, module in model.named_modules():
        if module_name.endswith(("q_proj", "k_proj", "v_proj")):
            out_dim, in_dim = module.out_features, module.in_features
            a = torch.randn(rank, in_dim, device=model.device) / math.sqrt(in_dim)
            # b를 0으로 둬서 첫 스텝에서 lora 출력이 0이 되도록.
            # 이로써 사전학습 모델 성능 그대로에서 출발 가능.
            b = torch.zeros(out_dim, rank, device=model.device)
            lora[f"{module_name}.a"] = Parameter(a)
            lora[f"{module_name}.b"] = Parameter(b)
    return lora


def main() -> None:
    config = parse(Config)
    set_seed(config.seed)
    model, tokenizer = load_model(config.model_name)
    samples = load_metamathqa(tokenizer, config.num_samples, config.max_seq_len)

    lora = init_lora(model, config.rank)
    hook_lora(model, lora, config.scale)
    train(model, samples, list(lora.values()), config)

    lora = {name: tensor * config.scale if name.endswith(".b") else tensor for name, tensor in lora.items()}
    path = f"lora/rank={config.rank},learning_rate={config.learning_rate:g},seed={config.seed}"
    save_lora(get_repo_name(config.model_name), path, lora)


if __name__ == "__main__":
    main()
