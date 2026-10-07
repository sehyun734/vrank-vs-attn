from collections import defaultdict
from dataclasses import dataclass

from simple_parsing import parse
import torch
from torch import Tensor
from torch.nn.functional import pad
from transformers import DynamicCache, PreTrainedModel

from primitives.dataset import load_metamathqa
from primitives.hub import get_pending_targets, load_lora, load_prefix, save_result
from primitives.model import hook_lora, load_model, make_prefix_cache


@dataclass
class Config:
    organization: str = "meta-llama"
    model_name: str = "Llama-3.2-3B"
    num_samples: int = 256
    max_sequence_len: int = 768
    mass_ratio: float = 0.9
    energy_ratio: float = 0.9


@torch.no_grad()
def capture_values(
    model: PreTrainedModel,
    samples: list[tuple[Tensor, Tensor]],
    sink: Tensor | None = None,
    prefix: Tensor | None = None,
) -> tuple[list[Tensor], list[Tensor]]:
    values = []
    hiddens = []
    for question_ids, answer_ids in samples:
        cache = make_prefix_cache(sink, prefix, 1) if prefix is not None else DynamicCache()
        cache_len = cache.get_seq_length()
        input_ids = torch.cat([question_ids, answer_ids]).unsqueeze(0).to(model.device)
        output = model(input_ids=input_ids, past_key_values=cache, output_hidden_states=True)
        values.append(torch.stack([layer.values[0, :, cache_len + 1 :] for layer in cache.layers]))
        hiddens.append(output.hidden_states[-1][0, 1:].float().cpu())
    return values, hiddens


def compute_basis(
    value: Tensor,
    energy_ratio: float,
) -> Tensor:
    # 실제 value는 noise 때문에 거의 항상 full rank.
    # 논문대로 energy 누적 합이 energy_ratio를 채우는 방향으로만.
    _, singular, basis = torch.linalg.svd(value.float(), full_matrices=False)
    energy = singular.square()
    return basis[: int((energy.cumsum(0) / energy.sum() < energy_ratio).sum()) + 1]


def compute_new_ratio(
    value: Tensor,
    basis: Tensor,
) -> float:
    # 논문대로 basis 밖 성분의 norm을 전체 norm으로 나눔.
    value = value.float()
    return ((value - value @ basis.T @ basis).norm() / value.norm()).item()


def compute_hidden_rank(
    hiddens: list[Tensor],
    mass_ratio: float,
) -> int:
    # 논문대로 각 sample의 마지막 레이어 hidden의 singular 정규화 후 평균.
    # 정규화는 sample마다 singular 크기가 다르고 누적합 비교에서는 합이 1이어야 하기 때문.
    # 또한 sample마다 길이가 다르므로 0 padding.
    # value의 새로운 방향이 마지막 레이어까지 이어지는지 확인하기 위함.
    spectra = []
    for hidden in hiddens:
        singular = torch.linalg.svdvals(hidden.float())
        spectra.append(pad(singular / singular.sum(), (0, hidden.shape[-1] - len(singular))))
    spectrum = torch.stack(spectra).mean(0)
    return int((spectrum.cumsum(0) < mass_ratio).sum()) + 1


def main() -> None:
    config = parse(Config)
    file_name = f"num_samples={config.num_samples}"
    targets = get_pending_targets(config.model_name, "vrank", file_name)
    # attention 구현에 따라 bf16 hidden 수치가 조금 달라짐.
    # 기존 probe 결과와 같은 기준으로 비교하기 위해 eager 유지.
    model, tokenizer = load_model(config.organization, config.model_name, attn_implementation="eager")
    samples = load_metamathqa(tokenizer, config.num_samples, config.max_sequence_len, held_out=True)

    num_samples = len(samples)
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    base_values, base_hiddens = capture_values(model, samples)
    for target in targets:
        method = target.split("/")[0]
        if method == "prefix":
            sink, prefix = load_prefix(target, model.device)
            values, hiddens = capture_values(model, samples, sink=sink, prefix=prefix)
            prefix_value = prefix[:, 1, 0]
        elif method == "lora":
            handles = hook_lora(model, load_lora(target, model.device))
            values, hiddens = capture_values(model, samples)
            for handle in handles:
                handle.remove()
            prefix_value = None
        elif method == "base":
            values, hiddens = base_values, base_hiddens
            prefix_value = None

        stats = defaultdict(lambda: torch.zeros(num_layers, num_kv_heads))
        for layer in range(num_layers):
            for head in range(num_kv_heads):
                # basis와 vrank는 sample별로 구해 평균.
                # 이는 각 sample마다 쓰는 주요 방향 basis가 다르기 때문.
                # value를 이어서 한 번에 계산하면 sample마다 다른 basis들이 섞임.
                # 그래서 vrank가 부풀려질 수 있음.
                for value, base_value in zip(values, base_values):
                    basis = compute_basis(base_value[layer, head], config.energy_ratio)
                    stats["vrank"][layer, head] += len(compute_basis(value[layer, head], config.energy_ratio)) / num_samples
                    stats["new_ratio"][layer, head] += compute_new_ratio(value[layer, head], basis) / num_samples
                    if prefix_value is not None:
                        stats["prefix_new_ratio"][layer, head] += compute_new_ratio(prefix_value[layer, head], basis) / num_samples
                if prefix_value is not None:
                    stats["prefix_vrank"][layer, head] = len(compute_basis(prefix_value[layer, head], config.energy_ratio))

        hidden_rank = compute_hidden_rank(hiddens, config.mass_ratio)
        result = {"hidden_rank": hidden_rank} | {name: stat.tolist() for name, stat in stats.items()}
        save_result("vrank", target, file_name, result)


if __name__ == "__main__":
    main()
