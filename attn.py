from collections import defaultdict
from dataclasses import dataclass

from simple_parsing import parse
import torch
from torch import Tensor
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


@torch.no_grad()
def capture_attns(
    model: PreTrainedModel,
    samples: list[tuple[Tensor, Tensor]],
    sink: Tensor | None = None,
    prefix: Tensor | None = None,
) -> list[Tensor]:
    attns = []
    for question_ids, answer_ids in samples:
        cache = make_prefix_cache(sink, prefix, 1) if prefix is not None else DynamicCache()
        input_ids = torch.cat([question_ids, answer_ids]).unsqueeze(0).to(model.device)
        output = model(input_ids=input_ids, past_key_values=cache, output_attentions=True)
        attns.append(torch.stack(output.attentions)[:, 0].cpu())
    return attns


def compute_entropy(
    prob: Tensor,
) -> Tensor:
    # sink나 prefix를 잘라낸 attn은 합이 1보다 작으므로 다시 정규화.
    prob = prob.float() / prob.float().sum(-1, keepdim=True)
    return -(prob * prob.clamp_min(1e-12).log()).sum(-1)


def compute_jsd(
    prob_a: Tensor,
    prob_b: Tensor,
) -> Tensor:
    prob_a = prob_a.float() / prob_a.float().sum(-1, keepdim=True)
    prob_b = prob_b.float() / prob_b.float().sum(-1, keepdim=True)
    return compute_entropy((prob_a + prob_b) / 2) - (compute_entropy(prob_a) + compute_entropy(prob_b)) / 2


def main() -> None:
    config = parse(Config)
    file_name = f"num_samples={config.num_samples}"
    targets = get_pending_targets(config.model_name, "attn", file_name)
    model, tokenizer = load_model(config.organization, config.model_name, attn_implementation="eager")
    samples = load_metamathqa(tokenizer, config.num_samples, config.max_sequence_len, held_out=True)

    num_query_tokens = sum(len(question_ids) + len(answer_ids) - 1 for question_ids, answer_ids in samples)
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    base_attns = capture_attns(model, samples)
    for target in targets:
        method = target.split("/")[0]
        if method == "prefix":
            sink, prefix = load_prefix(target, model.device)
            attns = capture_attns(model, samples, sink=sink, prefix=prefix)
            prefix_len = prefix.shape[-2]
        elif method == "lora":
            handles = hook_lora(model, load_lora(target, model.device))
            attns = capture_attns(model, samples)
            for handle in handles:
                handle.remove()
        elif method == "base":
            attns = base_attns

        stats = defaultdict(lambda: torch.zeros(num_layers, num_heads))
        for layer in range(num_layers):
            for head in range(num_heads):
                for attn, base_attn in zip(attns, base_attns):
                    if method == "prefix":
                        # prefix에서는 맨 앞 cache sink가 sink 역할 가져갈 수 있음.
                        # 따라서 cache sink와 input bos를 더해 sink로 봄.
                        # 그래야 bos 하나가 sink인 base와 같은 기준으로 비교 가능.
                        sink_attn = attn[layer, head, 1:, 0]
                        bos_attn = attn[layer, head, 1:, prefix_len + 1]
                        prefix_attn = attn[layer, head, 1:, 1 : prefix_len + 1]
                        input_attn = attn[layer, head, 1:, prefix_len + 2 :]
                        stats["sink_ratio"][layer, head] += (sink_attn + bos_attn).sum() / num_query_tokens
                        stats["prefix_ratio"][layer, head] += prefix_attn.sum() / num_query_tokens
                    else:
                        bos_attn = attn[layer, head, 1:, 0]
                        input_attn = attn[layer, head, 1:, 1:]
                        stats["sink_ratio"][layer, head] += bos_attn.sum() / num_query_tokens
                    base_input_attn = base_attn[layer, head, 1:, 1:]
                    stats["entropy"][layer, head] += compute_entropy(input_attn).sum() / num_query_tokens
                    stats["jsd"][layer, head] += compute_jsd(input_attn, base_input_attn).sum() / num_query_tokens

        result = {name: stat.tolist() for name, stat in stats.items()}
        save_result("attn", target, file_name, result)


if __name__ == "__main__":
    main()
