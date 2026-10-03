from collections import defaultdict
from dataclasses import dataclass

from simple_parsing import parse
import torch

from primitives.capture import capture_acts
from primitives.data import load_metamathqa
from primitives.hub import get_repo_name, get_unprobed_paths, load_lora, load_prefix, save_json
from primitives.model import hook_lora, load_model
from primitives.stats import calc_basis, calc_entropy, calc_hidden_rank, calc_jsd, calc_new_ratio


@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.2-3B"
    num_samples: int = 256
    max_seq_len: int = 768
    mass_ratio: float = 0.9
    energy_ratio: float = 0.9


def main() -> None:
    config = parse(Config)
    model, tokenizer = load_model(config.model_name, attn_impl="eager")
    samples = load_metamathqa(tokenizer, config.num_samples, config.max_seq_len, held_out=True)
    num_samples = len(samples)
    num_query_tokens = sum(len(question) + len(answer) - 1 for question, answer in samples)
    base_values, base_attns, base_hiddens = capture_acts(model, samples)
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    num_heads = model.config.num_attention_heads
    repo_name = get_repo_name(config.model_name)

    for path in get_unprobed_paths(repo_name):
        method = path.split("/")[0]
        if method == "prefix":
            sink, prefix = load_prefix(repo_name, path, model.device)
            values, attns, hiddens = capture_acts(model, samples, sink, prefix)
            use_prefix = True
            prefix_value = prefix[:, 1, 0]
            prefix_len = prefix_value.shape[-2]
        elif method == "lora":
            use_prefix = False
            handles = hook_lora(model, load_lora(repo_name, path, model.device))
            values, attns, hiddens = capture_acts(model, samples)
            for handle in handles:
                handle.remove()
        else:
            use_prefix = False
            values, attns, hiddens = base_values, base_attns, base_hiddens
        vrank_stats = defaultdict(lambda: torch.zeros(num_layers, num_kv_heads))
        attn_stats = defaultdict(lambda: torch.zeros(num_layers, num_heads))

        for layer in range(num_layers):
            for head in range(num_kv_heads):
                # basis와 vrank는 sample별로 구해 평균.
                # 이는 각 sample마다 쓰는 주요 방향 basis가 다르기 때문.
                # value를 이어서 한 번에 계산하면 sample마다 다른 basis들이 섞임.
                # 그래서 vrank가 부풀려질 수 있음.
                for value, base_value in zip(values, base_values):
                    basis = calc_basis(base_value[layer, head], config.energy_ratio)
                    vrank_stats["vrank"][layer, head] += len(calc_basis(value[layer, head], config.energy_ratio)) / num_samples
                    vrank_stats["new_ratio"][layer, head] += calc_new_ratio(value[layer, head], basis) / num_samples
                    if use_prefix:
                        vrank_stats["prefix_new_ratio"][layer, head] += calc_new_ratio(prefix_value[layer, head], basis) / num_samples
                if use_prefix:
                    vrank_stats["prefix_vrank"][layer, head] = len(calc_basis(prefix_value[layer, head], config.energy_ratio))

            for head in range(num_heads):
                for attn, base_attn in zip(attns, base_attns):
                    if use_prefix:
                        # prefix에서는 맨 앞 cache sink가 sink 역할 가져갈 수 있음.
                        # 따라서 cache sink와 input bos를 더해 sink로 봄.
                        # 그래야 bos 하나가 sink인 base와 같은 기준으로 비교 가능.
                        sink_attn = attn[layer, head, 1:, 0]
                        bos_attn = attn[layer, head, 1:, prefix_len + 1]
                        prefix_attn = attn[layer, head, 1:, 1 : prefix_len + 1]
                        input_attn = attn[layer, head, 1:, prefix_len + 2 :]
                        attn_stats["sink_ratio"][layer, head] += (sink_attn + bos_attn).sum() / num_query_tokens
                        attn_stats["prefix_ratio"][layer, head] += prefix_attn.sum() / num_query_tokens
                    else:
                        bos_attn = attn[layer, head, 1:, 0]
                        input_attn = attn[layer, head, 1:, 1:]
                        attn_stats["sink_ratio"][layer, head] += bos_attn.sum() / num_query_tokens
                    base_input_attn = base_attn[layer, head, 1:, 1:]
                    attn_stats["entropy"][layer, head] += calc_entropy(input_attn).sum() / num_query_tokens
                    attn_stats["jsd"][layer, head] += calc_jsd(input_attn, base_input_attn).sum() / num_query_tokens

        hidden_rank = calc_hidden_rank(hiddens, config.mass_ratio)
        save_json(repo_name, path, "vrank", {"hidden_rank": hidden_rank} | {key: stat.tolist() for key, stat in vrank_stats.items()})
        save_json(repo_name, path, "attn", {key: stat.tolist() for key, stat in attn_stats.items()})


if __name__ == "__main__":
    main()
