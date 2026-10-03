from dataclasses import dataclass
import math
import re
import time

from simple_parsing import parse
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from primitives.data import load_gsm8k
from primitives.hub import get_repo_name, get_unbenched_paths, load_lora, load_prefix, save_json
from primitives.model import generate, hook_lora, load_model
from primitives.utils import format_duration


@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.2-3B"
    num_samples: int = 1319
    batch_size: int = 64
    max_gen_len: int = 512


def parse_answer(
    text: str,
) -> float | None:
    before, _, after = text.replace(",", "").partition("The answer is")
    numbers = re.findall(r"-?\d*\.?\d+", after)[:1] if after else re.findall(r"-?\d*\.?\d+", before)[-1:]
    return float(numbers[0]) if numbers else None


def bench_gsm8k(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[tuple[Tensor, float]],
    config: Config,
    sink: Tensor | None = None,
    prefix: Tensor | None = None,
) -> float:
    num_batches = math.ceil(len(samples) / config.batch_size)
    num_correct = 0

    log_time = time.perf_counter()
    for step, offset in enumerate(range(0, len(samples), config.batch_size)):
        batch = samples[offset : offset + config.batch_size]
        output_ids = generate(model, tokenizer, [question for question, _ in batch], config.max_gen_len, sink, prefix)
        texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        num_correct += sum(parse_answer(text) == answer for text, (_, answer) in zip(texts, batch))

        print(
            f"step={step + 1}/{num_batches} "
            f"accuracy={num_correct / (offset + len(batch)):.2%} "
            f"dt={format_duration(time.perf_counter() - log_time)}",
            flush=True,
        )
        log_time = time.perf_counter()

    return num_correct / len(samples)


def main() -> None:
    config = parse(Config)
    model, tokenizer = load_model(config.model_name)
    samples = load_gsm8k(tokenizer, config.num_samples)
    repo_name = get_repo_name(config.model_name)

    for path in get_unbenched_paths(repo_name):
        method = path.split("/")[0]
        if method == "prefix":
            sink, prefix = load_prefix(repo_name, path, model.device)
            accuracy = bench_gsm8k(model, tokenizer, samples, config, sink, prefix)
        elif method == "lora":
            handles = hook_lora(model, load_lora(repo_name, path, model.device))
            accuracy = bench_gsm8k(model, tokenizer, samples, config)
            for handle in handles:
                handle.remove()
        else:
            accuracy = bench_gsm8k(model, tokenizer, samples, config)

        print(f"accuracy={accuracy:.2%}", flush=True)
        save_json(repo_name, path, "gsm8k", {"accuracy": accuracy})


if __name__ == "__main__":
    main()
