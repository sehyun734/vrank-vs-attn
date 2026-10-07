from dataclasses import dataclass
import math
import re
import time

from simple_parsing import parse

from primitives.dataset import load_gsm8k
from primitives.hub import get_pending_targets, load_lora, load_prefix, save_result
from primitives.model import generate, hook_lora, load_model
from primitives.utils import format_duration


@dataclass
class Config:
    organization: str = "meta-llama"
    model_name: str = "Llama-3.2-3B"
    num_samples: int = 1319
    batch_size: int = 64
    max_generation_len: int = 512


def parse_answer(
    text: str,
) -> float | None:
    before, _, after = text.replace(",", "").partition("The answer is")
    numbers = re.findall(r"-?\d*\.?\d+", after)[:1] if after else re.findall(r"-?\d*\.?\d+", before)[-1:]
    return float(numbers[0]) if numbers else None


def main() -> None:
    config = parse(Config)
    file_name = f"num_samples={config.num_samples}"
    targets = get_pending_targets(config.model_name, "gsm8k", file_name)
    model, tokenizer = load_model(config.organization, config.model_name)
    samples = load_gsm8k(tokenizer, config.num_samples)

    num_batches = math.ceil(len(samples) / config.batch_size)
    for target in targets:
        method = target.split("/")[0]
        if method == "prefix":
            sink, prefix = load_prefix(target, model.device)
            handles = []
        elif method == "lora":
            sink = None
            prefix = None
            handles = hook_lora(model, load_lora(target, model.device))
        elif method == "base":
            sink = None
            prefix = None
            handles = []

        corrects = []
        log_time = time.perf_counter()
        for step, offset in enumerate(range(0, len(samples), config.batch_size)):
            batch = samples[offset : offset + config.batch_size]
            output_ids = generate(model, tokenizer, [question_ids for question_ids, _ in batch], config.max_generation_len, sink=sink, prefix=prefix)
            texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            corrects.extend(parse_answer(text) == answer for text, (_, answer) in zip(texts, batch))

            print(f"batch {step + 1}/{num_batches} accuracy {sum(corrects) / len(corrects):.2%} time {format_duration(time.perf_counter() - log_time)}", flush=True)
            log_time = time.perf_counter()

        for handle in handles:
            handle.remove()
        accuracy = sum(corrects) / len(samples)
        print(f"{target} accuracy {accuracy:.2%}", flush=True)
        result = {"accuracy": accuracy}
        save_result("gsm8k", target, file_name, result)


if __name__ == "__main__":
    main()
