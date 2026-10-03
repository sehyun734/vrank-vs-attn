from dataclasses import dataclass
import math
import random
import time

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import PreTrainedModel, get_cosine_schedule_with_warmup

from primitives.model import make_prefix_cache
from primitives.utils import format_duration, format_memory


@dataclass
class TrainConfig:
    num_epochs: int = 3
    batch_size: int = 128
    micro_size: int = 4
    learning_rate: float = 1e-3
    warmup_ratio: float = 0.05


def backward(
    model: PreTrainedModel,
    batch: list[tuple[Tensor, Tensor]],
    micro_size: int,
    sink: Tensor | None = None,
    prefix_mlp: Module | None = None,
) -> float:
    num_answer_tokens = sum(len(answer) for _, answer in batch)
    total_loss = 0.0

    for offset in range(0, len(batch), micro_size):
        micro_batch = batch[offset : offset + micro_size]
        input_ids = [torch.cat([question, answer]) for question, answer in micro_batch]
        labels = [torch.cat([torch.full_like(question, -100), answer]) for question, answer in micro_batch]
        input_ids = pad_sequence(input_ids, batch_first=True).to(model.device)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(model.device)

        cache = make_prefix_cache(sink, prefix_mlp(), len(micro_batch)) if prefix_mlp is not None else None
        loss = model(input_ids=input_ids, labels=labels, past_key_values=cache, num_items_in_batch=num_answer_tokens).loss
        loss.backward()
        total_loss += loss.item()

    return total_loss


def train(
    model: PreTrainedModel,
    samples: list[tuple[Tensor, Tensor]],
    params: list[Parameter],
    config: TrainConfig,
    sink: Tensor | None = None,
    prefix_mlp: Module | None = None,
) -> None:
    num_batches = math.ceil(len(samples) / config.batch_size)
    num_steps = num_batches * config.num_epochs
    # prefix_mlp의 마지막 층과 lora의 b는 0에서 시작해 커져야 함.
    # weight_decay는 이를 0 쪽으로 당겨 수렴을 늦추므로 쓰지 않음.
    optimizer = AdamW(params, lr=config.learning_rate, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, round(num_steps * config.warmup_ratio), num_steps)

    log_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(config.num_epochs):
        random.shuffle(samples)
        for step, offset in enumerate(range(0, len(samples), config.batch_size)):
            batch = samples[offset : offset + config.batch_size]
            loss = backward(model, batch, config.micro_size, sink, prefix_mlp)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 16 == 0:
                print(
                    f"epoch={epoch + 1}/{config.num_epochs} "
                    f"step={step + 1}/{num_batches} "
                    f"loss={loss:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} "
                    f"dt={format_duration(time.perf_counter() - log_time)} "
                    f"mem={format_memory(torch.cuda.max_memory_allocated())}",
                    flush=True,
                )
                log_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
