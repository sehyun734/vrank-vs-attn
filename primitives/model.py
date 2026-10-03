from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel, PreTrainedTokenizerBase


def load_model(
    model_name: str,
    attn_impl: str = "sdpa",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    # llama tokenizer는 pad 토큰이 없어 generate의 pad_token_id에서 오류.
    # 따라서 없으면 eos를 pad로 씀.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=attn_impl, dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    return model, tokenizer


def make_prefix_cache(
    sink: Tensor,
    prefix: Tensor,
    batch_size: int,
) -> DynamicCache:
    key_value = torch.cat([sink, prefix.to(sink.dtype)], dim=-2)
    return DynamicCache(ddp_cache_data=key_value.repeat_interleave(batch_size, dim=2))


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: list[Tensor],
    max_gen_len: int,
    sink: Tensor | None = None,
    prefix: Tensor | None = None,
) -> Tensor:
    cache = make_prefix_cache(sink, prefix, len(questions)) if prefix is not None else None
    cache_len = cache.get_seq_length() if cache is not None else 0
    input_ids = pad_sequence(questions, batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left")
    attn_mask = pad_sequence([torch.ones_like(question) for question in questions], batch_first=True, padding_side="left")
    input_ids = pad(input_ids, (cache_len, 0), value=tokenizer.pad_token_id)
    attn_mask = pad(attn_mask, (cache_len, 0), value=1)

    output_ids = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attn_mask.to(model.device),
        past_key_values=cache,
        max_new_tokens=max_gen_len,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        stop_strings=["\nQuestion:"],
        tokenizer=tokenizer,
    )
    return output_ids[:, input_ids.shape[1] :]


def hook_lora(
    model: PreTrainedModel,
    lora: dict[str, Tensor],
    scale: float = 1.0,
) -> list[RemovableHandle]:
    def make_hook(a: Tensor, b: Tensor) -> Callable:
        def hook(module: Module, args: tuple[Tensor], output: Tensor) -> Tensor:
            return output + args[0] @ a.to(output.dtype).T @ b.to(output.dtype).T * scale

        return hook

    handles = []
    for module_name in {key.rsplit(".", 1)[0] for key in lora}:
        hook = make_hook(lora[f"{module_name}.a"], lora[f"{module_name}.b"])
        handles.append(model.get_submodule(module_name).register_forward_hook(hook))
    return handles
