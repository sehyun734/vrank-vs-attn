import torch
from torch import Tensor
from transformers import DynamicCache, PreTrainedModel

from primitives.model import make_prefix_cache


@torch.no_grad()
def capture_acts(
    model: PreTrainedModel,
    samples: list[tuple[Tensor, Tensor]],
    sink: Tensor | None = None,
    prefix: Tensor | None = None,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    values = []
    attns = []
    hiddens = []
    for question, answer in samples:
        cache = make_prefix_cache(sink, prefix, 1) if prefix is not None else DynamicCache()
        cache_len = cache.get_seq_length()
        input_ids = torch.cat([question, answer]).unsqueeze(0).to(model.device)
        output = model(input_ids=input_ids, past_key_values=cache, output_attentions=True, output_hidden_states=True)
        values.append(torch.stack([layer.values[0, :, cache_len + 1 :] for layer in cache.layers]))
        attns.append(torch.stack(output.attentions)[:, 0].cpu())
        hiddens.append(output.hidden_states[-1][0, 1:].float().cpu())
    return values, attns, hiddens
