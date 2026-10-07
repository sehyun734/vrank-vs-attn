import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save
import torch
from torch import Tensor


def get_repo_name() -> str:
    return f"{HfApi().whoami()['name']}/vrank-vs-attn"


def save_prefix(
    path: str,
    sink: Tensor,
    prefix: Tensor,
) -> None:
    data = save({"sink": sink.detach().cpu().contiguous(), "prefix": prefix.detach().cpu().contiguous()})
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=data, path_in_repo=f"{path}/prefix.safetensors", repo_id=repo_name)
    print(f"saved {path}/prefix.safetensors", flush=True)


def load_prefix(
    path: str,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    tensors = load_file(hf_hub_download(get_repo_name(), f"{path}/prefix.safetensors"), device=str(device))
    return tensors["sink"], tensors["prefix"]


def save_lora(
    path: str,
    lora: dict[str, Tensor],
) -> None:
    data = save({name: tensor.detach().cpu().contiguous() for name, tensor in lora.items()})
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=data, path_in_repo=f"{path}/lora.safetensors", repo_id=repo_name)
    print(f"saved {path}/lora.safetensors", flush=True)


def load_lora(
    path: str,
    device: torch.device,
) -> dict[str, Tensor]:
    return load_file(hf_hub_download(get_repo_name(), f"{path}/lora.safetensors"), device=str(device))


def get_pending_targets(
    model_name: str,
    result_name: str,
    file_name: str,
) -> list[str]:
    files = set(HfApi().list_repo_files(get_repo_name()))
    targets = [f"base/{model_name}"]
    for file in sorted(files):
        if file.startswith((f"lora/{model_name}/", f"prefix/{model_name}/")) and file.endswith(".safetensors"):
            targets.append(str(Path(file).parent))
    targets = [target for target in targets if f"{result_name}/{target}/{file_name}.json" not in files]
    print(f"{result_name} pending {len(targets)}", *targets, sep="\n  ", flush=True)
    return targets


def save_result(
    result_name: str,
    target: str,
    file_name: str,
    result: dict,
) -> None:
    path = f"{result_name}/{target}/{file_name}.json"
    api = HfApi()
    repo_name = get_repo_name()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=json.dumps(result, indent=2).encode(), path_in_repo=path, repo_id=repo_name)
    print(f"saved {path}", flush=True)
