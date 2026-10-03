import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save
import torch
from torch import Tensor


def get_repo_name(
    model_name: str,
) -> str:
    return f"{HfApi().whoami()['name']}/{model_name.split('/')[-1]}-vrank-vs-attn"


def get_unprobed_paths(
    repo_name: str,
) -> list[str]:
    files = set(HfApi().list_repo_files(repo_name))
    paths = {"base"} | {str(Path(file).parent) for file in files if file.endswith(".safetensors")}
    paths = sorted(path for path in paths if f"{path}/vrank.json" not in files or f"{path}/attn.json" not in files)
    print(f"pending={len(paths)}", *paths, sep="\n  ", flush=True)
    return paths


def get_unbenched_paths(
    repo_name: str,
) -> list[str]:
    files = set(HfApi().list_repo_files(repo_name))
    paths = {"base"} | {str(Path(file).parent) for file in files if file.endswith(".safetensors")}
    paths = sorted(path for path in paths if f"{path}/gsm8k.json" not in files)
    print(f"pending={len(paths)}", *paths, sep="\n  ", flush=True)
    return paths


def save_prefix(
    repo_name: str,
    path: str,
    sink: Tensor,
    prefix: Tensor,
) -> None:
    data = save({"sink": sink.detach().cpu().contiguous(), "prefix": prefix.detach().cpu().contiguous()})
    api = HfApi()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=data, path_in_repo=f"{path}/prefix.safetensors", repo_id=repo_name)
    print(f"save={repo_name}/{path}/prefix.safetensors", flush=True)


def load_prefix(
    repo_name: str,
    path: str,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    file = hf_hub_download(repo_name, f"{path}/prefix.safetensors")
    tensors = load_file(file, device=str(device))
    return tensors["sink"], tensors["prefix"]


def save_lora(
    repo_name: str,
    path: str,
    lora: dict[str, Tensor],
) -> None:
    data = save({name: tensor.detach().cpu().contiguous() for name, tensor in lora.items()})
    api = HfApi()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=data, path_in_repo=f"{path}/lora.safetensors", repo_id=repo_name)
    print(f"save={repo_name}/{path}/lora.safetensors", flush=True)


def load_lora(
    repo_name: str,
    path: str,
    device: torch.device,
) -> dict[str, Tensor]:
    file = hf_hub_download(repo_name, f"{path}/lora.safetensors")
    return load_file(file, device=str(device))


def save_json(
    repo_name: str,
    path: str,
    file_name: str,
    data: dict,
) -> None:
    api = HfApi()
    api.create_repo(repo_name, exist_ok=True)
    api.upload_file(path_or_fileobj=json.dumps(data, indent=2).encode(), path_in_repo=f"{path}/{file_name}.json", repo_id=repo_name)
    print(f"save={repo_name}/{path}/{file_name}.json", flush=True)
