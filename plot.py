from dataclasses import dataclass
import json
from pathlib import Path

from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from simple_parsing import parse

from primitives.hub import get_repo_name

METHOD_COLORS = {"prefix": "tab:blue", "lora": "tab:orange", "base": "tab:gray"}
STAT_NAMES = ["vrank", "new_ratio", "prefix_vrank", "prefix_new_ratio", "hidden_rank", "sink_ratio", "prefix_ratio", "entropy", "jsd"]


@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.2-1B"


def parse_run(
    path: str,
) -> dict:
    if path == "base":
        return {"method": "base", "size": 0, "learning_rate": None}
    method, run_name = path.split("/")
    run_config = dict(param.split("=") for param in run_name.split(","))
    size = int(run_config["prefix_len"] if method == "prefix" else run_config["rank"])
    return {"method": method, "size": size, "learning_rate": float(run_config["learning_rate"])}


def main() -> None:
    config = parse(Config)
    result_dir = Path(snapshot_download(get_repo_name(config.model_name), allow_patterns="*.json"))

    runs = []
    for run_dir in sorted(file.parent for file in result_dir.glob("**/gsm8k.json")):
        if not (run_dir / "vrank.json").exists() or not (run_dir / "attn.json").exists():
            continue
        path = str(run_dir.relative_to(result_dir))
        run = {"path": path} | parse_run(path)
        run |= json.loads((run_dir / "gsm8k.json").read_text())
        stats = json.loads((run_dir / "vrank.json").read_text())
        run |= {stat_name: float(np.mean(stat[-1])) if isinstance(stat, list) else stat for stat_name, stat in stats.items()}
        stats = json.loads((run_dir / "attn.json").read_text())
        run |= {stat_name: float(np.mean(stat)) for stat_name, stat in stats.items()}
        runs.append(run)
    base_accuracy = next((run["accuracy"] for run in runs if run["method"] == "base"), 0.0)
    for run in runs:
        if run["method"] != "base" and run["accuracy"] <= base_accuracy:
            print(f"exclude={run['path']} accuracy={run['accuracy']:.2%}", flush=True)
    runs = [run for run in runs if run["method"] == "base" or run["accuracy"] > base_accuracy]
    for run in runs:
        if run["method"] == "base":
            run["marker_size"], run["alpha"] = 90, 1.0
            continue
        others = [other for other in runs if other["method"] == run["method"]]
        sizes = sorted({other["size"] for other in others})
        learning_rates = sorted({other["learning_rate"] for other in others})
        run["marker_size"] = 20 + 70 * sizes.index(run["size"]) / max(len(sizes) - 1, 1)
        run["alpha"] = 0.25 + 0.75 * learning_rates.index(run["learning_rate"]) / max(len(learning_rates) - 1, 1)

    fig, axes = plt.subplots(2, 5, figsize=(25, 9), sharey=True, layout="constrained")
    axes[1, 4].axis("off")
    for ax, stat_name in zip(axes.flat, STAT_NAMES):
        stat_runs = [run for run in runs if stat_name in run]
        for run in stat_runs:
            ax.scatter(
                run[stat_name],
                run["accuracy"],
                s=run["marker_size"],
                color=METHOD_COLORS[run["method"]],
                alpha=run["alpha"],
                linewidth=0,
                zorder=3,
            )

        # 두 방법을 합치면 방법 간 차이가 상관처럼 보이므로 방법별로 따로 구함.
        # run이 적고 튀는 값이 있어 순위 기반인 spearman을 씀.
        rho_texts = []
        for method in ["prefix", "lora"]:
            method_runs = [run for run in stat_runs if run["method"] == method]
            if len(method_runs) > 2:
                rho = spearmanr([run[stat_name] for run in method_runs], [run["accuracy"] for run in method_runs]).statistic
                rho_texts.append(f"{method} spearman={rho:+.2f}")
        ax.set_title(", ".join(rho_texts), fontsize=11)
        ax.set_xlabel(stat_name)
        ax.margins(x=0.15, y=0.12)
        ax.grid(color="#e5e4df", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("gsm8k accuracy")
    for method, color in METHOD_COLORS.items():
        axes[0, 0].scatter([], [], s=36, color=color, label=method)
    prefix_sizes = sorted({run["size"] for run in runs if run["method"] == "prefix"})
    lora_sizes = sorted({run["size"] for run in runs if run["method"] == "lora"})
    for i, (prefix_size, lora_size) in enumerate(zip(prefix_sizes, lora_sizes)):
        marker_size = 20 + 70 * i / max(len(prefix_sizes) - 1, 1)
        axes[0, 0].scatter([], [], s=marker_size, color="tab:gray", label=f"prefix_len={prefix_size} / rank={lora_size}")
    fig.legend(
        loc="outside upper center",
        ncol=len(METHOD_COLORS) + len(prefix_sizes),
        frameon=False,
        title="size = prefix_len/rank, transparent = learning_rate",
    )
    fig.savefig(f"figures/{config.model_name.split('/')[-1]}-corr.png", dpi=200)


if __name__ == "__main__":
    main()
