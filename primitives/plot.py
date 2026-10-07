import json
from pathlib import Path

from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from primitives.hub import get_repo_name

METHOD_COLORS = {"prefix": "tab:blue", "lora": "tab:orange", "base": "tab:gray"}
GRID_COLOR = "#e5e4df"


def load_runs(
    model_name: str,
    result_name: str,
    file_name: str,
    gsm8k_file_name: str,
) -> list[dict]:
    allow_patterns = [f"gsm8k/*/{model_name}/*{gsm8k_file_name}.json", f"{result_name}/*/{model_name}/*{file_name}.json"]
    result_dir = Path(snapshot_download(get_repo_name(), allow_patterns=allow_patterns))
    runs = []
    for file in sorted(result_dir.glob(f"{result_name}/*/{model_name}/**/{file_name}.json")):
        target = str(file.parent.relative_to(result_dir / result_name))
        gsm8k_file = result_dir / "gsm8k" / target / f"{gsm8k_file_name}.json"
        if not gsm8k_file.exists():
            continue
        method, _, *names = target.split("/")
        run_config = dict(param.split("=") for param in names[0].split(",")) if names else {}
        run = {"target": target, "method": method, "size": int(run_config.get("prefix_len", run_config.get("rank", 0))), "learning_rate": float(run_config.get("learning_rate", 0))}
        run |= json.loads(gsm8k_file.read_text())
        run["stats"] = json.loads(file.read_text())
        runs.append(run)

    base_accuracy = next((run["accuracy"] for run in runs if run["method"] == "base"), 0.0)
    for run in runs:
        if run["method"] != "base" and run["accuracy"] <= base_accuracy:
            print(f"excluded {run['target']} accuracy {run['accuracy']:.2%}", flush=True)
    runs = [run for run in runs if run["method"] == "base" or run["accuracy"] > base_accuracy]
    for run in runs:
        if run["method"] == "base":
            run["marker_size"] = 90
            run["alpha"] = 1.0
            continue
        others = [other for other in runs if other["method"] == run["method"]]
        sizes = sorted({other["size"] for other in others})
        learning_rates = sorted({other["learning_rate"] for other in others})
        run["marker_size"] = 20 + 70 * sizes.index(run["size"]) / max(len(sizes) - 1, 1)
        run["alpha"] = 0.25 + 0.75 * learning_rates.index(run["learning_rate"]) / max(len(learning_rates) - 1, 1)
    return runs


def plot_correlations(
    runs: list[dict],
    stat_names: list[str],
    title: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(stat_names), figsize=(5 * len(stat_names), 5), sharey=True, layout="constrained")
    for ax, stat_name in zip(axes, stat_names):
        stat_runs = [run for run in runs if stat_name in run]
        for run in stat_runs:
            ax.scatter(run[stat_name], run["accuracy"], s=run["marker_size"], color=METHOD_COLORS[run["method"]], alpha=run["alpha"], linewidth=0, zorder=3)

        # 두 방법을 합치면 방법 간 차이가 상관처럼 보이므로 방법별로 따로 구함.
        # run이 적고 튀는 값이 있어 순위 기반인 spearman을 씀.
        rho_texts = []
        for method in ["prefix", "lora"]:
            method_runs = [run for run in stat_runs if run["method"] == method]
            if len(method_runs) > 2:
                rho = spearmanr([run[stat_name] for run in method_runs], [run["accuracy"] for run in method_runs]).statistic
                rho_texts.append(f"{method} spearman {rho:+.2f}")
        ax.set_title(", ".join(rho_texts), fontsize=11)
        ax.set_xlabel(stat_name)
        ax.margins(x=0.15, y=0.12)
        ax.grid(color=GRID_COLOR, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("gsm8k accuracy")
    for method, color in METHOD_COLORS.items():
        axes[0].scatter([], [], s=36, color=color, label=method)
    prefix_sizes = sorted({run["size"] for run in runs if run["method"] == "prefix"})
    lora_sizes = sorted({run["size"] for run in runs if run["method"] == "lora"})
    for i, (prefix_size, lora_size) in enumerate(zip(prefix_sizes, lora_sizes)):
        axes[0].scatter([], [], s=20 + 70 * i / max(len(prefix_sizes) - 1, 1), color=METHOD_COLORS["base"], label=f"prefix_len={prefix_size} / rank={lora_size}")
    fig.legend(loc="outside upper center", ncol=len(METHOD_COLORS) + len(prefix_sizes), frameon=False, title=f"{title} (size = prefix_len/rank, transparent = learning_rate)")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"saved {path}", flush=True)
