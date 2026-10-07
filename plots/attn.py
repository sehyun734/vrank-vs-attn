from dataclasses import dataclass
from pathlib import Path

import numpy as np
from simple_parsing import parse

from primitives.plot import load_runs, plot_correlations


@dataclass
class Config:
    model_name: str = "Llama-3.2-1B"
    gsm8k_num_samples: int = 1319
    num_samples: int = 256


def main() -> None:
    config = parse(Config)
    runs = load_runs(config.model_name, "attn", f"num_samples={config.num_samples}", f"num_samples={config.gsm8k_num_samples}")
    for run in runs:
        run |= {name: float(np.mean(stat)) for name, stat in run.pop("stats").items()}
    stat_names = ["sink_ratio", "prefix_ratio", "entropy", "jsd"]
    plot_correlations(runs, stat_names, f"{config.model_name} attn", Path(f"figures/{config.model_name}/attn.png"))


if __name__ == "__main__":
    main()
