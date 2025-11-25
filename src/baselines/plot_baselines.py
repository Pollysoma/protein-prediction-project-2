from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


PAIR_TO_DIFFICULTY = {
    "N_vs_S1": "easy",
    "S2_vs_S3": "medium",
    "S4_vs_U4": "hard",
}


def load_metrics(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, float]]]]:
    """Return (models, metrics[model][difficulty][metric] = (mean,std)), keeping random baseline if present."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    models: List[str] = []
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for pair, model_stats in raw.items():
        difficulty = PAIR_TO_DIFFICULTY.get(pair, pair)
        for model_name, stat in model_stats.items():
            summary = stat.get("summary", {})
            if model_name not in models:
                models.append(model_name)
            metrics.setdefault(model_name, {}).setdefault(difficulty, {})
            for metric_name, vals in summary.items():
                metrics[model_name][difficulty][metric_name] = (vals.get("mean"), vals.get("std"))
    return models, metrics


def plot_metric(
    metric: str,
    models: List[str],
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
    difficulties: Tuple[str, str, str] = ("easy", "medium", "hard"),
):
    x = np.arange(len(models))
    width = 0.2
    offsets = np.linspace(-width, width, num=len(difficulties))

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, diff in zip(offsets, difficulties):
        means = []
        stds = []
        for model in models:
            diff_stats = metrics.get(model, {}).get(diff, {})
            mean, std = diff_stats.get(metric, (np.nan, np.nan))
            means.append(mean)
            stds.append(std)
        ax.bar(x + offset, means, width=width, yerr=stds, label=diff, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric} by model and difficulty")
    ax.legend(title="difficulty")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot baseline metrics with error bars.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/baselines_nv_8M.json"),
        help="Path to JSON results file produced by run_baselines.py",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/plots"),
        help="Directory to write plots into",
    )
    args = parser.parse_args()

    models, metric_table = load_metrics(args.results)
    metrics_to_plot = ["auc", "balanced_accuracy", "precision", "recall"]

    for metric in metrics_to_plot:
        out_path = args.outdir / f"{metric}.png"
        plot_metric(metric, models, metric_table, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
