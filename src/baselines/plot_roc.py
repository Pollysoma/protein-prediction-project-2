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


def load_results(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_mean_curve(fprs: List[List[float]], tprs: List[List[float]], grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not fprs or not tprs:
        return np.array([]), np.array([])
    interp = []
    for fpr, tpr in zip(fprs, tprs):
        fpr_arr = np.asarray(fpr)
        tpr_arr = np.asarray(tpr)
        # ensure monotonic and unique for interpolation
        order = np.argsort(fpr_arr)
        fpr_sorted = fpr_arr[order]
        tpr_sorted = tpr_arr[order]
        interp.append(np.interp(grid, fpr_sorted, tpr_sorted))
    interp = np.vstack(interp)
    return interp.mean(axis=0), interp.std(axis=0, ddof=1)


def plot_roc(results: Dict[str, dict], outdir: Path):
    grid = np.linspace(0, 1, 200)
    outdir.mkdir(parents=True, exist_ok=True)

    # Organize by difficulty -> model -> list of curves
    curves: Dict[str, Dict[str, Dict[str, List[List[float]]]]] = {}
    for pair, models in results.items():
        diff = PAIR_TO_DIFFICULTY.get(pair, pair)
        for model_name, stats in models.items():
            folds = stats.get("fold_metrics", [])
            fprs = []
            tprs = []
            for fold in folds:
                fpr = fold.get("fpr")
                tpr = fold.get("tpr")
                if fpr is None or tpr is None:
                    continue
                fprs.append(fpr)
                tprs.append(tpr)
            if not fprs:
                continue
            curves.setdefault(diff, {}).setdefault(model_name, {"fprs": [], "tprs": []})
            curves[diff][model_name]["fprs"].extend(fprs)
            curves[diff][model_name]["tprs"].extend(tprs)

    for diff, model_data in curves.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        for model_name, data in model_data.items():
            mean, std = aggregate_mean_curve(data["fprs"], data["tprs"], grid)
            if mean.size == 0:
                continue
            ax.plot(grid, mean, label=model_name)
            ax.fill_between(grid, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC - {diff}")
        ax.legend()
        ax.grid(alpha=0.4)
        fig.tight_layout()
        out_path = outdir / f"roc_{diff}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ROC curves (mean/std across folds) by difficulty.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/baselines_nv_8M.json"),
        help="Path to JSON results with per-fold fpr/tpr (use --include-curves when running baselines).",
    )
    parser.add_argument("--outdir", type=Path, default=Path("outputs/plots"), help="Directory for plots.")
    args = parser.parse_args()

    results = load_results(args.results)
    plot_roc(results, args.outdir)


if __name__ == "__main__":
    main()

