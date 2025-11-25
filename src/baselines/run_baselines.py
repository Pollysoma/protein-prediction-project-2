from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from safetensors import safe_open
import torch
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import auc, balanced_accuracy_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

try:
    from .simple_mlp import SimpleMLP
except Exception:  # pragma: no cover - optional torch baseline
    SimpleMLP = None


class RandomBaseline:
    """Produces random membership scores; useful sanity check."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng: Optional[np.random.Generator] = None

    def get_params(self, deep: bool = True):
        return {"seed": self.seed}

    def set_params(self, **params):
        if "seed" in params:
            self.seed = params["seed"]
        return self

    def fit(self, X, y):
        self._rng = np.random.default_rng(self.seed)
        return self

    def predict_proba(self, X):
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        probs = self._rng.random(len(X))
        return np.vstack([1 - probs, probs]).T

# Dataset pairs: (non_member_set, member_set)
PAIR_DEFINITIONS: Dict[str, Tuple[str, str]] = {
    "N_vs_S1": ("N", "S1"),   # new distribution vs member subset
    "S2_vs_S3": ("S3", "S2"),  # train vs validation distribution
    "S4_vs_U4": ("U4", "S4"),  # exact train vs hard homologs
}


def load_embeddings(path: Path) -> np.ndarray:
    """Load embeddings from a safetensors file as numpy array."""
    with safe_open(str(path), framework="pt", device="cpu") as f:
        emb = f.get_tensor("embeddings").to(dtype=torch.float32)
    return emb.cpu().numpy()


def build_features_labels(pair_name: str, embedding_size: str, root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load member/non-member embeddings for a given pair using training splits only."""
    if pair_name not in PAIR_DEFINITIONS:
        raise ValueError(f"Unknown pair '{pair_name}'. Choose from {list(PAIR_DEFINITIONS)}")
    non_member, member = PAIR_DEFINITIONS[pair_name]

    def path_for(split: str) -> Path:
        path = root / f"{split}_train_{embedding_size}_embeddings.safetensors"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing embeddings for '{split}' at {path}. "
                f"Run BioNeMo embedding generation first (see scripts/run_bionemo_embeddings.sh)."
            )
        return path

    X_non = load_embeddings(path_for(non_member))
    X_mem = load_embeddings(path_for(member))
    y_non = np.zeros(len(X_non), dtype=np.int64)
    y_mem = np.ones(len(X_mem), dtype=np.int64)
    X = np.concatenate([X_non, X_mem], axis=0)
    y = np.concatenate([y_non, y_mem], axis=0)
    return X, y


def build_dual_features_labels(
    pair_name: str, embedding_from: str, embedding_to: str, root: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load paired embeddings (from->to) for member/non-member train splits."""
    if pair_name not in PAIR_DEFINITIONS:
        raise ValueError(f"Unknown pair '{pair_name}'. Choose from {list(PAIR_DEFINITIONS)}")
    non_member, member = PAIR_DEFINITIONS[pair_name]

    def path_for(split: str, size: str) -> Path:
        path = root / f"{split}_train_{size}_embeddings.safetensors"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing embeddings for '{split}' size {size} at {path}. "
                f"Run BioNeMo embedding generation first (see scripts/run_bionemo_embeddings.sh)."
            )
        return path

    X_from_non = load_embeddings(path_for(non_member, embedding_from))
    X_from_mem = load_embeddings(path_for(member, embedding_from))
    X_to_non = load_embeddings(path_for(non_member, embedding_to))
    X_to_mem = load_embeddings(path_for(member, embedding_to))

    X_from = np.concatenate([X_from_non, X_from_mem], axis=0)
    X_to = np.concatenate([X_to_non, X_to_mem], axis=0)
    y = np.concatenate(
        [np.zeros(len(X_from_non), dtype=np.int64), np.ones(len(X_from_mem), dtype=np.int64)],
        axis=0,
    )
    return X_from, X_to, y


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Squash decision_function outputs into [0,1] to mimic probabilities."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    min_s, max_s = float(scores.min()), float(scores.max())
    if max_s - min_s < 1e-8:
        return np.full_like(scores, 0.5)
    return (scores - min_s) / (max_s - min_s)


def run_procrustes_ridge(
    X_from: np.ndarray,
    X_to: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    member_train_frac: float = 0.5,
    alpha: float = 1.0,
    random_state: int = 42,
) -> Dict[str, Dict[str, list]]:
    """Fit ridge mapping on a subset of member training data; score on unseen members/non-members."""
    results: Dict[str, Dict[str, list]] = {"procrustes_ridge": {"folds": []}}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_from, y), start=1):
        # Only members from the training fold are used to fit the mapping.
        member_train_indices = [i for i in train_idx if y[i] == 1]
        if len(member_train_indices) < 2:
            raise RuntimeError("Not enough member samples to train Procrustes mapping in this fold.")
        mem_train_fit, mem_train_holdout = train_test_split(
            member_train_indices,
            train_size=member_train_frac,
            shuffle=True,
            random_state=random_state + fold_idx,
        )

        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_from[mem_train_fit], X_to[mem_train_fit])

        # Evaluation members: only those not used for fitting (holdout + validation members)
        val_member_idx = [i for i in val_idx if y[i] == 1]
        eval_member_idx = np.array(mem_train_holdout + val_member_idx, dtype=np.int64)
        eval_nonmember_idx = np.array([i for i in val_idx if y[i] == 0], dtype=np.int64)

        eval_idx = np.concatenate([eval_nonmember_idx, eval_member_idx], axis=0)
        if eval_idx.size == 0:
            continue

        preds = ridge.predict(X_from[eval_idx])
        residuals = np.linalg.norm(preds - X_to[eval_idx], axis=1)
        # Higher score should mean more likely member -> negate residual
        scores = -residuals
        eval_trues = np.concatenate(
            [np.zeros(len(eval_nonmember_idx), dtype=np.int64), np.ones(len(eval_member_idx), dtype=np.int64)],
            axis=0,
        )
        results["procrustes_ridge"]["folds"].append(
            {"trues": eval_trues.tolist(), "preds": scores.tolist()}
        )
        print(f"Completed Procrustes fold {fold_idx}/{n_splits}")

    return results


def run_models(
    models: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, list]]:
    """Fit/evaluate models with stratified CV; returns aggregated truths/preds."""
    results: Dict[str, Dict[str, list]] = {name: {"folds": []} for name in models}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        for model_name, model in models.items():
            estimator = clone(model)
            estimator.fit(X_train, y_train)
            if hasattr(estimator, "predict_proba"):
                y_scores = estimator.predict_proba(X_val)[:, 1]
            else:
                y_scores = _normalize_scores(estimator.decision_function(X_val))
            results[model_name]["folds"].append(
                {"trues": y_val.tolist(), "preds": np.asarray(y_scores).tolist()}
            )
        print(f"Completed fold {fold_idx}/{n_splits}")

    return results


def evaluate_results(
    results: Dict[str, Dict[str, list]], include_curves: bool = False
) -> Dict[str, dict]:
    """Compute ROC-AUC and thresholded metrics per fold, plus mean/std."""
    eval_results: Dict[str, dict] = {}
    for model_name, res in results.items():
        fold_metrics = []
        for fold in res["folds"]:
            y_true = np.asarray(fold["trues"])
            y_scores = np.asarray(fold["preds"])
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            y_pred = (y_scores >= 0.5).astype(int)

            fold_stat = {
                "auc": float(roc_auc),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }
            if include_curves:
                fold_stat["fpr"] = fpr.tolist()
                fold_stat["tpr"] = tpr.tolist()
            fold_metrics.append(fold_stat)

        def summarize(metric: str) -> Dict[str, float]:
            vals = [fm[metric] for fm in fold_metrics]
            mean = float(np.mean(vals)) if vals else float("nan")
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            return {"mean": mean, "std": std}

        eval_results[model_name] = {
            "fold_metrics": fold_metrics,
            "summary": {
                "auc": summarize("auc"),
                "balanced_accuracy": summarize("balanced_accuracy"),
                "precision": summarize("precision"),
                "recall": summarize("recall"),
            },
        }
    return eval_results


def default_models(
    input_dim: int,
    include_mlp: bool = False,
    include_xgb: bool = False,
    include_random: bool = True,
) -> Dict[str, object]:
    """Return a small suite of baselines."""
    models: Dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=500, n_jobs=-1, class_weight="balanced"),
        "knn_cosine": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
    }
    if include_random:
        models["random"] = RandomBaseline()
    if include_mlp and SimpleMLP is not None:
        models["simple_mlp"] = SimpleMLP(input_size=input_dim, hidden_size=512, epochs=5)
    if include_xgb and XGBClassifier is not None:
        # Small tree ensemble; hist mode keeps it fast on CPU.
        models["xgboost"] = XGBClassifier(
            n_estimators=80,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
        )
    return models


def main():
    parser = argparse.ArgumentParser(
        description="Run membership baselines on NVIDIA BioNeMo embeddings (train splits only)."
    )
    parser.add_argument(
        "--embedding-size",
        type=str,
        default="8M",
        help="Embedding size token used in filenames (default: 8M BioNeMo outputs).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs_splits_final"),
        help="Directory containing *_train_<size>_embeddings.safetensors files.",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        choices=PAIR_DEFINITIONS.keys(),
        default=list(PAIR_DEFINITIONS.keys()),
        help="Dataset pairs to evaluate.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--include-mlp", action="store_true", help="Also run the small torch MLP baseline.")
    parser.add_argument("--include-xgb", action="store_true", help="Also run the XGBoost baseline (requires xgboost).")
    parser.add_argument("--no-random", action="store_true", help="Disable the random baseline.")
    parser.add_argument(
        "--include-procrustes",
        action="store_true",
        help="Run Procrustes-style ridge mapping (requires paired embeddings, e.g., 35M->8M).",
    )
    parser.add_argument(
        "--procrustes-from-size",
        type=str,
        default="35M",
        help="Source embedding size token (default: 35M).",
    )
    parser.add_argument(
        "--procrustes-to-size",
        type=str,
        default=None,
        help="Target embedding size token (default: uses --embedding-size).",
    )
    parser.add_argument(
        "--procrustes-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength for Procrustes mapping.",
    )
    parser.add_argument(
        "--procrustes-member-train-frac",
        type=float,
        default=0.5,
        help="Fraction of member training fold used to fit the mapping (rest is held out for scoring).",
    )
    parser.add_argument(
        "--include-curves",
        action="store_true",
        help="Include FPR/TPR curves in the JSON output (larger files).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write metrics (default: outputs/baselines_nv_<size>.json).",
    )

    args = parser.parse_args()
    root = args.root
    if not root.exists():
        raise SystemExit(f"Embeddings directory not found: {root}")

    pair_metrics: Dict[str, dict] = {}
    for pair_name in args.pairs:
        print(f"\n=== {pair_name} (embedding size: {args.embedding_size}) ===")
        X, y = build_features_labels(pair_name, args.embedding_size, root)
        models = default_models(
            X.shape[1],
            include_mlp=args.include_mlp,
            include_xgb=args.include_xgb,
            include_random=not args.no_random,
        )
        results = run_models(models, X, y, n_splits=args.folds)
        stats = evaluate_results(results, include_curves=args.include_curves)

        # Optional Procrustes baseline
        if args.include_procrustes:
            target_size = args.procrustes_to_size or args.embedding_size
            try:
                X_from, X_to, y_dual = build_dual_features_labels(
                    pair_name, args.procrustes_from_size, target_size, root
                )
                proc_results = run_procrustes_ridge(
                    X_from,
                    X_to,
                    y_dual,
                    n_splits=args.folds,
                    member_train_frac=args.procrustes_member_train_frac,
                    alpha=args.procrustes_alpha,
                )
                proc_stats = evaluate_results(proc_results, include_curves=args.include_curves)
                stats.update(proc_stats)
            except FileNotFoundError as e:
                print(f"Skipping Procrustes for {pair_name}: {e}")
            except RuntimeError as e:
                print(f"Skipping Procrustes for {pair_name}: {e}")

        pair_metrics[pair_name] = stats
        for model_name, stat in stats.items():
            s = stat["summary"]
            print(
                f"{model_name}: "
                f"AUC={s['auc']['mean']:.4f}+/-{s['auc']['std']:.4f} "
                f"bal_acc={s['balanced_accuracy']['mean']:.4f}+/-{s['balanced_accuracy']['std']:.4f} "
                f"precision={s['precision']['mean']:.4f}+/-{s['precision']['std']:.4f} "
                f"recall={s['recall']['mean']:.4f}+/-{s['recall']['std']:.4f}"
            )

    out_path = args.output_json or Path("outputs") / f"baselines_nv_{args.embedding_size}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(pair_metrics, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
