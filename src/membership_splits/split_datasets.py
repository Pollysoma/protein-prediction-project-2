from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def stratified_split(df: pd.DataFrame, *, bin_col: str = "bin", train_frac: float = 0.8, seed: int = 123):
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for _, group in df.groupby(bin_col):
        idx = np.arange(len(group))
        rng.shuffle(idx)
        cutoff = int(np.ceil(train_frac * len(group)))
        train_parts.append(group.iloc[idx[:cutoff]])
        test_parts.append(group.iloc[idx[cutoff:]])
    return pd.concat(train_parts), pd.concat(test_parts)


def run(args: argparse.Namespace) -> None:
    base = Path(args.s_members_dir)
    val_base = Path(args.s_members_val_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    S1 = pd.read_parquet(base / "S1.parquet")
    S2 = pd.read_parquet(base / "S2.parquet")
    S3 = pd.read_parquet(val_base / "S3.parquet")
    N = pd.read_parquet(args.annotated_n)

    for df in (S1, S2, S3, N):
        df["bin"] = df["kingdom"].astype(str) + "|" + df["length_bin"].astype(int).astype(str)

    A = pd.concat(
        [
            S1.assign(label=1)[["sequence", "length", "length_bin", "kingdom", "bin", "label"]],
            N[["sequence", "length", "length_bin", "kingdom", "bin"]].assign(label=0),
        ],
        ignore_index=True,
    )
    A_train, A_test = stratified_split(A, train_frac=args.train_frac, seed=args.seed)

    B = pd.concat([S2.assign(label=1), S3.assign(label=0)], ignore_index=True)
    B_train, B_test = stratified_split(B, train_frac=args.train_frac, seed=args.seed)

    A_train.to_parquet(out_dir / "A_train.parquet", index=False)
    A_test.to_parquet(out_dir / "A_test.parquet", index=False)
    B_train.to_parquet(out_dir / "B_train.parquet", index=False)
    B_test.to_parquet(out_dir / "B_test.parquet", index=False)
    print(f"Saved splits to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-members-dir", type=Path, default=Path("outputs_samples"))
    parser.add_argument("--s-members-val-dir", type=Path, default=Path("outputs_samples_val"))
    parser.add_argument("--annotated-n", type=Path, default=Path("outputs/annotated_N.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_splits"))
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
