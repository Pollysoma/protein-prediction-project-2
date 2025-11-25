from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download nvidia/esm2_uniref_pretraining_data train/validation shards to data/hf_cache."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/hf_cache"),
        help="Destination directory for cached parquet shards.",
    )
    parser.add_argument(
        "--pattern",
        nargs="+",
        default=["train/*.parquet", "validation/*.parquet"],
        help="File patterns to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        "nvidia/esm2_uniref_pretraining_data",
        repo_type="dataset",
        local_dir=str(args.dest),
        local_dir_use_symlinks=False,
        allow_patterns=args.pattern,
    )
    print(f"Downloaded patterns {args.pattern} to {args.dest}")


if __name__ == "__main__":
    main()
