from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from .streaming import ParquetShardStreamer


def run_precheck(
    shard_dir: Path,
    *,
    limit: int | None,
    top_k: int,
    show_progress: bool,
) -> dict:
    streamer = ParquetShardStreamer(shard_dir, show_progress=show_progress)
    counter: Counter[str] = Counter()
    lengths: list[int] = []

    start = time.perf_counter()
    for idx, record in enumerate(streamer, start=1):
        lengths.append(record.length)
        if record.ur50_id:
            counter[record.ur50_id] += 1
        if limit and idx >= limit:
            break
    elapsed = time.perf_counter() - start

    summary = {
        "processed_sequences": len(lengths),
        "elapsed_seconds": elapsed,
        "throughput_per_second": len(lengths) / elapsed if elapsed else 0.0,
    }

    if lengths:
        arr = np.asarray(lengths)
        summary["length_stats"] = {
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
    else:
        summary["length_stats"] = {}

    summary["unique_ur50_clusters"] = len(counter)
    summary["top_ur50_clusters"] = [
        {"ur50_id": key, "count": value} for key, value in counter.most_common(top_k)
    ]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick sanity scan over HF shards.")
    parser.add_argument("--shard-dir", type=Path, default=Path("data/hf_cache/train"))
    parser.add_argument("--limit", type=int, help="Optional cap on processed sequences.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs_precheck/precheck.json"),
        help="Where to store the JSON summary.",
    )
    parser.add_argument("--top-k", type=int, default=25, help="How many clusters to record.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_precheck(
        args.shard_dir,
        limit=args.limit,
        top_k=args.top_k,
        show_progress=not args.quiet,
    )
    summary["shard_dir"] = str(args.shard_dir)
    summary["limit"] = args.limit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"[precheck-fullscan] {summary['processed_sequences']} sequences -> {args.output}")


if __name__ == "__main__":
    main()
