from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple

from .streaming import ParquetShardStreamer


@dataclass
class ScanReport:
    total_sequences: int
    elapsed_seconds: float
    throughput: float


def count_ur50_clusters(
    streamer: ParquetShardStreamer,
    *,
    limit: Optional[int] = None,
) -> Tuple[Counter, ScanReport]:
    """
    Scan shards and count how often each UR50 cluster occurs. Returns the
    counts and a ScanReport summarizing runtime so we can estimate future runs.
    """
    counter: Counter = Counter()
    start = time.perf_counter()

    processed = 0
    for processed, record in enumerate(streamer, start=1):
        counter[record.ur50_id or "UNKNOWN"] += 1
        if limit and processed >= limit:
            break

    elapsed = time.perf_counter() - start
    processed = min(processed, streamer.total_rows) if processed else 0
    report = ScanReport(
        total_sequences=processed,
        elapsed_seconds=elapsed,
        throughput=processed / elapsed if elapsed and processed else 0.0,
    )
    return counter, report
