from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np


def make_length_bins(
    lengths: Sequence[int],
    *,
    max_bins: int = 10,
    min_per_bin: int = 25,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Derive monotonically increasing bin edges from a set of sequence lengths.
    The first return value is a list of inclusive lower bounds. The second
    return value pairs the lower/upper bound of each interval for reporting.
    """
    if not lengths:
        return [0], []

    arr = np.asarray(lengths, dtype=np.int64)
    arr = np.sort(arr)
    if len(arr) <= min_per_bin:
        upper = int(arr[-1])
        return [0, upper + 1], [(0, upper)]

    quantiles = np.linspace(0.0, 1.0, num=max_bins + 1)
    raw_edges = np.quantile(arr, quantiles, method="nearest").astype(int).tolist()
    raw_edges[0] = 0

    edges = [raw_edges[0]]
    for edge in raw_edges[1:]:
        edges.append(max(edge, edges[-1] + 1))

    max_length = int(arr[-1])
    if edges[-1] <= max_length:
        edges.append(max_length + 1)

    bins: List[Tuple[int, int]] = []
    for i in range(len(edges) - 1):
        bins.append((edges[i], edges[i + 1] - 1))

    return edges, bins


def assign_length_bin(length: int, edges: Sequence[int]) -> int:
    """
    Map a length to a bin index given the inclusive edges returned by
    `make_length_bins`. The final bin is open-ended.
    """
    if not edges:
        return 0
    idx = np.searchsorted(edges, length, side="right") - 1
    if idx < 0:
        return 0
    if idx >= len(edges) - 1:
        return len(edges) - 2
    return idx


def compute_bin_distribution(
    kingdoms: Sequence[str],
    length_bins: Sequence[int],
) -> Dict[Tuple[str, int], float]:
    """
    Convert raw assignments into proportions keyed by (kingdom, length_bin).
    """
    if not kingdoms:
        return {}
    if len(kingdoms) != len(length_bins):
        raise ValueError("kingdoms and length_bins must have equal lengths")

    counts: Counter = Counter()
    for kingdom, bin_idx in zip(kingdoms, length_bins):
        counts[(kingdom, bin_idx)] += 1

    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}
