from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .bins import assign_length_bin
from .streaming import ParquetShardStreamer, SequenceHasher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Ur50KingdomCache:
    """
    Loads the UR50 metadata parquet and keeps a lightweight dictionary
    mapping ur50_id -> kingdom. This currently loads everything into memory,
    so ensure the host has sufficient RAM. Future work: swap to an on-disk KV.
    """

    def __init__(self, metadata_path: Path) -> None:
        self.metadata_path = metadata_path
        self._cache: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        pf = pq.ParquetFile(self.metadata_path)
        progress = tqdm(
            range(pf.num_row_groups),
            desc="load-ur50-kingdom-cache",
            unit="rg",
        )
        for row_group in progress:
            table = pf.read_row_group(row_group, columns=["ur50_id", "kingdom"])
            ids = table.column("ur50_id").to_pylist()
            kingdoms = table.column("kingdom").to_pylist()
            for idx, ur50 in enumerate(ids):
                self._cache[ur50] = kingdoms[idx] or "Unknown"

    def lookup(self, ur50_id: Optional[str]) -> str:
        if not ur50_id:
            return "Unknown"
        return self._cache.get(ur50_id, "Unknown")


class DatasetWriter:
    def __init__(self, path: Path, flush_every: int = 1000) -> None:
        self.path = path
        self.flush_every = flush_every
        self.buffer: List[Dict[str, object]] = []
        self.schema = pa.schema(
            [
                ("dataset", pa.string()),
                ("sequence", pa.string()),
                ("ur50_id", pa.string()),
                ("ur90_id", pa.string()),
                ("kingdom", pa.string()),
                ("length", pa.int32()),
                ("length_bin", pa.int32()),
                ("shard", pa.string()),
                ("row_index", pa.int64()),
            ]
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(self.path, self.schema, compression="snappy")

    def write(self, row: Dict[str, object]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        self.writer.write_table(table)
        self.buffer = []

    def close(self) -> None:
        self.flush()
        self.writer.close()


class BinTargetTracker:
    def __init__(self, targets: Dict[str, Dict[str, int]]) -> None:
        self.targets = targets
        self.counts: Dict[str, Dict[str, int]] = {
            name: defaultdict(int) for name in targets
        }
        self.total_targets = {name: sum(bins.values()) for name, bins in targets.items()}
        self.total_counts = {name: 0 for name in targets}

    def needs(self, dataset: str, bin_key: str) -> bool:
        target = self.targets.get(dataset, {})
        if not target:
            return False
        return self.counts[dataset][bin_key] < target.get(bin_key, 0)

    def add(self, dataset: str, bin_key: str) -> None:
        self.counts[dataset][bin_key] += 1
        self.total_counts[dataset] += 1

    def dataset_done(self, dataset: str) -> bool:
        return self.total_counts[dataset] >= self.total_targets.get(dataset, 0)

    def all_done(self, datasets: List[str]) -> bool:
        return all(self.dataset_done(name) for name in datasets)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


@dataclass
class SamplerConfig:
    shard_dir: Path
    length_edges: List[int]
    bin_targets: Dict[str, Dict[str, int]]
    datasets: List[str]
    metadata_path: Path
    output_dir: Path
    shuffle_row_groups: bool = True
    seed: Optional[int] = 123
    flush_every: int = 1000
    limit: Optional[int] = None
    preload_used: List[Path] | None = None
    skip_upi_ur50: bool = False
    allowed_ur50_per_bin: Optional[Dict[str, set[str]]] = None
    disallow_ur50: Optional[set[str]] = None


class MultiDatasetSampler:
    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self.metadata = Ur50KingdomCache(config.metadata_path)
        self.targets = BinTargetTracker(config.bin_targets)
        self.sequence_hasher = SequenceHasher()
        self._preload_used()
        self.writers = {
            dataset: DatasetWriter(
                config.output_dir / f"{dataset}.parquet", flush_every=config.flush_every
            )
            for dataset in config.datasets
        }

    def _preload_used(self) -> None:
        if not self.config.preload_used:
            return
        for path in self.config.preload_used:
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                df = pd.read_parquet(path)
            elif suffix in {".csv", ".tsv"}:
                df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
            else:
                continue
            if "sequence" not in df.columns:
                continue
            for seq in df["sequence"]:
                self.sequence_hasher.add(seq)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()

    def run(self) -> Dict[str, object]:
        stats = {
            "assigned": {dataset: 0 for dataset in self.config.datasets},
            "skipped_seen": 0,
            "skipped_missing_bin": 0,
            "skipped_upi_ur50": 0,
            "skipped_disallowed_ur50": 0,
            "processed_sequences": 0,
        }

        streamer = ParquetShardStreamer(
            self.config.shard_dir,
            shuffle_row_groups=self.config.shuffle_row_groups,
            seed=self.config.seed,
        )

        try:
            for idx, record in enumerate(streamer, start=1):
                if self.targets.all_done(self.config.datasets):
                    break
                stats["processed_sequences"] = idx
                if self.config.limit and idx > self.config.limit:
                    break

                if self.config.skip_upi_ur50 and record.ur50_id and "UPI" in record.ur50_id:
                    stats["skipped_upi_ur50"] += 1
                    continue

                if self.config.disallow_ur50 and record.ur50_id in self.config.disallow_ur50:
                    stats.setdefault("skipped_disallow_ur50", 0)
                    stats["skipped_disallow_ur50"] += 1
                    continue

                if self.sequence_hasher.seen(record.sequence):
                    stats["skipped_seen"] += 1
                    continue

                length = len(record.sequence)
                bin_idx = assign_length_bin(length, self.config.length_edges)
                kingdom = self.metadata.lookup(record.ur50_id)
                bin_key = f"{kingdom}|{bin_idx}"

                allowed = self.config.allowed_ur50_per_bin
                if allowed is not None:
                    allowed_set = allowed.get(bin_key)
                    if not allowed_set or record.ur50_id not in allowed_set:
                        stats["skipped_disallowed_ur50"] += 1
                        continue

                assigned_dataset = None
                for dataset in self.config.datasets:
                    if self.targets.dataset_done(dataset):
                        continue
                    if self.targets.needs(dataset, bin_key):
                        assigned_dataset = dataset
                        break

                if not assigned_dataset:
                    stats["skipped_missing_bin"] += 1
                    continue

                self.sequence_hasher.add(record.sequence)
                self.targets.add(assigned_dataset, bin_key)
                stats["assigned"][assigned_dataset] += 1

                self.writers[assigned_dataset].write(
                    {
                        "dataset": assigned_dataset,
                        "sequence": record.sequence,
                        "ur50_id": record.ur50_id,
                        "ur90_id": record.ur90_id,
                        "kingdom": kingdom,
                        "length": length,
                        "length_bin": bin_idx,
                        "shard": record.shard_path.name,
                        "row_index": record.row_index,
                    }
                )

                if self.targets.all_done(self.config.datasets):
                    break
        finally:
            self.close()

        stats["targets"] = {
            dataset: self.targets.total_targets.get(dataset, 0)
            for dataset in self.config.datasets
        }
        return stats


def load_length_edges(path: Path) -> List[int]:
    payload = json.loads(path.read_text())
    return payload["length_edges"]


def load_bin_targets(path: Path, datasets: List[str]) -> Dict[str, Dict[str, int]]:
    payload = json.loads(path.read_text())
    targets = payload.get("targets", {})
    return {dataset: targets.get(dataset, {}) for dataset in datasets}
