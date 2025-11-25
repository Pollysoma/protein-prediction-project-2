from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import pyarrow.parquet as pq
from tqdm.auto import tqdm


@dataclass(frozen=True)
class SequenceRecord:
    source: str
    shard_path: Path
    row_index: int
    sequence: str
    ur50_id: Optional[str]
    ur90_id: Optional[str]

    @property
    def length(self) -> int:
        return len(self.sequence)


class ParquetShardStreamer:
    """
    Iterate over gigantic parquet shards while keeping memory bounded.
    The class reports progress with an ETA so we never run blind jobs.
    """

    def __init__(
        self,
        shard_dir: Path,
        *,
        pattern: str = "*.parquet",
        columns: Sequence[str] = ("sequence", "ur50_id", "ur90_id"),
        show_progress: bool = True,
        shuffle_row_groups: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.shard_dir = shard_dir
        self.pattern = pattern
        self.columns = columns
        self.show_progress = show_progress
        self.shuffle_row_groups = shuffle_row_groups
        self.random = random.Random(seed)
        self.shards = sorted(self.shard_dir.glob(self.pattern))
        if not self.shards:
            raise FileNotFoundError(f"No shards matching {pattern} in {shard_dir}")

        self._shard_meta: List[pq.FileMetaData] = []
        total = 0
        for path in self.shards:
            file_meta = pq.ParquetFile(path).metadata
            self._shard_meta.append(file_meta)
            total += file_meta.num_rows
        self.total_rows = total

    def __iter__(self) -> Iterator[SequenceRecord]:
        progress = tqdm(
            total=self.total_rows,
            desc=f"scan:{self.shard_dir.name}",
            unit="seq",
            disable=not self.show_progress,
        )
        processed = 0
        for shard in self.shards:
            parquet = pq.ParquetFile(shard)
            for row_group in range(parquet.num_row_groups):
                table = parquet.read_row_group(row_group, columns=self.columns)
                seqs = table.column("sequence").to_pylist()
                ur50 = (
                    table.column("ur50_id").to_pylist()
                    if "ur50_id" in table.schema.names
                    else [None] * len(seqs)
                )
                ur90 = (
                    table.column("ur90_id").to_pylist()
                    if "ur90_id" in table.schema.names
                    else [None] * len(seqs)
                )
                records: List[SequenceRecord] = []
                for local_idx, seq in enumerate(seqs):
                    records.append(
                        SequenceRecord(
                            source=self.shard_dir.name,
                            shard_path=shard,
                            row_index=processed + local_idx,
                            sequence=seq,
                            ur50_id=ur50[local_idx],
                            ur90_id=ur90[local_idx],
                        )
                    )
                if self.shuffle_row_groups:
                    self.random.shuffle(records)
                for record in records:
                    yield record
                processed += len(records)
                progress.update(len(records))
        progress.close()


class SequenceHasher:
    """
    Hash-based set for enforcing global disjointness between sampled splits.
    Stores SHA1 hashes of sequences so memory stays bounded.
    """

    def __init__(self) -> None:
        self._hashes: set[str] = set()

    def _hash(self, sequence: str) -> str:
        return hashlib.sha1(sequence.encode("utf-8")).hexdigest()

    def add(self, sequence: str) -> None:
        self._hashes.add(self._hash(sequence))

    def seen(self, sequence: str) -> bool:
        return self._hash(sequence) in self._hashes

    def __len__(self) -> int:
        return len(self._hashes)
