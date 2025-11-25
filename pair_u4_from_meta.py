from __future__ import annotations

"""
Pair S4_repaired with metadata fetched via fetch_u4_meta_parallel.py to build U4.

Inputs:
  - outputs_samples/S4_final.parquet
  - outputs_samples/U4_metadata.tsv  (accession, Length, Taxonomic lineage, Sequence)
Outputs:
  - outputs_samples/U4.parquet
  - outputs_samples/U4_log.parquet  (counts of matched/missed)
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from membership_splits.bins import assign_length_bin
from membership_splits.streaming import SequenceHasher

S4_PATH = Path("outputs_samples/S4_final.parquet")
META_PATH = Path("outputs_samples/U4_metadata.tsv")
LENGTH_BINS = Path("outputs_precheck/n_length_bins.json")
OUT_U4 = Path("outputs_samples/U4.parquet")
OUT_LOG = Path("outputs_samples/U4_log.parquet")
OUT_MISSES = Path("outputs_samples/U4_unmatched.parquet")
# Optional: hashes of sequences to avoid (e.g., sequences known to be in Train)
DISALLOW_HASHES = Path("outputs_samples/train_u4_collision_hashes.txt")


def lineage_to_kingdom(lineage: str) -> str:
    lower = lineage.lower()
    if "archaea" in lower:
        return "Archaea"
    # Check Archaea before Bacteria because many archaeal lineages contain
    # substrings like "methanobacteria" that would otherwise be mis-classified.
    if "metazoa" in lower or "animalia" in lower:
        return "Metazoa"
    if "viridiplantae" in lower or "plantae" in lower:
        return "Viridiplantae"
    if "fungi" in lower:
        return "Fungi"
    if "virus" in lower:
        return "Viruses"
    if "bacteria" in lower:
        return "Bacteria"
    if "eukaryota" in lower:
        return "Other Eukaryota"
    return "Unknown"


def main() -> None:
    s4 = pd.read_parquet(S4_PATH)
    meta = pd.read_csv(META_PATH, sep="\t")
    length_edges = json.loads(LENGTH_BINS.read_text())["length_edges"]

    # Preload any disallowed sequence hashes (e.g., sequences present in Train)
    disallow = set()
    if DISALLOW_HASHES.exists():
        for line in DISALLOW_HASHES.read_text().splitlines():
            line = line.strip()
            if line:
                disallow.add(line)

    meta["kingdom"] = meta["Taxonomic lineage"].astype(str).apply(lineage_to_kingdom)
    meta["length_bin"] = meta["Length"].astype(int).apply(lambda x: assign_length_bin(x, length_edges))
    meta_lookup = meta.set_index("Entry")

    # Build UR50 -> accessions map
    idmap = pd.read_parquet("outputs_samples/U4_idmap_raw.parquet")
    clusters = idmap.groupby("ur50_id")["accession"].apply(list).to_dict()

    used = SequenceHasher()
    for h in disallow:
        used._hashes.add(h)

    for seq in s4["sequence"]:
        used.add(seq)

    rows = []
    missed = 0
    miss_reasons: Counter[str] = Counter()
    miss_rows = []
    for row in tqdm(s4.itertuples(), total=len(s4), desc="pairing-u4", unit="s4"):
        ur50 = row.ur50_id
        kingdom = row.kingdom
        bin_idx = int(row.length_bin)
        cands = clusters.get(ur50, [])
        buddy = None

        if not cands:
            miss_reasons["no_cluster"] += 1
            missed += 1
            miss_rows.append(
                {"s4_idx": row.Index, "ur50_id": ur50, "kingdom": kingdom, "length_bin": bin_idx, "reason": "no_cluster"}
            )
            continue

        meta_rows = meta_lookup.reindex(cands)
        meta_rows = meta_rows.dropna(subset=["Length", "Sequence"], how="any")
        if meta_rows.empty:
            miss_reasons["no_metadata"] += 1
            missed += 1
            miss_rows.append(
                {"s4_idx": row.Index, "ur50_id": ur50, "kingdom": kingdom, "length_bin": bin_idx, "reason": "no_metadata"}
            )
            continue

        kingdom_rows = meta_rows[meta_rows["kingdom"] == kingdom]
        if kingdom_rows.empty:
            miss_reasons["kingdom_mismatch"] += 1
            missed += 1
            miss_rows.append(
                {
                    "s4_idx": row.Index,
                    "ur50_id": ur50,
                    "kingdom": kingdom,
                    "length_bin": bin_idx,
                    "reason": "kingdom_mismatch",
                    "cluster_size": len(meta_rows),
                }
            )
            continue

        bin_rows = kingdom_rows[kingdom_rows["length_bin"] == bin_idx]
        if bin_rows.empty:
            miss_reasons["length_bin_mismatch"] += 1
            missed += 1
            miss_rows.append(
                {
                    "s4_idx": row.Index,
                    "ur50_id": ur50,
                    "kingdom": kingdom,
                    "length_bin": bin_idx,
                    "reason": "length_bin_mismatch",
                    "cluster_size": len(meta_rows),
                    "kingdom_ok": len(kingdom_rows),
                }
            )
            continue

        duped = 0
        for acc, m in bin_rows.iterrows():
            seq = m["Sequence"]
            if used.seen(seq):
                duped += 1
                continue
            used.add(seq)
            buddy = {
                "sequence": seq,
                "ur50_id": ur50,
                "kingdom": m["kingdom"],
                "length": int(m["Length"]),
                "length_bin": int(m["length_bin"]),
                "u4_accession": acc,
                "s4_idx": int(row.Index),
            }
            break

        if buddy:
            rows.append(buddy)
        else:
            missed += 1
            miss_reasons["all_used"] += 1
            miss_rows.append(
                {
                    "s4_idx": row.Index,
                    "ur50_id": ur50,
                    "kingdom": kingdom,
                    "length_bin": bin_idx,
                    "reason": "all_used",
                    "cluster_size": len(meta_rows),
                    "kingdom_ok": len(kingdom_rows),
                    "bin_ok": len(bin_rows),
                    "dupe_candidates": duped,
                }
            )

    pd.DataFrame(rows).to_parquet(OUT_U4, index=False)
    log_row = {"matched": len(rows), "missed": missed}
    for reason, count in sorted(miss_reasons.items()):
        log_row[f"reason_{reason}"] = count
    pd.DataFrame([log_row]).to_parquet(OUT_LOG, index=False)
    if miss_rows:
        pd.DataFrame(miss_rows).to_parquet(OUT_MISSES, index=False)

    print(f"[pair] matched={len(rows)} missed={missed} -> {OUT_U4}")


if __name__ == "__main__":
    main()
