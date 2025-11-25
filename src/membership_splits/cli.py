from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .bins import assign_length_bin, compute_bin_distribution, make_length_bins
from .idmapping_client import (
    _session,
    fetch_idmap_results,
    fetch_uniprot_metadata,
    stream_results,
    submit_id_mapping,
    wait_for_job,
)
from .sampler import (
    MultiDatasetSampler,
    SamplerConfig,
    load_bin_targets,
    load_length_edges,
)
from .stats import count_ur50_clusters
from .streaming import ParquetShardStreamer
from .taxonomy import kingdom_from_local, load_taxdump_minimal
from .uniref50_metadata import iter_uniref50_entries
from .u4_builder import U4Builder, fetch_uniref_cluster, extract_members
from .streaming import SequenceHasher


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".tsv"}:
        sep = "," if path.suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported extension for {path}")


def _parquet_counts_to_weights(counts_path: Path, output_path: Path) -> None:
    counts_path = counts_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(counts_path)
    schema = pf.schema_arrow.append(pa.field("weight", pa.float64()))
    writer = pq.ParquetWriter(output_path, schema, compression="snappy")
    try:
        for rg in tqdm(range(pf.num_row_groups), desc="ur50-weights", unit="rg"):
            table = pf.read_row_group(rg)
            counts = table.column("count").to_numpy(zero_copy_only=False)
            weights = np.divide(
                1.0,
                counts,
                out=np.zeros_like(counts, dtype=np.float64),
                where=counts > 0,
            )
            table = table.append_column("weight", pa.array(weights))
            writer.write_table(table)
    finally:
        writer.close()


def _table_counts_to_weights(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["count"].astype(float)
    df = df.copy()
    df["weight"] = np.where(counts > 0, 1.0 / counts, 0.0)
    return df


def _allocate_bin_targets(distribution: Dict[str, float], total: int) -> List[Tuple[str, int]]:
    weights = list(distribution.items())
    raw_counts = [(key, value * total) for key, value in weights]
    floors = [(key, int(np.floor(rc))) for key, rc in raw_counts]
    assigned = sum(count for _, count in floors)
    remainders = sorted(
        ((key, rc - int(np.floor(rc))) for (key, rc) in raw_counts),
        key=lambda x: x[1],
        reverse=True,
    )
    remaining = total - assigned
    counts = {key: count for key, count in floors}
    for key, _ in remainders:
        if remaining <= 0:
            break
        counts[key] += 1
        remaining -= 1
    if remaining != 0:
        raise ValueError("Failed to allocate counts to match total.")
    return sorted(counts.items())


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_compute_n_bins(args: argparse.Namespace) -> None:
    df = _load_table(args.annotated)
    lengths = df[args.length_column].astype(int).tolist()
    edges, ranges = make_length_bins(lengths, max_bins=args.max_bins)
    length_bins = [assign_length_bin(length, edges) for length in lengths]
    kingdoms = df[args.kingdom_column].tolist()
    distribution = compute_bin_distribution(kingdoms, length_bins)

    ranges_payload = [{"index": idx, "start": start, "end": end} for idx, (start, end) in enumerate(ranges)]
    dist_payload = {
        f"{kingdom}|{bin_idx}": value for (kingdom, bin_idx), value in distribution.items()
    }
    _write_json(
        args.length_bins_out,
        {
            "length_edges": edges,
            "bin_ranges": ranges_payload,
            "source": str(args.annotated),
        },
    )
    _write_json(
        args.bin_dist_out,
        {
            "distribution": dist_payload,
            "num_sequences": len(df),
            "kingdom_column": args.kingdom_column,
            "length_column": args.length_column,
        },
    )
    print(f"[compute-n-bins] Saved {args.length_bins_out} and {args.bin_dist_out}")


def cmd_scan_ur50(args: argparse.Namespace) -> None:
    streamer = ParquetShardStreamer(Path(args.shard_dir), show_progress=not args.quiet)
    counts, report = count_ur50_clusters(streamer, limit=args.limit)
    df = (
        pd.DataFrame(
            {"ur50_id": list(counts.keys()), "count": list(counts.values())}
        ).sort_values("count", ascending=False)
    )
    if args.counts_out:
        _write_table(Path(args.counts_out), df)
        print(f"[count-ur50-clusters] Wrote counts to {args.counts_out}")

    if args.stats_out:
        _write_json(
            Path(args.stats_out),
            {
                "total_sequences": report.total_sequences,
                "elapsed_seconds": report.elapsed_seconds,
                "throughput_per_second": report.throughput,
                "shard_dir": str(args.shard_dir),
                "limit": args.limit,
            },
        )
        print(f"[count-ur50-clusters] Stats -> {args.stats_out}")


def cmd_build_ur50_metadata(args: argparse.Namespace) -> None:
    taxa, names = load_taxdump_minimal(Path(args.taxdump))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("ur50_id", pa.string()),
            ("rep_id", pa.string()),
            ("tax_name", pa.string()),
            ("taxid", pa.int64()),
            ("length", pa.int64()),
            ("kingdom", pa.string()),
        ]
    )
    writer = pq.ParquetWriter(output_path, schema, compression="snappy")
    rows: List[Dict[str, object]] = []
    progress = tqdm(
        total=args.limit,
        desc="uniref50-metadata",
        unit="seq",
        disable=False,
    )
    try:
        for record in iter_uniref50_entries(Path(args.fasta), limit=args.limit):
            taxid = record.get("taxid")
            kingdom = "Unknown"
            if taxid and taxa:
                kingdom = kingdom_from_local(int(taxid), taxa, names)
            rows.append(
                {
                    "ur50_id": record.get("ur50_id"),
                    "rep_id": record.get("rep_id"),
                    "tax_name": record.get("tax_name"),
                    "taxid": taxid,
                    "length": record.get("length"),
                    "kingdom": kingdom,
                }
            )
            if len(rows) >= args.chunk_size:
                writer.write_table(pa.Table.from_pylist(rows))
                rows = []
            progress.update(1)
    finally:
        if rows:
            writer.write_table(pa.Table.from_pylist(rows))
        writer.close()
        progress.close()
    print(f"[build-ur50-metadata] Saved {output_path}")


def cmd_build_ur50_weights(args: argparse.Namespace) -> None:
    counts_path = Path(args.counts)
    output_path = Path(args.output)
    if counts_path.suffix == ".parquet":
        if output_path.suffix != ".parquet":
            raise ValueError("Parquet counts require a parquet output to preserve schema.")
        _parquet_counts_to_weights(counts_path, output_path)
    else:
        df = _load_table(counts_path)
        df = _table_counts_to_weights(df)
        _write_table(output_path, df)
    print(f"[build-ur50-weights] Weights written to {output_path}")


def cmd_derive_bin_targets(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.distribution).read_text())
    dist = payload["distribution"]
    targets = {}
    for target in args.target:
        name, value = target.split("=")
        counts = _allocate_bin_targets(dist, int(value))
        targets[name] = {key: count for key, count in counts}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"targets": targets, "distribution_source": str(args.distribution)}, indent=2))
    print(f"[derive-bin-targets] Saved {out}")


def cmd_sample_members(args: argparse.Namespace) -> None:
    length_edges = load_length_edges(Path(args.length_bins))
    bin_targets = load_bin_targets(Path(args.bin_targets), args.datasets)
    config = SamplerConfig(
        shard_dir=Path(args.shard_dir),
        length_edges=length_edges,
        bin_targets=bin_targets,
        datasets=args.datasets,
        metadata_path=Path(args.metadata),
        output_dir=Path(args.output_dir),
        shuffle_row_groups=not args.no_shuffle,
        seed=args.seed,
        flush_every=args.flush_every,
        limit=args.limit,
        preload_used=[Path(p) for p in (args.preload_used or [])],
        skip_upi_ur50=args.skip_upi_ur50,
    )
    sampler = MultiDatasetSampler(config)
    stats = sampler.run()
    _write_json(Path(args.stats_out), stats)
    print(f"[sample-members] Stats -> {args.stats_out}")


def cmd_build_u4(args: argparse.Namespace) -> None:
    length_edges = load_length_edges(Path(args.length_bins))
    s4 = pd.read_parquet(args.s4)
    # Prevent selecting identical sequences by seeding the hasher with S4.
    builder = U4Builder(
        taxdump=Path(args.taxdump),
        length_edges=length_edges,
        cache_path=Path(args.cache),
        seed=args.seed,
    )
    for seq in s4["sequence"]:
        builder.used.add(seq)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        out_path,
        pa.schema(
            [
                ("sequence", pa.string()),
                ("ur90_id", pa.string()),
                ("kingdom", pa.string()),
                ("length", pa.int64()),
                ("length_bin", pa.int32()),
                ("paired_ur90", pa.string()),
            ]
        ),
        compression="snappy",
    )
    written = 0
    progress = tqdm(total=len(s4), desc="u4-sampling", unit="s4")
    for idx, row in s4.iterrows():
        if args.limit and written >= args.limit:
            break
        match = builder.find_match(
            row["ur50_id"],
            row["kingdom"],
            int(row["length_bin"]),
        )
        if match:
            seq, kingdom, length, bin_idx = match
            writer.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "sequence": seq,
                            "ur90_id": row["ur90_id"],
                            "kingdom": kingdom,
                            "length": length,
                            "length_bin": bin_idx,
                            "paired_ur90": row["ur90_id"],
                        }
                    ]
                )
            )
            written += 1
        progress.update(1)
    writer.close()
    builder._save_cache()
    progress.close()
    print(f"[build-u4] Saved {written} homologs to {out_path}")


def _lineage_to_kingdom(lineage: str) -> str:
    """
    Best-effort kingdom classifier from a lineage string.
    """
    lower = lineage.lower()
    if "metazoa" in lower or "animalia" in lower:
        return "Metazoa"
    if "viridiplantae" in lower or "plantae" in lower:
        return "Viridiplantae"
    if "fungi" in lower:
        return "Fungi"
    if "bacteria" in lower:
        return "Bacteria"
    if "archaea" in lower:
        return "Archaea"
    if "viruses" in lower or "virus" in lower:
        return "Viruses"
    if "eukaryota" in lower:
        return "Other Eukaryota"
    return "Unknown"


def cmd_build_u4_idmap(args: argparse.Namespace) -> None:
    def chunked(seq: List[str], size: int) -> List[List[str]]:
        return [seq[i : i + size] for i in range(0, len(seq), size)]

    s4 = pd.read_parquet(args.s4)
    if args.limit:
        s4 = s4.iloc[: args.limit]
    length_edges = load_length_edges(Path(args.length_bins))

    ur50_ids = s4["ur50_id"].dropna().unique().tolist()
    if not ur50_ids:
        print("[build-u4-idmap] No ur50_ids found in S4")
        return

    # Step 1: submit ID mapping in chunks to avoid huge payloads.
    all_maps: List[pd.DataFrame] = []
    jobs = chunked(ur50_ids, args.idmap_batch)
    print(f"[build-u4-idmap] Submitting {len(ur50_ids)} UR50 IDs across {len(jobs)} job(s)...")
    for chunk in tqdm(jobs, desc="idmap-jobs", unit="job"):
        job_id = submit_id_mapping(chunk, from_ns="UniRef50", to_ns="UniProtKB")
        wait_for_job(job_id, poll=5.0, timeout=args.idmap_timeout)
        df_chunk = fetch_idmap_results(job_id, batch_size=args.idmap_page_size)
        all_maps.append(df_chunk)
    idmap_df = pd.concat(all_maps, ignore_index=True) if all_maps else pd.DataFrame(columns=["from", "to"])
    idmap_df.rename(columns={"from": "ur50_id", "to": "accession"}, inplace=True)

    if idmap_df.empty:
        print("[build-u4-idmap] No mapping results returned.")
        return

    if args.idmap_save:
        Path(args.idmap_save).parent.mkdir(parents=True, exist_ok=True)
        idmap_df.to_parquet(args.idmap_save, index=False)

    if args.skip_meta:
        # Produce a log with only mapping counts and early-exit.
        grouped_counts = idmap_df.groupby("ur50_id").size().to_dict()
        log_rows = []
        progress = tqdm(total=len(s4), desc="pairing-u4", unit="s4")
        for idx, row in s4.iterrows():
            ur50 = row["ur50_id"]
            cnt = grouped_counts.get(ur50, 0)
            status = "has_members" if cnt > 0 else "no_members"
            log_rows.append(
                {
                    "s4_row": int(idx),
                    "ur50_id": ur50,
                    "status": status,
                    "u4_accession": None,
                    "total_cluster_members": cnt,
                    "kingdom_matches": None,
                    "kingdom_length_matches": None,
                }
            )
            progress.update(1)
        progress.close()
        log_path = Path(args.log_output) if args.log_output else Path(args.output).with_name(Path(args.output).stem + "_log.parquet")
        pd.DataFrame(log_rows).to_parquet(log_path, index=False)
        print(f"[build-u4-idmap] Skip meta enabled. Log with counts -> {log_path}")
        return

    # Step 2: fetch UniProt metadata (length, lineage, sequence) for all member accessions.
    accessions = sorted(idmap_df["accession"].dropna().unique().tolist())
    meta_df = fetch_uniprot_metadata(
        accessions,
        fields="accession,length,lineage,sequence",
        batch_size=args.meta_batch,
        desc="uniprot-meta",
    )
    meta_df.rename(
        columns={
            "Entry": "accession",
            "Length": "length",
            "Taxonomic lineage": "lineage",
            "Sequence": "sequence",
        },
        inplace=True,
    )
    if meta_df.empty:
        print("[build-u4-idmap] UniProt metadata retrieval returned no rows.")
        return
    meta_df["kingdom"] = meta_df["lineage"].astype(str).apply(_lineage_to_kingdom)
    meta_df["length_bin"] = meta_df["length"].astype(int).apply(lambda x: assign_length_bin(x, length_edges))
    meta_lookup = meta_df.set_index("accession")

    # Fallback helpers
    taxa = names = None
    seq_used = SequenceHasher()
    for seq in s4["sequence"]:
        seq_used.add(seq)
    if args.fallback_uniref:
        taxa, names = load_taxdump_minimal(Path(args.taxdump))

    def fallback_from_uniref(cluster_id: str, kingdom: str, bin_idx: int) -> Optional[Dict]:
        if not args.fallback_uniref:
            return None
        members: List[Tuple[str, Optional[int]]] = []
        cursor = None
        while True:
            payload = fetch_uniref_cluster(_session(), cluster_id, cursor=cursor)
            if not payload:
                break
            chunk_members, next_cursor, size = extract_members(payload)
            members.extend(chunk_members)
            if size and len(members) >= size:
                break
            if next_cursor:
                cursor = next_cursor
            else:
                break
            time.sleep(0.05)
        if not members:
            return None
        random.shuffle(members)
        for seq, taxid in members:
            if seq_used.seen(seq):
                continue
            k = kingdom_from_local(int(taxid), taxa, names) if taxid and taxa else "Unknown"
            if k != kingdom:
                continue
            b = assign_length_bin(len(seq), length_edges)
            if b != bin_idx:
                continue
            seq_used.add(seq)
            return {
                "sequence": seq,
                "kingdom": k,
                "length": len(seq),
                "length_bin": b,
                "u4_accession": None,
            }
        return None

    # Step 3: choose a buddy per S4 sequence.
    grouped = idmap_df.groupby("ur50_id")["accession"].apply(list).to_dict()
    results = []
    log_rows = []
    progress = tqdm(total=len(s4), desc="pairing-u4", unit="s4")
    for idx, row in s4.iterrows():
        ur50 = row["ur50_id"]
        bin_idx = int(row["length_bin"])
        kingdom = row["kingdom"]
        candidates = grouped.get(ur50, [])
        total_cands = len(candidates)
        same_kingdom = 0
        same_kingdom_bin = 0
        buddy = None
        for acc in candidates:
            if acc not in meta_lookup.index:
                continue
            meta = meta_lookup.loc[acc]
            if meta["kingdom"] != kingdom:
                continue
            same_kingdom += 1
            if int(meta["length_bin"]) != bin_idx:
                continue
            same_kingdom_bin += 1
            buddy = meta
            break
        if buddy is None:
            # Try fallback via UniRef if enabled
            fb = fallback_from_uniref(ur50, kingdom, bin_idx)
            if fb:
                buddy = fb
                same_kingdom = 1
                same_kingdom_bin = 1
                total_cands = len(fb)

        if buddy is not None:
            seq_used.add(buddy["sequence"])
            results.append(
                {
                    "sequence": buddy["sequence"],
                    "ur50_id": ur50,
                    "kingdom": buddy["kingdom"],
                    "length": int(buddy["length"]),
                    "length_bin": int(buddy["length_bin"]),
                    "u4_accession": buddy["u4_accession"] if isinstance(buddy, dict) else buddy.name,
                    "s4_row": int(idx),
                }
            )
            log_rows.append(
                {
                    "s4_row": int(idx),
                    "ur50_id": ur50,
                    "status": "matched",
                    "u4_accession": buddy["u4_accession"] if isinstance(buddy, dict) else buddy.name,
                    "total_cluster_members": total_cands,
                    "kingdom_matches": same_kingdom,
                    "kingdom_length_matches": same_kingdom_bin,
                }
            )
        else:
            status = "no_members"
            if total_cands > 0 and same_kingdom == 0:
                status = "kingdom_mismatch"
            elif total_cands > 0 and same_kingdom > 0 and same_kingdom_bin == 0:
                status = "length_mismatch"
            log_rows.append(
                {
                    "s4_row": int(idx),
                    "ur50_id": ur50,
                    "status": status,
                    "u4_accession": None,
                    "total_cluster_members": total_cands,
                    "kingdom_matches": same_kingdom,
                    "kingdom_length_matches": same_kingdom_bin,
                }
            )
        progress.update(1)
    progress.close()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(out_path, index=False)
    log_path = Path(args.log_output) if args.log_output else out_path.with_name(out_path.stem + "_log.parquet")
    pd.DataFrame(log_rows).to_parquet(log_path, index=False)
    print(f"[build-u4-idmap] Wrote {len(results)} paired homologs to {out_path}")
    print(f"[build-u4-idmap] Log with reasons -> {log_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Membership split orchestration CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compute-n-bins
    p_bins = subparsers.add_parser(
        "compute-n-bins",
        help="Compute kingdom/length-bin distribution for annotated N sequences.",
    )
    p_bins.add_argument("--annotated", type=Path, required=True, help="Annotated N parquet/csv.")
    p_bins.add_argument("--kingdom-column", default="kingdom")
    p_bins.add_argument("--length-column", default="length")
    p_bins.add_argument("--max-bins", type=int, default=10)
    p_bins.add_argument(
        "--length-bins-out",
        type=Path,
        default=Path("outputs_precheck/n_length_bins.json"),
    )
    p_bins.add_argument(
        "--bin-dist-out",
        type=Path,
        default=Path("outputs_precheck/n_bin_distribution.json"),
    )
    p_bins.set_defaults(func=cmd_compute_n_bins)

    # count-ur50-clusters
    p_scan = subparsers.add_parser(
        "count-ur50-clusters",
        help="Iterate over HF train shards and count UR50 cluster frequencies.",
    )
    p_scan.add_argument("--shard-dir", type=Path, default=Path("data/hf_cache/train"))
    p_scan.add_argument("--counts-out", type=Path, help="Where to store the counts table (.csv or .parquet).")
    p_scan.add_argument(
        "--stats-out",
        type=Path,
        default=Path("outputs_precheck/ur50_scan_stats.json"),
    )
    p_scan.add_argument("--limit", type=int, help="Optional maximum number of sequences to process.")
    p_scan.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    p_scan.set_defaults(func=cmd_scan_ur50)

    # build-ur50-metadata
    p_meta = subparsers.add_parser(
        "build-ur50-metadata",
        help="Parse UniRef50 FASTA and produce a ur50_id -> metadata parquet.",
    )
    p_meta.add_argument("--fasta", type=Path, default=Path("data/raw/uniref50.fasta.gz"))
    p_meta.add_argument("--taxdump", type=Path, default=Path("data/raw/taxdump.tar.gz"))
    p_meta.add_argument("--output", type=Path, default=Path("data/cache/uniref50_metadata.parquet"))
    p_meta.add_argument("--chunk-size", type=int, default=50000)
    p_meta.add_argument("--limit", type=int, help="Optional maximum number of sequences to parse.")
    p_meta.set_defaults(func=cmd_build_ur50_metadata)

    # build-ur50-weights
    p_weights = subparsers.add_parser(
        "build-ur50-weights",
        help="Convert UR50 count tables into weight tables (weight = 1/count).",
    )
    p_weights.add_argument("--counts", type=Path, required=True, help="Path to count table (.parquet/.csv).")
    p_weights.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output table with weight column (format inferred from suffix).",
    )
    p_weights.set_defaults(func=cmd_build_ur50_weights)

    # derive-bin-targets
    p_targets = subparsers.add_parser(
        "derive-bin-targets",
        help="Compute integer per-bin targets for desired dataset sizes.",
    )
    p_targets.add_argument(
        "--distribution",
        type=Path,
        required=True,
        help="JSON file containing 'distribution' map (kingdom|bin -> proportion).",
    )
    p_targets.add_argument(
        "--target",
        action="append",
        required=True,
        help="Name=Size pair (e.g., S1=1237). Can be provided multiple times.",
    )
    p_targets.add_argument(
        "--output",
        type=Path,
        default=Path("outputs_precheck/bin_targets.json"),
    )
    p_targets.set_defaults(func=cmd_derive_bin_targets)

    # sample-members
    p_sample = subparsers.add_parser(
        "sample-members",
        help="Sample datasets from train shards using bin targets.",
    )
    p_sample.add_argument("--shard-dir", type=Path, default=Path("data/hf_cache/train"))
    p_sample.add_argument("--metadata", type=Path, default=Path("data/cache/uniref50_metadata.parquet"))
    p_sample.add_argument("--length-bins", type=Path, default=Path("outputs_precheck/n_length_bins.json"))
    p_sample.add_argument("--bin-targets", type=Path, default=Path("outputs_precheck/bin_targets.json"))
    p_sample.add_argument(
        "--datasets",
        nargs="+",
        default=["S1", "S2", "S4"],
        help="Datasets to sample (must exist in bin targets JSON).",
    )
    p_sample.add_argument("--output-dir", type=Path, default=Path("outputs_samples"))
    p_sample.add_argument("--stats-out", type=Path, default=Path("outputs_samples/sampling_stats.json"))
    p_sample.add_argument("--seed", type=int, default=123)
    p_sample.add_argument("--flush-every", type=int, default=1000)
    p_sample.add_argument("--no-shuffle", action="store_true", help="Disable row-group shuffling.")
    p_sample.add_argument("--limit", type=int, help="Optional limit on processed sequences (useful for dry runs).")
    p_sample.add_argument(
        "--preload-used",
        nargs="+",
        help="Parquet/CSV files with a 'sequence' column to mark as already used (avoids resampling collisions).",
    )
    p_sample.add_argument(
        "--skip-upi-ur50",
        action="store_true",
        help="Skip sequences whose ur50_id contains UPI (those clusters do not map cleanly to UniProtKB).",
    )
    p_sample.set_defaults(func=cmd_sample_members)

    # build-u4
    p_u4 = subparsers.add_parser(
        "build-u4",
        help="Build U4 homolog negatives for S4 using UniRef90 API.",
    )
    p_u4.add_argument("--s4", type=Path, default=Path("outputs_samples/S4.parquet"))
    p_u4.add_argument("--taxdump", type=Path, default=Path("data/raw/taxdump.tar.gz"))
    p_u4.add_argument("--length-bins", type=Path, default=Path("outputs_precheck/n_length_bins.json"))
    p_u4.add_argument("--cache", type=Path, default=Path("data/cache/uniref90_cache.json"))
    p_u4.add_argument("--output", type=Path, default=Path("outputs_samples/U4.parquet"))
    p_u4.add_argument("--seed", type=int, default=123)
    p_u4.add_argument("--limit", type=int, help="Optional number of homologs to generate (for smoke tests).")
    p_u4.set_defaults(func=cmd_build_u4)

    # build-u4-idmap
    p_u4map = subparsers.add_parser(
        "build-u4-idmap",
        help="Trial run: map S4 ur50_ids to UniProtKB and pick homologs locally from mapping results.",
    )
    p_u4map.add_argument("--s4", type=Path, default=Path("outputs_samples/S4.parquet"))
    p_u4map.add_argument("--length-bins", type=Path, default=Path("outputs_precheck/n_length_bins.json"))
    p_u4map.add_argument("--output", type=Path, default=Path("outputs_samples/U4_idmap_trial.parquet"))
    p_u4map.add_argument("--limit", type=int, help="Optional limit of S4 rows for the trial.")
    p_u4map.add_argument("--idmap-batch", type=int, default=500, help="UR50 IDs per idmapping job.")
    p_u4map.add_argument("--idmap-page-size", type=int, default=500, help="Page size when streaming idmapping results.")
    p_u4map.add_argument("--meta-batch", type=int, default=100, help="Accessions per UniProt metadata request.")
    p_u4map.add_argument("--idmap-timeout", type=int, default=1800, help="Timeout (s) for each idmapping job.")
    p_u4map.add_argument("--log-output", type=Path, help="Optional path for detailed trial log (defaults to output with _log).")
    p_u4map.add_argument("--taxdump", type=Path, default=Path("data/raw/taxdump.tar.gz"), help="NCBI taxdump for fallback kingdom lookup.")
    p_u4map.add_argument("--fallback-uniref", action="store_true", help="If set, try UniRef JSON members when ID mapping returns nothing.")
    p_u4map.add_argument("--skip-meta", action="store_true", help="Skip UniProt metadata fetch; just log presence/absence of mappings.")
    p_u4map.add_argument("--idmap-save", type=Path, help="Optional path to save raw ID-mapping results (ur50_id, accession).")
    p_u4map.set_defaults(func=cmd_build_u4_idmap)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
