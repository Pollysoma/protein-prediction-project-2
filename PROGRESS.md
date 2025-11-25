# Progress Log

## Data and lookups
- Downloaded full `nvidia/esm2_uniref_pretraining_data` train/validation shards to `data/hf_cache/` (512 train parquet files, ~38.4 GB; 1 validation shard ~0.05 GB). Command (resumable): `uv run python -m membership_splits.download_train`.

## Phase 1 scaffolding (2025-11-20)
- Implemented the `membership_splits` package with FASTA parsing, reproducible length binning, UniProt batch mapper, hashed-sequence tracking, and a tqdm-backed parquet streamer for the 180M-sequence shards.
- Added `membership_splits.cli` with `compute-n-bins` and `count-ur50-clusters` subcommands plus `membership_splits.precheck_fullscan` for fast data quality checks (all commands emit ETA-aware progress bars). Sample validation runs:
  - `python -m membership_splits.precheck_fullscan --limit 10000` → `outputs_precheck/precheck.json` (length stats + top clusters).
  - `python -m membership_splits.cli count-ur50-clusters --limit 5000 --counts-out outputs_precheck/ur50_counts_sample.csv` → counts snapshot + `outputs_precheck/ur50_scan_stats.json`.

### Next executable subtasks
1. Run `annotate_n_kingdoms.py` to label the 1,237 N sequences (`outputs/annotated_N.parquet`) and feed that file to `membership_splits.cli compute-n-bins` to lock the binning scheme.
2. Extend the CLI with a job that builds `ur50` cluster weights for the full train split (streaming all shards, writing to parquet to avoid huge JSON).
3. Derive kingdom metadata for all `ur50` clusters referenced in T via a streamed pass over `data/raw/uniref50.fasta.gz` (store a cache for re-use in samplers).

## Phase 1 math + UR50 weights (2025-11-21)
- Added CLI helpers to convert massive UR50 count tables to weights without materializing them (`build-ur50-weights`) and to derive integer per-bin targets for any dataset size (`derive-bin-targets`). Outputs:
  - Train counts → weights: `outputs_precheck/train_ur50_weights.parquet` (63 row groups, snappy compressed) with stats captured in `outputs_precheck/train_ur50_scan_stats_full.json`.
  - Validation counts → weights: `outputs_precheck/val_ur50_weights.csv`.
  - Bin targets derived from the N distribution for S1=1,237 and S2/S3/S4=12,000 each: `outputs_precheck/bin_targets.json`.
- Downloaded `taxdump.tar.gz` and implemented `build-ur50-metadata` to stream `data/raw/uniref50.fasta.gz`, classify each UR50 cluster via the local taxonomy, and store the result in `data/cache/uniref50_metadata.parquet` (~59M rows, progress-barred run took ~17 minutes).
- Added `sample-members` CLI: loads the bin targets + length bins, shuffles row-groups from `data/hf_cache/train`, looks up kingdoms via the UR50 metadata cache, enforces disjointness via hashed sequences, and writes `S1/S2/S4` parquet outputs with progress reporting. The sampler is modular (configurable datasets, limits, flush size) and ready for a full production run once we decide the final quotas (currently set to 1,237 / 12,370).
- Sampled `S3` from validation using the same sampler into `outputs_samples_val/S3.parquet` (stats: `outputs_samples_val/sampling_stats.json`) and produced bin-stratified train/test splits:
  - Dataset A (S1 vs N): `outputs_splits/A_train.parquet`, `outputs_splits/A_test.parquet`.
  - Dataset B (S2 vs S3): `outputs_splits/B_train.parquet`, `outputs_splits/B_test.parquet`.
