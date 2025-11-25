# Membership splits – reproducible recipe

This README documents how to regenerate the final stratified splits in `outputs_splits_final/` (N, S1, S2, S3, S4, U4) using the helper CLIs and scripts in this repo. All paths are workspace‑relative.

## Prerequisites
- Python 3.10+ with `pandas`, `pyarrow`, `requests`, `tqdm` installed (e.g., `pip install pandas pyarrow requests tqdm`).
- Train shards directory (HF/FASTA parquet shards) providing ~180M sequences with columns: `sequence`, `length`, `kingdom`, `ur50_id`, `ur90_id`, `shard`, `row_index`.
- Annotated N set at `outputs/annotated_N.parquet` (produced by `dataset_preparation/annotate_n_kingdoms.py`).

## Length bins and bin targets
1) Compute length bins and N distribution:
```bash
python -m membership_splits.cli compute-n-bins \
  --annotated outputs/annotated_N.parquet \
  --length-bins-out outputs_precheck/n_length_bins.json \
  --bin-dist-out outputs_precheck/n_bin_dist.json
```
2) Derive integer per-bin targets for each split size (used here: S1=1,237; S2/S3/S4/U4=12,370):
```bash
python -m membership_splits.cli derive-bin-targets \
  --distribution outputs_precheck/n_bin_dist.json \
  --target S1=1237 --target S2=12370 --target S3=12370 --target S4=12370 --target U4=12370 \
  --output outputs_precheck/bin_targets.json
```

## Sample S1, S2, S3, S4
Use the row-group streaming sampler (progress bars enabled by default). Skip UPI UR50s and keep a disallow list to avoid reusing clusters across refills.
```bash
# S1 + S2 (train subset)
python -m membership_splits.cli sample-members \
  --shard-dir <train_shards_dir> \
  --length-bins outputs_precheck/n_length_bins.json \
  --bin-targets outputs_precheck/bin_targets.json \
  --datasets S1 S2 \
  --output-dir outputs_samples \
  --stats-out outputs_samples/sampling_stats.json \
  --skip-upi-ur50

# S3 (validation negatives) – same targets as S2, saved separately
python -m membership_splits.cli sample-members \
  --shard-dir <train_shards_dir> \
  --length-bins outputs_precheck/n_length_bins.json \
  --bin-targets outputs_precheck/bin_targets.json \
  --datasets S3 \
  --output-dir outputs_samples_val \
  --skip-upi-ur50

# S4 (final version after refills to match bins)
python -m membership_splits.cli sample-members \
  --shard-dir <train_shards_dir> \
  --length-bins outputs_precheck/n_length_bins.json \
  --bin-targets outputs_precheck/bin_targets.json \
  --datasets S4 \
  --output-dir outputs_samples \
  --skip-upi-ur50 \
  --preload-used outputs_samples_refill/disallow_ur50_all.txt
```
The resulting files used for the final splits are:
- `outputs_samples/S1.parquet` (1,237)
- `outputs_samples/S2.parquet` (12,370)
- `outputs_samples_val/S3.parquet` (12,370 after the small manual refill in `outputs_samples_val/S3_refill_manual.parquet`)
- `outputs_samples/S4_final.parquet` (12,370; matches bin targets)

## Build U4 (homolog negatives for S4)
1) Map S4 UR50 IDs to UniProtKB accessions (UniProt ID mapping service). Save raw pairs and a log:
```bash
python -m membership_splits.cli build-u4-idmap \
  --s4 outputs_samples/S4_final.parquet \
  --length-bins outputs_precheck/n_length_bins.json \
  --output outputs_samples/U4_idmap_raw.parquet \
  --log-output outputs_samples/U4_idmap_full_log.parquet \
  --idmap-batch 500 --meta-batch 80 --skip-meta
```
2) Fetch UniProt metadata in parallel (accession, length, lineage, sequence):
```bash
python fetch_u4_meta_parallel.py  # reads U4_idmap_raw.parquet, writes outputs_samples/U4_metadata.tsv
```
3) Pair S4 with U4 buddies, enforcing same kingdom/length_bin, no sequence reuse, and train-disjointness via hashes:
```bash
python pair_u4_from_meta.py \
  # uses defaults:
  #   S4: outputs_samples/S4_final.parquet
  #   metadata: outputs_samples/U4_metadata.tsv
  #   length bins: outputs_precheck/n_length_bins.json
  #   disallow hashes: outputs_samples/train_u4_collision_hashes.txt
  # outputs:
  #   outputs_samples/U4.parquet
  #   outputs_samples/U4_log.parquet (matched/missed summary)
```
Helper artifacts:
- `outputs_samples_refill/disallow_ur50_all.txt`: UR50 clusters already used/failed during refills.
- `outputs_samples/train_u4_collision_hashes.txt`: sequence hashes seen in Train to keep U4 disjoint from Train.

## Final stratified splits (80/20)
Run the simple stratifier to mirror bin proportions for each dataset:
```bash
python - <<'PY'
import pandas as pd, numpy as np
from pathlib import Path
def stratified_split(df, bin_col='bin', train_frac=0.8, seed=123):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for _, g in df.groupby(bin_col):
        idx = np.arange(len(g)); rng.shuffle(idx)
        cut = int(np.ceil(train_frac * len(g)))
        train.append(g.iloc[idx[:cut]]); test.append(g.iloc[idx[cut:]])
    return pd.concat(train), pd.concat(test)

sources = {
    'N': Path('outputs/annotated_N.parquet'),
    'S1': Path('outputs_samples/S1.parquet'),
    'S2': Path('outputs_samples/S2.parquet'),
    'S3': Path('outputs_samples_val/S3.parquet'),
    'S4': Path('outputs_samples/S4_final.parquet'),
    'U4': Path('outputs_samples/U4.parquet'),
}
out = Path('outputs_splits_final'); out.mkdir(parents=True, exist_ok=True)
for name, path in sources.items():
    df = pd.read_parquet(path)
    if 'bin' not in df.columns:
        df = df.copy()
        df['bin'] = df['kingdom'].astype(str) + '|' + df['length_bin'].astype(int).astype(str)
    tr, te = stratified_split(df, train_frac=0.8, seed=123)
    tr.to_parquet(out / f'{name}_train.parquet', index=False)
    te.to_parquet(out / f'{name}_test.parquet', index=False)
PY
```
Resulting files in `outputs_splits_final/`:
- `N_train/test.parquet` (1,007 / 230)
- `S1_train/test.parquet` (1,007 / 230)
- `S2_train/test.parquet` (9,896 / 2,474)
- `S3_train/test.parquet` (9,896 / 2,474; Metazoa|10 count 8/2)
- `S4_train/test.parquet` (9,896 / 2,474)
- `U4_train/test.parquet` (9,896 / 2,474)

## Notes and sanity checks
- All datasets are stratified on `bin = kingdom|length_bin`.
- S4 is contained in Train by design; U4 is made disjoint from Train at the sequence level via hash filtering.
- Progress bars are shown by default for sampling, ID mapping, metadata fetch, and pairing.
- If any step fails mid-way, rerun: samplers and fetchers append or resume safely (driven by parquet/TSV presence and hash/disallow lists).
