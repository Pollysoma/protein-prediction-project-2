BioNeMo ESM2nv Embeddings (8M-650M)
===================================

Run BioNeMo ESM2nv (8M, 35M, 150M, 650M) over all parquet splits in `outputs_splits_final` and emit safetensors embeddings.

Model sources
-------------
- From NGC (needs `NGC_CLI_API_KEY`): 8M `nvidia/clara/esm2nv8m:2.1`, 650M `nvidia/clara/esm2nv650m:2.1`.
- From Hugging Face (auto-download via `huggingface_hub`, optional `HF_TOKEN` if needed):
  - 35M: repo `nvidia/esm2_t12_35M_UR50D`, file `esm2_35m_checkpoint.tar.gz`
  - 150M: repo `nvidia/esm2_t30_150M_UR50D`, file `esm2_150m_checkpoint.tar.gz`
  - You can override by pre-placing the tar under `models/<size>/`, setting alt repos `HF_REPO_35M` / `HF_REPO_150M`, or pointing env vars to tar paths:
    - `HF_35M_TAR=/abs/path/to/esm2_35m_checkpoint.tar.gz`
    - `HF_150M_TAR=/abs/path/to/esm2_150m_checkpoint.tar.gz`

Prerequisites
-------------
- Docker with GPU support (NVIDIA Container Toolkit).
- For NGC downloads: export `NGC_CLI_API_KEY`.
- Enough disk for checkpoints (`models/`) and outputs (`outputs_splits_final/`).
- Run from the repo root (same level as `outputs_splits_final`).

Run (Linux/macOS shell)
-----------------------
```bash
export NGC_CLI_API_KEY=<your-ngc-key>   # needed for 8M/650M
# export HF_TOKEN=<hf-token-if-required> # optional, if the HF repos need auth
# export HF_REPO_35M=nvidia/esm2_t12_35M_UR50D  # optional override
# export HF_REPO_150M=nvidia/esm2_t30_150M_UR50D
# export HF_35M_TAR=/path/to/esm2_35m_checkpoint.tar.gz
# export HF_150M_TAR=/path/to/esm2_150m_checkpoint.tar.gz
chmod +x scripts/run_bionemo_embeddings.sh
./scripts/run_bionemo_embeddings.sh
```

Run (PowerShell)
----------------
```powershell
$env:NGC_CLI_API_KEY="<your-ngc-key>"  # needed for 8M/650M
$env:HF_TOKEN="<hf-token-if-required>" # optional
$env:HF_REPO_35M="nvidia/esm2_t12_35M_UR50D"   # optional override
$env:HF_REPO_150M="nvidia/esm2_t30_150M_UR50D"
# $env:HF_35M_TAR="C:\path\to\esm2_35m_checkpoint.tar.gz"
# $env:HF_150M_TAR="C:\path\to\esm2_150m_checkpoint.tar.gz"
powershell -NoLogo -ExecutionPolicy Bypass -File ".\scripts\run_bionemo_embeddings.ps1"
```

What the scripts do
-------------------
- Pull the BioNeMo framework container `nvcr.io/nvidia/clara/bionemo-framework:2.7.1`.
- Download checkpoints (8M/650M from NGC; 35M/150M from local Hugging Face tarballs).
- Convert each `outputs_splits_final/*.parquet` to CSV with `sequences` and `accession` columns.
- Run `bionemo.esm2.scripts.infer_esm2` inside the container to produce embeddings.
- Convert predictions to safetensors with `ids_json` metadata.

Outputs
-------
- Embeddings: `outputs_splits_final/<file>_<size>_embeddings.safetensors` for each size present.
- Intermediate: `outputs_splits_final/<file>_<size>_pred/`
- Checkpoints: `models/<size>/unpacked/`

Notes
-----
- Default micro-batch sizes (for ~6 GB VRAM): 8M=8, 35M=4, 150M=2, 650M=1. Increase only if you have more VRAM.
- Edit the model list at the top of the scripts if you want to run a subset.
- Input parquet must have `sequence` or `sequences`; `accession` defaults to row index if missing.
