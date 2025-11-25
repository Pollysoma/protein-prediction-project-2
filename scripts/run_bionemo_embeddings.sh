#!/usr/bin/env bash
set -euo pipefail

# Run BioNeMo ESM2nv inference (8M/35M/150M/650M) over all parquet files in outputs_splits_final.
# Requirements:
#   - Docker with GPU support (NVIDIA Container Toolkit)
#   - env NGC_CLI_API_KEY set to your NGC key (for 8M and 650M)
#   - For 35M and 150M, Hugging Face download via huggingface_hub (set HF_TOKEN if needed or place tarballs locally)
#   - Working directory: repo root (same level as outputs_splits_final)
#
# Outputs:
#   - models/<size>/unpacked/                 # downloaded checkpoints
#   - outputs_splits_final/<file>_<size>_embeddings.safetensors
#   - intermediate predictions in outputs_splits_final/<file>_<size>_pred/
#
# Usage:
#   export NGC_CLI_API_KEY=<your-key>
#   chmod +x scripts/run_bionemo_embeddings.sh
#   ./scripts/run_bionemo_embeddings.sh

IMAGE="nvcr.io/nvidia/clara/bionemo-framework:2.7.1"
NGC_ORG="nvidia"
WORKDIR="$(pwd)"
MODEL_DIR="${WORKDIR}/models"
DATA_DIR="${WORKDIR}/outputs_splits_final"

# Model list: size | source | source ref | checkpoint tarball name
#   source:
#     - ngc: download via NGC model ID (needs NGC_CLI_API_KEY)
#     - hf: download a tarball from Hugging Face (optionally set HF_TOKEN for private repos)
#     - local: read from local tarball (set env var to an absolute path or place file under models/<size>/)
MODELS=(
  "8M ngc nvidia/clara/esm2nv8m:2.1 esm2_8m_checkpoint.tar.gz"
  "650M ngc nvidia/clara/esm2nv650m:2.1 esm2_650m_checkpoint.tar.gz"
)

# Conservative micro-batch sizes for a ~6 GB GPU (e.g., RTX 4050). Increase if you have more VRAM.
mbs_for_size() {
  case "$1" in
    8M) echo 8 ;;
    35M) echo 4 ;;
    150M) echo 2 ;;
    650M) echo 1 ;;
    *) echo 1 ;;
  esac
}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

download_model() {
  local size="$1" source="$2" source_ref="$3" tarname="$4"
  local dest="${MODEL_DIR}/${size}"
  local unpack="${dest}/unpacked"
  if [[ -d "${unpack}" ]]; then
    log "Model ${size} already present at ${unpack}"
    return
  fi
  mkdir -p "${dest}"

  if [[ "${source}" == "ngc" ]]; then
    if [[ -z "${NGC_CLI_API_KEY:-}" ]]; then
      echo "NGC_CLI_API_KEY not set. Needed to fetch ${size} (${source_ref})." >&2
      exit 1
    fi
    log "Downloading ${source_ref} -> ${dest}"
    docker run --rm -e NGC_CLI_API_KEY --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      -v "${WORKDIR}:/host" "${IMAGE}" bash -lc "
        set -euo pipefail
        apt-get update >/dev/null && apt-get install -y unzip >/dev/null
        curl -fSL https://ngc.nvidia.com/downloads/ngccli_linux.zip -o /tmp/ngccli_linux.zip
        unzip -o /tmp/ngccli_linux.zip -d /tmp >/dev/null
        chmod +x /tmp/ngc-cli/ngc
        export NGC_CLI_ORG=${NGC_ORG}
        export NGC_CLI_ACCEPT_LICENSE=yes
        /tmp/ngc-cli/ngc registry model download-version ${source_ref} --dest /host/models/${size}
      "
  fi

  if [[ "${source}" == "hf" ]]; then
    local repo="${source_ref}"
    local repo_env="HF_REPO_${size}"
    if [[ -n "${!repo_env:-}" ]]; then
      repo="${!repo_env}"
    fi
    log "Downloading ${tarname} from Hugging Face repo ${repo}"
    python - "$repo" "$tarname" "$dest" <<'PY'
import os, shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id, filename, dest_dir = sys.argv[1:4]
token = os.environ.get("HF_TOKEN") or None
dest = Path(dest_dir)
dest.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(repo_id=repo_id, filename=filename, token=token, local_dir=None)
target = dest / Path(path).name
if Path(path).resolve() != target.resolve():
    shutil.copy2(path, target)
print(target)
PY
  fi

  local tar_path=""
  if [[ "${source}" == "local" ]]; then
    local override="${!source_ref:-}"
    if [[ -n "${override}" && -f "${override}" ]]; then
      tar_path="${override}"
    fi
  fi
  if [[ -z "${tar_path}" ]]; then
    tar_path="$(find "${dest}" -maxdepth 2 -type f -name "${tarname}" -print -quit)"
  fi
  if [[ -z "${tar_path}" ]]; then
    if [[ "${source}" == "hf" ]]; then
      echo "Checkpoint for ${size} not found. Ensure ${tarname} exists in ${dest} or override with HF_TOKEN if needed." >&2
    else
      echo "Checkpoint for ${size} not found. Download ${tarname} from Hugging Face, place it under ${dest}, or set ${source_ref}=/abs/path/to/${tarname}." >&2
    fi
    return 1
  fi

  mkdir -p "${unpack}"
  tar -xf "${tar_path}" -C "${unpack}"
  log "Unpacked ${size} to ${unpack}"
}

parquet_to_csv() {
  local parquet="$1"
  local csv="$2"
  python - "$parquet" "$csv" <<'PY'
import sys, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_parquet(src)
if 'sequence' in df.columns and 'sequences' not in df.columns:
    df = df.rename(columns={'sequence': 'sequences'})
if 'sequences' not in df.columns:
    raise SystemExit("Input needs a 'sequences' or 'sequence' column")
if 'accession' not in df.columns:
    df['accession'] = df.index.astype(str)
df[['sequences','accession']].to_csv(dst, index=False)
print(f"wrote {dst} rows={len(df)}")
PY
}

convert_predictions_to_safetensors() {
  local pred_dir="$1" out_path="$2" csv="$3"
  python - "$pred_dir" "$out_path" "$csv" <<'PY'
import sys, json, torch
from pathlib import Path
from safetensors.torch import save_file

pred_dir, out_path, csv_path = map(Path, sys.argv[1:4])
pt_files = sorted(pred_dir.glob("predictions__rank_*__dp_rank_*.pt"))
if not pt_files:
    raise SystemExit(f"No predictions__rank_* files in {pred_dir}")
data = torch.load(pt_files[0], map_location="cpu")
emb = data.get("embeddings")
if emb is None:
    raise SystemExit("No embeddings key in predictions file")
import pandas as pd
df = pd.read_csv(csv_path)
ids = df["accession"].tolist()
if emb.shape[0] != len(ids):
    raise SystemExit(f"Mismatch: embeddings {emb.shape[0]} vs ids {len(ids)}")
meta = {"ids_json": json.dumps(ids)}
save_file({"embeddings": emb}, out_path, metadata=meta)
print(f"saved {out_path} shape {tuple(emb.shape)}")
PY
}

run_infer() {
  local size="$1" ckpt_path="$2" csv="$3" pred_dir="$4" mbs="$5"
  mkdir -p "${pred_dir}"
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${WORKDIR}:/host" "${IMAGE}" bash -lc "
      set -euo pipefail
      python /workspace/bionemo2/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/infer_esm2.py \
        --checkpoint-path /host/${ckpt_path} \
        --data-path /host/${csv} \
        --results-path /host/${pred_dir} \
        --include-embeddings \
        --micro-batch-size ${mbs} \
        --precision bf16-mixed \
        --num-gpus 1 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1
    "
}

main() {
  mkdir -p "${MODEL_DIR}"
  mapfile -t parquet_files < <(find "${DATA_DIR}" -maxdepth 1 -type f -name "*.parquet" | sort)
  if [[ ${#parquet_files[@]} -eq 0 ]]; then
    echo "No parquet files in ${DATA_DIR}" >&2
    exit 1
  fi

  for entry in "${MODELS[@]}"; do
    read -r size source source_ref tarname <<<"${entry}"
    if ! download_model "${size}" "${source}" "${source_ref}" "${tarname}"; then
      log "Skipping ${size}; checkpoint missing (set ${source_ref} or place the tarball)."
      continue
    fi
    ckpt="models/${size}/unpacked"
    for pq in "${parquet_files[@]}"; do
      base="$(basename "${pq}" .parquet)"
      csv="${DATA_DIR}/${base}.csv"
      [[ -f "${csv}" ]] || parquet_to_csv "${pq}" "${csv}"
      pred_dir="${DATA_DIR}/${base}_${size}_pred"
      out_path="${DATA_DIR}/${base}_${size}_embeddings.safetensors"
      if [[ -f "${out_path}" ]]; then
        log "Skipping ${size} on ${base}; embeddings already exist at ${out_path}"
        continue
      fi
      mbs="$(mbs_for_size "${size}")"
      log "Running ${size} on ${base} (micro-batch ${mbs})"
      run_infer "${size}" "${ckpt}" "${csv}" "${pred_dir}" "${mbs}"
      convert_predictions_to_safetensors "${pred_dir}" "${out_path}" "${csv}"
    done
  done
}

main "$@"
