#!/usr/bin/env bash
# Raw EAT with the authors' precomputed embeddings of all Table 1 pLMs, Euclidean and cosine.
# Usage: WORK=/path/to/scratch bash run.sh   (CPU is enough, ~12 GB of scratch space)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL="$HERE/../01_prott5_cath_reproduction/evaluate.py"
WORK="${WORK:?set WORK to a scratch directory}"
EAT_COMMIT=b82ea03688a894d3717019972a455b55b785cd53
ZENODO=https://zenodo.org/api/records/14675997/files
mkdir -p "$WORK" && cd "$WORK"

[ -d EAT ] || git clone -q https://github.com/Rostlab/EAT
git -C EAT checkout -q "$EAT_COMMIT"
[ -f cath-domain-list.txt ] || curl -sL -o cath-domain-list.txt "$ZENODO/cath-domain-list.txt/content"
if [ ! -d embeddings ]; then
    curl -sL -C - --retry 5 -o emb.tar.gz "$ZENODO/prottucker_training_embeddings.tar.gz/content"
    tar xzf emb.tar.gz && rm emb.tar.gz
fi

EMB=(
    ProSE-DLM=embeddings/proseDLM_cathS100.h5
    ESM-1b=embeddings/ESM1b_cath_S100.h5
    ProtBERT=embeddings/protBERT_bfd_half_cath_S100.h5
    ProtT5=embeddings/prott5_cath_S100.h5
    ProSE-MT=embeddings/proseMT_cathS100.h5
)
for metric in euclidean cosine; do
    python "$EVAL" --eat-dir EAT --labels cath-domain-list.txt --metric "$metric" \
        --embeddings "${EMB[@]}" --out "$HERE/results/$metric.csv"
done
