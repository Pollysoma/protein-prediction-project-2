#!/usr/bin/env bash
# Reproduce raw ProtT5 EAT and the MMseqs2 baseline on CATH test219 vs lookup69k.
# Usage: WORK=/path/to/scratch bash run.sh   (needs a CUDA GPU, ~10 GB of scratch space)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:?set WORK to a scratch directory}"
EAT_COMMIT=b82ea03688a894d3717019972a455b55b785cd53
ZENODO=https://zenodo.org/api/records/14675997/files
mkdir -p "$WORK" && cd "$WORK"

# 1. Code and data: Rostlab/EAT (sets) and Zenodo record 14675997 (labels, authors' embeddings)
[ -d EAT ] || git clone -q https://github.com/Rostlab/EAT
git -C EAT checkout -q "$EAT_COMMIT"
SETS=EAT/data/ProtTucker
[ -f cath-domain-list.txt ] || wget -q -O cath-domain-list.txt "$ZENODO/cath-domain-list.txt/content"
if [ ! -f embeddings/prott5_cath_S100.h5 ]; then
    wget -q -O prottucker_training_embeddings.tar.gz "$ZENODO/prottucker_training_embeddings.tar.gz/content"
    tar xzf prottucker_training_embeddings.tar.gz embeddings/prott5_cath_S100.h5
fi
md5sum cath-domain-list.txt  # expected 6df99c76b576d7fda8c47f0d43187ce1

# 2. Our own ProtT5 embeddings
for set in lookup69k test219; do
    [ -f "prott5_ours_$set.h5" ] || python "$HERE/embed_prott5.py" "$SETS/$set.fasta" "prott5_ours_$set.h5"
done
python - <<'EOF'
import h5py
with h5py.File("prott5_ours.h5", "w") as out:
    for name in ["prott5_ours_lookup69k.h5", "prott5_ours_test219.h5"]:
        with h5py.File(name) as f:
            for key in f:
                if key not in out:
                    out.create_dataset(key, data=f[key][()])
EOF

# 3. MMseqs2: single-sequence search, highest sensitivity, hits up to E-value 10
[ -x mmseqs/bin/mmseqs ] || { wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz && tar xzf mmseqs-linux-avx2.tar.gz; }
MMSEQS=mmseqs/bin/mmseqs
$MMSEQS version
mkdir -p mm
$MMSEQS createdb "$SETS/test219.fasta" mm/query >/dev/null
$MMSEQS createdb "$SETS/lookup69k.fasta" mm/lookup >/dev/null
$MMSEQS search mm/query mm/lookup mm/result mm/tmp -s 7.5 -e 10 >mm/search.log
$MMSEQS convertalis mm/query mm/lookup mm/result mm/hits.m8 \
    --format-output query,target,evalue,bits,pident,alnlen >/dev/null

# 4. Scoring with the original EAT evaluation code
python "$HERE/evaluate.py" --eat-dir EAT --labels cath-domain-list.txt \
    --embeddings prott5_authors=embeddings/prott5_cath_S100.h5 prott5_ours=prott5_ours.h5 \
    --mmseqs-m8 mm/hits.m8 --out "$HERE/results/cath_test219.csv"
