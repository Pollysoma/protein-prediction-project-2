#!/usr/bin/env bash
# Survey the candidate pool for a CATH calibration set.
#
# Pool = CATH-S100 v4.3 minus lookup69k, test300 and val200. For every candidate
# we measure sequence similarity to lookup69k (how remote it is, i.e. whether it
# resembles the test219 queries) and to test219 (leakage into the final test),
# with the same MMseqs2 iterative profile search the paper used for redundancy
# reduction. Nothing is sampled or frozen here; this only produces the numbers
# the construction rules need.
#
# Usage: WORK=/path/to/scratch bash survey_pool.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:?set WORK to a scratch directory}"
THREADS="${THREADS:-64}"
EAT_COMMIT=b82ea03688a894d3717019972a455b55b785cd53
CATH=https://download.cathdb.info/cath/releases/all-releases/v4_3_0/sequence-data/cath-domain-seqs-S100-v4_3_0.fa
mkdir -p "$WORK" && cd "$WORK"

[ -d EAT ] || git clone -q https://github.com/Rostlab/EAT
git -C EAT checkout -q "$EAT_COMMIT"
SETS=EAT/data/ProtTucker
[ -f cath-domain-seqs-S100.fa ] || curl -sL -o cath-domain-seqs-S100.fa "$CATH"

# Candidate pool with sequences from the CATH release.
python3 - "$SETS" <<'EOF'
import sys
from pathlib import Path

sets = Path(sys.argv[1])
excluded = set()
for name in ("lookup69k.fasta", "test300.fasta", "val200.fasta"):
    excluded |= {l[1:].split()[0] for l in open(sets / name) if l.startswith(">")}

kept = 0
with open("cath-domain-seqs-S100.fa") as src, open("pool.fasta", "w") as out:
    write = False
    for line in src:
        if line.startswith(">"):
            # header: >cath|4_3_0|107lA00/1-162
            domain = line[1:].split("|")[2].split("/")[0]
            write = domain not in excluded
            kept += write
            if write:
                out.write(f">{domain}\n")
        elif write:
            out.write(line)
print(f"pool: {kept} domains")
EOF

MMSEQS="${MMSEQS:-mmseqs/bin/mmseqs}"
[ -x "$MMSEQS" ] || { curl -sLO https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz && tar xzf mmseqs-linux-avx2.tar.gz; }
mkdir -p survey
$MMSEQS createdb pool.fasta survey/pool >/dev/null
$MMSEQS createdb "$SETS/lookup69k.fasta" survey/lookup >/dev/null
$MMSEQS createdb "$SETS/test219.fasta" survey/test219 >/dev/null

# Same settings as the paper's redundancy reduction: iterative profile search,
# highest sensitivity, bidirectional coverage.
for target in lookup test219; do
    $MMSEQS search survey/pool "survey/$target" "survey/res_$target" "survey/tmp_$target" \
        --num-iterations 3 -s 7.5 --cov-mode 0 -e 10 --threads "$THREADS" > "survey/search_$target.log"
    $MMSEQS convertalis survey/pool "survey/$target" "survey/res_$target" "survey/hits_$target.m8" \
        --format-output query,target,fident,alnlen,evalue,bits --threads "$THREADS" >/dev/null
done

mkdir -p "$HERE/results"
for target in lookup test219; do
    python3 "$HERE/hval.py" "survey/hits_$target.m8" --queries pool.fasta --out "survey/hval_$target.csv"
done
python3 "$HERE/summarise_pool.py" --hval-lookup survey/hval_lookup.csv --hval-test219 survey/hval_test219.csv \
    --labels EAT/data/cath_v430_dom_seqs_S100_161121_labels.txt --lookup "$SETS/lookup69k.fasta" --out-dir "$HERE/results"
