#!/usr/bin/env bash
# Run the CATH v4.4 pilot split on the LMU GPU server and copy summaries back.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RESULTS="$HERE/results"
HVAL="$ROOT/experiments/03_calibration_set/hval.py"
PILOT="$HERE/pilot_split.py"
REMOTE_USER=ra62ler
REMOTE_HOST=aranuka.plai.ifi.lmu.de
REMOTE_PORT=54321
REMOTE_SCRATCH=/tmp/ra62ler_pp2/pilot_split
REMOTE_PYTHON=/tmp/ra62ler_pp2/venv/bin/python
REMOTE_MMSEQS=/tmp/ra62ler_pp2/mmseqs/bin/mmseqs
REMOTE_FASTA=/tmp/ra62ler_pp2/v44/cath-domain-seqs-S100-v4_4_0.fa
REMOTE_CLASSIFICATION=/tmp/ra62ler_pp2/v44/cath-domain-list-S100-v4_4_0.txt
THREADS="${THREADS:-96}"

mkdir -p "$RESULTS"
sshpass -p 'Mosbyboys123' ssh -p "$REMOTE_PORT" \
  "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_SCRATCH/results'"
sshpass -p 'Mosbyboys123' scp -P "$REMOTE_PORT" \
  "$PILOT" "$HVAL" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRATCH/"

sshpass -p 'Mosbyboys123' ssh -p "$REMOTE_PORT" \
  "$REMOTE_USER@$REMOTE_HOST" \
  "'$REMOTE_PYTHON' '$REMOTE_SCRATCH/pilot_split.py' \
    --fasta '$REMOTE_FASTA' \
    --classification '$REMOTE_CLASSIFICATION' \
    --work '$REMOTE_SCRATCH' \
    --results '$REMOTE_SCRATCH/results' \
    --hval-script '$REMOTE_SCRATCH/hval.py' \
    --mmseqs '$REMOTE_MMSEQS' \
    --seed 42 \
    --calibration-target 1000 \
    --test-target 1000 \
    --identity 0.20 \
    --coverage 0.80 \
    --evalue 10 \
    --threads '$THREADS' \
    --max-seqs 20000"

sshpass -p 'Mosbyboys123' scp -P "$REMOTE_PORT" \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRATCH/results/*" "$RESULTS/"
