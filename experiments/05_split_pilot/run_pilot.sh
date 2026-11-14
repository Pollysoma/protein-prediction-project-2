#!/usr/bin/env bash
# Run the CATH v4.4 pilot split on the LMU GPU server and copy summaries back.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RESULTS="$HERE/results"
HVAL="$ROOT/experiments/03_calibration_set/hval.py"
PILOT="$HERE/pilot_split.py"

# Server connection details live in SERVER.local.sh, which is gitignored.
# This repository is public, so no host, user or password belongs in a tracked file.
# Template:
#   REMOTE_USER=...
#   REMOTE_HOST=...
#   REMOTE_PORT=...
#   REMOTE_PASSWORD=...
#   REMOTE_BASE=/tmp/<user>_pp2
CONFIG="$ROOT/SERVER.local.sh"
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing $CONFIG - see the template in the comment above and in SERVER.local.md." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONFIG"
: "${REMOTE_USER:?set in SERVER.local.sh}"
: "${REMOTE_HOST:?set in SERVER.local.sh}"
: "${REMOTE_PORT:?set in SERVER.local.sh}"
: "${REMOTE_PASSWORD:?set in SERVER.local.sh}"
: "${REMOTE_BASE:?set in SERVER.local.sh}"

REMOTE_SCRATCH="$REMOTE_BASE/pilot_split"
REMOTE_PYTHON="$REMOTE_BASE/venv/bin/python"
REMOTE_MMSEQS="$REMOTE_BASE/mmseqs/bin/mmseqs"
REMOTE_FASTA="$REMOTE_BASE/v44/cath-domain-seqs-S100-v4_4_0.fa"
REMOTE_CLASSIFICATION="$REMOTE_BASE/v44/cath-domain-list-S100-v4_4_0.txt"
THREADS="${THREADS:-96}"

# sshpass -e reads the password from SSHPASS, so it never appears in the
# process list the way sshpass -p does.
export SSHPASS="$REMOTE_PASSWORD"

mkdir -p "$RESULTS"
sshpass -e ssh -p "$REMOTE_PORT" \
  "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_SCRATCH/results'"
sshpass -e scp -P "$REMOTE_PORT" \
  "$PILOT" "$HVAL" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRATCH/"

sshpass -e ssh -p "$REMOTE_PORT" \
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

sshpass -e scp -P "$REMOTE_PORT" \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SCRATCH/results/*" "$RESULTS/"
