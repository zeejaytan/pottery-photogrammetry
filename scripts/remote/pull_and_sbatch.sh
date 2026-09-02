#!/usr/bin/env bash
# Pull latest code on Spartan (ff-only), then sbatch a job.
# Usage: ./scripts/remote/pull_and_sbatch.sh <job.slurm> [extra sbatch args...]
set -euo pipefail

SPARTAN_HOST="${SPARTAN_HOST:-spartan}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/gpfs/projects/punim2657/Photogrammetry}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <job.slurm> [extra sbatch args...]" >&2
  echo "  e.g. $0 slurm/milo_train.slurm 16062025" >&2
  exit 1
fi

JOB_SCRIPT="$1"
shift

ssh "$SPARTAN_HOST" "cd $(printf '%q' "$REMOTE_ROOT") && git pull --ff-only && sbatch $(printf '%q' "$JOB_SCRIPT") $(printf '%q ' "$@")"
