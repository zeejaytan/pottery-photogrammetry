#!/usr/bin/env bash
# Show Slurm queue / accounting for this user on Spartan.
# Usage: ./scripts/remote/job_status.sh [jobid]
set -euo pipefail

SPARTAN_HOST="${SPARTAN_HOST:-spartan}"
JOBID="${1:-}"

if [[ -n "$JOBID" ]]; then
  ssh "$SPARTAN_HOST" "squeue -j $(printf '%q' "$JOBID") 2>/dev/null || true; sacct -j $(printf '%q' "$JOBID") --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS -P"
else
  ssh "$SPARTAN_HOST" "squeue -u \"\$USER\"; echo '---'; sacct -u \"\$USER\" --starttime=today --format=JobID,JobName,State,Elapsed,ExitCode -P | tail -n 30"
fi
