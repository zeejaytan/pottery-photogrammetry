#!/usr/bin/env bash
# Rsync a relative path from the Spartan Photogrammetry tree to a local directory.
# Usage: ./scripts/remote/fetch_artifacts.sh <remote-relpath> [local-dir]
#
# WHAT MAY COME DOWN: small derived outputs only — comparison renders (PNG),
# metrics CSV/JSON, job logs. Photographs, COLMAP datasets, trained scenes and
# full-resolution meshes stay on Spartan. If you find yourself reaching for a
# multi-hundred-MB path here, that is the signal to inspect it in place with
# `ssh spartan` instead.
set -euo pipefail

SPARTAN_HOST="${SPARTAN_HOST:-spartan}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/gpfs/projects/punim2657/Photogrammetry}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <remote-relpath> [local-dir]" >&2
  echo "  remote-relpath: path relative to REMOTE_ROOT" >&2
  echo "                  (e.g. pipeline/logs, or a work_colmap_openmvs subdir)" >&2
  echo "  local-dir:      default ./artifacts/" >&2
  exit 1
fi

REMOTE_REL="${1%/}"
LOCAL_DIR="${2:-./artifacts}"

mkdir -p "$LOCAL_DIR"

if command -v rsync >/dev/null 2>&1; then
  rsync -avz --progress \
    "${SPARTAN_HOST}:${REMOTE_ROOT}/${REMOTE_REL}" \
    "${LOCAL_DIR}/"
else
  # Git Bash on Windows has no rsync; scp -r (Windows OpenSSH) is a fine
  # fallback for the small artifacts this script is meant for.
  echo "rsync not found; falling back to scp -r" >&2
  scp -r \
    "${SPARTAN_HOST}:${REMOTE_ROOT}/${REMOTE_REL}" \
    "${LOCAL_DIR}/"
fi

echo "Fetched → ${LOCAL_DIR}/$(basename "$REMOTE_REL")"
