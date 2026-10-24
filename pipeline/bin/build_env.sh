#!/usr/bin/env bash
# Build the pipeline's own Python environment: a venv on the system Python, filled from
# pipeline/requirements.txt.
#
#   pipeline/bin/build_env.sh [<prefix>]      default: <repo>/envs/pipeline
#
# Run it on the login node (Spartan's Python page installs with pip there); it takes a
# minute or two. It builds into <prefix>.new and only moves that into place once every
# import works, so a failed build never leaves a half-filled environment where the
# pipeline would find it. An existing <prefix> is kept as <prefix>.old-<date>, not deleted.
#
# WHY THE SYSTEM PYTHON (/usr/bin/python3, 3.9 on RHEL 9) AND NOT A MODULE. pipeline_main.sh
# runs this interpreter before any module is loaded (bin/pipeline_python.sh says why). A
# venv on the Python/3.10.4 module links libpython from the module tree and, with the
# module unloaded, exits 127 before running a line (checked on Spartan 2026-10-24). The
# system Python needs nothing loaded, and it is the same 3.9 the borrowed MILo environment
# ran, so the same wheels install. It changes only with the operating system; if Spartan
# moves off RHEL 9, rebuild this and re-run the parity check.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PREFIX="${1:-${REPO_DIR}/envs/pipeline}"
BASE_PY="${BASE_PYTHON:-/usr/bin/python3}"
REQS="${REPO_DIR}/pipeline/requirements.txt"
NEW="${PREFIX}.new"

[[ -x "$BASE_PY" ]] || { echo "build_env: base Python $BASE_PY not found" >&2; exit 1; }
[[ -f "$REQS" ]] || { echo "build_env: $REQS not found" >&2; exit 1; }
[[ -e "$NEW" ]] && { echo "build_env: $NEW exists from an earlier attempt; look at it, then remove it" >&2; exit 1; }

# Nothing from modules or ~/.local may leak into the build.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

mkdir -p "$(dirname "$PREFIX")"
"$BASE_PY" -m venv "$NEW"
"$NEW/bin/python" -m pip install --quiet --no-cache-dir -r "$REQS"

# The imports every stage needs, and the ArUco module the scale step uses.
"$NEW/bin/python" - <<'PY'
import sys, yaml, numpy, pandas, scipy, networkx, trimesh, cv2
assert hasattr(cv2, "aruco"), "this OpenCV build has no aruco module"
print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| trimesh", trimesh.__version__,
      "| scipy", scipy.__version__, "| networkx", networkx.__version__, "| cv2", cv2.__version__)
PY

{
    echo "# built $(date -Iseconds) on $(hostname) from $REQS"
    echo "# base $("$BASE_PY" --version 2>&1) at $BASE_PY; $(head -n 1 /etc/redhat-release 2>/dev/null || true)"
    "$NEW/bin/python" -m pip freeze
} > "$NEW/FREEZE.txt"

if [[ -e "$PREFIX" ]]; then
    mv "$PREFIX" "${PREFIX}.old-$(date +%Y%m%d-%H%M)"
fi
# Renaming a venv leaves bin/pip and bin/activate naming the .new path. The pipeline only
# ever runs $PREFIX/bin/python, which works after the move; to add a package by hand, use
# `$PREFIX/bin/python -m pip`, or better, add it to requirements.txt and rebuild.
mv "$NEW" "$PREFIX"
echo "build_env: ready at $PREFIX (package list in $PREFIX/FREEZE.txt)"
