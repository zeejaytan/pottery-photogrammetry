#!/usr/bin/env bash
# The interpreter every pipeline stage runs on, insulated from module-provided Python
# packages but nothing else.
#
# WHY THIS EXISTS AT ALL. pipeline_main.sh has to read the configuration to find out
# WHICH MODULES TO LOAD, so the first Python it runs is necessarily a Python chosen
# before any module is loaded. That read imports lib.pipeline_utils, which imports yaml.
# It worked for as long as the default python3 happened to have PyYAML; on Spartan
# `python3` now resolves to the `graphify` conda environment, which does not, and the
# pipeline dies at its first line with a bare ModuleNotFoundError. Naming the interpreter
# explicitly is the fix — inferring it from $PATH is what broke.
#
# WHY IT FILTERS PYTHONPATH. Loading COLMAP/3.9-CUDA-11.7.0 drags in SciPy-bundle/2022.05,
# which prepends a Python 3.10 site-packages directory to PYTHONPATH. Running any other
# Python with that in place picks up the wrong numpy and dies with:
#
#   ImportError: Importing the numpy C-extensions failed.
#   ... No module named 'numpy.core._multiarray_umath'
#
# The message blames numpy, which is fair — it is the right numpy being shadowed by one
# built for a different Python.
#
# WHY IT FILTERS RATHER THAN CLEARS. run_colmap.sh and friends set
#   export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"
# so that `from lib.pipeline_utils import ...` resolves. Clearing the variable outright
# would throw that away along with the offending entries. Only paths under the module
# tree are dropped; anything the caller deliberately put there survives.
#
# CHOOSING THE INTERPRETER, in order:
#   1. $PIPELINE_PYTHON            — explicit override, wins over everything
#   2. environment.python_interpreter in the config
#   3. python3 from $PATH          — the historical behaviour, kept as a last resort
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PIPELINE_DIR}/config/pipeline_config.yaml}"

PY="${PIPELINE_PYTHON:-}"

if [[ -z "$PY" && -f "$CONFIG_PATH" ]]; then
    # Read one scalar with sed rather than a YAML parser. Using Python here would be
    # circular: this script exists precisely because we cannot yet assume a Python that
    # can parse YAML. `python_interpreter` appears once in the config, under environment.
    PY="$(sed -n 's/^[[:space:]]*python_interpreter:[[:space:]]*["'\'']\{0,1\}\([^"'\''#]*[^"'\''[:space:]#]\)["'\'']\{0,1\}[[:space:]]*\(#.*\)\{0,1\}$/\1/p' "$CONFIG_PATH" | head -n 1)"
fi

PY="${PY:-python3}"

if ! command -v "$PY" >/dev/null 2>&1 && [[ ! -x "$PY" ]]; then
    cat >&2 <<EOF
ERROR: pipeline interpreter not found: $PY

Set one of these to a Python that has PyYAML, trimesh, numpy, pandas, scipy and networkx:
  - \$PIPELINE_PYTHON
  - environment.python_interpreter in $CONFIG_PATH
EOF
    exit 1
fi

clean=""
if [[ -n "${PYTHONPATH:-}" ]]; then
    IFS=':' read -r -a parts <<< "$PYTHONPATH"
    for p in "${parts[@]}"; do
        [[ -z "$p" ]] && continue
        # Module-system packages are built for a different Python and must not shadow
        # this environment's. Anything else is the caller's business.
        case "$p" in
            /apps/easybuild*|/apps/*/easybuild/*) continue ;;
        esac
        clean="${clean:+$clean:}$p"
    done
fi

export PYTHONPATH="$clean"
# The pipeline's dependencies must come from the named environment, not from whatever
# happens to be in ~/.local/lib/pythonX.Y/site-packages for the calling user.
export PYTHONNOUSERSITE=1
unset PYTHONHOME
exec "$PY" "$@"
