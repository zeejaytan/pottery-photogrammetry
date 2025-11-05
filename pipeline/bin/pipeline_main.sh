#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] <tree_path> [config_path]

Options:
  --resume              Resume from last completed stage (default).
  --no-resume           Disable resume; rerun every stage.
  --force-colmap        Re-run COLMAP even if outputs exist.
  --force-openmvs       Re-run OpenMVS even if outputs exist.
  --force-validate      Re-run split/validation even if outputs exist.
  --config <path>       Override configuration path.
  -h, --help            Show this help message.
EOF
}

RESUME=1
FORCE_COLMAP=0
FORCE_OPENMVS=0
FORCE_VALIDATE=0
CONFIG_OVERRIDE=""
TREE_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)
      RESUME=1
      shift
      ;;
    --no-resume)
      RESUME=0
      shift
      ;;
    --force-colmap|--rerun-colmap)
      FORCE_COLMAP=1
      shift
      ;;
    --force-openmvs|--rerun-openmvs)
      FORCE_OPENMVS=1
      shift
      ;;
    --force-validate|--rerun-validate)
      FORCE_VALIDATE=1
      shift
      ;;
    --config)
      shift
      CONFIG_OVERRIDE="${1:-}"
      if [[ -z "${CONFIG_OVERRIDE}" ]]; then
        echo "ERROR: --config expects a path" >&2
        usage
        exit 1
      fi
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR: Unknown option $1" >&2
      usage
      exit 1
      ;;
    *)
      TREE_PATH="$1"
      shift
      break
      ;;
  esac
done

if [[ -z "${TREE_PATH}" ]]; then
  if [[ $# -gt 0 ]]; then
    TREE_PATH="$1"
    shift
  else
    echo "ERROR: Missing tree_path argument" >&2
    usage
    exit 1
  fi
fi

if [[ -z "${CONFIG_OVERRIDE}" && $# -gt 0 ]]; then
  CONFIG_OVERRIDE="$1"
  shift
fi

CONFIG_PATH="${CONFIG_OVERRIDE:-${PIPELINE_DIR}/config/pipeline_config.yaml}"
PYTHON_BIN="${PYTHON:-python3}"

export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

# Helper to evaluate python snippet
get_python_value() {
  local snippet="$1"
  "${PYTHON_BIN}" - "$CONFIG_PATH" "$snippet" <<'PY'
import sys
from lib.pipeline_utils import PipelineContext, resolve_path, build_module_load_commands

config_path = sys.argv[1]
expression = sys.argv[2]
ctx = PipelineContext.from_config_path(config_path)
locals_dict = {
    "ctx": ctx,
    "resolve_path": resolve_path,
    "build_module_load_commands": build_module_load_commands,
}
value = eval(expression, {}, locals_dict)
if isinstance(value, (list, tuple)):
    for item in value:
        print(item)
elif value is not None:
    print(value)
PY
}

DATA_ROOT="$(get_python_value 'str(ctx.data_root)')"
LOG_DIR="$(get_python_value "str(resolve_path(ctx.config, 'paths.log_dir'))")"
mkdir -p "${LOG_DIR}"

mapfile -t MODULE_CMDS < <(get_python_value 'build_module_load_commands(ctx.config)')

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  echo "=== Loading modules ==="
  for module_cmd in "${MODULE_CMDS[@]}"; do
    if [[ -n "${module_cmd}" ]]; then
      echo "+ ${module_cmd}"
      eval "${module_cmd}"
    fi
  done
  module list
else
  echo "WARNING: Environment modules unavailable; proceeding without module loads."
fi

WORK_DIR="${DATA_ROOT}/${TREE_PATH}/work_colmap_openmvs"
mkdir -p "${WORK_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${WORK_DIR}/pipeline_${TIMESTAMP}.log"
mkdir -p "$(dirname "${LOG_FILE}")"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== Processing ${TREE_PATH} ==="
echo "Config: ${CONFIG_PATH}"
echo "Data root: ${DATA_ROOT}"
echo "Working dir: ${WORK_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU Info ==="
  nvidia-smi || echo "nvidia-smi failed"
fi

record_stage_completion() {
  local stage="$1"
  "${PYTHON_BIN}" - "$WORK_DIR" "$stage" <<'PY'
import sys
from pathlib import Path
from lib.stage_tracker import mark_stage

work_dir = Path(sys.argv[1])
stage = sys.argv[2]
mark_stage(work_dir, stage, completed=True)
PY
}

declare -A STAGE_STATUS=()

if [[ "${RESUME}" -eq 1 ]]; then
  mapfile -t STAGE_INFO < <("${PYTHON_BIN}" - "$CONFIG_PATH" "$WORK_DIR" <<'PY'
import sys
from pathlib import Path
from lib.pipeline_utils import load_config
from lib.stage_tracker import evaluate_stage

config = load_config(Path(sys.argv[1]))
work_dir = Path(sys.argv[2])
status = evaluate_stage(config, work_dir)
data = status.to_dict()
print(f"work_dir={work_dir}")
for key, value in data.items():
    if isinstance(value, bool):
        value = str(value).lower()
    print(f"{key}={value}")
PY
)
  for line in "${STAGE_INFO[@]}"; do
    key="${line%%=*}"
    value="${line#*=}"
    STAGE_STATUS["$key"]="$value"
  done
  echo "Resume mode: COLMAP=${STAGE_STATUS[colmap_complete]:-false}, OpenMVS=${STAGE_STATUS[openmvs_complete]:-false}, Split=${STAGE_STATUS[split_complete]:-false}"
fi

RUN_COLMAP=1
RUN_OPENMVS=1
RUN_VALIDATE=1

if [[ "${RESUME}" -eq 1 ]]; then
  if [[ "${FORCE_COLMAP}" -eq 0 && "${STAGE_STATUS[colmap_complete]:-false}" == "true" ]]; then
    RUN_COLMAP=0
  fi
  if [[ "${FORCE_OPENMVS}" -eq 0 && "${STAGE_STATUS[openmvs_complete]:-false}" == "true" ]]; then
    RUN_OPENMVS=0
  fi
  if [[ "${FORCE_VALIDATE}" -eq 0 && "${STAGE_STATUS[split_complete]:-false}" == "true" ]]; then
    RUN_VALIDATE=0
  fi
fi

if [[ "${RUN_COLMAP}" -eq 1 ]]; then
  RUN_OPENMVS=1
  RUN_VALIDATE=1
elif [[ "${RUN_OPENMVS}" -eq 1 ]]; then
  RUN_VALIDATE=1
fi

EXIT_CODE=0

if [[ "${RUN_COLMAP}" -eq 1 ]]; then
  echo "=== Stage 1: COLMAP ==="
  if ! bash "${PIPELINE_DIR}/bin/run_colmap.sh" "${TREE_PATH}" "${CONFIG_PATH}"; then
    echo "ERROR: COLMAP stage failed" >&2
    exit 1
  fi
  record_stage_completion "colmap"
else
  echo "=== Stage 1: COLMAP (skipped: resume) ==="
  if [[ "${RESUME}" -eq 1 ]]; then
    record_stage_completion "colmap"
  fi
fi

if [[ "${RUN_OPENMVS}" -eq 1 ]]; then
  echo "=== Stage 2: OpenMVS ==="
  if ! bash "${PIPELINE_DIR}/bin/run_openmvs.sh" "${WORK_DIR}" "${CONFIG_PATH}"; then
    echo "ERROR: OpenMVS stage failed" >&2
    exit 2
  fi
  record_stage_completion "openmvs"
else
  echo "=== Stage 2: OpenMVS (skipped: resume) ==="
  if [[ "${RESUME}" -eq 1 ]]; then
    record_stage_completion "openmvs"
  fi
fi

if [[ "${RUN_VALIDATE}" -eq 1 ]]; then
  echo "=== Stage 3: Split and Validate ==="
  if ! "${PYTHON_BIN}" "${PIPELINE_DIR}/bin/split_and_validate.py" \
    --config "${CONFIG_PATH}" \
    --tree "${TREE_PATH}"; then
    SPLIT_EXIT=$?
    echo "WARNING: Validation issues detected (exit code ${SPLIT_EXIT})" >&2
    EXIT_CODE=${SPLIT_EXIT}
  else
    record_stage_completion "split"
    EXIT_CODE=0
  fi
else
  echo "=== Stage 3: Split and Validate (skipped: resume) ==="
  if [[ "${RESUME}" -eq 1 ]]; then
    record_stage_completion "split"
  fi
fi

echo "=== Pipeline completed for ${TREE_PATH} (exit code ${EXIT_CODE}) ==="
exit "${EXIT_CODE}"
