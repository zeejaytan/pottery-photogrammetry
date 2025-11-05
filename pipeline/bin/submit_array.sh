#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${PIPELINE_DIR}/config/pipeline_config.yaml"
TARGETS_OVERRIDE=""
DRY_RUN=0
PYTHON_BIN="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --targets)
      TARGETS_OVERRIDE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

readarray -t CONFIG_INFO < <("${PYTHON_BIN}" - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path
from lib.pipeline_utils import PipelineContext, resolve_path

config_path = sys.argv[1]
ctx = PipelineContext.from_config_path(config_path)
cfg = ctx.config

try:
    log_dir = resolve_path(cfg, "paths.log_dir")
except KeyError:
    log_dir = Path(ctx.project_root) / "pipeline/logs"

targets_path = (
    Path(ctx.project_root)
    / cfg.get("targets", {}).get("targets_file", "pipeline/targets.txt")
)

print(f"targets_file={targets_path}")
print(f"log_dir={log_dir}")

slurm_cfg = cfg.get("slurm", {})
for key, value in slurm_cfg.items():
    print(f"slurm.{key}={value}")
PY
)

declare -A CFG_MAP
for line in "${CONFIG_INFO[@]}"; do
  key="${line%%=*}"
  value="${line#*=}"
  CFG_MAP["$key"]="$value"
done

TARGETS_FILE="${TARGETS_OVERRIDE:-${CFG_MAP[targets_file]}}"
LOG_DIR="${CFG_MAP[log_dir]}"
mkdir -p "${LOG_DIR}"

if [[ "${DRY_RUN}" -eq 0 && -z "${TARGETS_OVERRIDE:-}" ]]; then
  "${PYTHON_BIN}" "${PIPELINE_DIR}/bin/scan_targets.py" --config "${CONFIG_PATH}" --output "${TARGETS_FILE}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[dry-run] Would run scan_targets.py to populate ${TARGETS_FILE}"
else
  echo "Using existing targets file: ${TARGETS_FILE}"
fi

if [[ ! -f "${TARGETS_FILE}" ]]; then
  echo "Targets file not found: ${TARGETS_FILE}" >&2
  exit 1
fi

NUM_TARGETS=$(grep -cve '^\s*$' "${TARGETS_FILE}" || true)
if [[ "${NUM_TARGETS}" -eq 0 ]]; then
  echo "No targets found in ${TARGETS_FILE}" >&2
  exit 1
fi

ARRAY_RANGE="1-${NUM_TARGETS}"
ARRAY_CHUNK="${CFG_MAP[slurm.array_chunk]:-}"
if [[ -n "${ARRAY_CHUNK}" && "${ARRAY_CHUNK}" != "0" ]]; then
  ARRAY_RANGE="${ARRAY_RANGE}%${ARRAY_CHUNK}"
fi

SBATCH_ARGS=()
if [[ -n "${CFG_MAP[slurm.account]:-}" ]]; then
  SBATCH_ARGS+=(--account="${CFG_MAP[slurm.account]}")
fi
if [[ -n "${CFG_MAP[slurm.partition]:-}" ]]; then
  SBATCH_ARGS+=(--partition="${CFG_MAP[slurm.partition]}")
fi
if [[ -n "${CFG_MAP[slurm.qos]:-}" && "${CFG_MAP[slurm.qos]}" != "NONE" ]]; then
  SBATCH_ARGS+=(--qos="${CFG_MAP[slurm.qos]}")
fi

TIME_HOURS="${CFG_MAP[slurm.time_hours]:-12}"
SBATCH_ARGS+=(--time="$(printf "%02d:00:00" "${TIME_HOURS}")")

MEMORY="${CFG_MAP[slurm.mem_gb]:-64}"
SBATCH_ARGS+=(--mem="${MEMORY}G")

CPUS="${CFG_MAP[slurm.cpus_per_task]:-8}"
SBATCH_ARGS+=(--cpus-per-task="${CPUS}")

if [[ -n "${CFG_MAP[slurm.gres]:-}" ]]; then
  SBATCH_ARGS+=(--gres="${CFG_MAP[slurm.gres]}")
else
  GPUS="${CFG_MAP[slurm.gpus]:-1}"
  SBATCH_ARGS+=(--gres="gpu:${GPUS}")
fi

if [[ -n "${CFG_MAP[slurm.mail_user]:-}" ]]; then
  SBATCH_ARGS+=(--mail-user="${CFG_MAP[slurm.mail_user]}")
fi
if [[ -n "${CFG_MAP[slurm.mail_type]:-}" && "${CFG_MAP[slurm.mail_type]}" != "NONE" ]]; then
  SBATCH_ARGS+=(--mail-type="${CFG_MAP[slurm.mail_type]}")
fi

JOB_NAME="${CFG_MAP[slurm.job_name]:-rabati_mesh}"
SBATCH_ARGS+=(--job-name="${JOB_NAME}")
SBATCH_ARGS+=(--output="${LOG_DIR}/rabati_%A_%a.log")
SBATCH_ARGS+=(--array="${ARRAY_RANGE}")

SBATCH_CMD=(sbatch "${SBATCH_ARGS[@]}" "${PIPELINE_DIR}/bin/slurm_array_job.sh")

echo "Submitting ${NUM_TARGETS} targets via Slurm array (${ARRAY_RANGE})"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '[dry-run] %q ' "${SBATCH_CMD[@]}"; printf '\n'
else
  "${SBATCH_CMD[@]}"
fi
