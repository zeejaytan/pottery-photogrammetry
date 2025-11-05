#!/usr/bin/env bash
#SBATCH --account=punim2657
#SBATCH --partition=gpu-a100
#SBATCH --qos=normal
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=openmvs_fix

set -euo pipefail

WORK_DIR="/data/gpfs/projects/punim2657/Rabati2025/16062025/work_colmap_openmvs"
PROJECT_ROOT="/data/gpfs/projects/punim2657/Photogrammetry"
PIPELINE_DIR="${PROJECT_ROOT}/pipeline"
CONFIG_PATH="${PIPELINE_DIR}/config/pipeline_config.yaml"

cd "${PROJECT_ROOT}"

# Load modules
module load Python/3.10.4 GCC/11.3.0 OpenMPI/4.1.4 ICU/71.1 CUDA/11.7.0

# Set PYTHONPATH
export PYTHONPATH="${PIPELINE_DIR}:${PYTHONPATH:-}"

# Remove old OpenMVS outputs
rm -f "${WORK_DIR}"/*.mvs "${WORK_DIR}"/*.ply "${WORK_DIR}"/sherd_*.ply "${WORK_DIR}"/*.log "${WORK_DIR}"/validation_report.csv

# Run OpenMVS pipeline
python pipeline/bin/run_openmvs.py "${WORK_DIR}" --config "${CONFIG_PATH}"

# Run validation
python pipeline/bin/split_and_validate.py --config "${CONFIG_PATH}" --tree 16062025
