#!/usr/bin/env bash
#SBATCH --account=punim2657
#SBATCH --partition=gpu-a100
#SBATCH --qos=normal
#SBATCH --time=4:00:00
# ReconstructMesh tetrahedralises A02's 9.8M dense points into 20.6M cells.
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=sh5_grid
#SBATCH --output=/data/gpfs/projects/punim2657/Photogrammetry/logs/sh5_grid_%j.log

# The 2x2 over OpenMVS's two smoothing stages. See sh5_smoothing_grid.sh for the design
# and why vertex count is not the variable under test.
set -euo pipefail

PROJECT_ROOT=/data/gpfs/projects/punim2657/Photogrammetry

echo "=== job ${SLURM_JOB_ID:-?} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# OpenMVS links against the CUDA runtime; without this the binaries fail on libcuda.so.1.
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.7.0 2>/dev/null || true
module list 2>&1 | head -20

bash "${PROJECT_ROOT}/scripts/experiments/sh5_smoothing_grid.sh"
