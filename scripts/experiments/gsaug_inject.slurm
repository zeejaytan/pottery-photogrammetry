#!/usr/bin/env bash
#SBATCH --account=punim2657
#SBATCH --partition=gpu-a100
#SBATCH --qos=normal
#SBATCH --time=6:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --job-name=gsaug_inject
#SBATCH --output=/data/gpfs/projects/punim2657/Photogrammetry/logs/gsaug_inject_%j.log

# GS-augment A02, known-pose injection route: the dense model copy already holds
# 162 originals + 15 renders with written-down poses (inject_renders.py). This
# runs ONLY the OpenMVS stages on it, with the baseline (A-cell) flags read back
# out of the A02 logs (see sh5_smoothing_grid.sh): the only difference between
# cells is the 15 extra views.
set -euo pipefail

DIR="/data/gpfs/projects/punim2657/Rabati2025/17062025/A02_gsaug/work_inject"
BIN="/data/gpfs/projects/punim2657/Photogrammetry/openmvs/install/bin/OpenMVS"

echo "=== job ${SLURM_JOB_ID:-?} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
module load GCC/11.3.0 OpenMPI/4.1.4 CUDA/11.7.0 2>/dev/null || true
ls "$DIR/dense/images" | wc -l

/usr/bin/time -f "Interface wall %E maxRSS %MkB" \
"$BIN/InterfaceCOLMAP" -w "$DIR" -i "$DIR/dense" -o "$DIR/scene_inject.mvs" \
    --image-folder "$DIR/dense/images"

/usr/bin/time -f "Densify wall %E maxRSS %MkB" \
"$BIN/DensifyPointCloud" -w "$DIR" -i "$DIR/scene_inject.mvs" -o "$DIR/scene_dense_inject.mvs" \
    --cuda-device 0 --resolution-level 0 --number-views 8 --number-views-fuse 5

/usr/bin/time -f "Reconstruct wall %E maxRSS %MkB" \
"$BIN/ReconstructMesh" -w "$DIR" -i "$DIR/scene_dense_inject.mvs" -p "$DIR/scene_dense_inject.ply" \
    -o "$DIR/scene_dense_mesh_inject.ply" --target-face-num 10000000 --crop-to-roi 0 --smooth 2

/usr/bin/time -f "Refine wall %E maxRSS %MkB" \
"$BIN/RefineMesh" -w "$DIR" -i "$DIR/scene_dense_inject.mvs" -m "$DIR/scene_dense_mesh_inject.ply" \
    -o "$DIR/scene_refined_inject.ply" --scales 2 --cuda-device 0 --decimate 1 \
    --regularity-weight 0.2 --export-type ply

echo "=== done ==="
ls -la "$DIR"/scene_*inject.ply | awk '{printf "%12d  %s\n",$5,$9}'
