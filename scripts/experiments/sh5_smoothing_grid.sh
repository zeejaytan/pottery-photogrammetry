#!/usr/bin/env bash
# Does OpenMVS lose SH5's fracture edge because of smoothing, and if so, which smoothing?
#
# BACKGROUND. On 17062025/A02 the refined mesh kept only 8 mm of fold steeper than 60 deg
# on sherd SH5, where COLMAP Poisson resolved 1,298 mm from the identical photographs. So
# the fracture edge is genuinely present in the images and OpenMVS removed it. The median
# across the seven sherds (141 mm) looks healthy and hides this completely.
#
# It is NOT a density problem. Measured on A02: OpenMVS vertices sit 0.461 mm apart,
# Poisson's 0.436 mm — 5% finer. And RefineMesh adds no vertices anyway (+302 of 1.93M,
# 0.016%); it moves the ones it is given. Two stages apply smoothing instead:
#
#   ReconstructMesh --smooth            default 2    passes over the graph-cut surface
#   RefineMesh      --regularity-weight default 0.2  smoothness prior vs photo-consistency
#
# THE DESIGN. A 2x2 over those two, so a change can be attributed rather than guessed at.
# One run per cell would not distinguish "the fix" from "one of two things I changed".
#
#   A  smooth 2  rw 0.20   the existing baseline, already on disk, NOT recomputed
#   B  smooth 0  rw 0.20   isolates ReconstructMesh's smoothing
#   C  smooth 0  rw 0.05   both knobs off
#   D  smooth 2  rw 0.05   isolates RefineMesh's smoothness prior
#
# COST. DensifyPointCloud is already done and cached (scene_dense.mvs + scene_dense.ply),
# which is the expensive stage. B and C share one ReconstructMesh; D re-refines the mesh
# that is already there. So: 1 x ReconstructMesh (~10 min) + 3 x RefineMesh (~6 min each).
#
# SAFETY. Nothing here overwrites the baseline. Every output carries its cell in the name,
# and the script refuses to start if the two cached inputs are missing.
set -euo pipefail

DIR="${WORK_DIR:-/data/gpfs/projects/punim2657/Rabati2025/17062025/A02/work_colmap_openmvs/dense_masked}"
BIN="${OPENMVS_BIN:-/data/gpfs/projects/punim2657/Photogrammetry/openmvs/install/bin/OpenMVS}"
CUDA="${CUDA_DEVICE:-0}"

for f in scene_dense.mvs scene_dense.ply scene_dense_mesh.ply scene_refined_mesh.ply; do
    [[ -f "$DIR/$f" ]] || { echo "ERROR: missing cached input $DIR/$f" >&2; exit 1; }
done

# Exactly the baseline's flags (read back out of RefineMesh-2608151550248D8C27.log and
# ReconstructMesh-2608151540448D5A33.log), so the only difference between cells is the
# knob under test. --crop-to-roi 0 and --decimate 1 are in there because the baseline had
# them, not because they are good defaults.
recon() {  # recon <smooth> <out.ply>
    echo "--- ReconstructMesh --smooth $1 -> $(basename "$2")"
    /usr/bin/time -f "    wall %E  maxRSS %MkB" \
    "$BIN/ReconstructMesh" -w "$DIR" -i "$DIR/scene_dense.mvs" -p "$DIR/scene_dense.ply" \
        -o "$2" --target-face-num 10000000 --crop-to-roi 0 --smooth "$1"
}
refine() {  # refine <in_mesh.ply> <regularity_weight> <out.ply>
    echo "--- RefineMesh --regularity-weight $2 on $(basename "$1") -> $(basename "$3")"
    /usr/bin/time -f "    wall %E  maxRSS %MkB" \
    "$BIN/RefineMesh" -w "$DIR" -i "$DIR/scene_dense.mvs" -m "$1" -o "$3" \
        --scales 2 --cuda-device "$CUDA" --decimate 1 --regularity-weight "$2" \
        --export-type ply
}

M_SM0="$DIR/scene_dense_mesh_sm0.ply"

# B and C share this one reconstruction. Skip it if a previous attempt got this far —
# a late failure should not repay an early cost.
if [[ -f "$M_SM0" ]]; then
    echo "=== reusing existing $(basename "$M_SM0")"
else
    recon 0 "$M_SM0"
fi

refine "$M_SM0"                     0.2  "$DIR/scene_refined_B_sm0_rw020.ply"   # B
refine "$M_SM0"                     0.05 "$DIR/scene_refined_C_sm0_rw005.ply"   # C
refine "$DIR/scene_dense_mesh.ply"  0.05 "$DIR/scene_refined_D_sm2_rw005.ply"   # D

echo
echo "=== done. A (baseline) = scene_refined_mesh.ply, untouched."
ls -la "$DIR"/scene_refined_*.ply "$DIR"/scene_dense_mesh*.ply | awk '{printf "%12d  %s\n",$5,$9}'
