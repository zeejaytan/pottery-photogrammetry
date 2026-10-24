#!/usr/bin/env bash
#SBATCH --account=punim2657
#SBATCH --partition=sapphire,cascade
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=env_parity
# Submit with pr-workspace scripts/new_run.sh, which sets RUN_DIR, SOURCE_COMMIT and CANONICAL
# and puts this job's output in <run>/logs/.
#
# Does the pipeline's own environment compute the same thing as the borrowed MILo one?
# Runs the sherd-splitting stage (split_and_validate.py, the pipeline's pure-Python stage)
# on the same refined mesh with each interpreter and compares the results byte for byte:
# the per-sherd report and every exported sherd mesh. Byte equality is the test; a
# difference is listed file by file, and the job exits 1.
#
#   MESH=...  the refined mesh to split (default: A02's shipping-default refined mesh)
set -euo pipefail

: "${RUN_DIR:?submit with new_run.sh}" "${SOURCE_COMMIT:?}" "${CANONICAL:?}"
source /data/gpfs/projects/punim2657/agent/pr-workspace/scripts/run_from_commit.sh \
    "$CANONICAL" "$SOURCE_COMMIT" "$RUN_DIR"

MESH="${MESH:-/data/gpfs/projects/punim2657/Rabati2025/17062025/A02/work_colmap_openmvs/dense_masked/scene_refined_mesh.ply}"
BORROWED=/data/gpfs/projects/punim2657/MILo/envs/milo/bin/python
OWN="${CANONICAL}/envs/pipeline/bin/python"

echo "=== job ${SLURM_JOB_ID:-?} on $(hostname), commit ${SOURCE_COMMIT} ==="
echo "mesh: $MESH ($(stat -c %s "$MESH") bytes)"
cd "$SRC"
export PYTHONPATH="$SRC/pipeline"

for name in borrowed own; do
    py=$([[ $name == borrowed ]] && echo "$BORROWED" || echo "$OWN")
    out="$RUN_DIR/out/$name"
    mkdir -p "$out"
    echo "--- $name: $py"
    # pipeline_python.sh is how every stage gets its interpreter; PIPELINE_PYTHON picks which.
    # Exit 3 means "some component below min_vertices", which is a result, not a crash.
    set +e
    PIPELINE_PYTHON="$py" pipeline/bin/pipeline_python.sh -c \
        'import sys, numpy, trimesh, scipy, networkx; print(sys.executable, sys.version.split()[0], "numpy", numpy.__version__, "trimesh", trimesh.__version__, "scipy", scipy.__version__, "networkx", networkx.__version__)'
    start=$SECONDS
    PIPELINE_PYTHON="$py" pipeline/bin/pipeline_python.sh pipeline/bin/split_and_validate.py \
        --config pipeline/config/pipeline_config.yaml --mesh "$MESH" --output-dir "$out" \
        2> "$RUN_DIR/logs/split_$name.err"
    rc=$?
    set -e
    echo "split exit $rc after $((SECONDS - start)) s"; grep -E "Exported" "$RUN_DIR/logs/split_$name.err" || true
    [[ $rc -eq 0 || $rc -eq 3 ]] || { echo "split failed under $name"; exit 2; }
    (cd "$out" && sha256sum * | sort -k2) > "$RUN_DIR/out/sha256_$name.txt"
done

echo "=== compare ==="
if diff "$RUN_DIR/out/sha256_borrowed.txt" "$RUN_DIR/out/sha256_own.txt"; then
    echo "IDENTICAL: $(wc -l < "$RUN_DIR/out/sha256_own.txt") files, report and every sherd mesh byte for byte"
else
    echo "DIFFERENT: see the lines above"; exit 1
fi
head -20 "$RUN_DIR/out/own/validation_report.csv"
