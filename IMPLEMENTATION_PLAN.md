# Rabati2025 COLMAP→OpenMVS Mesh Pipeline - Implementation Plan

**Document Version**: 1.0
**Date**: 2025-10-22
**Target System**: Spartan HPC (University of Melbourne)

## Executive Summary

This document outlines the complete implementation plan for an automated, GPU-accelerated photogrammetry pipeline that processes pottery sherd photographs from the Rabati2025 archaeological dataset. The pipeline uses COLMAP for Structure-from-Motion (SfM) and OpenMVS for Multi-View Stereo (MVS) dense reconstruction, outputting separate high-density meshes (≥100,000 vertices) for each pottery sherd.

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Directory Structure](#directory-structure)
3. [Component Specifications](#component-specifications)
4. [Workflow](#workflow)
5. [Configuration](#configuration)
6. [Implementation Checklist](#implementation-checklist)
7. [Testing Strategy](#testing-strategy)
8. [Parameter Tuning Guide](#parameter-tuning-guide)

---

## System Architecture

### Infrastructure
- **Compute Environment**: Spartan HPC, GPU partition (gpu-v100)
- **Execution Model**: Slurm array jobs, one job per pottery-tree folder
- **GPU Acceleration**: NVIDIA V100 via `--gres=gpu:1`
- **COLMAP**: Module system (`COLMAP/3.9-CUDA-11.7.0`)
- **OpenMVS**: Local build at `/data/gpfs/projects/punim2657/Photogrammetry/openmvs/install/bin/OpenMVS/`

### Data Flow
```
Input: /data/gpfs/projects/punim2657/Rabati2025/<date>/<tree>/*.JPG
  ↓
COLMAP: Feature extraction → Matching → Sparse reconstruction → Undistortion
  ↓
OpenMVS: InterfaceCOLMAP → Densify → Reconstruct mesh → Refine mesh
  ↓
Post-process: Split by connected components → Validate vertex counts
  ↓
Output: <tree>/work_colmap_openmvs/sherd_NNN.ply (N separate meshes)
```

---

## Directory Structure

```
/data/gpfs/projects/punim2657/Photogrammetry/
├── pipeline/
│   ├── bin/
│   │   ├── scan_targets.py           # Discover pottery-tree folders
│   │   ├── run_colmap.sh             # COLMAP workflow wrapper
│   │   ├── run_openmvs.sh            # OpenMVS workflow wrapper
│   │   ├── split_and_validate.py     # Mesh splitting + QA
│   │   ├── pipeline_main.sh          # Master orchestrator
│   │   ├── submit_array.sh           # Slurm submission script
│   │   ├── slurm_array_job.sh        # Slurm batch template
│   │   └── aggregate_results.py      # Results collector (bonus)
│   ├── lib/
│   │   └── pipeline_utils.py         # Shared Python utilities
│   ├── config/
│   │   └── pipeline_config.yaml      # Central configuration
│   ├── logs/                         # Execution logs
│   └── targets.txt                   # List of pottery-tree paths (generated)
├── colmap/                           # Local COLMAP build (exists)
├── openmvs/                          # Local OpenMVS build (exists)
└── docs/
    └── IMPLEMENTATION_PLAN.md        # This document
```

### Per-Tree Output Structure
```
/data/gpfs/projects/punim2657/Rabati2025/<date>/<tree>/
├── *.JPG                             # Input photos
├── work_colmap_openmvs/              # Working directory
│   ├── database.db                   # COLMAP database
│   ├── sparse/                       # Sparse reconstruction
│   │   └── 0/
│   │       ├── cameras.bin
│   │       ├── images.bin
│   │       └── points3D.bin
│   ├── dense/                        # MVS-ready workspace
│   │   ├── images/                   # Undistorted images
│   │   ├── sparse/                   # Cameras for MVS
│   │   └── stereo/                   # (not used - OpenMVS replaces this)
│   ├── scene.mvs                     # OpenMVS scene
│   ├── scene_dense.mvs               # Densified point cloud
│   ├── scene_dense_mesh.ply          # Reconstructed mesh
│   ├── scene_refined_mesh.ply        # Refined mesh (multi-object)
│   ├── sherd_001.ply                 # Split sherd meshes
│   ├── sherd_002.ply
│   ├── ...
│   ├── sherd_NNN.ply
│   ├── validation_report.csv         # QA results
│   └── pipeline_<timestamp>.log      # Execution log
```

---

## Component Specifications

### 1. `scan_targets.py`

**Purpose**: Scan Rabati2025 directory and build target list for Slurm array job.

**Inputs**:
- Root directory: `/data/gpfs/projects/punim2657/Rabati2025/`

**Logic**:
1. Walk all date-named folders (e.g., `04052025`, `16062025`)
2. For each subfolder (pottery-tree), count JPEG files (`.jpg`, `.jpeg`, case-insensitive)
3. Include tree if `jpeg_count >= min_jpegs_per_tree` (config: 10)
4. Write relative path to `targets.txt` (e.g., `04052025/O01`)

**Output**:
- `pipeline/targets.txt` (one path per line)

**CLI**:
```bash
python pipeline/bin/scan_targets.py --config pipeline/config/pipeline_config.yaml
```

**Key Functions**:
```python
def find_pottery_trees(root_path, min_jpegs=10):
    """Recursively find pottery-tree folders with sufficient JPEGs."""

def count_jpegs(folder_path, extensions=['.jpg', '.jpeg']):
    """Count JPEG files in folder (case-insensitive)."""
```

---

### 2. `run_colmap.sh`

**Purpose**: Execute COLMAP workflow to create MVS-ready dense workspace.

**Inputs**:
- `TREE_PATH`: Relative path to pottery-tree (e.g., `04052025/O01`)
- `CONFIG`: Path to YAML config

**Workflow**:
```bash
# 1. Load COLMAP module
module purge
module load GCC/11.3.0 OpenMPI/4.1.4
module load COLMAP/3.9-CUDA-11.7.0

# 2. Setup paths
RABATI_ROOT=/data/gpfs/projects/punim2657/Rabati2025
WORK_DIR=$RABATI_ROOT/$TREE_PATH/work_colmap_openmvs
IMAGE_DIR=$RABATI_ROOT/$TREE_PATH

# 3. Feature extraction (GPU)
colmap feature_extractor \
  --database_path $WORK_DIR/database.db \
  --image_path $IMAGE_DIR \
  --ImageReader.mask_path "" \
  --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu 1

# 4. Exhaustive matching (GPU)
colmap exhaustive_matcher \
  --database_path $WORK_DIR/database.db \
  --SiftMatching.use_gpu 1

# 5. Sparse reconstruction
colmap mapper \
  --database_path $WORK_DIR/database.db \
  --image_path $IMAGE_DIR \
  --output_path $WORK_DIR/sparse

# 6. Undistort images for MVS (creates dense/ workspace)
colmap image_undistorter \
  --image_path $IMAGE_DIR \
  --input_path $WORK_DIR/sparse/0 \
  --output_path $WORK_DIR/dense \
  --output_type COLMAP
```

**Exit Codes**:
- `0`: Success
- `1`: COLMAP command failed

---

### 3. `run_openmvs.sh`

**Purpose**: Execute OpenMVS workflow from COLMAP dense workspace to refined mesh.

**Inputs**:
- `WORK_DIR`: Path to `work_colmap_openmvs/`
- `CONFIG`: Path to YAML config

**Workflow**:
```bash
# 1. Export OpenMVS binary path
OPENMVS_BIN=/data/gpfs/projects/punim2657/Photogrammetry/openmvs/install/bin/OpenMVS
export LD_LIBRARY_PATH=$OPENMVS_BIN:$LD_LIBRARY_PATH

# 2. Convert COLMAP workspace to OpenMVS scene
$OPENMVS_BIN/InterfaceCOLMAP \
  -i $WORK_DIR/dense \
  -o $WORK_DIR/scene.mvs

# 3. Densify point cloud (full resolution)
$OPENMVS_BIN/DensifyPointCloud \
  -i $WORK_DIR/scene.mvs \
  -o $WORK_DIR/scene_dense.mvs \
  --resolution-level 0 \
  --number-views 8 \
  --number-views-fuse 8 \
  --cuda-device 0

# 4. Reconstruct mesh (no decimation)
$OPENMVS_BIN/ReconstructMesh \
  -i $WORK_DIR/scene_dense.mvs \
  -o $WORK_DIR/scene_dense_mesh.ply \
  --min-point-distance 1.5 \
  --decimate 1.0

# 5. Refine mesh (multi-scale, no decimation)
$OPENMVS_BIN/RefineMesh \
  -i $WORK_DIR/scene_dense_mesh.ply \
  -o $WORK_DIR/scene_refined_mesh.ply \
  -m $WORK_DIR/scene_dense.mvs \
  --scales 3 \
  --decimate 1.0 \
  --cuda-device 0
```

**Exit Codes**:
- `0`: Success
- `2`: OpenMVS command failed

---

### 4. `split_and_validate.py`

**Purpose**: Split multi-object mesh into per-sherd PLY files and validate vertex counts.

**Inputs**:
- `--mesh`: Path to refined mesh (e.g., `scene_refined_mesh.ply`)
- `--output_dir`: Directory to save split meshes
- `--min_vertices`: Minimum vertex count per sherd (default: 100,000)

**Algorithm**:
```python
import trimesh
import pandas as pd

def split_and_validate(mesh_path, output_dir, min_vertices=100000):
    # 1. Load mesh
    mesh = trimesh.load(mesh_path)

    # 2. Split by connected components
    components = mesh.split(only_watertight=False)

    # 3. Sort by vertex count (descending)
    components = sorted(components, key=lambda c: len(c.vertices), reverse=True)

    # 4. Save each component
    results = []
    for i, component in enumerate(components, start=1):
        sherd_name = f"sherd_{i:03d}.ply"
        output_path = os.path.join(output_dir, sherd_name)
        component.export(output_path)

        status = "PASS" if len(component.vertices) >= min_vertices else "FAIL"
        results.append({
            'sherd_id': sherd_name,
            'vertex_count': len(component.vertices),
            'face_count': len(component.faces),
            'status': status
        })

    # 5. Write validation report
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'validation_report.csv'), index=False)

    # 6. Return overall status
    all_pass = all(r['status'] == 'PASS' for r in results)
    return 0 if all_pass else 3
```

**Exit Codes**:
- `0`: All sherds meet vertex requirement
- `3`: At least one sherd below threshold

---

### 5. `pipeline_main.sh`

**Purpose**: Master orchestrator that runs full pipeline for one pottery-tree.

**Inputs**:
- `$1`: Pottery-tree path (e.g., `04052025/O01`)

**Workflow**:
```bash
#!/bin/bash
set -euo pipefail

TREE_PATH=$1
CONFIG=/data/gpfs/projects/punim2657/Photogrammetry/pipeline/config/pipeline_config.yaml
RABATI_ROOT=/data/gpfs/projects/punim2657/Rabati2025
WORK_DIR=$RABATI_ROOT/$TREE_PATH/work_colmap_openmvs

# Setup
echo "=== Processing $TREE_PATH ==="
mkdir -p $WORK_DIR
LOG_FILE=$WORK_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a $LOG_FILE) 2>&1

# Log GPU info
nvidia-smi

# Stage 1: COLMAP
echo "=== Stage 1: COLMAP ==="
bash /data/gpfs/projects/punim2657/Photogrammetry/pipeline/bin/run_colmap.sh $TREE_PATH $CONFIG
if [ $? -ne 0 ]; then
    echo "ERROR: COLMAP failed"
    exit 1
fi

# Stage 2: OpenMVS
echo "=== Stage 2: OpenMVS ==="
bash /data/gpfs/projects/punim2657/Photogrammetry/pipeline/bin/run_openmvs.sh $WORK_DIR $CONFIG
if [ $? -ne 0 ]; then
    echo "ERROR: OpenMVS failed"
    exit 2
fi

# Stage 3: Split and validate
echo "=== Stage 3: Split and Validate ==="
python /data/gpfs/projects/punim2657/Photogrammetry/pipeline/bin/split_and_validate.py \
  --mesh $WORK_DIR/scene_refined_mesh.ply \
  --output_dir $WORK_DIR \
  --config $CONFIG

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "WARNING: Validation failed (exit code $EXIT_CODE)"
    echo "Check $WORK_DIR/validation_report.csv for details"
fi

echo "=== Pipeline completed for $TREE_PATH (exit code $EXIT_CODE) ==="
exit $EXIT_CODE
```

---

### 6. `submit_array.sh`

**Purpose**: Submit Slurm array job for all pottery-trees.

**Workflow**:
```bash
#!/bin/bash
PIPELINE_ROOT=/data/gpfs/projects/punim2657/Photogrammetry/pipeline
TARGETS=$PIPELINE_ROOT/targets.txt

# Count targets
NUM_TARGETS=$(wc -l < $TARGETS)
echo "Found $NUM_TARGETS pottery-trees to process"

# Submit array job
sbatch --array=1-${NUM_TARGETS}%10 \
       --partition=gpu-v100 \
       --gres=gpu:1 \
       --mem=64G \
       --cpus-per-task=8 \
       --time=12:00:00 \
       --job-name=rabati_mesh \
       --output=$PIPELINE_ROOT/logs/rabati_%A_%a.log \
       $PIPELINE_ROOT/bin/slurm_array_job.sh
```

---

### 7. `slurm_array_job.sh`

**Purpose**: Slurm batch script executed for each array task.

```bash
#!/bin/bash
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

set -euo pipefail

PIPELINE_ROOT=/data/gpfs/projects/punim2657/Photogrammetry/pipeline
TARGETS=$PIPELINE_ROOT/targets.txt

# Get tree path for this array task
TREE_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $TARGETS)

echo "=== Slurm Array Job ${SLURM_ARRAY_TASK_ID} ==="
echo "Tree: $TREE_PATH"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Run pipeline
bash $PIPELINE_ROOT/bin/pipeline_main.sh $TREE_PATH
```

---

### 8. `pipeline_utils.py`

**Purpose**: Shared Python utilities.

```python
import os
import yaml
import logging
from datetime import datetime
from pathlib import Path

def setup_logging(log_file=None, level=logging.INFO):
    """Configure structured logging."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def count_jpegs(directory, extensions=['.jpg', '.jpeg']):
    """Count JPEG files in directory (case-insensitive)."""
    extensions_lower = [ext.lower() for ext in extensions]
    count = 0
    for file in Path(directory).iterdir():
        if file.is_file() and file.suffix.lower() in extensions_lower:
            count += 1
    return count

def create_work_dir(tree_path, work_dir_name='work_colmap_openmvs'):
    """Safely create working directory."""
    work_dir = Path(tree_path) / work_dir_name
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir

def log_gpu_info():
    """Log GPU information via nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        logging.info(f"GPU detected: {result.stdout.strip()}")
    except Exception as e:
        logging.warning(f"Could not detect GPU: {e}")
```

---

## Configuration

### `pipeline_config.yaml`

```yaml
# Rabati2025 Mesh Pipeline Configuration
version: "1.0"

paths:
  rabati_root: /data/gpfs/projects/punim2657/Rabati2025
  pipeline_root: /data/gpfs/projects/punim2657/Photogrammetry/pipeline
  colmap_module: COLMAP/3.9-CUDA-11.7.0
  colmap_module_deps:
    - GCC/11.3.0
    - OpenMPI/4.1.4
  openmvs_bin: /data/gpfs/projects/punim2657/Photogrammetry/openmvs/install/bin/OpenMVS

colmap:
  image_extensions: [.jpg, .jpeg, .JPG, .JPEG]
  ignore_extensions: [.nef, .NEF, .raw, .RAW]
  camera_model: OPENCV
  feature_extractor:
    use_gpu: true
  matcher:
    use_gpu: true
    type: exhaustive  # exhaustive, sequential, or vocab_tree

openmvs:
  densify:
    resolution_level: 0        # 0=full res, 1=half res, 2=quarter res
    number_views: 8            # Supporting views per depth estimate (default: 5)
    number_views_fuse: 8       # Views for depth fusion (default: 3)
    cuda_device: 0             # GPU device ID
  reconstruct:
    min_point_distance: 1.5    # Lower = denser mesh (default: 2.5)
    decimate: 1.0              # 1.0 = no decimation
  refine:
    scales: 3                  # Multi-scale refinement levels
    decimate: 1.0              # 1.0 = no decimation
    cuda_device: 0

validation:
  min_vertices_per_sherd: 100000
  min_jpegs_per_tree: 10

slurm:
  partition: gpu-v100
  gpus_per_task: 1
  mem_gb: 64
  cpus_per_task: 8
  time_hours: 12
  concurrent_jobs: 10          # Max concurrent array tasks (%N)
```

---

## Workflow

### Full Pipeline Execution

**1. Preparation**
```bash
cd /data/gpfs/projects/punim2657/Photogrammetry
# Scan for targets
python pipeline/bin/scan_targets.py --config pipeline/config/pipeline_config.yaml
```

**2. Submission**
```bash
# Submit array job
bash pipeline/bin/submit_array.sh
# Note the job ID (e.g., 1234567)
```

**3. Monitoring**
```bash
# Check job status
squeue -u $USER

# Watch specific tree log
tail -f /data/gpfs/projects/punim2657/Rabati2025/04052025/O01/work_colmap_openmvs/pipeline_*.log

# Check array job logs
tail -f pipeline/logs/rabati_1234567_*.log
```

**4. Results Collection**
```bash
# After completion, aggregate results
python pipeline/bin/aggregate_results.py --targets pipeline/targets.txt
# Outputs: pipeline/summary_report.csv
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure (Day 1)
- [ ] Create directory structure (`pipeline/{bin,lib,config,logs}`)
- [ ] Write `pipeline_config.yaml`
- [ ] Implement `pipeline_utils.py`
- [ ] Write `requirements.txt` and install Python dependencies
- [ ] Test module loading: `module load COLMAP/3.9-CUDA-11.7.0`
- [ ] Test OpenMVS binary: `$OPENMVS_BIN/InterfaceCOLMAP --help`

### Phase 2: Scanner (Day 1)
- [ ] Implement `scan_targets.py`
- [ ] Test on Rabati2025 directory
- [ ] Verify `targets.txt` output

### Phase 3: COLMAP Wrapper (Day 2)
- [ ] Write `run_colmap.sh`
- [ ] Test on single pottery-tree (e.g., `04052025/O01`)
- [ ] Verify `sparse/` and `dense/` outputs
- [ ] Check undistorted images in `dense/images/`

### Phase 4: OpenMVS Wrapper (Day 2-3)
- [ ] Write `run_openmvs.sh`
- [ ] Test `InterfaceCOLMAP` conversion
- [ ] Test `DensifyPointCloud` with full resolution
- [ ] Test `ReconstructMesh`
- [ ] Test `RefineMesh`
- [ ] Verify final PLY output

### Phase 5: Splitting & Validation (Day 3)
- [ ] Implement `split_and_validate.py`
- [ ] Test connected component separation with `trimesh`
- [ ] Verify vertex counts
- [ ] Test validation report CSV generation

### Phase 6: Integration (Day 4)
- [ ] Write `pipeline_main.sh`
- [ ] Test end-to-end on one pottery-tree
- [ ] Verify all outputs and logs

### Phase 7: Slurm Integration (Day 4)
- [ ] Write `slurm_array_job.sh`
- [ ] Write `submit_array.sh`
- [ ] Test array job with `--array=1-2` (small subset)
- [ ] Check GPU allocation with `nvidia-smi`

### Phase 8: Production Run (Day 5+)
- [ ] Submit full array job for all targets
- [ ] Monitor progress
- [ ] Handle failures
- [ ] Collect results

---

## Testing Strategy

### Unit Tests

**Test 1: JPEG counting**
```bash
python -c "
from pipeline.lib.pipeline_utils import count_jpegs
count = count_jpegs('/data/gpfs/projects/punim2657/Rabati2025/04052025/O01')
print(f'Found {count} JPEGs')
assert count >= 10, 'Insufficient JPEGs'
"
```

**Test 2: Config loading**
```bash
python -c "
from pipeline.lib.pipeline_utils import load_config
cfg = load_config('pipeline/config/pipeline_config.yaml')
assert cfg['openmvs']['densify']['resolution_level'] == 0
print('Config OK')
"
```

### Integration Tests

**Test 3: Single tree (COLMAP only)**
```bash
bash pipeline/bin/run_colmap.sh 04052025/O01 pipeline/config/pipeline_config.yaml
# Verify: database.db, sparse/0/, dense/ exist
```

**Test 4: Single tree (OpenMVS only)**
```bash
# Assumes COLMAP already ran
bash pipeline/bin/run_openmvs.sh /data/gpfs/projects/punim2657/Rabati2025/04052025/O01/work_colmap_openmvs pipeline/config/pipeline_config.yaml
# Verify: scene.mvs, scene_dense.mvs, *.ply exist
```

**Test 5: Full pipeline (single tree)**
```bash
bash pipeline/bin/pipeline_main.sh 04052025/O01
# Verify: All outputs + validation_report.csv
```

**Test 6: Array job (2 trees)**
```bash
# Edit targets.txt to include only 2 trees
sbatch --array=1-2 pipeline/bin/slurm_array_job.sh
```

---

## Parameter Tuning Guide

### Scenario 1: Low Vertex Counts (<100k)

**Diagnosis**: Check `validation_report.csv` for failed sherds.

**Solution**: Increase density parameters in `pipeline_config.yaml`:
```yaml
openmvs:
  densify:
    number_views: 10              # Was: 8
    number_views_fuse: 10         # Was: 8
  reconstruct:
    min_point_distance: 1.0       # Was: 1.5 (lower = denser)
```

**Re-run**:
```bash
bash pipeline/bin/pipeline_main.sh <failed_tree>
```

### Scenario 2: Sherds Incorrectly Merged

**Diagnosis**: Fewer components than expected (e.g., 8 instead of 10).

**Solution**: Reduce hole-closing aggressiveness in `ReconstructMesh`:
```bash
# In run_openmvs.sh, add:
--close-holes 30  # Default is higher
```

### Scenario 3: Out of Memory

**Diagnosis**: OpenMVS crashes with CUDA OOM error.

**Solution**:
1. Request more GPU memory:
   ```yaml
   slurm:
     mem_gb: 96  # Was: 64
   ```
2. Or reduce view count:
   ```yaml
   densify:
     number_views: 6  # Was: 8
   ```

### Scenario 4: Slow Processing

**Diagnosis**: Jobs timeout after 12 hours.

**Solution**:
1. Increase time limit:
   ```yaml
   slurm:
     time_hours: 24  # Was: 12
   ```
2. Or use sequential matcher for large sets:
   ```yaml
   colmap:
     matcher:
       type: sequential  # Was: exhaustive
   ```

---

## Dependencies

### Python (`requirements.txt`)
```
trimesh>=3.20.0
numpy>=1.24.0
pyyaml>=6.0
pandas>=2.0.0
```

**Installation**:
```bash
module load Python/3.10.4
pip install --user -r pipeline/requirements.txt
```

### System Modules
```bash
module load GCC/11.3.0
module load OpenMPI/4.1.4
module load COLMAP/3.9-CUDA-11.7.0
```

---

## Expected Outputs

### Per Pottery-Tree
- **Meshes**: `sherd_001.ply` through `sherd_NNN.ply` (typically N=10)
- **Report**: `validation_report.csv`
- **Log**: `pipeline_YYYYMMDD_HHMMSS.log`

### Global
- **Target List**: `pipeline/targets.txt`
- **Summary**: `pipeline/summary_report.csv` (from `aggregate_results.py`)
- **Slurm Logs**: `pipeline/logs/rabati_<jobid>_<taskid>.log`

---

## References

1. COLMAP Documentation: https://colmap.github.io/cli.html
2. OpenMVS Wiki: https://github.com/cdcseacave/openMVS/wiki/Usage
3. Trimesh Documentation: https://trimsh.org/trimesh.html
4. Spartan HPC User Guide: https://dashboard.hpc.unimelb.edu.au/

---

**END OF DOCUMENT**
