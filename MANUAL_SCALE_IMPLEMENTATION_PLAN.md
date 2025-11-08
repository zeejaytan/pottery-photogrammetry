# Implementation Plan: Manual Scale Measurement Workflow

## Overview

This plan implements a manual scale measurement workflow that:
1. **Exports sparse model** to PLY/TXT after COLMAP reconstruction
2. **Allows user measurement** on laptop (CloudCompare/MeshLab)
3. **Applies scale factor** to sparse model on Spartan (headless)
4. **Regenerates dense workspace** from scaled sparse for OpenMVS
5. **Maintains full provenance** and reproducibility

This approach scales the entire reconstruction chain (cameras, depth maps, meshes) coherently by transforming the sparse model before undistortion, rather than post-processing the final mesh.

---

## Architecture

### New Files to Create

```
pipeline/
├── bin/
│   ├── scale_export.py              # Export sparse to PLY/TXT (NEW)
│   ├── scale_apply.py               # Apply scale from measurement (NEW)
│   └── run_colmap.py                # Modified for scale integration
├── lib/
│   ├── manual_scale.py              # Scale computation utilities (NEW)
│   └── pipeline_utils.py            # Extended with scale helpers
├── config/
│   └── pipeline_config.yaml         # Add manual_scale section
└── docs/
    └── MANUAL_SCALE_USAGE.md        # User guide (NEW)
```

### Output Directory Structure

```
work_colmap_openmvs/
├── sparse/
│   └── 0/                           # Original sparse (unscaled)
├── sparse_txt/                      # TXT export for provenance (NEW)
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.txt
├── sparse_ply/                      # PLY export for measurement (NEW)
│   └── points3D.ply
├── scale/                           # Scale measurement workspace (NEW)
│   ├── MANIFEST.txt                 # Run context & metadata
│   ├── measurement.env              # User input: d_real_m, d_rec_units
│   ├── SCALE.txt                    # Computed scale factor
│   └── scale_log.txt                # Scaling operation log
├── sparse_scaled/                   # Scaled sparse model (NEW)
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── dense_scaled/                    # Dense workspace from scaled sparse (NEW)
│   ├── images/
│   ├── sparse/
│   ├── stereo/
│   └── fused.ply
├── dense/                           # Original dense (kept for reference)
└── pipeline_RUNLOG.txt              # Audit log (EXTENDED)
```

---

## Implementation Details

### 1. Sparse Model Export Script (`pipeline/bin/scale_export.py`)

**Purpose**: Export COLMAP sparse model to PLY and TXT formats for measurement

**Interface**:
```python
#!/usr/bin/env python3
"""
Export COLMAP sparse model to PLY and TXT for scale measurement.

Usage:
    scale_export.py --work <path> [--config <path>]

Outputs:
    - sparse_ply/points3D.ply  (for laptop measurement)
    - sparse_txt/*.txt         (for provenance)
    - scale/MANIFEST.txt       (run metadata)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.manual_scale import export_sparse_for_measurement
from lib.pipeline_utils import PipelineError


def main():
    parser = argparse.ArgumentParser(
        description="Export COLMAP sparse model for scale measurement"
    )

    parser.add_argument(
        "--work",
        required=True,
        type=Path,
        help="Path to work_colmap_openmvs directory"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline/config/pipeline_config.yaml"),
        help="Path to pipeline configuration"
    )

    parser.add_argument(
        "--sparse-model",
        type=Path,
        default=None,
        help="Path to sparse model (default: work/sparse/0)"
    )

    parser.add_argument(
        "--colmap",
        default="colmap",
        help="COLMAP executable path"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.work.exists():
        print(f"ERROR: Work directory does not exist: {args.work}", file=sys.stderr)
        return 1

    # Determine sparse model path
    if args.sparse_model:
        sparse_path = args.sparse_model
    else:
        sparse_path = args.work / "sparse" / "0"

    if not sparse_path.exists():
        print(f"ERROR: Sparse model not found: {sparse_path}", file=sys.stderr)
        print("Run COLMAP Stage 1 first to generate sparse reconstruction", file=sys.stderr)
        return 1

    try:
        print("=" * 70)
        print("SPARSE MODEL EXPORT FOR SCALE MEASUREMENT")
        print("=" * 70)
        print(f"Work directory: {args.work}")
        print(f"Sparse model: {sparse_path}")
        print()

        # Export sparse to PLY and TXT
        export_paths = export_sparse_for_measurement(
            work_dir=args.work,
            sparse_path=sparse_path,
            colmap_exec=args.colmap
        )

        print()
        print("✓ Export complete")
        print()
        print("Next steps:")
        print(f"  1. Copy {export_paths['ply']} to your laptop")
        print(f"  2. Open in CloudCompare or MeshLab")
        print(f"  3. Measure a known distance (e.g., base diameter)")
        print(f"     - Note the REAL-WORLD distance in metres (e.g., 0.100 for 100mm)")
        print(f"     - Note the RECONSTRUCTED distance from the viewer")
        print(f"  4. On Spartan, edit {args.work}/scale/measurement.env:")
        print(f"       d_real_m=0.100")
        print(f"       d_rec_units=0.0XYZ")
        print(f"  5. Run: python pipeline/bin/scale_apply.py --work {args.work}")
        print()

        return 0

    except PipelineError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

### 2. Manual Scale Library (`lib/manual_scale.py`)

**Purpose**: Core functions for exporting, computing, and applying scale

```python
"""
Manual scale measurement and application.

This module provides functions to export COLMAP sparse models for measurement,
compute scale factors from user measurements, and apply scaling transformations.
"""

import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

from .pipeline_utils import PipelineError, run_command

logger = logging.getLogger(__name__)


def export_sparse_for_measurement(
    work_dir: Path,
    sparse_path: Path,
    colmap_exec: str = "colmap"
) -> Dict[str, Path]:
    """
    Export COLMAP sparse model to PLY and TXT formats.

    Args:
        work_dir: Work directory (work_colmap_openmvs)
        sparse_path: Path to sparse/0 directory
        colmap_exec: COLMAP executable

    Returns:
        Dict with paths: {'ply': ..., 'txt': ..., 'manifest': ...}
    """
    logger.info("Exporting sparse model for scale measurement...")

    # Create output directories
    sparse_ply_dir = work_dir / "sparse_ply"
    sparse_txt_dir = work_dir / "sparse_txt"
    scale_dir = work_dir / "scale"

    sparse_ply_dir.mkdir(exist_ok=True)
    sparse_txt_dir.mkdir(exist_ok=True)
    scale_dir.mkdir(exist_ok=True)

    # Export to PLY (for laptop measurement)
    logger.info(f"Exporting PLY to {sparse_ply_dir}...")
    ply_cmd = [
        colmap_exec,
        "model_converter",
        "--input_path", str(sparse_path),
        "--output_path", str(sparse_ply_dir),
        "--output_type", "PLY"
    ]
    run_command(ply_cmd, logger=logger)

    # Export to TXT (for provenance)
    logger.info(f"Exporting TXT to {sparse_txt_dir}...")
    txt_cmd = [
        colmap_exec,
        "model_converter",
        "--input_path", str(sparse_path),
        "--output_path", str(sparse_txt_dir),
        "--output_type", "TXT"
    ]
    run_command(txt_cmd, logger=logger)

    # Create manifest
    manifest_path = scale_dir / "MANIFEST.txt"
    tree_name = work_dir.parent.name if work_dir.parent.name != "work_colmap_openmvs" else "unknown"

    with manifest_path.open("w") as f:
        f.write(f"tree={tree_name}\n")
        f.write(f"sparse_model={sparse_path.relative_to(work_dir)}\n")
        f.write(f"export_time={datetime.now().isoformat()}\n")
        f.write(f"ply_export={sparse_ply_dir.relative_to(work_dir)}\n")
        f.write(f"txt_export={sparse_txt_dir.relative_to(work_dir)}\n")

    logger.info(f"✓ Manifest: {manifest_path}")

    return {
        'ply': sparse_ply_dir / "points3D.ply",
        'txt': sparse_txt_dir,
        'manifest': manifest_path
    }


def read_measurement_file(measurement_path: Path) -> Tuple[float, float]:
    """
    Read measurement.env file and extract d_real_m and d_rec_units.

    Args:
        measurement_path: Path to measurement.env

    Returns:
        Tuple of (d_real_m, d_rec_units)
    """
    if not measurement_path.exists():
        raise PipelineError(f"Measurement file not found: {measurement_path}")

    d_real = None
    d_rec = None

    with measurement_path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith('d_real_m='):
                d_real = float(line.split('=')[1])
            elif line.startswith('d_rec_units='):
                d_rec = float(line.split('=')[1])

    if d_real is None or d_rec is None:
        raise PipelineError(
            f"Invalid measurement file. Expected:\n"
            f"  d_real_m=0.100\n"
            f"  d_rec_units=0.0XYZ"
        )

    return d_real, d_rec


def compute_scale_factor(
    d_real_m: float,
    d_rec_units: float,
    sanity_min: float = 0.01,
    sanity_max: float = 100.0
) -> float:
    """
    Compute scale factor and validate it's within sane bounds.

    Args:
        d_real_m: Real-world distance in metres
        d_rec_units: Reconstructed distance in model units
        sanity_min: Minimum allowed scale factor
        sanity_max: Maximum allowed scale factor

    Returns:
        Scale factor

    Raises:
        PipelineError if scale is outside sane bounds
    """
    if d_rec_units == 0:
        raise PipelineError("Reconstructed distance is zero - check measurement")

    scale = d_real_m / d_rec_units

    if not (sanity_min <= scale <= sanity_max):
        raise PipelineError(
            f"Scale factor {scale:.6f} is outside sane bounds [{sanity_min}, {sanity_max}].\n"
            f"  d_real_m = {d_real_m}\n"
            f"  d_rec_units = {d_rec_units}\n"
            f"  scale = {scale:.6f}\n"
            f"Check your measurements for typos or unit errors."
        )

    logger.info(f"Computed scale factor: {scale:.9f}")
    logger.info(f"  Real-world distance: {d_real_m} m ({d_real_m*1000:.1f} mm)")
    logger.info(f"  Reconstructed distance: {d_rec_units:.6f} units")

    return scale


def apply_scale_to_sparse(
    sparse_input: Path,
    sparse_output: Path,
    scale_factor: float,
    colmap_exec: str = "colmap"
) -> None:
    """
    Apply scale transformation to COLMAP sparse model.

    Args:
        sparse_input: Input sparse/0 directory
        sparse_output: Output sparse_scaled/0 directory
        scale_factor: Scale factor to apply
        colmap_exec: COLMAP executable
    """
    logger.info(f"Applying scale {scale_factor:.9f} to sparse model...")

    # Create output directory
    sparse_output.parent.mkdir(parents=True, exist_ok=True)
    sparse_output.mkdir(parents=True, exist_ok=True)

    # Apply transformation
    transform_cmd = [
        colmap_exec,
        "model_transformer",
        "--input_path", str(sparse_input),
        "--output_path", str(sparse_output),
        "--transform_type", "scale",
        "--scale", str(scale_factor)
    ]

    run_command(transform_cmd, logger=logger)

    logger.info(f"✓ Scaled sparse model: {sparse_output}")


def regenerate_dense_workspace(
    image_path: Path,
    sparse_scaled_path: Path,
    dense_output: Path,
    max_image_size: int = 6000,
    colmap_exec: str = "colmap"
) -> None:
    """
    Regenerate undistorted dense workspace from scaled sparse model.

    Args:
        image_path: Path to original images
        sparse_scaled_path: Path to scaled sparse/0
        dense_output: Output dense_scaled directory
        max_image_size: Max image size for undistortion
        colmap_exec: COLMAP executable
    """
    logger.info("Regenerating dense workspace from scaled sparse...")

    # Remove old dense_scaled if it exists
    if dense_output.exists():
        logger.info(f"Removing existing {dense_output}")
        import shutil
        shutil.rmtree(dense_output)

    dense_output.mkdir(parents=True, exist_ok=True)

    # Run image_undistorter
    undistort_cmd = [
        colmap_exec,
        "image_undistorter",
        "--image_path", str(image_path),
        "--input_path", str(sparse_scaled_path),
        "--output_path", str(dense_output),
        "--max_image_size", str(max_image_size)
    ]

    run_command(undistort_cmd, logger=logger)

    logger.info(f"✓ Dense workspace regenerated: {dense_output}")


def write_scale_log(
    scale_dir: Path,
    d_real_m: float,
    d_rec_units: float,
    scale_factor: float
) -> None:
    """
    Write scale computation details to log file.

    Args:
        scale_dir: Scale directory
        d_real_m: Real-world distance
        d_rec_units: Reconstructed distance
        scale_factor: Computed scale
    """
    log_path = scale_dir / "scale_log.txt"

    with log_path.open("w") as f:
        f.write("=" * 70 + "\n")
        f.write("MANUAL SCALE APPLICATION\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Application time: {datetime.now().isoformat()}\n\n")

        f.write("Measurements:\n")
        f.write(f"  Real-world distance:    {d_real_m:.6f} m ({d_real_m*1000:.1f} mm)\n")
        f.write(f"  Reconstructed distance: {d_rec_units:.6f} units\n\n")

        f.write(f"Computed scale factor: {scale_factor:.9f}\n\n")

        f.write("This scale factor transforms the sparse reconstruction\n")
        f.write("to real-world metric units (metres).\n\n")

        f.write("Applied to: sparse/0 → sparse_scaled/0\n")
        f.write("Regenerated: dense_scaled/ from sparse_scaled/0\n\n")

        f.write("=" * 70 + "\n")

    logger.info(f"✓ Scale log: {log_path}")


def append_to_runlog(
    work_dir: Path,
    scale_factor: float,
    d_real_m: float,
    d_rec_units: float
) -> None:
    """
    Append scale application to pipeline runlog.

    Args:
        work_dir: Work directory
        scale_factor: Applied scale factor
        d_real_m: Real-world distance
        d_rec_units: Reconstructed distance
    """
    runlog_path = work_dir / "pipeline_RUNLOG.txt"

    log_line = (
        f"scaled=1 scale={scale_factor:.9f} "
        f"d_real_m={d_real_m:.6f} d_rec={d_rec_units:.6f} "
        f"time={datetime.now().isoformat()}\n"
    )

    with runlog_path.open("a") as f:
        f.write(log_line)

    logger.info(f"✓ Runlog updated: {runlog_path}")
```

---

### 3. Scale Application Script (`pipeline/bin/scale_apply.py`)

**Purpose**: Read user measurements and apply scale to sparse model

```python
#!/usr/bin/env python3
"""
Apply manual scale measurement to COLMAP sparse model.

Reads measurement.env, computes scale factor, applies to sparse model,
and regenerates dense workspace.

Usage:
    scale_apply.py --work <path> [--config <path>]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.manual_scale import (
    read_measurement_file,
    compute_scale_factor,
    apply_scale_to_sparse,
    regenerate_dense_workspace,
    write_scale_log,
    append_to_runlog
)
from lib.pipeline_utils import PipelineError, PipelineContext


def main():
    parser = argparse.ArgumentParser(
        description="Apply manual scale measurement to sparse model"
    )

    parser.add_argument(
        "--work",
        required=True,
        type=Path,
        help="Path to work_colmap_openmvs directory"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline/config/pipeline_config.yaml"),
        help="Path to pipeline configuration"
    )

    parser.add_argument(
        "--colmap",
        default="colmap",
        help="COLMAP executable path"
    )

    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Path to original images (default: auto-detect from tree)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.work.exists():
        print(f"ERROR: Work directory does not exist: {args.work}", file=sys.stderr)
        return 1

    measurement_file = args.work / "scale" / "measurement.env"
    if not measurement_file.exists():
        print(f"ERROR: Measurement file not found: {measurement_file}", file=sys.stderr)
        print("Create it with:", file=sys.stderr)
        print("  d_real_m=0.100", file=sys.stderr)
        print("  d_rec_units=0.0XYZ", file=sys.stderr)
        return 1

    try:
        print("=" * 70)
        print("MANUAL SCALE APPLICATION")
        print("=" * 70)
        print(f"Work directory: {args.work}")
        print()

        # Step 1: Read measurements
        print("Reading measurements from", measurement_file)
        d_real, d_rec = read_measurement_file(measurement_file)
        print(f"  Real-world distance: {d_real} m ({d_real*1000:.1f} mm)")
        print(f"  Reconstructed distance: {d_rec}")
        print()

        # Step 2: Compute scale factor
        print("Computing scale factor...")
        scale_factor = compute_scale_factor(d_real, d_rec)
        print(f"  Scale factor: {scale_factor:.9f}")
        print()

        # Save scale factor
        scale_file = args.work / "scale" / "SCALE.txt"
        with scale_file.open("w") as f:
            f.write(f"{scale_factor:.9f}\n")
        print(f"✓ Scale factor saved: {scale_file}")
        print()

        # Step 3: Apply scale to sparse
        sparse_input = args.work / "sparse" / "0"
        sparse_output = args.work / "sparse_scaled" / "0"

        print(f"Applying scale to sparse model...")
        print(f"  Input:  {sparse_input}")
        print(f"  Output: {sparse_output}")
        apply_scale_to_sparse(
            sparse_input,
            sparse_output,
            scale_factor,
            colmap_exec=args.colmap
        )
        print()

        # Step 4: Regenerate dense workspace
        # Determine image path
        if args.image_path:
            image_path = args.image_path
        else:
            # Try to find tree directory
            tree_dir = args.work.parent
            if (tree_dir / "images_jpg").exists():
                image_path = tree_dir / "images_jpg"
            elif any(tree_dir.glob("*.jpg")) or any(tree_dir.glob("*.JPG")):
                image_path = tree_dir
            else:
                print("ERROR: Cannot determine image path. Use --image-path", file=sys.stderr)
                return 1

        dense_output = args.work / "dense_scaled"

        print(f"Regenerating dense workspace...")
        print(f"  Images: {image_path}")
        print(f"  Output: {dense_output}")

        # Load config for max_image_size
        context = PipelineContext.from_config_path(args.config)
        max_image_size = context.config.get("colmap", {}).get("image_undistorter", {}).get("max_image_size", 6000)

        regenerate_dense_workspace(
            image_path,
            sparse_output,
            dense_output,
            max_image_size=max_image_size,
            colmap_exec=args.colmap
        )
        print()

        # Step 5: Write logs
        print("Writing logs...")
        write_scale_log(args.work / "scale", d_real, d_rec, scale_factor)
        append_to_runlog(args.work, scale_factor, d_real, d_rec)
        print()

        print("=" * 70)
        print("SCALE APPLICATION COMPLETE")
        print("=" * 70)
        print(f"Scaled sparse model: {sparse_output}")
        print(f"Scaled dense workspace: {dense_output}")
        print()
        print("Next steps:")
        print(f"  1. Verify scale in CloudCompare (open dense_scaled/fused.ply)")
        print(f"  2. Continue to OpenMVS using dense_scaled/ as input")
        print(f"     InterfaceCOLMAP -i {dense_output} -o scene.mvs")
        print()

        return 0

    except PipelineError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

### 4. Integration with `run_colmap.py`

Add optional flags to export sparse model after mapper:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute COLMAP SfM pipeline.")
    # ... existing args ...

    parser.add_argument(
        "--export-sparse",
        action="store_true",
        help="Export sparse model to PLY/TXT for scale measurement after mapper.",
    )

    return parser.parse_args()


def main() -> int:
    # ... existing code ...

    # After mapper completes successfully
    model_dir = model_dirs[0]

    # Optional: Export sparse for manual scaling
    if args.export_sparse:
        logger.info("Exporting sparse model for scale measurement...")
        try:
            from lib.manual_scale import export_sparse_for_measurement

            export_paths = export_sparse_for_measurement(
                work_dir=work_dir,
                sparse_path=model_dir,
                colmap_exec=colmap_exec
            )

            logger.info(f"✓ Sparse exported to PLY: {export_paths['ply']}")
            logger.info("Run scale measurement workflow before continuing to dense reconstruction")
            logger.info("  1. Copy PLY to laptop and measure known distance")
            logger.info("  2. Edit work_dir/scale/measurement.env with measurements")
            logger.info("  3. Run: python pipeline/bin/scale_apply.py --work <work_dir>")

            # Optionally stop here to let user apply scale
            if not args.skip_dense:
                response = input("\nContinue to dense reconstruction without scaling? [y/N]: ")
                if response.lower() != 'y':
                    logger.info("Stopping before dense reconstruction. Apply scale and re-run.")
                    return 0

        except Exception as e:
            logger.warning(f"Sparse export failed: {e}")

    # Check if scaled sparse exists and prefer it
    scaled_sparse = work_dir / "sparse_scaled" / "0"
    scaled_dense = work_dir / "dense_scaled"

    if scaled_sparse.exists() and not scaled_dense.exists():
        logger.info("Found scaled sparse model, using it for undistortion")
        model_dir = scaled_sparse
        dense_dir = scaled_dense

    # Continue with undistortion...
```

---

### 5. Configuration (`pipeline_config.yaml` additions)

```yaml
manual_scale:
  enabled: false                    # Enable manual scale workflow
  export_after_mapper: false        # Auto-export sparse after mapper
  sanity_bounds:
    min_scale: 0.01                 # Minimum allowed scale factor
    max_scale: 100.0                # Maximum allowed scale factor
  prefer_scaled_sparse: true        # Use sparse_scaled/0 if it exists
  dense_output_name: dense_scaled   # Output directory for scaled dense
```

---

### 6. User Documentation (`docs/MANUAL_SCALE_USAGE.md`)

```markdown
# Manual Scale Measurement Workflow

This guide explains how to apply real-world scale to COLMAP reconstructions
using manual measurement on your laptop.

## Overview

Instead of coded targets, you can:
1. Export the sparse model after COLMAP Stage 1
2. Measure a known distance on your laptop (e.g., pottery base diameter)
3. Feed the measurement back to Spartan
4. Pipeline applies scale to sparse and regenerates dense workspace
5. All downstream products (meshes, textures) are in metric units

## Quick Start

### Step 1: Run COLMAP Stage 1

```bash
python pipeline/bin/run_colmap.py /path/to/tree --export-sparse
```

This runs feature extraction, matching, mapper, and then exports:
- `work_colmap_openmvs/sparse_ply/points3D.ply` (for measurement)
- `work_colmap_openmvs/sparse_txt/*.txt` (for provenance)

### Step 2: Measure on Laptop

Copy `sparse_ply/points3D.ply` to your laptop:

```bash
scp spartan:/path/to/work_colmap_openmvs/sparse_ply/points3D.ply ./
```

Open in CloudCompare or MeshLab:
1. Select two points on a known feature (e.g., pottery base diameter)
2. Use "Point Picking" or "Measure Distance" tool
3. Note the **reconstructed distance** shown (e.g., `0.0543271`)
4. You know the **real-world distance** (e.g., `100mm = 0.100m`)

### Step 3: Apply Scale on Spartan

Create measurement file:

```bash
cd /path/to/work_colmap_openmvs/scale
cat > measurement.env <<EOF
d_real_m=0.100
d_rec_units=0.0543271
EOF
```

Apply scale:

```bash
python pipeline/bin/scale_apply.py \
  --work /path/to/work_colmap_openmvs
```

This will:
- Compute scale: `0.100 / 0.0543271 = 1.840...`
- Apply to sparse: `sparse/0` → `sparse_scaled/0`
- Regenerate dense: `dense_scaled/`

### Step 4: Continue to OpenMVS

Use the scaled dense workspace:

```bash
InterfaceCOLMAP \
  -i work_colmap_openmvs/dense_scaled \
  -o scene.mvs \
  --image-folder work_colmap_openmvs/dense_scaled/images

DensifyPointCloud -i scene.mvs -o scene_dense.mvs
ReconstructMesh -i scene_dense.mvs -o mesh.mvs
```

All outputs will be in metres.

## Detailed Workflow

### Export Sparse (Alternative)

If you didn't use `--export-sparse`:

```bash
python pipeline/bin/scale_export.py \
  --work /path/to/work_colmap_openmvs
```

### Verify Scale

After applying scale, verify in CloudCompare:

```bash
# Open fused.ply from dense_scaled
# Measure the same feature you measured before
# Should now read ~0.100m instead of 0.054m
```

### Reapply Scale

If you made a mistake, just update `measurement.env` and rerun:

```bash
# Edit measurement.env with corrected values
python pipeline/bin/scale_apply.py --work /path/to/work_colmap_openmvs
```

The script will regenerate `sparse_scaled/` and `dense_scaled/`.

## Masking Considerations

### During Scale Measurement

If you're measuring the pottery base, ensure it's visible in the sparse:
- Don't use masks during Stage 1, OR
- Use masks that leave a thin band of base exposed

### After Scale Application

Once scale is applied, you can rerun Stage 1 with production masks:

```bash
python pipeline/bin/run_colmap.py /path/to/tree \
  --mask-path work_colmap_openmvs/masks_user \
  --rebuild-from-matching
```

This will use the masked images but you've already captured scale.

## Provenance

All scale information is logged:

- `scale/measurement.env` - Your input measurements
- `scale/SCALE.txt` - Computed scale factor
- `scale/scale_log.txt` - Full calculation details
- `scale/MANIFEST.txt` - Export metadata
- `pipeline_RUNLOG.txt` - Audit trail entry

## Troubleshooting

### "Scale factor outside sane bounds"

Check your measurements:
- Did you use metres for `d_real_m`? (100mm = 0.100, not 100)
- Did you copy the full precision from the viewer for `d_rec_units`?
- Are you measuring the same feature in both cases?

### "Measurement file not found"

Create the file:

```bash
cd work_colmap_openmvs/scale
nano measurement.env
# Add your measurements
```

### Dense workspace not regenerated

Check logs in `scale/scale_log.txt` for errors. Common issues:
- Image path not found (use `--image-path`)
- COLMAP not in PATH (use `--colmap /path/to/colmap`)

## Why Scale Before Dense?

Scaling after OpenMVS mesh export is tempting but problematic:
- Camera poses remain in old units
- Depth maps need recomputation
- Texturing may fail
- Hard to re-export with correct scale

By scaling the sparse and regenerating dense, the entire pipeline is coherent.

## References

- COLMAP model_transformer: https://colmap.github.io/cli.html#model-transformer
- COLMAP model_converter: https://colmap.github.io/cli.html#model-converter
```

---

## Integration Workflow

### Modified Pipeline Execution

```bash
# Stage 1: COLMAP with sparse export
python pipeline/bin/run_colmap.py /path/to/tree \
  --config pipeline/config/pipeline_config.yaml \
  --export-sparse

# Pipeline stops and asks if you want to apply scale
# Copy sparse_ply/points3D.ply to laptop, measure, return to Spartan

# Create measurement file
cd work_colmap_openmvs/scale
cat > measurement.env <<EOF
d_real_m=0.100
d_rec_units=0.0543271
EOF

# Apply scale
python pipeline/bin/scale_apply.py \
  --work work_colmap_openmvs

# Continue with OpenMVS (uses dense_scaled automatically)
python pipeline/bin/run_openmvs.py work_colmap_openmvs/dense_scaled
```

---

## Implementation Roadmap

### Phase 1: Core Export/Apply (Week 1)
- [ ] Create `lib/manual_scale.py` with export functions
- [ ] Implement `scale_export.py` CLI tool
- [ ] Test sparse export to PLY and TXT
- [ ] Validate manifest generation

### Phase 2: Scale Application (Week 1-2)
- [ ] Implement scale computation with sanity checks
- [ ] Create `scale_apply.py` CLI tool
- [ ] Test `model_transformer` integration
- [ ] Verify dense workspace regeneration

### Phase 3: Pipeline Integration (Week 2)
- [ ] Add `--export-sparse` flag to `run_colmap.py`
- [ ] Auto-detect and prefer `sparse_scaled/0`
- [ ] Update `run_openmvs.py` to use `dense_scaled/`
- [ ] Test end-to-end workflow

### Phase 4: Documentation & Testing (Week 2-3)
- [ ] Write `MANUAL_SCALE_USAGE.md`
- [ ] Add configuration section to `pipeline_config.yaml`
- [ ] Test with real pottery data
- [ ] Validate scale accuracy (±0.1%)
- [ ] User acceptance testing

---

## Success Criteria

1. **Export Quality**: PLY loads in CloudCompare/MeshLab, points are measurable
2. **Scale Accuracy**: Applied scale reproduces known dimensions within ±0.5%
3. **Dense Consistency**: `dense_scaled/` has same image count as original `dense/`
4. **OpenMVS Compatibility**: `InterfaceCOLMAP` reads `dense_scaled/` without errors
5. **Measurement Reproducibility**: Re-measuring scaled output gives expected values
6. **Provenance Complete**: All logs, measurements, and metadata are recorded

---

## Failure Handling

### Export Phase
- **Sparse model missing**: Error early with clear message to run Stage 1 first
- **COLMAP converter fails**: Log error, keep original sparse intact

### Measurement Phase
- **Invalid measurement file**: Clear error showing expected format
- **Scale outside bounds**: Abort with sanity-check message
- **Zero or negative values**: Error before calling `model_transformer`

### Application Phase
- **model_transformer fails**: Keep unscaled sparse/dense, log error
- **image_undistorter fails**: Keep scaled sparse, log error, don't delete original dense
- **Disk full**: Detect and abort before removing old dense workspace

---

## Performance Estimates

### Time Overhead (150 images, A100)

| Stage | Time | Notes |
|-------|------|-------|
| Export to PLY/TXT | 30 sec | One-time per tree |
| Laptop measurement | 2 min | Manual user step |
| Scale computation | <1 sec | Instant |
| model_transformer | 5 sec | Scales sparse in-place |
| image_undistorter | 2 min | Regenerates dense |
| **Total overhead** | **~3 min** | Plus manual measurement |

This is negligible compared to the ~10 minute COLMAP Stage 1 runtime.

---

## Comparison: Manual Scale vs Coded Targets

| Feature | Manual Scale | Coded Targets |
|---------|-------------|---------------|
| **Setup** | None (use existing sparse) | Print/place board |
| **User effort** | 2 min measurement | Minimal after setup |
| **Accuracy** | ±0.5% (user-dependent) | ±2mm (board-dependent) |
| **Automation** | Semi-automated | Fully automated |
| **Flexibility** | Measure any feature | Limited to board |
| **Overhead** | ~3 min per tree | +30 sec per tree |
| **Dependencies** | None (laptop + viewer) | OpenCV ArUco module |
| **Use case** | Retrospective scaling | Production pipeline |

**Recommendation**: Use manual scale for existing datasets or one-off projects. Use coded targets for large-scale production pipelines.

---

## Open Questions / Design Decisions

1. **Auto-detect scaled sparse?**
   - **Decision**: Yes, prefer `sparse_scaled/0` if it exists and `dense_scaled/` doesn't

2. **Prompt user during pipeline run?**
   - **Decision**: Optional prompt with `--export-sparse`, can skip for batch

3. **Allow multiple measurements?**
   - **Decision**: Single measurement for simplicity, user can rerun if needed

4. **Validate scale against expected range?**
   - **Decision**: Yes, sanity bounds [0.01, 100.0] to catch typos

5. **Keep or delete original dense/?**
   - **Decision**: Keep it, only ~2GB per tree, useful for comparison

6. **Integrate with maskbuild workflow?**
   - **Decision**: User chooses: measure base → apply scale → apply production masks

---

## Dependencies

### Existing Dependencies
- COLMAP 3.9+ with `model_converter` and `model_transformer`
- Python 3.7+
- Existing pipeline utilities

### New Dependencies
None! This workflow uses only COLMAP built-ins and Python stdlib.

### External Tools (User's Laptop)
- CloudCompare or MeshLab for measurement (free, cross-platform)

---

## References

- [COLMAP model_converter](https://colmap.github.io/cli.html#model-converter)
- [COLMAP model_transformer](https://colmap.github.io/cli.html#model-transformer)
- [CloudCompare Point Picking](https://www.cloudcompare.org/doc/wiki/index.php?title=Point_picking)
- [MeshLab Measuring Tool](https://www.meshlab.net/#description)

---

## Next Steps

1. **Review this plan** and adjust as needed
2. **Implement Phase 1** (export functionality)
3. **Test export** on a sample sparse model
4. **Implement Phase 2** (scale application)
5. **Validate scale accuracy** with known test object
6. **Document workflow** with screenshots
7. **User training** on measurement technique

This plan provides a simple, robust manual scaling workflow that integrates cleanly with your existing pipeline and requires no additional dependencies or hardware!
