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


def read_measurement_file(measurement_path: Path) -> Dict[str, float]:
    """
    Read measurement.env file and extract both measurement pairs.

    Args:
        measurement_path: Path to measurement.env

    Returns:
        Dict with keys: d1_real_m, d1_rec_units, d2_real_m, d2_rec_units
    """
    if not measurement_path.exists():
        raise PipelineError(f"Measurement file not found: {measurement_path}")

    measurements = {}

    with measurement_path.open() as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()

                if key in ['d1_real_m', 'd1_rec_units', 'd2_real_m', 'd2_rec_units']:
                    try:
                        measurements[key] = float(val)
                    except ValueError:
                        raise PipelineError(f"Invalid number for {key}: {val}")

    # Validate all required keys present
    required = ['d1_real_m', 'd1_rec_units', 'd2_real_m', 'd2_rec_units']
    missing = [k for k in required if k not in measurements]

    if missing:
        raise PipelineError(
            f"Invalid measurement file. Missing: {', '.join(missing)}\n"
            f"Expected format:\n"
            f"  d1_real_m=0.100\n"
            f"  d1_rec_units=0.0543271\n"
            f"  d2_real_m=0.150\n"
            f"  d2_rec_units=0.0814906"
        )

    return measurements


def compute_scale_factor(
    measurements: Dict[str, float],
    sanity_min: float = 0.01,
    sanity_max: float = 100.0,
    agreement_tolerance_pct: float = 2.0
) -> Tuple[float, Dict[str, float]]:
    """
    Compute scale factor from two measurements and validate agreement.

    Args:
        measurements: Dict with d1_real_m, d1_rec_units, d2_real_m, d2_rec_units
        sanity_min: Minimum allowed scale factor
        sanity_max: Maximum allowed scale factor
        agreement_tolerance_pct: Maximum allowed disagreement between scales (%)

    Returns:
        Tuple of (mean_scale, validation_info)

    Raises:
        PipelineError if scales are outside bounds or disagree
    """
    d1_real = measurements['d1_real_m']
    d1_rec = measurements['d1_rec_units']
    d2_real = measurements['d2_real_m']
    d2_rec = measurements['d2_rec_units']

    # Check for zero denominators
    if d1_rec == 0 or d2_rec == 0:
        raise PipelineError("Reconstructed distance is zero - check measurements")

    # Compute both scales
    scale1 = d1_real / d1_rec
    scale2 = d2_real / d2_rec

    logger.info(f"Measurement 1: {d1_real:.6f} m / {d1_rec:.6f} units = {scale1:.9f}")
    logger.info(f"Measurement 2: {d2_real:.6f} m / {d2_rec:.6f} units = {scale2:.9f}")

    # Sanity check individual scales
    if not (sanity_min <= scale1 <= sanity_max):
        raise PipelineError(
            f"Scale 1 ({scale1:.6f}) is outside sane bounds [{sanity_min}, {sanity_max}].\n"
            f"  Measurement 1: {d1_real:.6f} m / {d1_rec:.6f} units\n"
            f"Check your measurements for typos or unit errors."
        )

    if not (sanity_min <= scale2 <= sanity_max):
        raise PipelineError(
            f"Scale 2 ({scale2:.6f}) is outside sane bounds [{sanity_min}, {sanity_max}].\n"
            f"  Measurement 2: {d2_real:.6f} m / {d2_rec:.6f} units\n"
            f"Check your measurements for typos or unit errors."
        )

    # Check agreement between scales
    mean_scale = (scale1 + scale2) / 2
    diff_pct = abs(scale1 - scale2) / mean_scale * 100

    logger.info(f"Scale agreement: {diff_pct:.2f}% difference")

    if diff_pct > agreement_tolerance_pct:
        raise PipelineError(
            f"Scales disagree by {diff_pct:.2f}% (tolerance: {agreement_tolerance_pct}%)\n"
            f"  Scale 1: {scale1:.9f} (from {d1_real:.3f}m / {d1_rec:.6f})\n"
            f"  Scale 2: {scale2:.9f} (from {d2_real:.3f}m / {d2_rec:.6f})\n"
            f"Check measurements for errors:\n"
            f"  - Did you measure the same features in the PLY and real world?\n"
            f"  - Are units consistent (metres, not mm)?\n"
            f"  - Did you read the full precision from the viewer?"
        )

    logger.info(f"✓ Scales agree within {agreement_tolerance_pct}% tolerance")
    logger.info(f"Using mean scale: {mean_scale:.9f}")

    validation_info = {
        'scale1': scale1,
        'scale2': scale2,
        'mean_scale': mean_scale,
        'diff_pct': diff_pct,
        'd1_real_m': d1_real,
        'd1_rec_units': d1_rec,
        'd2_real_m': d2_real,
        'd2_rec_units': d2_rec
    }

    return mean_scale, validation_info


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
    scale_factor: float,
    validation_info: Dict[str, float]
) -> None:
    """
    Write scale computation details to log file.

    Args:
        scale_dir: Scale directory
        scale_factor: Applied scale factor (mean of two measurements)
        validation_info: Dict with both measurements and validation details
    """
    log_path = scale_dir / "scale_log.txt"

    with log_path.open("w") as f:
        f.write("=" * 70 + "\n")
        f.write("MANUAL SCALE APPLICATION (DUAL-MEASUREMENT VALIDATION)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Application time: {datetime.now().isoformat()}\n\n")

        f.write("Measurement 1:\n")
        f.write(f"  Real-world distance:    {validation_info['d1_real_m']:.6f} m ")
        f.write(f"({validation_info['d1_real_m']*1000:.1f} mm)\n")
        f.write(f"  Reconstructed distance: {validation_info['d1_rec_units']:.6f} units\n")
        f.write(f"  Scale 1:                {validation_info['scale1']:.9f}\n\n")

        f.write("Measurement 2:\n")
        f.write(f"  Real-world distance:    {validation_info['d2_real_m']:.6f} m ")
        f.write(f"({validation_info['d2_real_m']*1000:.1f} mm)\n")
        f.write(f"  Reconstructed distance: {validation_info['d2_rec_units']:.6f} units\n")
        f.write(f"  Scale 2:                {validation_info['scale2']:.9f}\n\n")

        f.write("Validation:\n")
        f.write(f"  Agreement:              {validation_info['diff_pct']:.2f}% difference\n")
        f.write(f"  Status:                 {'PASS' if validation_info['diff_pct'] < 2.0 else 'FAIL'}\n\n")

        f.write(f"Applied scale factor:     {scale_factor:.9f} (mean of both)\n\n")

        f.write("This scale factor transforms the sparse reconstruction\n")
        f.write("to real-world metric units (metres).\n\n")

        f.write("Applied to: sparse/0 → sparse_scaled/0\n")
        f.write("Regenerated: dense_scaled/ from sparse_scaled/0\n\n")

        f.write("=" * 70 + "\n")

    logger.info(f"✓ Scale log: {log_path}")


def append_to_runlog(
    work_dir: Path,
    scale_factor: float,
    validation_info: Dict[str, float]
) -> None:
    """
    Append scale application to pipeline runlog.

    Args:
        work_dir: Work directory
        scale_factor: Applied scale factor (mean)
        validation_info: Validation details from both measurements
    """
    runlog_path = work_dir / "pipeline_RUNLOG.txt"

    log_line = (
        f"scaled=1 "
        f"scale={scale_factor:.9f} "
        f"scale1={validation_info['scale1']:.9f} "
        f"scale2={validation_info['scale2']:.9f} "
        f"diff_pct={validation_info['diff_pct']:.2f} "
        f"time={datetime.now().isoformat()}\n"
    )

    with runlog_path.open("a") as f:
        f.write(log_line)

    logger.info(f"✓ Runlog updated: {runlog_path}")
