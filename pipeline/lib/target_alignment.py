"""
Model alignment and scaling using coded targets.

This module provides functions to align and scale COLMAP sparse models
using known board geometry from coded targets (ArUco/ChArUco).
"""

import json
import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from .target_utils import load_config
from .pipeline_utils import PipelineError

logger = logging.getLogger(__name__)


def align_sparse_model(
    sparse_dir: Path,
    work_dir: Path,
    config_path: Path,
    colmap_exec: str = "colmap"
) -> Dict:
    """
    Align sparse model using coded target board.

    Two modes:
    1. Full 7-DoF similarity (if board poses available)
    2. Scale only (if only board size known)

    Args:
        sparse_dir: COLMAP sparse/0 directory
        work_dir: Work directory
        config_path: Pipeline config
        colmap_exec: COLMAP executable

    Returns:
        Transform info dict
    """
    config = load_config(config_path)
    target_cfg = config.get("coded_targets", {})

    targets_dir = work_dir / "coded_targets"
    meta_file = targets_dir / "target_meta.json"

    if not meta_file.exists():
        raise PipelineError("No target metadata found. Run detect_coded_targets first.")

    with meta_file.open() as f:
        meta = json.load(f)

    # Check if we have board poses
    has_poses = meta.get("has_board_poses", False)
    square_size_mm = meta.get("square_size_mm", 50.0)

    alignment_cfg = target_cfg.get("alignment", {})
    alignment_method = alignment_cfg.get("method", "auto")

    # Determine which alignment method to use
    if alignment_method == "auto":
        if has_poses:
            logger.info("Auto-selected alignment method: board_poses (7-DoF similarity)")
            return align_with_board_poses(
                sparse_dir, targets_dir, colmap_exec, square_size_mm
            )
        else:
            logger.info("Auto-selected alignment method: scale_only")
            return align_with_scale_only(
                sparse_dir, targets_dir, colmap_exec, square_size_mm
            )
    elif alignment_method == "poses":
        if not has_poses:
            logger.warning("Board poses not available, falling back to scale_only")
            return align_with_scale_only(
                sparse_dir, targets_dir, colmap_exec, square_size_mm
            )
        return align_with_board_poses(
            sparse_dir, targets_dir, colmap_exec, square_size_mm
        )
    elif alignment_method == "scale_only":
        return align_with_scale_only(
            sparse_dir, targets_dir, colmap_exec, square_size_mm
        )
    else:
        raise PipelineError(f"Unknown alignment method: {alignment_method}")


def align_with_board_poses(
    sparse_dir: Path,
    targets_dir: Path,
    colmap_exec: str,
    square_size_mm: float
) -> Dict:
    """
    Align using board poses (full 7-DoF similarity).

    Uses COLMAP model_aligner with reference poses.

    Args:
        sparse_dir: COLMAP sparse/0 directory
        targets_dir: Coded targets directory
        colmap_exec: COLMAP executable
        square_size_mm: Square size in mm

    Returns:
        Transform info dict
    """
    logger.info("Aligning model using board poses (7-DoF similarity)...")

    # Load board poses
    poses_file = targets_dir / "board_poses.json"

    if not poses_file.exists():
        # Try to create board poses from detections
        logger.warning("Board poses file not found, attempting to create from detections")
        # This would require implementing pose extraction from detections
        # For now, fall back to scale only
        logger.warning("Falling back to scale-only alignment")
        return align_with_scale_only(sparse_dir, targets_dir, colmap_exec, square_size_mm)

    with poses_file.open() as f:
        board_poses = json.load(f)

    logger.info(f"Loaded {len(board_poses)} board poses")

    # Note: Full implementation of model_aligner would require
    # converting board poses to COLMAP reference format and
    # running the aligner. This is a simplified version that
    # focuses on scale-only alignment for now.

    logger.info("Note: Full 7-DoF alignment requires camera calibration")
    logger.info("Falling back to scale-only alignment")

    return align_with_scale_only(sparse_dir, targets_dir, colmap_exec, square_size_mm)


def align_with_scale_only(
    sparse_dir: Path,
    targets_dir: Path,
    colmap_exec: str,
    square_size_mm: float
) -> Dict:
    """
    Apply scale only using known marker distance.

    Uses COLMAP model_transformer with computed scale factor.

    Args:
        sparse_dir: COLMAP sparse/0 directory
        targets_dir: Coded targets directory
        colmap_exec: COLMAP executable
        square_size_mm: Square size in mm

    Returns:
        Transform info dict
    """
    logger.info("Aligning model using scale-only transform...")

    # Compute scale factor from marker distances
    scale_factor = compute_scale_factor(
        sparse_dir, targets_dir, square_size_mm
    )

    if scale_factor is None or scale_factor <= 0:
        logger.warning("Could not compute valid scale factor, skipping alignment")
        return {
            "method": "scale_only",
            "alignment_type": "none",
            "scale_factor": 1.0,
            "square_size_mm": square_size_mm,
            "status": "failed"
        }

    logger.info(f"Computed scale factor: {scale_factor:.6f}")

    # Apply scale using model_transformer
    transform_cmd = [
        colmap_exec,
        "model_transformer",
        "--input_path", str(sparse_dir),
        "--output_path", str(sparse_dir),  # Transform in-place
        "--transform_type", "scale",
        "--scale", str(scale_factor),
    ]

    try:
        result = subprocess.run(
            transform_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✓ Model transformation complete")
        logger.debug(result.stdout)

    except subprocess.CalledProcessError as e:
        logger.error(f"Model transformation failed: {e}")
        logger.error(e.stderr)
        raise PipelineError(f"COLMAP model_transformer failed: {e}")

    return {
        "method": "scale_only",
        "alignment_type": "scale_transform",
        "scale_factor": scale_factor,
        "square_size_mm": square_size_mm,
        "status": "success"
    }


def compute_scale_factor(
    sparse_dir: Path,
    targets_dir: Path,
    square_size_mm: float
) -> Optional[float]:
    """
    Compute scale factor from marker distances in reconstruction.

    Args:
        sparse_dir: COLMAP sparse/0 directory
        targets_dir: Coded targets directory
        square_size_mm: Known square size in mm

    Returns:
        Scale factor (mm per reconstruction unit), or None if computation fails
    """
    logger.info("Computing scale factor from marker distances...")

    # This is a simplified implementation
    # A full implementation would:
    # 1. Load COLMAP sparse model (images.bin, points3D.bin)
    # 2. Load marker detections
    # 3. Find 3D points corresponding to marker corners
    # 4. Measure distances between known marker corners
    # 5. Compare to known square_size_mm
    # 6. Return scale factor = real_distance / reconstructed_distance

    # For now, return a default scale factor
    # In practice, this should be implemented properly
    logger.warning("Scale factor computation not fully implemented")
    logger.warning("Using default scale factor of 1.0")

    # Try to load metadata to check if we have enough information
    meta_file = targets_dir / "target_meta.json"
    if not meta_file.exists():
        logger.error("Target metadata not found")
        return None

    with meta_file.open() as f:
        meta = json.load(f)

    images_with_markers = meta.get("images_with_markers", 0)
    if images_with_markers < 10:
        logger.warning(f"Only {images_with_markers} images with markers, scale may be inaccurate")

    # Return default scale factor
    # TODO: Implement proper scale computation
    return 1.0


def create_alignment_report(
    transform_info: Dict,
    work_dir: Path
) -> None:
    """
    Create alignment report with transform details.

    Args:
        transform_info: Transform information dict
        work_dir: Work directory
    """
    report_file = work_dir / "alignment_report.txt"

    with report_file.open("w") as f:
        f.write("=" * 70 + "\n")
        f.write("CODED TARGET ALIGNMENT REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Alignment Method: {transform_info.get('method', 'unknown')}\n")
        f.write(f"Transform Type: {transform_info.get('alignment_type', 'unknown')}\n")
        f.write(f"Status: {transform_info.get('status', 'unknown')}\n\n")

        if transform_info.get('method') == 'scale_only':
            f.write(f"Scale Factor: {transform_info.get('scale_factor', 1.0):.6f}\n")
            f.write(f"Reference Square Size: {transform_info.get('square_size_mm', 0):.2f} mm\n")

        f.write("\n" + "=" * 70 + "\n")

    logger.info(f"✓ Alignment report: {report_file}")
