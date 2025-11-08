#!/usr/bin/env python3
"""
Phase 2: Mask projection from edited mesh.

Validates edited mesh and projects into all camera views to generate masks.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

from pipeline.lib.maskbuild_utils import (
    compute_bundle_hash,
    compute_file_hash,
    read_cameras_binary,
    read_images_binary,
    validate_mesh_frame,
)
from pipeline.lib.mesh_projection import project_mesh_to_mask
from pipeline.lib.pipeline_utils import (
    PipelineError,
    configure_logging,
    ensure_directory,
    load_config,
)


def run_phase2(
    work_dir: Path,
    mesh_path: Path,
    config_path: Path,
    pad_pixels: int = 2,
    erosion: int = 0
) -> int:
    """
    Phase 2: Validate edited mesh and project to masks.

    Steps:
    1. Load edited mesh
    2. Load sparse cameras from Phase 1
    3. Validate mesh is in same coordinate frame
    4. For each camera:
       a. Load camera parameters
       b. Project mesh to image plane
       c. Rasterize silhouette
       d. Apply padding and optional erosion
       e. Write PNG mask
    5. Compute coverage statistics
    6. Generate manifest with checksums
    7. Write summary report

    Args:
        work_dir: Work directory (work_colmap_openmvs)
        mesh_path: Path to edited mesh (PLY or OBJ)
        config_path: Path to pipeline config
        pad_pixels: Padding around mask edges (default: 2)
        erosion: Erosion iterations (default: 0)

    Returns:
        0 on success, non-zero on failure
    """
    config = load_config(config_path)
    proj_cfg = config.get("maskbuild", {}).get("projection", {})

    # Override with command-line args if provided
    if pad_pixels is None:
        pad_pixels = proj_cfg.get("pad_pixels", 2)
    if erosion is None:
        erosion = proj_cfg.get("erosion", 0)

    mask_build_dir = work_dir / "mask_build_user"
    masks_dir = work_dir / "masks_user"

    # Setup logging (append to Phase 1 log)
    log_file = mask_build_dir / "RUNLOG.txt"
    logger = configure_logging(log_file, console=True)
    logger.info("")
    logger.info("=" * 70)
    logger.info("MASKBUILD PHASE 2: Mask Projection")
    logger.info("=" * 70)
    logger.info(f"Edited mesh: {mesh_path}")
    logger.info(f"Pad pixels: {pad_pixels}")
    logger.info(f"Erosion: {erosion}")

    start_time = time.time()

    # 1. Load edited mesh
    if not mesh_path.exists():
        raise PipelineError(
            f"Edited mesh not found: {mesh_path}\n"
            f"Expected location: {mask_build_dir}/edited_model.ply\n"
            "Please edit the coarse model and save as edited_model.ply"
        )

    logger.info("Loading edited mesh...")
    try:
        edited_mesh = trimesh.load(mesh_path)
    except Exception as e:
        raise PipelineError(f"Failed to load mesh: {e}")

    logger.info(f"  Vertices: {len(edited_mesh.vertices)}")
    logger.info(f"  Faces: {len(edited_mesh.faces)}")

    # Compute mesh hash
    mesh_hash = compute_file_hash(mesh_path)
    logger.info(f"  SHA-256: {mesh_hash[:16]}...")

    # 2. Load sparse reconstruction
    sparse_dir = mask_build_dir / "sparse" / "0"
    if not sparse_dir.exists():
        raise PipelineError(
            f"Sparse reconstruction not found: {sparse_dir}\n"
            "Run Phase 1 (user-init) first to build the coarse model."
        )

    logger.info("Loading COLMAP sparse model...")
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")
    logger.info(f"  Cameras: {len(cameras)}")
    logger.info(f"  Images: {len(images)}")

    # Compute camera bundle hash
    bundle_hash = compute_bundle_hash(sparse_dir)
    logger.info(f"  Bundle SHA-256: {bundle_hash[:16]}...")

    # 3. Validate coordinate frame
    logger.info("Validating coordinate frame...")
    original_mesh_path = mask_build_dir / "coarse_model.ply"

    bbox_tol = proj_cfg.get("bbox_tolerance", 0.1)
    dist_tol = proj_cfg.get("distance_tolerance", 0.05)

    validation = validate_mesh_frame(
        mesh=edited_mesh,
        sparse_dir=sparse_dir,
        original_mesh_path=original_mesh_path,
        bbox_tolerance=bbox_tol,
        distance_tolerance=dist_tol
    )

    if not validation["is_valid"]:
        raise PipelineError(
            f"Mesh coordinate frame validation failed:\n"
            f"  {validation['error']}\n\n"
            "The edited mesh appears to have been moved, rotated, or scaled.\n"
            "Please re-edit the mesh WITHOUT applying any transformations.\n"
            "Only delete unwanted geometry - do not move or transform the mesh."
        )

    logger.info(f"  ✓ {validation['message']}")

    # Save validation results
    validation_path = mask_build_dir / "frame_validation.json"
    with validation_path.open("w") as f:
        json.dump(validation, f, indent=2)

    # 4. Project masks
    ensure_directory(masks_dir)
    logger.info(f"Projecting masks to {masks_dir}...")

    mask_records = []
    failed_images = []
    coverage_stats = []

    for idx, (image_id, image_info) in enumerate(images.items(), 1):
        image_name = image_info["name"]
        camera_id = image_info["camera_id"]
        camera = cameras[camera_id]

        # Get camera pose (world-to-camera)
        qvec = image_info["qvec"]
        tvec = image_info["tvec"]

        try:
            # Project mesh to mask
            mask = project_mesh_to_mask(
                mesh=edited_mesh,
                camera=camera,
                qvec=qvec,
                tvec=tvec,
                image_width=camera["width"],
                image_height=camera["height"],
                pad_pixels=pad_pixels,
                erosion=erosion
            )

            # Save mask (same basename as image, but .png extension)
            mask_filename = Path(image_name).stem + ".png"
            mask_path = masks_dir / mask_filename

            cv2.imwrite(str(mask_path), mask)

            # Compute coverage statistics
            mask_area = (mask > 0).sum()
            total_pixels = mask.shape[0] * mask.shape[1]
            coverage_pct = 100.0 * mask_area / total_pixels

            coverage_stats.append(coverage_pct)

            # Compute mask checksum
            mask_hash = compute_file_hash(mask_path)

            mask_records.append({
                "image_name": image_name,
                "mask_filename": mask_filename,
                "mask_sha256": mask_hash,
                "coverage_percent": float(coverage_pct),
                "mask_pixels": int(mask_area),
                "image_pixels": int(total_pixels)
            })

            if idx % 20 == 0:
                logger.info(f"  Projected {idx}/{len(images)} masks...")

        except Exception as e:
            logger.warning(f"  Failed to project {image_name}: {e}")
            failed_images.append(image_name)

    logger.info(f"✓ Projected {len(mask_records)}/{len(images)} masks")
    if failed_images:
        logger.warning(f"⚠ Failed to project {len(failed_images)} images")

    # 5. Coverage statistics
    coverage_array = np.array(coverage_stats)

    stats_summary = {
        "total_images": len(images),
        "masks_generated": len(mask_records),
        "masks_failed": len(failed_images),
        "coverage": {
            "mean_percent": float(coverage_array.mean()),
            "median_percent": float(np.median(coverage_array)),
            "std_percent": float(coverage_array.std()),
            "min_percent": float(coverage_array.min()),
            "max_percent": float(coverage_array.max()),
            "p25_percent": float(np.percentile(coverage_array, 25)),
            "p75_percent": float(np.percentile(coverage_array, 75))
        }
    }

    logger.info("Coverage statistics:")
    logger.info(f"  Mean: {stats_summary['coverage']['mean_percent']:.1f}%")
    logger.info(f"  Median: {stats_summary['coverage']['median_percent']:.1f}%")
    logger.info(f"  Range: {stats_summary['coverage']['min_percent']:.1f}% - {stats_summary['coverage']['max_percent']:.1f}%")

    # Detect outliers (coverage < 5% or > 95%)
    outliers = [
        r for r in mask_records
        if r["coverage_percent"] < 5.0 or r["coverage_percent"] > 95.0
    ]

    if outliers:
        logger.warning(f"⚠ {len(outliers)} outliers detected (coverage <5% or >95%)")

    # 6. Write manifest
    manifest_path = work_dir / "masks_manifest.json"
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "edited_mesh_path": str(mesh_path),
        "edited_mesh_sha256": mesh_hash,
        "sparse_bundle_sha256": bundle_hash,
        "masks_directory": str(masks_dir),
        "projection_params": {
            "pad_pixels": pad_pixels,
            "erosion": erosion
        },
        "statistics": stats_summary,
        "outliers": outliers,
        "failed_images": failed_images,
        "masks": mask_records
    }

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"✓ Manifest: {manifest_path}")

    # 7. Write summary report
    report_path = work_dir / "masks_report.txt"
    with report_path.open("w") as f:
        f.write("=" * 70 + "\n")
        f.write("MASK GENERATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {manifest['generated_at']}\n")
        f.write(f"Edited mesh: {mesh_path.name}\n")
        f.write(f"Mesh SHA-256: {mesh_hash}\n")
        f.write(f"Bundle SHA-256: {bundle_hash}\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total images:      {stats_summary['total_images']}\n")
        f.write(f"Masks generated:   {stats_summary['masks_generated']}\n")
        f.write(f"Failed:            {stats_summary['masks_failed']}\n\n")

        f.write("COVERAGE STATISTICS\n")
        f.write("-" * 70 + "\n")
        cov = stats_summary['coverage']
        f.write(f"Mean:              {cov['mean_percent']:.1f}%\n")
        f.write(f"Median:            {cov['median_percent']:.1f}%\n")
        f.write(f"Std Dev:           {cov['std_percent']:.1f}%\n")
        f.write(f"Range:             {cov['min_percent']:.1f}% - {cov['max_percent']:.1f}%\n")
        f.write(f"IQR (25-75):       {cov['p25_percent']:.1f}% - {cov['p75_percent']:.1f}%\n\n")

        if outliers:
            f.write("OUTLIERS (coverage <5% or >95%)\n")
            f.write("-" * 70 + "\n")
            for rec in outliers[:10]:
                f.write(f"  {rec['image_name']:<40} {rec['coverage_percent']:>6.1f}%\n")
            if len(outliers) > 10:
                f.write(f"  ... and {len(outliers) - 10} more\n")
            f.write("\n")

        if failed_images:
            f.write("FAILED IMAGES\n")
            f.write("-" * 70 + "\n")
            for img in failed_images:
                f.write(f"  {img}\n")
            f.write("\n")

        f.write("NEXT STEPS\n")
        f.write("-" * 70 + "\n")
        f.write("The masks have been generated and are ready for use.\n\n")
        f.write("To rerun COLMAP Stage 1 with these masks:\n\n")
        f.write(f"  python pipeline/bin/run_colmap.py <tree_dir>\n\n")
        f.write("The pipeline will automatically detect and use masks_user/\n\n")
        f.write("Or explicitly specify the mask path:\n\n")
        f.write(f"  python pipeline/bin/run_colmap.py <tree_dir> \\\n")
        f.write(f"    --mask-path {masks_dir}\n\n")
        f.write("Expected improvements:\n")
        f.write("  ✓ More images registered (fewer unregistered)\n")
        f.write("  ✓ Lower reprojection errors\n")
        f.write("  ✓ Features only detected on sherd surfaces\n")
        f.write("  ✓ Cleaner sparse point cloud\n")
        f.write("  ✓ Better dense reconstruction (no background)\n")

    logger.info(f"✓ Report: {report_path}")

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"✓ Phase 2 complete in {elapsed:.1f}s")
    logger.info(f"✓ Masks: {masks_dir} ({len(mask_records)} files)")
    logger.info(f"✓ Manifest: {manifest_path}")
    logger.info(f"✓ Report: {report_path}")
    logger.info("")
    logger.info("NEXT: Rerun COLMAP with masks to improve reconstruction")
    logger.info("=" * 70)

    # One-line summary for batch logs
    print(f"MASKBUILD_PHASE2_OK masks={len(mask_records)} coverage={cov['median_percent']:.1f}% time={elapsed:.1f}s")

    return 0
