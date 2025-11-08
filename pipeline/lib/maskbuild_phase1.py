#!/usr/bin/env python3
"""
Phase 1: Quick model builder for user editing.

Builds a fast, coarse 3D reconstruction optimized for speed over quality.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import List

import trimesh

from pipeline.lib.maskbuild_utils import list_images
from pipeline.lib.pipeline_utils import (
    PipelineError,
    configure_logging,
    ensure_directory,
    load_config,
    run_command,
)


def run_phase1(
    images_dir: Path,
    work_dir: Path,
    config_path: Path
) -> int:
    """
    Phase 1: Build quick coarse model for user editing.

    Steps:
    1. Create mask_build_user/ workspace
    2. Run COLMAP feature extraction (small images, fewer features)
    3. Run COLMAP exhaustive matching (GPU)
    4. Run COLMAP mapper (lenient thresholds)
    5. Undistort images at low resolution
    6. Run OpenMVS densify (low-res depth maps)
    7. Run OpenMVS reconstruct (coarse mesh)
    8. Simplify mesh if >500k faces
    9. Export as coarse_model.ply with README
    10. Log summary and wait for user edit

    Args:
        images_dir: Directory containing JPEG images
        work_dir: Work directory (work_colmap_openmvs)
        config_path: Path to pipeline config

    Returns:
        0 on success, non-zero on failure
    """
    # Load config
    config = load_config(config_path)
    env_cfg = config.get("environment", {})
    mb_cfg = config.get("maskbuild", {}).get("quick_build", {})

    # Get executables
    colmap_exec = env_cfg.get("colmap_executable", "colmap")
    openmvs_bin_dir = Path(env_cfg.get("openmvs_bin_dir", ""))

    # Setup workspace
    mask_build_dir = work_dir / "mask_build_user"
    ensure_directory(mask_build_dir)

    sparse_dir = mask_build_dir / "sparse"
    dense_dir = mask_build_dir / "dense"
    database_path = mask_build_dir / "database.db"

    # Setup logging
    log_file = mask_build_dir / "RUNLOG.txt"
    logger = configure_logging(log_file, console=True)
    logger.info("=" * 70)
    logger.info("MASKBUILD PHASE 1: Quick Model Build")
    logger.info("=" * 70)
    logger.info(f"Images: {images_dir}")
    logger.info(f"Work dir: {work_dir}")
    logger.info(f"Config: {config_path}")

    start_time = time.time()

    # Count images
    image_files = list_images(images_dir)
    logger.info(f"Found {len(image_files)} images")

    if len(image_files) == 0:
        raise PipelineError(f"No images found in {images_dir}")

    # 1. Feature extraction (low-res)
    logger.info("Step 1/8: Feature extraction (low-res)")
    logger.info(f"  max_image_size: {mb_cfg.get('max_image_size', 2048)}")
    logger.info(f"  max_num_features: {mb_cfg.get('max_num_features', 8000)}")

    feature_cmd = [
        colmap_exec,
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.gpu_index", "0",
        "--SiftExtraction.max_image_size", str(mb_cfg.get("max_image_size", 2048)),
        "--SiftExtraction.max_num_features", str(mb_cfg.get("max_num_features", 8000)),
        "--SiftExtraction.domain_size_pooling", "1",
    ]
    run_command(feature_cmd, logger=logger)

    # 2. Exhaustive matching
    logger.info("Step 2/8: Exhaustive matching (GPU)")
    logger.info(f"  guided_matching: {mb_cfg.get('guided_matching', 1)}")

    match_cmd = [
        colmap_exec,
        "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "1",
        "--SiftMatching.gpu_index", "0",
        "--SiftMatching.guided_matching", str(mb_cfg.get("guided_matching", 1)),
    ]
    run_command(match_cmd, logger=logger)

    # 3. Sparse reconstruction (lenient)
    logger.info("Step 3/8: Sparse reconstruction (lenient thresholds)")
    logger.info(f"  init_min_tri_angle: {mb_cfg.get('mapper_min_tri_angle', 2.0)}")
    logger.info(f"  abs_pose_min_num_inliers: {mb_cfg.get('mapper_min_inliers', 12)}")

    ensure_directory(sparse_dir)

    mapper_cmd = [
        colmap_exec,
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.init_min_tri_angle", str(mb_cfg.get("mapper_min_tri_angle", 2.0)),
        "--Mapper.abs_pose_min_num_inliers", str(mb_cfg.get("mapper_min_inliers", 12)),
        "--Mapper.abs_pose_min_inlier_ratio", str(mb_cfg.get("mapper_min_inlier_ratio", 0.15)),
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_extra_params", "1",
    ]
    run_command(mapper_cmd, logger=logger)

    # Check for single model
    model_dirs = sorted([d for d in sparse_dir.iterdir() if d.is_dir()])
    if len(model_dirs) == 0:
        raise PipelineError("No sparse model created. Check image quality and overlap.")

    if len(model_dirs) > 1:
        logger.warning(f"Multiple models detected ({len(model_dirs)}), retrying with even looser thresholds...")
        shutil.rmtree(sparse_dir)
        ensure_directory(sparse_dir)

        # Retry with very lenient settings
        mapper_cmd_retry = [
            colmap_exec,
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.init_min_tri_angle", "1.0",
            "--Mapper.abs_pose_min_num_inliers", "8",
            "--Mapper.abs_pose_min_inlier_ratio", "0.12",
            "--Mapper.ba_refine_principal_point", "0",
        ]
        run_command(mapper_cmd_retry, logger=logger)

        model_dirs = sorted([d for d in sparse_dir.iterdir() if d.is_dir()])
        if len(model_dirs) != 1:
            raise PipelineError(
                f"Cannot create single unified model even with very lenient thresholds. "
                f"Found {len(model_dirs)} models. "
                "Check image overlap and quality. "
                "For quick masking, we need >90% of images to register."
            )

    model_dir = model_dirs[0]
    logger.info(f"Sparse model created: {model_dir}")

    # 4. Image undistortion (low-res)
    logger.info("Step 4/8: Image undistortion (low-res)")
    logger.info(f"  max_image_size: {mb_cfg.get('undistort_max_image_size', 2048)}")

    undistort_cmd = [
        colmap_exec,
        "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(model_dir),
        "--output_path", str(dense_dir),
        "--max_image_size", str(mb_cfg.get("undistort_max_image_size", 2048)),
    ]
    run_command(undistort_cmd, logger=logger)

    # 5. OpenMVS InterfaceCOLMAP
    logger.info("Step 5/8: OpenMVS InterfaceCOLMAP")

    scene_mvs = mask_build_dir / "scene.mvs"
    interface_bin = openmvs_bin_dir / "InterfaceCOLMAP"

    interface_cmd = [
        str(interface_bin),
        "-w", str(mask_build_dir),
        "-i", str(dense_dir),
        "-o", str(scene_mvs),
    ]
    run_command(interface_cmd, logger=logger)

    # 6. Densify point cloud (low-res)
    logger.info("Step 6/8: Densify point cloud (low-res)")
    logger.info(f"  resolution_level: {mb_cfg.get('densify_resolution_level', 2)}")
    logger.info(f"  number_views: {mb_cfg.get('densify_number_views', 4)}")

    scene_dense = mask_build_dir / "scene_dense.mvs"
    densify_bin = openmvs_bin_dir / "DensifyPointCloud"

    densify_cmd = [
        str(densify_bin),
        str(scene_mvs),
        "-o", str(scene_dense),
        "--resolution-level", str(mb_cfg.get("densify_resolution_level", 2)),
        "--number-views", str(mb_cfg.get("densify_number_views", 4)),
        "--number-views-fuse", str(mb_cfg.get("densify_number_views_fuse", 3)),
    ]
    run_command(densify_cmd, logger=logger)

    # 7. Reconstruct mesh (coarse)
    logger.info("Step 7/8: Reconstruct mesh (coarse)")

    scene_mesh = mask_build_dir / "scene_mesh.mvs"
    reconstruct_bin = openmvs_bin_dir / "ReconstructMesh"

    reconstruct_cmd = [
        str(reconstruct_bin),
        str(scene_dense),
        "-o", str(scene_mesh),
    ]
    run_command(reconstruct_cmd, logger=logger)

    # 8. Export and simplify
    logger.info("Step 8/8: Export coarse model")

    # Find the reconstructed mesh
    mesh_ply = mask_build_dir / "scene_dense_mesh.ply"
    if not mesh_ply.exists():
        raise PipelineError(f"Mesh not found: {mesh_ply}")

    # Load mesh
    mesh = trimesh.load(mesh_ply)
    original_faces = len(mesh.faces)
    logger.info(f"Original mesh: {len(mesh.vertices)} vertices, {original_faces} faces")

    # Simplify if too large
    simplify_ratio = mb_cfg.get("reconstruct_simplify", 0.5)
    if original_faces > 500000 and simplify_ratio < 1.0:
        logger.info(f"Simplifying mesh (ratio={simplify_ratio})...")
        target_faces = int(original_faces * simplify_ratio)
        mesh = mesh.simplify_quadric_decimation(target_faces)
        logger.info(f"Simplified: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Export coarse model
    coarse_mesh_path = mask_build_dir / "coarse_model.ply"
    mesh.export(coarse_mesh_path)
    logger.info(f"Exported: {coarse_mesh_path}")

    # Write README
    readme_path = mask_build_dir / "README_EDIT_MESH.txt"
    readme_content = f"""
MESH EDITING INSTRUCTIONS
==========================

Coarse model: {coarse_mesh_path.name}
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

EDITING STEPS:
-------------
1. Copy {coarse_mesh_path.name} to your workstation

2. Open in a mesh editor:
   - MeshLab (recommended)
   - Blender
   - CloudCompare
   - Any 3D editing software

3. DELETE all geometry that is NOT pottery sherds:
   ✗ Turntable surface
   ✗ Support rods and clamps
   ✗ Background walls
   ✗ Any reconstruction artifacts
   ✗ Thin floating geometry

4. KEEP only the pottery sherd surfaces
   ✓ All visible pottery fragments
   ✓ Internal and external surfaces

5. Save the edited mesh as: edited_model.ply
   - Use PLY format
   - Binary or ASCII is fine

6. Copy edited_model.ply back to this directory:
   {mask_build_dir}/edited_model.ply

CRITICAL WARNINGS:
-----------------
⚠️  DO NOT translate, rotate, or scale the mesh!
⚠️  DO NOT change the coordinate system!
⚠️  DO NOT apply any transformations!
⚠️  ONLY delete unwanted geometry!

The mesh must remain in the same coordinate frame as the sparse
reconstruction for mask projection to work correctly.

NEXT STEP:
---------
Once edited_model.ply is in place, run Phase 2:

    maskbuild user-project \\
        --work {work_dir} \\
        --mesh {mask_build_dir}/edited_model.ply

This will:
1. Validate the edited mesh is in the correct frame
2. Project the mesh into all camera views
3. Generate binary masks for each image
4. Create a manifest and coverage report

For help: See docs/MASKBUILD_USAGE.md
"""

    readme_path.write_text(readme_content)
    logger.info(f"README: {readme_path}")

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"✓ Phase 1 complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"✓ Coarse model: {coarse_mesh_path}")
    logger.info(f"✓ Instructions: {readme_path}")
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info(f"  1. Edit {coarse_mesh_path.name} (remove non-sherd geometry)")
    logger.info(f"  2. Save as edited_model.ply")
    logger.info(f"  3. Run: maskbuild user-project --work {work_dir} --mesh {mask_build_dir}/edited_model.ply")
    logger.info("=" * 70)

    # One-line summary for batch logs
    print(f"MASKBUILD_PHASE1_OK images={len(image_files)} vertices={len(mesh.vertices)} time={elapsed:.1f}s")

    return 0
