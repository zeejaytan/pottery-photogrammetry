#!/usr/bin/env python3
"""
Execute the COLMAP SfM pipeline for a single pottery-tree folder.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.pipeline_utils import (
    PipelineContext,
    PipelineError,
    get_tree_workdir,
    run_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute COLMAP SfM pipeline.")
    parser.add_argument(
        "tree",
        help=(
            "Path to the pottery-tree directory. "
            "Relative paths are resolved against project.data_root."
        ),
    )
    parser.add_argument(
        "--config",
        default="pipeline/config/pipeline_config.yaml",
        help="Path to pipeline configuration file.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Reuse existing COLMAP outputs instead of overwriting.",
    )
    parser.add_argument(
        "--rebuild-from-matching",
        action="store_true", 
        help="Keep database if complete, but rebuild from matching stage onwards.",
    )
    return parser.parse_args()


def remove_path(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[COLMAP] %(message)s")
    logger = logging.getLogger("colmap")

    context = PipelineContext.from_config_path(args.config)
    tree_dir = Path(args.tree)
    if not tree_dir.is_absolute():
        tree_dir = (context.data_root / tree_dir).resolve()

    if not tree_dir.exists():
        raise PipelineError(f"Tree directory not found: {tree_dir}")

    colmap_cfg = context.config.get("colmap", {})
    env_cfg = context.config.get("environment", {})
    colmap_exec = env_cfg.get("colmap_executable", "colmap")

    work_dir = get_tree_workdir(tree_dir)
    database_path = work_dir / colmap_cfg.get("database_name", "database.db")
    sparse_dir = work_dir / colmap_cfg.get("sparse_dir", "sparse")
    undistort_cfg = colmap_cfg.get("image_undistorter", {}) or {}

    workspace_cfg = undistort_cfg.get("output_path", "dense")
    workspace_path = Path(workspace_cfg)
    if not workspace_path.is_absolute():
        workspace_path = work_dir / workspace_path

    images_cfg = undistort_cfg.get("image_path")
    if images_cfg:
        images_path = Path(images_cfg)
        if not images_path.is_absolute():
            images_path = work_dir / images_path
    else:
        images_path = workspace_path / "images"

    dense_dir = workspace_path

    # Check if database is complete (has all expected JPG images)
    should_rebuild_database = True
    if args.rebuild_from_matching and database_path.exists():
        # Count JPG images in source directory
        expected_images = len([f for f in tree_dir.iterdir() 
                            if f.is_file() and f.suffix.upper() == '.JPG'])
        
        # Count images in database
        try:
            import sqlite3
            conn = sqlite3.connect(str(database_path))
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM images')
            db_images = cursor.fetchone()[0]
            conn.close()
            
            logger.info(f"Database has {db_images} images, expected {expected_images} JPG files")
            if db_images >= expected_images * 0.95:  # Allow 5% tolerance
                should_rebuild_database = False
                logger.info("Database is complete, keeping for rebuild from matching stage")
        except Exception as e:
            logger.warning(f"Could not verify database completeness: {e}")

    # Step 1: Feature Extraction (only if needed)
    if not args.keep_existing and not args.rebuild_from_matching and should_rebuild_database:
        if database_path.exists():
            logger.info("Removing previous database at %s", database_path)
            remove_path(database_path)
        if sparse_dir.exists():
            logger.info("Removing previous sparse reconstruction at %s", sparse_dir)
            remove_path(sparse_dir)
        if dense_dir.exists():
            logger.info("Removing previous dense workspace at %s", dense_dir)
            remove_path(dense_dir)

        sparse_dir.mkdir(parents=True, exist_ok=True)
        dense_dir.mkdir(parents=True, exist_ok=True)
        images_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Feature Extraction
        feature_cfg = colmap_cfg.get("feature_extractor", {}) or {}
        feature_cmd = [
            colmap_exec,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(tree_dir),
        ]

        # Single camera flag
        if "single_camera" in feature_cfg:
            feature_cmd.extend(
                ["--ImageReader.single_camera", str(feature_cfg["single_camera"])]
            )

        # GPU settings
        if "use_gpu" in feature_cfg:
            feature_cmd.extend(
                ["--SiftExtraction.use_gpu", str(feature_cfg["use_gpu"])]
            )
        if "gpu_index" in feature_cfg:
            feature_cmd.extend(
                ["--SiftExtraction.gpu_index", str(feature_cfg["gpu_index"])]
            )

        # Feature density settings
        if "max_image_size" in feature_cfg:
            feature_cmd.extend(
                ["--SiftExtraction.max_image_size", str(feature_cfg["max_image_size"])]
            )
        if "max_num_features" in feature_cfg:
            feature_cmd.extend(
                ["--SiftExtraction.max_num_features", str(feature_cfg["max_num_features"])]
            )
        if "domain_size_pooling" in feature_cfg:
            feature_cmd.extend(
                ["--SiftExtraction.domain_size_pooling", str(feature_cfg["domain_size_pooling"])]
            )

        # Optional threshold tuning
        if "sift_peak_threshold" in feature_cfg:
            feature_cmd.extend(
                ["--SiftExtraction.peak_threshold", str(feature_cfg["sift_peak_threshold"])]
            )
        if "sift_edge_threshold" in feature_cfg:
            feature_cmd.extend(
                ["--SiftExtraction.edge_threshold", str(feature_cfg["sift_edge_threshold"])]
            )

        logger.info("Running feature extraction...")
        logger.debug("PATH=%s", os.environ.get("PATH", ""))
        logger.debug("LD_LIBRARY_PATH=%s", os.environ.get("LD_LIBRARY_PATH", ""))
        run_command(feature_cmd, logger=logger)
    else:
        logger.info("Keeping existing database and directories")
        sparse_dir.mkdir(parents=True, exist_ok=True)
        dense_dir.mkdir(parents=True, exist_ok=True)
        images_path.mkdir(parents=True, exist_ok=True)

    # Step 2: Exhaustive Matching (essential for multi-ring turntable captures)
    if not args.keep_existing or args.rebuild_from_matching:
        # Always use exhaustive matcher for turntable captures (120-170 images)
        # This ensures cross-ring connections that sequential matching often misses
        matcher_cfg = colmap_cfg.get("exhaustive_matcher", {}) or {}
        matcher_cmd = [
            colmap_exec,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
        ]

        # GPU settings
        if "use_gpu" in matcher_cfg:
            matcher_cmd.extend(
                ["--SiftMatching.use_gpu", str(matcher_cfg["use_gpu"])]
            )
        if "gpu_index" in matcher_cfg:
            matcher_cmd.extend(
                ["--SiftMatching.gpu_index", str(matcher_cfg["gpu_index"])]
            )

        # Guided matching - critical for turntable stability across height rings
        if "guided_matching" in matcher_cfg:
            matcher_cmd.extend(
                ["--SiftMatching.guided_matching", str(matcher_cfg["guided_matching"])]
            )

        # Threading
        if "num_workers" in matcher_cfg:
            matcher_cmd.extend(
                ["--SiftMatching.num_threads", str(matcher_cfg["num_workers"])]
            )

        logger.info("Running exhaustive matcher with guided matching...")
        run_command(matcher_cmd, logger=logger)

    # Step 3: Sparse Reconstruction with relaxed thresholds for turntable captures
    if not args.keep_existing or args.rebuild_from_matching:
        mapper_cfg = colmap_cfg.get("mapper", {}) or {}
        mapper_cmd = [
            colmap_exec,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(tree_dir),
            "--output_path",
            str(sparse_dir),
        ]

        # Map config keys to COLMAP flags
        mapper_flags = {
            "multiple_models": "--Mapper.multiple_models",
            "init_min_tri_angle": "--Mapper.init_min_tri_angle",
            "abs_pose_min_num_inliers": "--Mapper.abs_pose_min_num_inliers",
            "abs_pose_min_inlier_ratio": "--Mapper.abs_pose_min_inlier_ratio",
            "ba_refine_focal_length": "--Mapper.ba_refine_focal_length",
            "ba_refine_principal_point": "--Mapper.ba_refine_principal_point",
            "ba_refine_extra_params": "--Mapper.ba_refine_extra_params",
            "min_num_matches": "--Mapper.min_num_matches",
            "triangulation_min_angle": "--Mapper.triangulation_min_angle",
            "max_extra_reproj_error": "--Mapper.max_extra_reproj_error",
            "max_reproj_error": "--Mapper.max_reproj_error",
            "num_threads": "--Mapper.num_threads",
        }

        for key, flag in mapper_flags.items():
            if key in mapper_cfg:
                mapper_cmd.extend([flag, str(mapper_cfg[key])])

        logger.info("Running sparse mapper with turntable-optimized thresholds...")
        run_command(mapper_cmd, logger=logger)

    if not any(child.is_dir() for child in sparse_dir.iterdir()):
        raise PipelineError(
            f"No sparse models produced at {sparse_dir}. "
            "Check COLMAP mapper output."
        )

    model_dir = next(child for child in sorted(sparse_dir.iterdir()) if child.is_dir())

    # Step 4: Image Undistortion
    if not args.keep_existing or args.rebuild_from_matching:
        undistort_cmd = [
            colmap_exec,
            "image_undistorter",
            "--image_path",
            str(tree_dir),
            "--input_path",
            str(model_dir),
            "--output_path",
            str(dense_dir),
        ]

        if "max_image_size" in undistort_cfg:
            undistort_cmd.extend(
                ["--max_image_size", str(undistort_cfg["max_image_size"])]
            )
        logger.info("Running image undistorter...")
        run_command(undistort_cmd, logger=logger)

    logger.info("COLMAP pipeline complete for %s", tree_dir.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
