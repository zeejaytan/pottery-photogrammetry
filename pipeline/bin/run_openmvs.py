#!/usr/bin/env python3
"""
Execute the OpenMVS pipeline on a COLMAP dense workspace.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

from lib.pipeline_utils import PipelineContext, PipelineError, run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenMVS workflow.")
    parser.add_argument(
        "work_dir",
        help="Path to tree working directory (typically work_colmap_openmvs).",
    )
    parser.add_argument(
        "--config",
        default="pipeline/config/pipeline_config.yaml",
        help="Path to pipeline configuration.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep pre-existing OpenMVS outputs if present.",
    )
    return parser.parse_args()


def ensure_binary(path: Path) -> Path:
    if not path.exists():
        raise PipelineError(f"Required binary not found: {path}")
    return path


def remove_path(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[OpenMVS] %(message)s")
    logger = logging.getLogger("openmvs")

    context = PipelineContext.from_config_path(args.config)
    work_dir = Path(args.work_dir).resolve()
    if not work_dir.exists():
        raise PipelineError(f"Work directory not found: {work_dir}")

    colmap_cfg = context.config.get("colmap", {}) or {}
    undistort_cfg = colmap_cfg.get("image_undistorter", {}) or {}
    workspace_cfg = undistort_cfg.get("output_path", "dense")
    workspace_path = Path(workspace_cfg)
    if not workspace_path.is_absolute():
        workspace_path = work_dir / workspace_path

    dense_dir = workspace_path
    if not dense_dir.exists():
        raise PipelineError(f"Dense workspace missing: {dense_dir}")

    env_cfg = context.config.get("environment", {})
    openmvs_cfg = context.config.get("openmvs", {})
    openmvs_bin_dir = Path(env_cfg.get("openmvs_bin_dir", ""))
    if not openmvs_bin_dir:
        raise PipelineError("environment.openmvs_bin_dir not set in configuration.")

    interface_bin = ensure_binary(openmvs_bin_dir / "InterfaceCOLMAP")
    densify_bin = ensure_binary(openmvs_bin_dir / "DensifyPointCloud")
    reconstruct_bin = ensure_binary(openmvs_bin_dir / "ReconstructMesh")
    refine_bin = ensure_binary(openmvs_bin_dir / "RefineMesh")

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        f"{openmvs_bin_dir}:{ld_library_path}" if ld_library_path else str(openmvs_bin_dir)
    )

    scene_path = work_dir / "scene.mvs"
    dense_scene_path = work_dir / "scene_dense.mvs"
    mesh_ply_path = work_dir / "scene_dense_mesh.ply"
    refined_project_path = work_dir / "scene_refined_mesh.mvs"
    refined_mesh_path = work_dir / "scene_refined_mesh.ply"

    if not args.keep_existing:
        for path in (
            scene_path,
            dense_scene_path,
            mesh_ply_path,
            refined_project_path,
            refined_mesh_path,
        ):
            if path.exists():
                logger.info("Removing previous artifact: %s", path)
                remove_path(path)

    cuda_device = str(openmvs_cfg.get("cuda_device", 0))

    # 1. InterfaceCOLMAP - convert COLMAP undistorted workspace
    interface_cfg = openmvs_cfg.get("interface", {})
    interface_cmd = [
        str(interface_bin),
        "-w",
        str(work_dir),
        "-i",
        str(dense_dir),
        "-o",
        str(scene_path),
    ]

    # Optional: specify image folder explicitly
    if "image_folder" in interface_cfg:
        image_folder = interface_cfg["image_folder"]
        if not Path(image_folder).is_absolute():
            image_folder = work_dir / image_folder
        interface_cmd.extend(["--image-folder", str(image_folder)])

    logger.info("Converting COLMAP workspace to OpenMVS scene...")
    run_command(interface_cmd, logger=logger, cwd=work_dir)

    # 2. DensifyPointCloud - full-resolution depth maps for ≥100k vertices/sherd
    densify_cfg = openmvs_cfg.get("densify", {})
    densify_cmd = [
        str(densify_bin),
        "-i",
        str(scene_path),
        "-o",
        str(dense_scene_path),
        "--cuda-device",
        cuda_device,
    ]

    # Resolution level: 0 = full resolution (primary density lever)
    if "resolution_level" in densify_cfg:
        densify_cmd.extend(
            ["--resolution-level", str(densify_cfg["resolution_level"])]
        )

    # Number of views: support for depth estimation per image
    if "number_views" in densify_cfg:
        densify_cmd.extend(["--number-views", str(densify_cfg["number_views"])])

    # Number of views fuse: views that must agree during fusion
    if "number_views_fuse" in densify_cfg:
        densify_cmd.extend(
            ["--number-views-fuse", str(densify_cfg["number_views_fuse"])]
        )

    logger.info("Generating dense point cloud (full resolution, %d support views)...",
                densify_cfg.get("number_views", 6))
    run_command(densify_cmd, logger=logger, cwd=work_dir)

    # 3. ReconstructMesh - no decimation, high target face count
    # Use correct parameters: -i for MVS scene, -p for dense PLY, -o for output
    reconstruct_cfg = openmvs_cfg.get("reconstruct", {})
    dense_ply_path = work_dir / "scene_dense.ply"
    reconstruct_cmd = [
        str(reconstruct_bin),
        "-i",
        str(dense_scene_path),
        "-p",
        str(dense_ply_path),
        "-o",
        str(mesh_ply_path),
    ]

    # High target face count prevents unintended simplification
    # For 10 sherds × 100k vertices ≈ 2-3M faces total expected
    if "target_face_num" in reconstruct_cfg:
        reconstruct_cmd.extend(
            ["--target-face-num", str(reconstruct_cfg["target_face_num"])]
        )

    # Legacy parameters (keep for compatibility with older OpenMVS builds)
    if "min_point_distance" in reconstruct_cfg:
        reconstruct_cmd.extend(
            ["--min-point-distance", str(reconstruct_cfg["min_point_distance"])]
        )
    if "decimate" in reconstruct_cfg:
        reconstruct_cmd.extend(["--decimate", str(reconstruct_cfg["decimate"])])

    logger.info("Reconstructing mesh without decimation (target: %d faces)...",
                reconstruct_cfg.get("target_face_num", 5000000))
    run_command(reconstruct_cmd, logger=logger, cwd=work_dir)

    # 4. RefineMesh - add detail without decimation
    refine_cfg = openmvs_cfg.get("refine", {})
    refine_cmd = [
        str(refine_bin),
        str(dense_scene_path),
        "-o",
        str(refined_project_path),
        "-m",
        str(mesh_ply_path),
        "--cuda-device",
        cuda_device,
    ]

    # Refinement scales (iterations to add detail)
    if "scales" in refine_cfg:
        refine_cmd.extend(["--scales", str(refine_cfg["scales"])])

    # Legacy decimation parameter (kept for compatibility)
    if "decimate" in refine_cfg:
        refine_cmd.extend(["--decimate", str(refine_cfg["decimate"])])

    refine_cmd.extend(["--export-type", "ply"])

    logger.info("Refining mesh (%d scales) without decimation...",
                refine_cfg.get("scales", 2))
    run_command(refine_cmd, logger=logger, cwd=work_dir)
    if not refined_mesh_path.exists():
        candidate = refined_project_path.with_suffix(".ply")
        if candidate.exists():
            refined_mesh_path = candidate
    if not refined_mesh_path.exists():
        raise PipelineError(
            f"RefineMesh completed but failed to create {refined_mesh_path}"
        )

    logger.info("OpenMVS pipeline complete at %s", work_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        logging.error("%s", exc)
        raise SystemExit(2)
