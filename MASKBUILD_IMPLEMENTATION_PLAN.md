# Implementation Plan: User-Edited Model Mask Generation

## Overview

This plan implements a two-phase headless masking system that:
1. **Phase 1**: Builds a quick, coarse 3D model from JPEGs for user editing
2. **Phase 2**: Projects the edited model into all camera views to generate pixel-perfect binary masks

The masks exclude stands, clamps, rods, and background, allowing COLMAP/OpenMVS to focus only on pottery sherds during feature detection, matching, and depth estimation.

---

## Architecture

### New Files to Create

```
pipeline/
├── bin/
│   ├── maskbuild.py                    # Main CLI entry point (NEW)
│   └── run_colmap.py                   # Modified to support mask_path
├── lib/
│   ├── maskbuild_phase1.py             # Quick model builder (NEW)
│   ├── maskbuild_phase2.py             # Mask projector (NEW)
│   ├── maskbuild_utils.py              # Shared utilities (NEW)
│   └── mesh_projection.py              # 3D→2D projection engine (NEW)
├── config/
│   └── pipeline_config.yaml            # Add maskbuild section
└── docs/
    └── MASKBUILD_USAGE.md              # User guide (NEW)
```

### Output Directory Structure

```
work_colmap_openmvs/
├── mask_build_user/                    # Working directory for masking
│   ├── sparse/                         # Quick sparse reconstruction
│   │   └── 0/                          # cameras.bin, images.bin, points3D.bin
│   ├── dense/                          # Low-res dense workspace
│   │   ├── images/                     # Undistorted images (small)
│   │   └── stereo/                     # Low-res depth maps
│   ├── coarse_model.ply                # Initial surface for editing
│   ├── edited_model.ply                # User-edited mesh (user provides)
│   ├── frame_validation.json           # Coordinate frame checks
│   └── RUNLOG.txt                      # Phase 1 & 2 logs
├── masks_user/                         # Final masks (NEW)
│   ├── IMG_0001.png
│   ├── IMG_0002.png
│   └── ...
├── masks_manifest.json                 # Checksums & metadata (NEW)
└── masks_report.txt                    # Coverage summary (NEW)
```

---

## Implementation Details

### 1. Main CLI (`pipeline/bin/maskbuild.py`)

**Purpose**: Entry point with two subcommands: `user-init` and `user-project`

**Interface**:
```python
#!/usr/bin/env python3
"""
User-edited model mask generation CLI.

Usage:
    maskbuild user-init --images <path> --work <path> [--config <path>]
    maskbuild user-project --work <path> --mesh <path> [--config <path>]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.maskbuild_phase1 import run_phase1
from lib.maskbuild_phase2 import run_phase2
from lib.pipeline_utils import PipelineError


def main():
    parser = argparse.ArgumentParser(
        description="Generate masks from user-edited 3D model"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Phase 1: Build coarse model
    init_parser = subparsers.add_parser(
        "user-init",
        help="Build quick coarse model for editing"
    )
    init_parser.add_argument(
        "--images",
        required=True,
        help="Path to JPEG image directory"
    )
    init_parser.add_argument(
        "--work",
        required=True,
        help="Path to work_colmap_openmvs directory"
    )
    init_parser.add_argument(
        "--config",
        default="pipeline/config/pipeline_config.yaml",
        help="Path to pipeline configuration"
    )

    # Phase 2: Project edited mesh to masks
    project_parser = subparsers.add_parser(
        "user-project",
        help="Project edited mesh to generate masks"
    )
    project_parser.add_argument(
        "--work",
        required=True,
        help="Path to work_colmap_openmvs directory"
    )
    project_parser.add_argument(
        "--mesh",
        required=True,
        help="Path to edited mesh file (PLY or OBJ)"
    )
    project_parser.add_argument(
        "--config",
        default="pipeline/config/pipeline_config.yaml",
        help="Path to pipeline configuration"
    )
    project_parser.add_argument(
        "--pad-pixels",
        type=int,
        default=2,
        help="Padding around mask edges (default: 2)"
    )
    project_parser.add_argument(
        "--erosion",
        type=int,
        default=0,
        help="Erosion iterations to prevent halos (default: 0)"
    )

    args = parser.parse_args()

    try:
        if args.command == "user-init":
            return run_phase1(
                images_dir=Path(args.images),
                work_dir=Path(args.work),
                config_path=Path(args.config)
            )
        elif args.command == "user-project":
            return run_phase2(
                work_dir=Path(args.work),
                mesh_path=Path(args.mesh),
                config_path=Path(args.config),
                pad_pixels=args.pad_pixels,
                erosion=args.erosion
            )
    except PipelineError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
```

---

### 2. Phase 1: Quick Model Builder (`lib/maskbuild_phase1.py`)

**Purpose**: Build fast, coarse reconstruction optimized for speed over quality

**Key Parameters** (in config):
```yaml
maskbuild:
  quick_build:
    max_image_size: 2048              # Smaller than main pipeline (4096)
    max_num_features: 8000            # Fewer features (main: 16000)
    matcher: exhaustive               # Still need cross-ring links
    mapper_min_tri_angle: 2.0         # Same as main pipeline
    mapper_min_inliers: 12            # More lenient (main: 15)
    densify_resolution_level: 2       # 1/4 resolution (main: 0 = full)
    densify_number_views: 4           # Fewer views (main: 8)
    reconstruct_simplify: 0.5         # Target 50% face reduction
```

**Implementation Steps**:

```python
def run_phase1(images_dir: Path, work_dir: Path, config_path: Path) -> int:
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

    Returns:
        0 on success, non-zero on failure
    """

    # Load config
    config = load_config(config_path)
    mb_cfg = config.get("maskbuild", {}).get("quick_build", {})

    # Setup workspace
    mask_build_dir = work_dir / "mask_build_user"
    mask_build_dir.mkdir(exist_ok=True)

    sparse_dir = mask_build_dir / "sparse"
    dense_dir = mask_build_dir / "dense"
    database_path = mask_build_dir / "database.db"

    # Setup logging
    log_file = mask_build_dir / "RUNLOG.txt"
    logger = configure_logging(log_file, console=True)
    logger.info("=== MASKBUILD PHASE 1: Quick Model Build ===")
    logger.info(f"Images: {images_dir}")
    logger.info(f"Work dir: {work_dir}")

    start_time = time.time()

    # Count images
    image_files = list_images(images_dir)
    logger.info(f"Found {len(image_files)} images")

    # 1. Feature extraction (low-res)
    logger.info("Step 1/8: Feature extraction (low-res)")
    run_colmap_feature_extractor(
        colmap_exec="colmap",
        database_path=database_path,
        image_path=images_dir,
        max_image_size=mb_cfg.get("max_image_size", 2048),
        max_num_features=mb_cfg.get("max_num_features", 8000),
        gpu_index=0,
        logger=logger
    )

    # 2. Exhaustive matching
    logger.info("Step 2/8: Exhaustive matching")
    run_colmap_matcher(
        colmap_exec="colmap",
        database_path=database_path,
        matcher_type="exhaustive",
        gpu_index=0,
        logger=logger
    )

    # 3. Sparse reconstruction (lenient)
    logger.info("Step 3/8: Sparse reconstruction (lenient thresholds)")
    sparse_dir.mkdir(exist_ok=True)
    run_colmap_mapper(
        colmap_exec="colmap",
        database_path=database_path,
        image_path=images_dir,
        output_path=sparse_dir,
        mapper_options={
            "Mapper.init_min_tri_angle": mb_cfg.get("mapper_min_tri_angle", 2.0),
            "Mapper.abs_pose_min_num_inliers": mb_cfg.get("mapper_min_inliers", 12),
            "Mapper.abs_pose_min_inlier_ratio": 0.15,
            "Mapper.ba_refine_principal_point": 0,
            "Mapper.multiple_models": 0  # Fail if multiple models
        },
        logger=logger
    )

    # Check for single model
    model_dirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
    if len(model_dirs) == 0:
        raise PipelineError("No sparse model created")
    if len(model_dirs) > 1:
        # Retry with even more lenient settings
        logger.warning(f"Multiple models detected ({len(model_dirs)}), retrying with looser thresholds")
        shutil.rmtree(sparse_dir)
        sparse_dir.mkdir()
        run_colmap_mapper(
            # ... even more lenient settings
            mapper_options={
                "Mapper.init_min_tri_angle": 1.0,
                "Mapper.abs_pose_min_num_inliers": 8,
                # ...
            }
        )
        model_dirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
        if len(model_dirs) != 1:
            raise PipelineError(
                "Cannot create single unified model even with lenient thresholds. "
                "Check image overlap and quality."
            )

    model_dir = model_dirs[0]

    # 4. Image undistortion (low-res)
    logger.info("Step 4/8: Image undistortion (low-res)")
    run_colmap_image_undistorter(
        colmap_exec="colmap",
        image_path=images_dir,
        input_path=model_dir,
        output_path=dense_dir,
        max_image_size=mb_cfg.get("max_image_size", 2048),
        logger=logger
    )

    # 5. OpenMVS interface
    logger.info("Step 5/8: OpenMVS interface")
    scene_mvs = mask_build_dir / "scene.mvs"
    run_openmvs_interface(
        openmvs_bin="InterfaceCOLMAP",
        working_folder=str(mask_build_dir),
        input_path=str(dense_dir),
        output_file=str(scene_mvs),
        logger=logger
    )

    # 6. Densify point cloud (low-res)
    logger.info("Step 6/8: Densify point cloud (low-res)")
    run_openmvs_densify(
        openmvs_bin="DensifyPointCloud",
        input_file=str(scene_mvs),
        output_file=str(mask_build_dir / "scene_dense.mvs"),
        resolution_level=mb_cfg.get("densify_resolution_level", 2),
        number_views=mb_cfg.get("densify_number_views", 4),
        logger=logger
    )

    # 7. Reconstruct mesh (coarse)
    logger.info("Step 7/8: Reconstruct mesh (coarse)")
    run_openmvs_reconstruct(
        openmvs_bin="ReconstructMesh",
        input_file=str(mask_build_dir / "scene_dense.mvs"),
        output_file=str(mask_build_dir / "scene_mesh.mvs"),
        logger=logger
    )

    # 8. Export and simplify
    logger.info("Step 8/8: Export coarse model")
    coarse_mesh_path = mask_build_dir / "coarse_model.ply"

    # Find the reconstructed mesh
    mesh_ply = mask_build_dir / "scene_dense_mesh.ply"
    if not mesh_ply.exists():
        raise PipelineError(f"Mesh not found: {mesh_ply}")

    # Load and optionally simplify
    import trimesh
    mesh = trimesh.load(mesh_ply)

    original_faces = len(mesh.faces)
    simplify_ratio = mb_cfg.get("reconstruct_simplify", 0.5)

    if original_faces > 500000 and simplify_ratio < 1.0:
        logger.info(f"Simplifying mesh from {original_faces} faces...")
        target_faces = int(original_faces * simplify_ratio)
        mesh = mesh.simplify_quadric_decimation(target_faces)
        logger.info(f"Simplified to {len(mesh.faces)} faces")

    mesh.export(coarse_mesh_path)

    # Write README
    readme_path = mask_build_dir / "README_EDIT_MESH.txt"
    readme_path.write_text(f"""
MESH EDITING INSTRUCTIONS
==========================

File to edit: {coarse_mesh_path}

STEPS:
1. Copy coarse_model.ply to your workstation
2. Open in MeshLab, Blender, or CloudCompare
3. DELETE all geometry that is NOT pottery sherds:
   - Turntable surface
   - Support rods/clamps
   - Background/walls
   - Any reconstruction artifacts
4. KEEP only the pottery sherd surfaces
5. Save the edited mesh as: edited_model.ply
6. Copy edited_model.ply back to: {mask_build_dir}/edited_model.ply

IMPORTANT:
- Do NOT translate, rotate, or scale the mesh
- Do NOT change coordinate system
- Only DELETE unwanted geometry
- The mesh must remain in the same coordinate frame

Once edited_model.ply is in place, run Phase 2:

    maskbuild user-project \\
      --work {work_dir} \\
      --mesh {mask_build_dir}/edited_model.ply

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
""")

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"Phase 1 complete in {elapsed:.1f}s")
    logger.info(f"Coarse model: {coarse_mesh_path}")
    logger.info(f"Instructions: {readme_path}")
    logger.info(f"Next: Edit mesh and run 'maskbuild user-project'")
    logger.info("=" * 70)

    # One-line summary for batch logs
    print(f"MASKBUILD_PHASE1_OK images={len(image_files)} time={elapsed:.1f}s mesh={coarse_mesh_path}")

    return 0
```

---

### 3. Phase 2: Mask Projector (`lib/maskbuild_phase2.py`)

**Purpose**: Validate edited mesh and project into all camera views

**Implementation**:

```python
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

    Returns:
        0 on success, non-zero on failure
    """

    config = load_config(config_path)

    mask_build_dir = work_dir / "mask_build_user"
    masks_dir = work_dir / "masks_user"

    # Setup logging (append to Phase 1 log)
    log_file = mask_build_dir / "RUNLOG.txt"
    logger = configure_logging(log_file, console=True, mode="a")
    logger.info("=== MASKBUILD PHASE 2: Mask Projection ===")
    logger.info(f"Edited mesh: {mesh_path}")

    start_time = time.time()

    # 1. Load edited mesh
    if not mesh_path.exists():
        raise PipelineError(f"Edited mesh not found: {mesh_path}")

    import trimesh
    import hashlib

    edited_mesh = trimesh.load(mesh_path)
    logger.info(f"Loaded mesh: {len(edited_mesh.vertices)} vertices, {len(edited_mesh.faces)} faces")

    # Compute mesh hash
    mesh_bytes = mesh_path.read_bytes()
    mesh_hash = hashlib.sha256(mesh_bytes).hexdigest()
    logger.info(f"Mesh SHA-256: {mesh_hash[:16]}...")

    # 2. Load sparse reconstruction
    sparse_dir = mask_build_dir / "sparse" / "0"
    if not sparse_dir.exists():
        raise PipelineError(f"Sparse reconstruction not found: {sparse_dir}")

    cameras, images = read_colmap_model(sparse_dir)
    logger.info(f"Loaded {len(cameras)} cameras, {len(images)} images")

    # Compute camera bundle hash
    bundle_hash = compute_bundle_hash(sparse_dir)
    logger.info(f"Bundle SHA-256: {bundle_hash[:16]}...")

    # 3. Validate coordinate frame
    logger.info("Validating coordinate frame...")
    validation = validate_mesh_frame(
        mesh=edited_mesh,
        sparse_dir=sparse_dir,
        original_mesh_path=mask_build_dir / "coarse_model.ply"
    )

    if not validation["is_valid"]:
        raise PipelineError(
            f"Mesh coordinate frame validation failed:\n"
            f"  {validation['error']}\n"
            f"Ensure mesh was not translated/rotated/scaled during editing."
        )

    logger.info(f"Frame validation passed: {validation['message']}")

    # Save validation
    validation_path = mask_build_dir / "frame_validation.json"
    with validation_path.open("w") as f:
        json.dump(validation, f, indent=2)

    # 4. Project masks
    masks_dir.mkdir(exist_ok=True)
    logger.info(f"Projecting masks to {masks_dir}...")

    mask_records = []
    failed_images = []
    coverage_stats = []

    for image_id, image_info in images.items():
        image_name = image_info["name"]
        camera_id = image_info["camera_id"]
        camera = cameras[camera_id]

        # Get camera pose (world-to-camera)
        qvec = image_info["qvec"]
        tvec = image_info["tvec"]

        try:
            # Project mesh to image
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

            # Save mask
            mask_filename = Path(image_name).stem + ".png"
            mask_path = masks_dir / mask_filename

            import cv2
            cv2.imwrite(str(mask_path), mask)

            # Compute stats
            mask_area = (mask > 0).sum()
            total_pixels = mask.shape[0] * mask.shape[1]
            coverage_pct = 100.0 * mask_area / total_pixels

            coverage_stats.append(coverage_pct)

            # Compute checksum
            mask_hash = hashlib.sha256(mask_path.read_bytes()).hexdigest()

            mask_records.append({
                "image_name": image_name,
                "mask_filename": mask_filename,
                "mask_sha256": mask_hash,
                "coverage_percent": coverage_pct,
                "mask_pixels": int(mask_area),
                "image_pixels": int(total_pixels)
            })

            if len(mask_records) % 20 == 0:
                logger.info(f"  Projected {len(mask_records)}/{len(images)} masks...")

        except Exception as e:
            logger.warning(f"Failed to project {image_name}: {e}")
            failed_images.append(image_name)

    logger.info(f"Projected {len(mask_records)} masks")
    if failed_images:
        logger.warning(f"Failed to project {len(failed_images)} images")

    # 5. Coverage statistics
    import numpy as np
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

    # Detect outliers (coverage < 5% or > 95%)
    outliers = [
        r for r in mask_records
        if r["coverage_percent"] < 5.0 or r["coverage_percent"] > 95.0
    ]

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

    logger.info(f"Manifest written: {manifest_path}")

    # 7. Write summary report
    report_path = work_dir / "masks_report.txt"
    with report_path.open("w") as f:
        f.write("=" * 70 + "\n")
        f.write("MASK GENERATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {manifest['generated_at']}\n")
        f.write(f"Edited mesh: {mesh_path}\n")
        f.write(f"Mesh hash: {mesh_hash}\n")
        f.write(f"Bundle hash: {bundle_hash}\n\n")

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
        f.write(f"IQR:               {cov['p25_percent']:.1f}% - {cov['p75_percent']:.1f}%\n\n")

        if outliers:
            f.write("OUTLIERS (coverage <5% or >95%)\n")
            f.write("-" * 70 + "\n")
            for rec in outliers[:10]:
                f.write(f"  {rec['image_name']}: {rec['coverage_percent']:.1f}%\n")
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
        f.write("Rerun COLMAP Stage 1 with masks:\n\n")
        f.write(f"  python pipeline/bin/run_colmap.py <tree_dir> \\\n")
        f.write(f"    --mask-path {masks_dir}\n\n")
        f.write("Expected improvements:\n")
        f.write("  - More images registered (fewer unregistered)\n")
        f.write("  - Lower reprojection errors\n")
        f.write("  - Features only detected on sherd surfaces\n")
        f.write("  - Cleaner sparse point cloud\n")

    logger.info(f"Report written: {report_path}")

    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"Phase 2 complete in {elapsed:.1f}s")
    logger.info(f"Masks: {masks_dir}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Report: {report_path}")
    logger.info("=" * 70)

    # One-line summary
    print(f"MASKBUILD_PHASE2_OK masks={len(mask_records)} coverage={cov['median_percent']:.1f}% time={elapsed:.1f}s")

    return 0
```

---

### 4. Mesh Projection Engine (`lib/mesh_projection.py`)

**Purpose**: Core 3D→2D projection and rasterization

```python
import numpy as np
import trimesh
from typing import Tuple

def project_mesh_to_mask(
    mesh: trimesh.Trimesh,
    camera: dict,
    qvec: np.ndarray,
    tvec: np.ndarray,
    image_width: int,
    image_height: int,
    pad_pixels: int = 2,
    erosion: int = 0
) -> np.ndarray:
    """
    Project 3D mesh into camera view and create binary mask.

    Args:
        mesh: Trimesh object in world coordinates
        camera: COLMAP camera dict with 'model', 'params', 'width', 'height'
        qvec: Quaternion rotation (world-to-camera)
        tvec: Translation vector (world-to-camera)
        image_width: Output mask width
        image_height: Output mask height
        pad_pixels: Dilation to protect thin edges
        erosion: Erosion iterations to prevent halos

    Returns:
        Binary mask (uint8, 255=keep, 0=discard)
    """

    # 1. Convert quaternion to rotation matrix
    R = qvec_to_rotation_matrix(qvec)

    # 2. Transform mesh vertices to camera frame
    vertices_world = mesh.vertices
    vertices_cam = (R @ vertices_world.T).T + tvec

    # 3. Remove back-facing geometry (Z <= 0)
    valid_faces = []
    for face in mesh.faces:
        v0, v1, v2 = vertices_cam[face]
        if v0[2] > 0 and v1[2] > 0 and v2[2] > 0:
            valid_faces.append(face)

    if len(valid_faces) == 0:
        # Mesh not visible in this view
        return np.zeros((image_height, image_width), dtype=np.uint8)

    # 4. Project to image plane
    fx, fy, cx, cy = parse_camera_params(camera)

    points_2d = []
    for face_idx in valid_faces:
        for vertex_idx in mesh.faces[face_idx]:
            v = vertices_cam[vertex_idx]
            x = (fx * v[0] / v[2]) + cx
            y = (fy * v[1] / v[2]) + cy
            points_2d.append([x, y])

    points_2d = np.array(points_2d)

    # 5. Rasterize using OpenCV
    import cv2

    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Create contours from projected triangles
    for face_idx in valid_faces:
        triangle = []
        for vertex_idx in mesh.faces[face_idx]:
            v = vertices_cam[vertex_idx]
            x = int((fx * v[0] / v[2]) + cx)
            y = int((fy * v[1] / v[2]) + cy)
            triangle.append([x, y])

        triangle = np.array(triangle, dtype=np.int32)
        cv2.fillConvexPoly(mask, triangle, 255)

    # 6. Apply padding (dilation)
    if pad_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad_pixels*2+1, pad_pixels*2+1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # 7. Apply erosion (if halos detected)
    if erosion > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=erosion)

    return mask


def qvec_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion to 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def parse_camera_params(camera: dict) -> Tuple[float, float, float, float]:
    """Extract fx, fy, cx, cy from COLMAP camera."""
    model = camera["model"]
    params = camera["params"]

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        return f, f, cx, cy
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        return fx, fy, cx, cy
    elif model in ["SIMPLE_RADIAL", "RADIAL"]:
        f, cx, cy = params[:3]
        return f, f, cx, cy
    elif model == "OPENCV":
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy
    else:
        raise ValueError(f"Unsupported camera model: {model}")


def read_colmap_model(sparse_dir: Path):
    """Read COLMAP cameras and images from binary files."""
    from lib.colmap_reader import (
        read_cameras_binary,
        read_images_binary
    )

    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")

    return cameras, images
```

---

### 5. Frame Validation (`lib/maskbuild_utils.py`)

**Purpose**: Ensure edited mesh is in same coordinate frame as sparse

```python
def validate_mesh_frame(
    mesh: trimesh.Trimesh,
    sparse_dir: Path,
    original_mesh_path: Path,
    bbox_tolerance: float = 0.1,
    distance_tolerance: float = 0.05
) -> dict:
    """
    Validate that edited mesh is in same coordinate frame as original.

    Checks:
    1. Bounding box overlap (>90%)
    2. Reference distances match within 5%
    3. Vertex count reasonable (10% - 150% of original)

    Returns:
        {
            "is_valid": bool,
            "message": str,
            "error": str or None,
            "checks": {...}
        }
    """

    checks = {}

    # Load original mesh
    original = trimesh.load(original_mesh_path)

    # Check 1: Bounding box
    bbox_edited = mesh.bounds
    bbox_original = original.bounds

    bbox_intersection = np.minimum(bbox_edited[1], bbox_original[1]) - \
                       np.maximum(bbox_edited[0], bbox_original[0])
    bbox_intersection_volume = np.prod(np.maximum(0, bbox_intersection))

    bbox_original_volume = np.prod(bbox_original[1] - bbox_original[0])

    bbox_overlap = bbox_intersection_volume / bbox_original_volume if bbox_original_volume > 0 else 0

    checks["bbox_overlap"] = bbox_overlap
    checks["bbox_threshold"] = 1.0 - bbox_tolerance
    checks["bbox_pass"] = bbox_overlap >= (1.0 - bbox_tolerance)

    # Check 2: Reference distances
    # Sample 10 random points from original, find distances
    np.random.seed(42)
    sample_indices = np.random.choice(len(original.vertices), min(10, len(original.vertices)), replace=False)
    sample_points = original.vertices[sample_indices]

    # Compute pairwise distances in original
    from scipy.spatial.distance import pdist
    original_distances = pdist(sample_points)

    # Find closest points in edited mesh
    edited_sample = []
    for pt in sample_points:
        closest_idx = np.argmin(np.linalg.norm(mesh.vertices - pt, axis=1))
        edited_sample.append(mesh.vertices[closest_idx])

    edited_sample = np.array(edited_sample)
    edited_distances = pdist(edited_sample)

    # Compute relative error
    distance_errors = np.abs(edited_distances - original_distances) / (original_distances + 1e-6)
    max_distance_error = distance_errors.max()

    checks["max_distance_error"] = float(max_distance_error)
    checks["distance_threshold"] = distance_tolerance
    checks["distance_pass"] = max_distance_error <= distance_tolerance

    # Check 3: Vertex count
    vertex_ratio = len(mesh.vertices) / len(original.vertices)
    checks["vertex_ratio"] = vertex_ratio
    checks["vertex_count_edited"] = len(mesh.vertices)
    checks["vertex_count_original"] = len(original.vertices)
    checks["vertex_pass"] = 0.1 <= vertex_ratio <= 1.5

    # Overall validation
    all_pass = checks["bbox_pass"] and checks["distance_pass"] and checks["vertex_pass"]

    if all_pass:
        return {
            "is_valid": True,
            "message": f"Frame validation passed (bbox_overlap={bbox_overlap:.2f}, max_dist_error={max_distance_error:.3f})",
            "error": None,
            "checks": checks
        }
    else:
        errors = []
        if not checks["bbox_pass"]:
            errors.append(f"Bounding box overlap {bbox_overlap:.2f} < {1.0-bbox_tolerance:.2f}")
        if not checks["distance_pass"]:
            errors.append(f"Distance error {max_distance_error:.3f} > {distance_tolerance:.3f}")
        if not checks["vertex_pass"]:
            errors.append(f"Vertex ratio {vertex_ratio:.2f} outside [0.1, 1.5]")

        return {
            "is_valid": False,
            "message": "Frame validation failed",
            "error": "; ".join(errors),
            "checks": checks
        }
```

---

### 6. COLMAP Integration (`run_colmap.py` modifications)

**Add mask support**:

```python
# In parse_args():
parser.add_argument(
    "--mask-path",
    help="Path to masks_user directory (if available)"
)

# In main():
def main() -> int:
    args = parse_args()

    # ... existing code ...

    # Check for user-provided masks
    mask_path = None
    if args.mask_path:
        mask_path = Path(args.mask_path)
    else:
        # Auto-detect masks_user in work directory
        work_dir = get_tree_workdir(ctx, tree_dir)
        auto_mask_dir = work_dir / "masks_user"
        if auto_mask_dir.exists():
            logger.info(f"Found user masks: {auto_mask_dir}")
            mask_path = auto_mask_dir

    # Pass to feature extractor and matcher
    if mask_path:
        feature_cmd.extend(["--ImageReader.mask_path", str(mask_path)])
        logger.info(f"Using masks from: {mask_path}")
```

---

### 7. Configuration (`pipeline_config.yaml` additions)

```yaml
maskbuild:
  quick_build:
    # Feature extraction (faster)
    max_image_size: 2048              # 50% of main pipeline
    max_num_features: 8000            # 50% of main pipeline

    # Matching (same strategy)
    matcher: exhaustive               # Still need cross-ring
    guided_matching: 1

    # Mapper (more lenient)
    mapper_min_tri_angle: 2.0
    mapper_min_inliers: 12            # Lower than main (15)
    mapper_min_inlier_ratio: 0.15

    # Dense reconstruction (low-res)
    undistort_max_image_size: 2048
    densify_resolution_level: 2       # 1/4 resolution
    densify_number_views: 4
    densify_number_views_fuse: 3

    # Mesh (simplified)
    reconstruct_simplify: 0.5         # Target 50% reduction

  projection:
    # Mask generation
    pad_pixels: 2                     # Protect thin rims
    erosion: 0                        # Set to 1 if halos appear

    # Frame validation
    bbox_tolerance: 0.1               # 10% bbox shrinkage OK
    distance_tolerance: 0.05          # 5% distance error OK
```

---

### 8. User Documentation (`docs/MASKBUILD_USAGE.md`)

```markdown
# User-Edited Model Masking

## Overview

Generate pixel-perfect binary masks from a user-edited 3D model to exclude turntable rigs, clamps, and background from COLMAP/OpenMVS processing.

## Workflow

### Phase 1: Build Quick Model

python pipeline/bin/maskbuild.py user-init \
  --images /path/to/photos_jpg \
  --work /path/to/work_colmap_openmvs

**Output:**
- `work_colmap_openmvs/mask_build_user/coarse_model.ply`
- `work_colmap_openmvs/mask_build_user/README_EDIT_MESH.txt`

**Time:** ~5-10 minutes on A100 for 150 images

### Edit the Mesh

1. Download `coarse_model.ply` to your workstation
2. Open in MeshLab, Blender, or CloudCompare
3. **Delete all non-sherd geometry** (turntable, rods, background)
4. **Keep only pottery surfaces**
5. Save as `edited_model.ply`
6. Upload to `work_colmap_openmvs/mask_build_user/edited_model.ply`

**CRITICAL:** Do not translate, rotate, or scale the mesh!

### Phase 2: Generate Masks

python pipeline/bin/maskbuild.py user-project \
  --work /path/to/work_colmap_openmvs \
  --mesh work_colmap_openmvs/mask_build_user/edited_model.ply

**Output:**
- `work_colmap_openmvs/masks_user/*.png` (one per input image)
- `work_colmap_openmvs/masks_manifest.json`
- `work_colmap_openmvs/masks_report.txt`

**Time:** ~1-2 minutes

### Rerun Pipeline with Masks

python pipeline/bin/run_colmap.py /path/to/tree \
  --mask-path work_colmap_openmvs/masks_user

Or let the pipeline auto-detect `masks_user/` directory.

## Troubleshooting

### "Frame validation failed"

The edited mesh was moved/rotated/scaled. Re-edit without transforming.

### "Multiple models detected"

Image overlap too low. Check photos or relax thresholds in config.

### Coverage outliers

Some masks have <5% or >95% coverage. Check `masks_report.txt` for list.

## Files Reference

- `mask_build_user/` - Working directory
  - `sparse/0/` - Quick COLMAP reconstruction
  - `dense/` - Low-res undistorted images
  - `coarse_model.ply` - Initial mesh
  - `edited_model.ply` - Your edited mesh
  - `RUNLOG.txt` - Phase 1 & 2 logs
  - `frame_validation.json` - Validation results

- `masks_user/` - Final masks (PNG, 8-bit)
- `masks_manifest.json` - Checksums and metadata
- `masks_report.txt` - Coverage summary
```

---

## Implementation Roadmap

### Milestone 1: Core Infrastructure (Week 1)
- [ ] Create `maskbuild.py` CLI entry point
- [ ] Implement `maskbuild_utils.py` with config loading
- [ ] Set up logging and workspace management
- [ ] Add `maskbuild` section to `pipeline_config.yaml`

### Milestone 2: Phase 1 - Quick Model Builder (Week 1-2)
- [ ] Implement COLMAP wrapper functions (feature/match/mapper)
- [ ] Implement OpenMVS wrapper functions (interface/densify/reconstruct)
- [ ] Add mesh simplification
- [ ] Generate README and export workflow
- [ ] Test on sample dataset

### Milestone 3: Phase 2 - Projection Engine (Week 2-3)
- [ ] Implement COLMAP binary readers (cameras.bin, images.bin)
- [ ] Implement quaternion→rotation conversion
- [ ] Implement mesh→mask projection (core algorithm)
- [ ] Add padding and erosion
- [ ] Test projection accuracy

### Milestone 4: Validation & Reporting (Week 3)
- [ ] Implement frame validation (bbox, distances)
- [ ] Generate coverage statistics
- [ ] Create manifest with SHA-256 checksums
- [ ] Write summary report
- [ ] Add outlier detection

### Milestone 5: Integration (Week 4)
- [ ] Modify `run_colmap.py` to accept `--mask-path`
- [ ] Add auto-detection of `masks_user/`
- [ ] Test end-to-end pipeline with masks
- [ ] Verify registration improvements

### Milestone 6: Documentation & Testing (Week 4)
- [ ] Write `MASKBUILD_USAGE.md`
- [ ] Add examples to README
- [ ] Create test suite
- [ ] Performance benchmarking
- [ ] User acceptance testing

---

## Success Criteria

1. **Phase 1 completes in <10 minutes** for 150 images on A100
2. **Phase 2 generates masks for all images** with <1% failure rate
3. **Median coverage is 20-60%** for typical pottery captures
4. **Frame validation catches** translated/rotated meshes
5. **Masked COLMAP run registers ≥5% more images** than unmasked
6. **Reprojection errors reduced** by ≥10% with masks
7. **Idempotent execution** - reruns produce identical masks

---

## Open Questions / Design Decisions

1. **COLMAP binary readers**: Use existing library (pycolmap) or implement minimal readers?
   - **Decision**: Implement minimal readers to avoid heavy dependency

2. **Mesh format**: Support only PLY or also OBJ?
   - **Decision**: Support both via trimesh (already a dependency)

3. **Retry logic**: If Phase 1 creates multiple models, auto-retry or fail?
   - **Decision**: Auto-retry once with looser thresholds, then fail with guidance

4. **Projection method**: Ray-casting vs rasterization?
   - **Decision**: Rasterization (faster, simpler)

5. **Mask format**: PNG 8-bit or 16-bit?
   - **Decision**: 8-bit (COLMAP standard)

6. **Failed images**: Omit masks or write empty masks?
   - **Decision**: Omit (don't create file), list in manifest

7. **Incremental re-projection**: Support updating subset of masks?
   - **Decision**: Phase 2 (future enhancement)

---

## Dependencies

### New Python Dependencies
- `opencv-python` - For mask rasterization and morphological operations
- `scipy` - For distance calculations in frame validation

### Existing Dependencies (already in pipeline)
- `trimesh` - Mesh loading and manipulation
- `numpy` - Numerical operations
- `pyyaml` - Config loading

### System Dependencies (already in pipeline)
- COLMAP (binary executable)
- OpenMVS (InterfaceCOLMAP, DensifyPointCloud, ReconstructMesh)

---

## Estimated Effort

- **Core implementation**: 3-4 weeks (1 developer)
- **Testing & refinement**: 1 week
- **Documentation**: 3-5 days
- **Total**: ~5-6 weeks

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Clarify open questions**
3. **Set up development branch**
4. **Begin Milestone 1** (infrastructure)
5. **Test with sample dataset** after Milestone 2
