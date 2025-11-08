#!/usr/bin/env python3
"""
ArUco marker-based scale measurement for COLMAP reconstructions.

Detects ArUco markers, triangulates 3D positions, prompts for real-world
distance, and applies scale to sparse model.

Usage:
    # Interactive mode (prompts for IDs and distance)
    scale_aruco.py --images images_jpg --sparse sparse/0

    # Non-interactive mode (fully scripted)
    scale_aruco.py --images images_jpg --sparse sparse/0 \
        --idA 11 --idB 23 --real-mm 100.0

    # With work directory
    scale_aruco.py --work work_colmap_openmvs
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.aruco_scale import (
    detect_markers_in_images,
    compute_marker_coverage,
    triangulate_marker_center,
    prompt_marker_ids,
    prompt_real_distance,
    compute_and_validate_scale,
    prompt_confirmation,
    write_aruco_scale_log
)
from lib.colmap_io import (
    load_cameras_txt,
    load_images_txt,
    export_sparse_to_txt
)
from lib.manual_scale import (
    apply_scale_to_sparse,
    regenerate_dense_workspace,
    append_to_runlog
)
from lib.pipeline_utils import PipelineError, PipelineContext


def main():
    parser = argparse.ArgumentParser(
        description="ArUco marker-based scale measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--work",
        type=Path,
        help="Work directory (auto-detects images and sparse)"
    )

    parser.add_argument(
        "--images",
        type=Path,
        help="Images directory"
    )

    parser.add_argument(
        "--sparse",
        type=Path,
        help="Sparse model directory (e.g., sparse/0)"
    )

    parser.add_argument(
        "--idA",
        type=int,
        help="Marker ID A (non-interactive mode)"
    )

    parser.add_argument(
        "--idB",
        type=int,
        help="Marker ID B (non-interactive mode)"
    )

    parser.add_argument(
        "--real-mm",
        type=float,
        help="Real-world distance in millimetres (non-interactive)"
    )

    parser.add_argument(
        "--real-m",
        type=float,
        help="Real-world distance in metres (non-interactive)"
    )

    parser.add_argument(
        "--dictionary",
        default="DICT_4X4_50",
        help="ArUco dictionary (default: DICT_4X4_50)"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline/config/pipeline_config.yaml"),
        help="Pipeline configuration"
    )

    parser.add_argument(
        "--colmap",
        default="colmap",
        help="COLMAP executable"
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Require all parameters (no prompts)"
    )

    args = parser.parse_args()

    # Determine paths
    if args.work:
        work_dir = args.work
        images_dir = args.images or work_dir.parent
        sparse_dir = args.sparse or work_dir / "sparse" / "0"
    else:
        if not args.images or not args.sparse:
            print("ERROR: Must provide --work OR both --images and --sparse", file=sys.stderr)
            return 1
        images_dir = args.images
        sparse_dir = args.sparse
        work_dir = sparse_dir.parent.parent  # Assume sparse/0 structure

    # Validate paths
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}", file=sys.stderr)
        return 1

    if not sparse_dir.exists():
        print(f"ERROR: Sparse model not found: {sparse_dir}", file=sys.stderr)
        print("Run COLMAP Stage 1 (mapper) first.", file=sys.stderr)
        return 1

    try:
        print("=" * 70)
        print("ARUCO MARKER-BASED SCALE MEASUREMENT")
        print("=" * 70)
        print(f"Images:  {images_dir}")
        print(f"Sparse:  {sparse_dir}")
        print()

        # Step 1: Export sparse to TXT format
        print("Step 1: Exporting sparse model to TXT format...")
        sparse_txt_dir = work_dir / "sparse_txt"
        export_sparse_to_txt(sparse_dir, sparse_txt_dir, args.colmap)
        print(f"✓ Exported to {sparse_txt_dir}")
        print()

        # Step 2: Load camera parameters and poses
        print("Step 2: Loading camera parameters and poses...")
        cameras = load_cameras_txt(sparse_txt_dir / "cameras.txt")
        images = load_images_txt(sparse_txt_dir / "images.txt")
        print(f"✓ Loaded {len(cameras)} cameras, {len(images)} images")
        print()

        # Step 3: Detect markers
        print("Step 3: Detecting ArUco markers...")
        detections = detect_markers_in_images(images_dir, args.dictionary)

        if not detections:
            print("\nERROR: No markers detected in any images.", file=sys.stderr)
            print("Possible issues:", file=sys.stderr)
            print("  - No markers visible in images", file=sys.stderr)
            print("  - Wrong dictionary specified", file=sys.stderr)
            print("  - Markers too small or blurry", file=sys.stderr)
            print("\nYou can fall back to manual PLY measurement instead.", file=sys.stderr)
            return 2

        coverage = compute_marker_coverage(detections)

        # Save detection info
        scale_dir = work_dir / "scale"
        scale_dir.mkdir(exist_ok=True, parents=True)

        import json
        with (scale_dir / "detected_ids.json").open("w") as f:
            json.dump(coverage, f, indent=2)

        print(f"✓ Detected {len(coverage)} unique markers")
        print()

        # Step 4: Choose markers (interactive or from args)
        if args.idA is not None and args.idB is not None:
            marker_idA = args.idA
            marker_idB = args.idB
            print(f"Using markers: {marker_idA}, {marker_idB} (from arguments)")
        else:
            if args.non_interactive:
                print("ERROR: --non-interactive requires --idA and --idB", file=sys.stderr)
                return 1
            marker_idA, marker_idB = prompt_marker_ids(coverage)

        print()

        # Step 5: Triangulate marker centers
        print(f"Step 4: Triangulating marker centers...")
        XA, views_A = triangulate_marker_center(marker_idA, detections, cameras, images)
        XB, views_B = triangulate_marker_center(marker_idB, detections, cameras, images)

        distance_reconstructed = float(np.linalg.norm(XA - XB))
        print(f"✓ Marker {marker_idA}: triangulated from {views_A} views")
        print(f"✓ Marker {marker_idB}: triangulated from {views_B} views")
        print(f"✓ Reconstructed distance: {distance_reconstructed:.6f} units")
        print()

        # Step 6: Get real-world distance
        if args.real_mm is not None:
            distance_real_m = args.real_mm / 1000.0
            print(f"Using distance: {args.real_mm:.2f} mm = {distance_real_m:.6f} m (from arguments)")
        elif args.real_m is not None:
            distance_real_m = args.real_m
            print(f"Using distance: {distance_real_m:.6f} m (from arguments)")
        else:
            if args.non_interactive:
                print("ERROR: --non-interactive requires --real-mm or --real-m", file=sys.stderr)
                return 1
            distance_real_m = prompt_real_distance()

        print()

        # Step 7: Compute scale
        print("Step 5: Computing scale factor...")
        scale_factor = compute_and_validate_scale(
            distance_reconstructed,
            distance_real_m
        )
        print()

        # Step 8: Confirm (if interactive)
        if not args.non_interactive:
            if not prompt_confirmation(
                marker_idA, marker_idB,
                distance_real_m, distance_reconstructed,
                scale_factor
            ):
                print("\nScale measurement cancelled by user.")
                return 0
            print()

        # Step 9: Write logs
        print("Step 6: Writing provenance logs...")
        write_aruco_scale_log(
            scale_dir,
            marker_idA, marker_idB,
            distance_real_m, distance_reconstructed,
            scale_factor,
            views_A, views_B
        )
        print()

        # Step 10: Apply scale to sparse
        print("Step 7: Applying scale to sparse model...")
        sparse_scaled = work_dir / "sparse_scaled" / "0"
        apply_scale_to_sparse(
            sparse_dir,
            sparse_scaled,
            scale_factor,
            args.colmap
        )
        print()

        # Step 11: Regenerate dense workspace
        print("Step 8: Regenerating dense workspace...")
        dense_scaled = work_dir / "dense_scaled"

        # Load config for max_image_size
        context = PipelineContext.from_config_path(args.config)
        max_image_size = context.config.get("colmap", {}).get("image_undistorter", {}).get("max_image_size", 6000)

        regenerate_dense_workspace(
            images_dir,
            sparse_scaled,
            dense_scaled,
            max_image_size,
            args.colmap
        )
        print()

        # Step 12: Update runlog
        print("Step 9: Updating pipeline runlog...")
        validation_info = {
            'scale1': scale_factor,
            'scale2': scale_factor,
            'diff_pct': 0.0,
            'd1_real_m': distance_real_m,
            'd1_rec_units': distance_reconstructed,
            'd2_real_m': distance_real_m,
            'd2_rec_units': distance_reconstructed
        }
        append_to_runlog(work_dir, scale_factor, validation_info)
        print()

        print("=" * 70)
        print("ARUCO SCALE MEASUREMENT COMPLETE")
        print("=" * 70)
        print(f"Scaled sparse: {sparse_scaled}")
        print(f"Scaled dense:  {dense_scaled}")
        print(f"Scale factor:  {scale_factor:.9f}")
        print()
        print("Next steps:")
        print("  1. Verify scale in CloudCompare (measure marker distance in dense)")
        print("  2. Continue to OpenMVS using dense_scaled/")
        print(f"     InterfaceCOLMAP -i {dense_scaled} -o scene.mvs")
        print()

        return 0

    except PipelineError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
