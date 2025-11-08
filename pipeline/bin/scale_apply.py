#!/usr/bin/env python3
"""
Apply manual scale measurement to COLMAP sparse model.

Reads measurement.env with two independent measurements, validates they agree,
computes scale factor, applies to sparse model, and regenerates dense workspace.

Usage:
    scale_apply.py --work <path> [--config <path>]

Expected measurement.env format:
    d1_real_m=0.100
    d1_rec_units=0.0543271
    d2_real_m=0.150
    d2_rec_units=0.0814906

Examples:
    # Apply scale from measurements
    python scale_apply.py --work /path/to/work_colmap_openmvs

    # With custom image path
    python scale_apply.py --work /path/to/work_colmap_openmvs \\
        --image-path /path/to/images
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for lib imports
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
        description="Apply manual scale measurement to sparse model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic application:
    %(prog)s --work work_colmap_openmvs/

  With custom image path:
    %(prog)s --work work_colmap_openmvs/ --image-path tree/images_jpg
        """
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
        print("  # Measurement 1", file=sys.stderr)
        print("  d1_real_m=0.100", file=sys.stderr)
        print("  d1_rec_units=0.0XYZ", file=sys.stderr)
        print("  # Measurement 2", file=sys.stderr)
        print("  d2_real_m=0.150", file=sys.stderr)
        print("  d2_rec_units=0.0XYZ", file=sys.stderr)
        return 1

    try:
        print("=" * 70)
        print("MANUAL SCALE APPLICATION")
        print("=" * 70)
        print(f"Work directory: {args.work}")
        print()

        # Step 1: Read measurements
        print("Reading measurements from", measurement_file)
        measurements = read_measurement_file(measurement_file)
        print("Measurement 1:")
        print(f"  Real-world: {measurements['d1_real_m']} m ({measurements['d1_real_m']*1000:.1f} mm)")
        print(f"  Reconstructed: {measurements['d1_rec_units']}")
        print("Measurement 2:")
        print(f"  Real-world: {measurements['d2_real_m']} m ({measurements['d2_real_m']*1000:.1f} mm)")
        print(f"  Reconstructed: {measurements['d2_rec_units']}")
        print()

        # Step 2: Compute and validate scale factor
        print("Computing and validating scale factors...")
        scale_factor, validation_info = compute_scale_factor(measurements)
        print(f"  Scale 1: {validation_info['scale1']:.9f}")
        print(f"  Scale 2: {validation_info['scale2']:.9f}")
        print(f"  Agreement: {validation_info['diff_pct']:.2f}% difference")
        print(f"  Using mean: {scale_factor:.9f}")
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
        write_scale_log(args.work / "scale", scale_factor, validation_info)
        append_to_runlog(args.work, scale_factor, validation_info)
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
