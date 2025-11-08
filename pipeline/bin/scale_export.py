#!/usr/bin/env python3
"""
Export COLMAP sparse model to PLY and TXT for scale measurement.

This script exports the sparse reconstruction to formats suitable for
manual measurement on a laptop (PLY) and provenance (TXT).

Usage:
    scale_export.py --work <path> [--config <path>]

Outputs:
    - sparse_ply/points3D.ply  (for laptop measurement)
    - sparse_txt/*.txt         (for provenance)
    - scale/MANIFEST.txt       (run metadata)

Examples:
    # Export sparse from default location
    python scale_export.py --work /path/to/work_colmap_openmvs

    # Export specific sparse model
    python scale_export.py --work /path/to/work_colmap_openmvs \\
        --sparse-model sparse/0
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.manual_scale import export_sparse_for_measurement
from lib.pipeline_utils import PipelineError


def main():
    parser = argparse.ArgumentParser(
        description="Export COLMAP sparse model for scale measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic export:
    %(prog)s --work work_colmap_openmvs/

  Export specific model:
    %(prog)s --work work_colmap_openmvs/ --sparse-model sparse/0
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
        if not sparse_path.is_absolute():
            sparse_path = args.work / sparse_path
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
        print(f"  3. Measure TWO known distances:")
        print(f"     Measurement 1 (e.g., base diameter):")
        print(f"       - Note the REAL-WORLD distance in metres (e.g., 0.100 for 100mm)")
        print(f"       - Note the RECONSTRUCTED distance from the viewer")
        print(f"     Measurement 2 (e.g., pottery height):")
        print(f"       - Note the REAL-WORLD distance in metres (e.g., 0.150 for 150mm)")
        print(f"       - Note the RECONSTRUCTED distance from the viewer")
        print(f"  4. On Spartan, edit {args.work}/scale/measurement.env:")
        print(f"       d1_real_m=0.100")
        print(f"       d1_rec_units=0.0XYZ")
        print(f"       d2_real_m=0.150")
        print(f"       d2_rec_units=0.0XYZ")
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
