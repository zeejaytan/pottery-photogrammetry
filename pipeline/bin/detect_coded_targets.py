#!/usr/bin/env python3
"""
Detect coded targets (ArUco/ChArUco) in images and generate COLMAP pair list.

This script detects ArUco or ChArUco markers in a set of images, refines corner
positions to sub-pixel accuracy, and generates a COLMAP-compatible pairs.txt
file for targeted feature matching.

Usage:
    detect_coded_targets.py --images <path> --work <path> [--config <path>]

Examples:
    # Basic usage
    python detect_coded_targets.py \\
        --images /path/to/photos \\
        --work /path/to/work_colmap_openmvs

    # With custom config
    python detect_coded_targets.py \\
        --images /path/to/photos \\
        --work /path/to/work_colmap_openmvs \\
        --config pipeline/config/custom_config.yaml

    # With camera calibration for pose estimation
    python detect_coded_targets.py \\
        --images /path/to/photos \\
        --work /path/to/work_colmap_openmvs \\
        --calibration /path/to/camera_calibration.npz
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.target_detection import detect_markers_in_images
from lib.target_pairing import generate_colmap_pairs, analyze_pair_coverage
from lib.pipeline_utils import PipelineError


def main():
    parser = argparse.ArgumentParser(
        description="Detect coded targets and generate COLMAP pair list",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic detection:
    %(prog)s --images photos/ --work work_colmap_openmvs/

  With custom config:
    %(prog)s --images photos/ --work work_colmap_openmvs/ --config my_config.yaml

  With camera calibration:
    %(prog)s --images photos/ --work work_colmap_openmvs/ --calibration calib.npz
        """
    )

    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Path to image directory"
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
        help="Path to pipeline configuration (default: pipeline/config/pipeline_config.yaml)"
    )

    parser.add_argument(
        "--calibration",
        type=Path,
        help="Optional camera calibration file (.npz) for pose estimation"
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze pair coverage statistics after generation"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.images.exists():
        print(f"ERROR: Images directory does not exist: {args.images}", file=sys.stderr)
        return 1

    if not args.images.is_dir():
        print(f"ERROR: Images path is not a directory: {args.images}", file=sys.stderr)
        return 1

    if not args.config.exists():
        print(f"ERROR: Config file does not exist: {args.config}", file=sys.stderr)
        return 1

    if args.calibration and not args.calibration.exists():
        print(f"ERROR: Calibration file does not exist: {args.calibration}", file=sys.stderr)
        return 1

    # Create work directory if it doesn't exist
    args.work.mkdir(parents=True, exist_ok=True)

    try:
        print("=" * 70)
        print("CODED TARGET DETECTION")
        print("=" * 70)
        print(f"Images directory: {args.images}")
        print(f"Work directory: {args.work}")
        print(f"Config file: {args.config}")
        if args.calibration:
            print(f"Calibration file: {args.calibration}")
        print()

        # Run marker detection
        print("Step 1: Detecting markers in images...")
        detections = detect_markers_in_images(
            images_dir=args.images,
            work_dir=args.work,
            config_path=args.config,
            calibration_path=args.calibration
        )

        if not detections:
            print("WARNING: No markers detected in any images", file=sys.stderr)
            print("Please check:")
            print("  - Board is visible in images")
            print("  - Correct dictionary specified in config")
            print("  - Board is not occluded or too blurry")
            return 1

        print(f"\n✓ Detected markers in {len(detections)} images")

        # Generate pairs.txt
        print("\nStep 2: Generating image pairs...")
        pairs_written = generate_colmap_pairs(
            detections=detections,
            work_dir=args.work,
            config_path=args.config
        )

        print(f"✓ Generated {pairs_written} image pairs")

        # Analyze coverage if requested
        if args.analyze:
            print("\nStep 3: Analyzing pair coverage...")
            pairs_file = args.work / "coded_targets" / "pairs.txt"
            stats = analyze_pair_coverage(detections, pairs_file)

        print("\n" + "=" * 70)
        print("DETECTION COMPLETE")
        print("=" * 70)
        print(f"Output directory: {args.work / 'coded_targets'}")
        print(f"Pairs file: {args.work / 'coded_targets' / 'pairs.txt'}")
        print(f"Metadata: {args.work / 'coded_targets' / 'target_meta.json'}")
        print(f"Detection log: {args.work / 'coded_targets' / 'detection_log.txt'}")
        print()
        print("Next steps:")
        print("  1. Run COLMAP with coded targets enabled in config")
        print("  2. COLMAP will use the generated pairs.txt for matching")
        print()

        return 0

    except PipelineError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
