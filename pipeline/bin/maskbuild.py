#!/usr/bin/env python3
"""
User-edited model mask generation CLI.

Two-phase workflow:
  1. user-init: Build quick coarse model for editing
  2. user-project: Project edited mesh to generate masks

Usage:
    maskbuild.py user-init --images <path> --work <path> [--config <path>]
    maskbuild.py user-project --work <path> --mesh <path> [--config <path>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.maskbuild_phase1 import run_phase1
from lib.maskbuild_phase2 import run_phase2
from lib.pipeline_utils import PipelineError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate masks from user-edited 3D model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Build coarse model
  %(prog)s user-init \\
    --images /data/pottery_001/photos_jpg \\
    --work /data/pottery_001/work_colmap_openmvs

  # Edit the mesh: coarse_model.ply -> edited_model.ply

  # Phase 2: Generate masks
  %(prog)s user-project \\
    --work /data/pottery_001/work_colmap_openmvs \\
    --mesh work_colmap_openmvs/mask_build_user/edited_model.ply
"""
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # =========================
    # Phase 1: user-init
    # =========================
    init_parser = subparsers.add_parser(
        "user-init",
        help="Build quick coarse model for editing",
        description="Build a fast, low-resolution 3D model for user editing"
    )
    init_parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Path to JPEG image directory"
    )
    init_parser.add_argument(
        "--work",
        required=True,
        type=Path,
        help="Path to work_colmap_openmvs directory"
    )
    init_parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline/config/pipeline_config.yaml"),
        help="Path to pipeline configuration (default: pipeline/config/pipeline_config.yaml)"
    )

    # =========================
    # Phase 2: user-project
    # =========================
    project_parser = subparsers.add_parser(
        "user-project",
        help="Project edited mesh to generate masks",
        description="Validate edited mesh and project to all camera views"
    )
    project_parser.add_argument(
        "--work",
        required=True,
        type=Path,
        help="Path to work_colmap_openmvs directory"
    )
    project_parser.add_argument(
        "--mesh",
        required=True,
        type=Path,
        help="Path to edited mesh file (PLY or OBJ)"
    )
    project_parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline/config/pipeline_config.yaml"),
        help="Path to pipeline configuration (default: pipeline/config/pipeline_config.yaml)"
    )
    project_parser.add_argument(
        "--pad-pixels",
        type=int,
        default=None,
        help="Padding around mask edges in pixels (default: from config, typically 2)"
    )
    project_parser.add_argument(
        "--erosion",
        type=int,
        default=None,
        help="Erosion iterations to prevent halos (default: from config, typically 0)"
    )

    args = parser.parse_args()

    try:
        if args.command == "user-init":
            # Resolve paths
            images_dir = args.images.resolve()
            work_dir = args.work.resolve()
            config_path = args.config.resolve()

            # Validate inputs
            if not images_dir.exists():
                raise PipelineError(f"Images directory not found: {images_dir}")

            if not config_path.exists():
                raise PipelineError(f"Config file not found: {config_path}")

            # Run Phase 1
            return run_phase1(
                images_dir=images_dir,
                work_dir=work_dir,
                config_path=config_path
            )

        elif args.command == "user-project":
            # Resolve paths
            work_dir = args.work.resolve()
            mesh_path = args.mesh.resolve()
            config_path = args.config.resolve()

            # Validate inputs
            if not work_dir.exists():
                raise PipelineError(f"Work directory not found: {work_dir}")

            if not mesh_path.exists():
                raise PipelineError(f"Mesh file not found: {mesh_path}")

            if not config_path.exists():
                raise PipelineError(f"Config file not found: {config_path}")

            # Run Phase 2
            return run_phase2(
                work_dir=work_dir,
                mesh_path=mesh_path,
                config_path=config_path,
                pad_pixels=args.pad_pixels,
                erosion=args.erosion
            )

    except PipelineError as e:
        print(f"\n❌ ERROR: {e}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
