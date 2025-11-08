#!/usr/bin/env python3
"""
Split pottery-tree refined mesh into individual sherd meshes.

Breaks thin bridges, computes connected components, filters noise,
and writes one mesh per sherd with full provenance.

Usage:
    # Standard splitting with defaults
    split_sherds.py --mesh dense_scaled/scene_dense_mesh_refine.ply

    # Custom thresholds
    split_sherds.py \
        --mesh dense_scaled/scene_dense_mesh_refine.ply \
        --out split_sherds \
        --min-faces 5000 \
        --max-elongation 25.0 \
        --bridge-quantile 0.995

    # Keep only top N largest sherds
    split_sherds.py --mesh scene.ply --keep-top 10

    # Keep all components (no filtering except min-faces)
    split_sherds.py --mesh scene.ply --keep-all
"""

import argparse
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.mesh_io import load_mesh, get_mesh_stats
from lib.mesh_splitter import (
    break_long_edge_bridges,
    split_into_components,
    filter_noise_components,
    validate_thresholds,
    write_component_meshes
)
from lib.pipeline_utils import PipelineError


def write_manifest(
    component_info: List[Dict],
    output_dir: Path,
    input_mesh: Path,
    params: Dict
) -> None:
    """
    Write CSV and JSON manifests with component metadata.

    Args:
        component_info: List of component statistics
        output_dir: Output directory
        input_mesh: Input mesh path
        params: Split parameters used
    """
    # CSV manifest
    csv_path = output_dir / "manifest.csv"

    fieldnames = [
        'component_id', 'kept', 'rejection_reason',
        'vertex_count', 'face_count', 'surface_area',
        'bbox_x', 'bbox_y', 'bbox_z', 'elongation'
    ]

    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for info in component_info:
            # Fill None with empty string for CSV
            row = {k: (v if v is not None else '') for k, v in info.items()}
            writer.writerow(row)

    # JSON manifest
    json_path = output_dir / "manifest.json"

    manifest = {
        'input_mesh': str(input_mesh),
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'components': component_info,
        'summary': {
            'total_components': len(component_info),
            'kept_components': sum(1 for info in component_info if info['kept']),
            'dropped_components': sum(1 for info in component_info if not info['kept']),
            'total_faces_kept': sum(info['face_count'] for info in component_info if info['kept']),
            'total_faces_dropped': sum(info['face_count'] for info in component_info if not info['kept'])
        }
    }

    with json_path.open('w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Wrote manifest: {csv_path}")
    print(f"✓ Wrote manifest: {json_path}")


def append_to_runlog(
    work_dir: Path,
    input_mesh: Path,
    params: Dict,
    num_kept: int,
    num_total: int,
    elapsed_seconds: float
) -> None:
    """
    Append split summary to pipeline runlog.

    Args:
        work_dir: Work directory
        input_mesh: Input mesh path
        params: Split parameters
        num_kept: Number of components kept
        num_total: Total number of components
        elapsed_seconds: Wall-clock time
    """
    runlog_path = work_dir / "pipeline_RUNLOG.txt"

    with runlog_path.open('a') as f:
        f.write(f"\n--- MESH SPLIT ({datetime.now().isoformat()}) ---\n")
        f.write(f"Input:        {input_mesh.name}\n")
        f.write(f"Min faces:    {params['min_faces']}\n")
        f.write(f"Max elong:    {params['max_elongation']:.1f}\n")
        f.write(f"Bridge quant: {params.get('bridge_quantile', 'N/A')}\n")
        f.write(f"Components:   {num_kept}/{num_total} kept\n")
        f.write(f"Time:         {elapsed_seconds:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(
        description="Split refined mesh into individual sherd meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--mesh",
        type=Path,
        required=True,
        help="Path to refined mesh (PLY or OBJ)"
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("split_sherds"),
        help="Output directory for sherd meshes (default: split_sherds)"
    )

    parser.add_argument(
        "--min-faces",
        type=int,
        default=5000,
        help="Minimum face count to keep component (default: 5000)"
    )

    parser.add_argument(
        "--max-elongation",
        type=float,
        default=25.0,
        help="Maximum elongation ratio to keep (default: 25.0)"
    )

    parser.add_argument(
        "--bridge-quantile",
        type=float,
        default=0.995,
        help="Edge length quantile for bridge breaking (default: 0.995)"
    )

    parser.add_argument(
        "--keep-top",
        type=int,
        help="Keep only top N largest components (overrides other filters)"
    )

    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Keep all components (only apply min-faces filter)"
    )

    parser.add_argument(
        "--format",
        choices=['ply', 'obj'],
        default='ply',
        help="Output mesh format (default: ply)"
    )

    parser.add_argument(
        "--no-bridge-break",
        action="store_true",
        help="Skip long-edge bridge breaking step"
    )

    args = parser.parse_args()

    # Resolve paths
    mesh_path = args.mesh.resolve()
    output_dir = args.out.resolve()

    # Determine work directory (for runlog)
    # Assume mesh is in work_dir/dense_scaled/scene.ply
    if 'work_' in str(mesh_path):
        # Find the work directory in the path
        parts = mesh_path.parts
        for i, part in enumerate(parts):
            if part.startswith('work_'):
                work_dir = Path(*parts[:i+1])
                break
        else:
            work_dir = mesh_path.parent
    else:
        work_dir = mesh_path.parent

    import time
    start_time = time.time()

    try:
        print("=" * 70)
        print("MESH SHERD SPLITTER")
        print("=" * 70)
        print(f"Input mesh: {mesh_path}")
        print(f"Output dir: {output_dir}")
        print()

        # Step 1: Load mesh
        print("Step 1: Loading mesh...")
        mesh = load_mesh(mesh_path)
        input_stats = get_mesh_stats(mesh)
        print(f"✓ Loaded mesh: {input_stats['vertex_count']} vertices, "
              f"{input_stats['face_count']} faces")
        print(f"  Surface area: {input_stats['surface_area']:.6f} sq units")
        print()

        # Step 2: Break bridges (optional)
        if not args.no_bridge_break:
            print(f"Step 2: Breaking long-edge bridges (quantile={args.bridge_quantile})...")
            mesh = break_long_edge_bridges(mesh, args.bridge_quantile)
            print()
        else:
            print("Step 2: Skipping bridge breaking (--no-bridge-break)")
            print()

        # Step 3: Split into components
        print("Step 3: Computing connected components...")
        components = split_into_components(mesh)
        print()

        # Step 4: Validate thresholds
        if not args.keep_all and not args.keep_top:
            print("Step 4: Validating thresholds...")
            validate_thresholds(components, args.min_faces, args.max_elongation)
            print("✓ Thresholds are valid")
            print()

        # Step 5: Filter components
        print("Step 5: Filtering noise components...")

        if args.keep_top:
            # Keep only top N
            kept_components = components[:args.keep_top]
            component_info = []

            for idx, comp in enumerate(components):
                stats = get_mesh_stats(comp)
                info = {
                    'component_id': idx,
                    'vertex_count': stats['vertex_count'],
                    'face_count': stats['face_count'],
                    'surface_area': stats['surface_area'],
                    'bbox_x': stats['bbox_x'],
                    'bbox_y': stats['bbox_y'],
                    'bbox_z': stats['bbox_z'],
                    'elongation': stats['elongation'],
                    'kept': idx < args.keep_top,
                    'rejection_reason': None if idx < args.keep_top else f"not_in_top_{args.keep_top}"
                }
                component_info.append(info)

            print(f"✓ Kept top {args.keep_top} components by size")

        elif args.keep_all:
            # Keep all components above min_faces
            kept_components, component_info = filter_noise_components(
                components,
                min_faces=args.min_faces,
                max_elongation=float('inf'),  # No elongation filter
                area_elongation_product_threshold=None  # No rod filter
            )
            print(f"✓ Kept all components with ≥{args.min_faces} faces")

        else:
            # Standard filtering
            kept_components, component_info = filter_noise_components(
                components,
                min_faces=args.min_faces,
                max_elongation=args.max_elongation
            )

        print()

        # Step 6: Write component meshes
        print(f"Step 6: Writing {len(kept_components)} sherd meshes...")
        written_paths = write_component_meshes(
            kept_components,
            component_info,
            output_dir,
            file_format=args.format
        )
        print()

        # Step 7: Write manifests
        print("Step 7: Writing manifests...")
        params = {
            'min_faces': args.min_faces,
            'max_elongation': args.max_elongation,
            'bridge_quantile': args.bridge_quantile if not args.no_bridge_break else None,
            'keep_top': args.keep_top,
            'keep_all': args.keep_all,
            'format': args.format
        }
        write_manifest(component_info, output_dir, mesh_path, params)
        print()

        # Step 8: Update runlog
        elapsed = time.time() - start_time
        if work_dir.exists():
            print("Step 8: Updating pipeline runlog...")
            append_to_runlog(
                work_dir,
                mesh_path,
                params,
                len(kept_components),
                len(components),
                elapsed
            )

        print()
        print("=" * 70)
        print("MESH SPLITTING COMPLETE")
        print("=" * 70)
        print(f"Components found:   {len(components)}")
        print(f"Components kept:    {len(kept_components)}")
        print(f"Components dropped: {len(components) - len(kept_components)}")
        print(f"Output directory:   {output_dir}")
        print(f"Time elapsed:       {elapsed:.1f}s")
        print()

        # Show summary table
        kept_info = [info for info in component_info if info['kept']]
        if kept_info:
            print("Kept components:")
            print(f"  {'ID':<6} {'Faces':<10} {'Vertices':<10} {'Area':<12} {'Elongation':<12}")
            print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
            for info in kept_info[:10]:  # Show first 10
                print(f"  {info['component_id']:<6} "
                      f"{info['face_count']:<10} "
                      f"{info['vertex_count']:<10} "
                      f"{info['surface_area']:<12.6f} "
                      f"{info['elongation']:<12.2f}")
            if len(kept_info) > 10:
                print(f"  ... and {len(kept_info) - 10} more")

        print()
        print("Next steps:")
        print(f"  1. Review sherds on laptop: {output_dir}")
        print(f"  2. Check manifest: {output_dir}/manifest.csv")
        print(f"  3. Texture individual sherds or full scene as needed")
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
