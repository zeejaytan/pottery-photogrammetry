#!/usr/bin/env python3
"""
Split refined mesh into per-sherd meshes and validate vertex counts.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Tuple

import trimesh

from lib.pipeline_utils import PipelineContext, PipelineError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split and validate pottery sherd meshes.")
    parser.add_argument(
        "--config",
        default="pipeline/config/pipeline_config.yaml",
        help="Path to pipeline configuration file.",
    )
    parser.add_argument(
        "--tree",
        help="Tree identifier relative to Rabati2025 root (e.g., 04052025/O01).",
    )
    parser.add_argument(
        "--mesh",
        help="Explicit path to refined mesh file. Overrides --tree.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to store split meshes. Defaults to mesh parent directory.",
    )
    parser.add_argument(
        "--min-vertices",
        type=int,
        help="Minimum vertex count per sherd. Overrides config.validation.min_vertices.",
    )
    return parser.parse_args()


def determine_paths(args: argparse.Namespace, context: PipelineContext) -> Tuple[Path, Path]:
    if args.mesh:
        mesh_path = Path(args.mesh).resolve()
        if not mesh_path.exists():
            raise PipelineError(f"Mesh file not found: {mesh_path}")
        output_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else mesh_path.parent
        )
        return mesh_path, output_dir

    if not args.tree:
        raise PipelineError("Either --tree or --mesh must be provided.")

    tree_dir = (context.data_root / args.tree).resolve()
    if not tree_dir.exists():
        raise PipelineError(f"Tree directory not found: {tree_dir}")
    work_dir = tree_dir / "work_colmap_openmvs"
    if not work_dir.exists():
        raise PipelineError(f"Work directory not found: {work_dir}")

    mesh_path = work_dir / "scene_refined_mesh.ply"
    if not mesh_path.exists():
        raise PipelineError(f"Refined mesh missing: {mesh_path}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else work_dir
    return mesh_path, output_dir


def export_components(
    mesh: trimesh.Trimesh,
    output_dir: Path,
    min_vertices: int,
    min_component_vertices: int,
    min_component_area: float,
) -> Tuple[List[dict], int]:
    components = mesh.split(only_watertight=False)
    if not components:
        raise PipelineError("No connected components found in mesh.")

    components.sort(key=lambda comp: len(comp.vertices), reverse=True)

    records: List[dict] = []
    exported = 0
    for index, component in enumerate(components, start=1):
        vertex_count = int(len(component.vertices))
        face_count = int(len(component.faces))
        surface_area = float(component.area)
        sherd_name = f"sherd_{index:03d}.ply"
        status = "PASS" if vertex_count >= min_vertices else "FAIL"

        keep_component = (
            vertex_count >= min_component_vertices and surface_area >= min_component_area
        )
        output_path = output_dir / sherd_name

        if keep_component:
            component.export(output_path)
            exported += 1
            file_path = output_path.relative_to(output_dir)
        else:
            status = "SKIP"
            file_path = Path("-")

        records.append(
            {
                "sherd_id": sherd_name,
                "vertex_count": vertex_count,
                "face_count": face_count,
                "surface_area": surface_area,
                "status": status,
                "file": str(file_path),
            }
        )

    return records, exported


def write_report(output_dir: Path, records: List[dict]) -> Path:
    report_path = output_dir / "validation_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sherd_id",
                "vertex_count",
                "face_count",
                "surface_area",
                "status",
                "file",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    return report_path


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[Split] %(message)s")
    logger = logging.getLogger("split")

    context = PipelineContext.from_config_path(args.config)
    mesh_path, output_dir = determine_paths(args, context)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_cfg = context.config.get("validation", {})
    min_vertices = args.min_vertices or validation_cfg.get("min_vertices", 100000)
    min_component_vertices = validation_cfg.get("min_component_vertices", 1000)
    min_component_area = validation_cfg.get("min_component_area", 1.0)

    logger.info("Loading mesh from %s", mesh_path)
    mesh = trimesh.load(mesh_path, process=False)

    records, exported = export_components(
        mesh,
        output_dir,
        min_vertices,
        min_component_vertices,
        min_component_area,
    )

    report_path = write_report(output_dir, records)
    logger.info("Wrote validation report to %s", report_path)
    logger.info("Exported %d components", exported)

    all_pass = all(record["status"] == "PASS" for record in records if record["status"] != "SKIP")
    if exported == 0:
        logger.warning("No components met export criteria; marking as failure.")
        return 3

    return 0 if all_pass else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        logging.error("%s", exc)
        raise SystemExit(3)
