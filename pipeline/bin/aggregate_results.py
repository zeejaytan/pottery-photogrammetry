#!/usr/bin/env python3
"""
Aggregate per-tree validation reports into a single CSV summary.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from lib.pipeline_utils import PipelineContext, PipelineError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate pipeline validation results.")
    parser.add_argument(
        "--config",
        default="pipeline/config/pipeline_config.yaml",
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument(
        "--targets",
        help="Optional path to targets file. Defaults to config.targets.targets_file.",
    )
    parser.add_argument(
        "--output",
        help="Optional output CSV path. Defaults to pipeline/summary_report.csv.",
    )
    return parser.parse_args()


def read_targets(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    context = PipelineContext.from_config_path(args.config)
    cfg = context.config

    targets_path = (
        Path(args.targets).resolve()
        if args.targets
        else Path(context.project_root)
        / cfg.get("targets", {}).get("targets_file", "pipeline/targets.txt")
    )

    if not targets_path.exists():
        raise PipelineError(f"Targets file not found: {targets_path}")

    summary_path = (
        Path(args.output).resolve()
        if args.output
        else Path(context.project_root) / "pipeline/summary_report.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    targets = read_targets(targets_path)
    rows: list[dict[str, str]] = []
    for target in targets:
        tree_dir = (context.data_root / target).resolve()
        report_path = tree_dir / "work_colmap_openmvs" / "validation_report.csv"
        if not report_path.exists():
            rows.append(
                {
                    "tree": target,
                    "status": "MISSING",
                    "sherd_id": "",
                    "vertex_count": "",
                    "face_count": "",
                    "surface_area": "",
                    "file": "",
                }
            )
            continue

        with report_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for entry in reader:
                rows.append(
                    {
                        "tree": target,
                        "status": entry.get("status", ""),
                        "sherd_id": entry.get("sherd_id", ""),
                        "vertex_count": entry.get("vertex_count", ""),
                        "face_count": entry.get("face_count", ""),
                        "surface_area": entry.get("surface_area", ""),
                        "file": entry.get("file", ""),
                    }
                )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tree",
                "status",
                "sherd_id",
                "vertex_count",
                "face_count",
                "surface_area",
                "file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary to {summary_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
