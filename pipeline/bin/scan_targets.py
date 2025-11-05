#!/usr/bin/env python3
"""
Discover pottery-tree folders containing enough images and produce targets.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lib.pipeline_utils import PipelineContext, count_images, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Rabati2025 dataset for valid pottery-tree targets."
    )
    parser.add_argument(
        "--config",
        default="pipeline/config/pipeline_config.yaml",
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional override for output targets file path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print targets instead of writing to file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = PipelineContext.from_config_path(args.config)
    config = context.config

    targets_cfg = config.get("targets", {})
    min_jpegs = targets_cfg.get("min_jpegs_per_tree", 10)
    extensions = targets_cfg.get("extensions", [".jpg", ".jpeg"])
    target_file = (
        Path(args.output).resolve()
        if args.output
        else Path(context.project_root) / targets_cfg.get("targets_file", "pipeline/targets.txt")
    )

    data_root = context.data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    valid_targets: list[str] = []
    for date_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for tree_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            image_count = count_images(tree_dir, extensions)
            if image_count < min_jpegs:
                continue
            relative_path = f"{date_dir.name}/{tree_dir.name}"
            valid_targets.append(relative_path)

    if args.dry_run:
        for target in valid_targets:
            print(target)
        return 0

    ensure_directory(target_file.parent)
    target_file.write_text("\n".join(valid_targets) + ("\n" if valid_targets else ""))
    print(f"Wrote {len(valid_targets)} targets to {target_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
