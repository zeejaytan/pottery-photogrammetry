#!/usr/bin/env python3
"""
Helpers to inspect pipeline stage progress and persist lightweight state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

STATE_FILE_NAME = "pipeline_state.json"


@dataclass
class StageStatus:
    """Represents completion status for each pipeline stage."""

    work_dir: Path
    colmap_complete: bool
    openmvs_complete: bool
    split_complete: bool
    notes: Dict[str, str] = field(default_factory=dict)

    @property
    def next_stage(self) -> str:
        if not self.colmap_complete:
            return "colmap"
        if not self.openmvs_complete:
            return "openmvs"
        if not self.split_complete:
            return "split"
        return "done"

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "colmap_complete": self.colmap_complete,
            "openmvs_complete": self.openmvs_complete,
            "split_complete": self.split_complete,
            "next_stage": self.next_stage,
        }
        if self.notes:
            data["notes"] = self.notes
        return data


def _resolve_dense_dir(config: Mapping[str, Any], work_dir: Path) -> Path:
    colmap_cfg = config.get("colmap", {}) or {}
    undistort_cfg = colmap_cfg.get("image_undistorter", {}) or {}
    workspace_cfg = undistort_cfg.get("output_path", "dense")
    workspace_path = Path(workspace_cfg)
    if not workspace_path.is_absolute():
        workspace_path = work_dir / workspace_path
    return workspace_path


def _resolve_dense_image_dir(config: Mapping[str, Any], work_dir: Path, dense_dir: Path) -> Path:
    colmap_cfg = config.get("colmap", {}) or {}
    undistort_cfg = colmap_cfg.get("image_undistorter", {}) or {}
    images_cfg = undistort_cfg.get("image_path")
    if images_cfg:
        images_path = Path(images_cfg)
        if not images_path.is_absolute():
            images_path = work_dir / images_path
    else:
        images_path = dense_dir / "images"
    return images_path


def evaluate_stage(config: Mapping[str, Any], work_dir: Path) -> StageStatus:
    """
    Inspect pipeline artifacts within ``work_dir`` and return stage completion flags.
    """
    work_dir = Path(work_dir)
    notes: Dict[str, str] = {}

    # First check if we have cached state
    cached_state = load_state(work_dir)
    
    colmap_cfg = config.get("colmap", {}) or {}
    database_name = colmap_cfg.get("database_name", "database.db")
    sparse_dir_name = colmap_cfg.get("sparse_dir", "sparse")

    # Use cached state if available and marked complete
    colmap_complete = cached_state.get("colmap", {}).get("completed", False)
    if not colmap_complete:
        # Fall back to directory inspection if no cached state
        database_exists = (work_dir / database_name).is_file()
        if not database_exists:
            notes["colmap"] = "database missing"

        sparse_dir = work_dir / sparse_dir_name
        sparse_complete = False
        if sparse_dir.exists():
            for child in sparse_dir.iterdir():
                if child.is_dir() and (child / "images.bin").exists() and (child / "points3D.bin").exists():
                    sparse_complete = True
                    break
        else:
            notes["colmap_sparse"] = "sparse directory missing"

        dense_dir = _resolve_dense_dir(config, work_dir)
        images_dir = _resolve_dense_image_dir(config, work_dir, dense_dir)

        dense_ready = dense_dir.exists() and images_dir.exists()
        if dense_ready:
            try:
                dense_has_images = any(images_dir.iterdir())
            except FileNotFoundError:
                dense_has_images = False
        else:
            dense_has_images = False
            notes.setdefault("dense", "undistorted images missing")

        if dense_has_images and "dense" in notes:
            notes.pop("dense", None)

        colmap_complete = database_exists and sparse_complete and dense_has_images

    # Check OpenMVS completion using cached state
    openmvs_complete = cached_state.get("openmvs", {}).get("completed", False)
    if not openmvs_complete:
        scene_mvs = work_dir / "scene_dense.mvs"
        dense_mesh = work_dir / "scene_dense_mesh.ply"
        refined_mesh = work_dir / "scene_refined_mesh.ply"
        openmvs_complete = scene_mvs.is_file() and dense_mesh.is_file() and refined_mesh.is_file()
        if not openmvs_complete:
            notes.setdefault("openmvs", "dense outputs incomplete")

    # Check split completion using cached state
    split_complete = cached_state.get("split", {}).get("completed", False)
    if not split_complete:
        validation_report = work_dir / "validation_report.csv"
        sherd_exists = any(work_dir.glob("sherd_*.ply"))
        split_complete = validation_report.is_file() and sherd_exists
        if not split_complete:
            notes.setdefault("split", "validation report or sherd meshes missing")

    return StageStatus(
        work_dir=work_dir,
        colmap_complete=colmap_complete,
        openmvs_complete=openmvs_complete,
        split_complete=split_complete,
        notes=notes,
    )


def _state_file(work_dir: Path) -> Path:
    return Path(work_dir) / STATE_FILE_NAME


def load_state(work_dir: Path) -> Dict[str, Any]:
    """
    Load cached stage state if present.
    """
    state_path = _state_file(work_dir)
    if not state_path.is_file():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def save_state(work_dir: Path, state: Mapping[str, Any]) -> None:
    """
    Persist stage state to ``pipeline_state.json``.
    """
    state_path = _state_file(work_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def mark_stage(work_dir: Path, stage: str, *, completed: bool = True, metadata: Mapping[str, Any] | None = None) -> None:
    """
    Record completion metadata for ``stage``.
    """
    stage = stage.lower()
    state = load_state(work_dir)
    entry: Dict[str, Any] = {
        "completed": bool(completed),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        entry.update(metadata)
    state[stage] = entry
    save_state(work_dir, state)
