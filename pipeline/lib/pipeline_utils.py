#!/usr/bin/env python3
"""
Shared helpers for the Rabati2025 photogrammetry pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import yaml


class PipelineError(RuntimeError):
    """Raised when a pipeline step fails."""


def load_config(config_path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load the YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config: Dict[str, Any] = yaml.safe_load(handle) or {}
    return config


def resolve_path(config: Mapping[str, Any], path_key: str) -> Path:
    """
    Resolve a path stored in the config under ``path_key`` relative to project root.
    ``path_key`` should be a dotted string (e.g. ``paths.log_dir``).
    """
    parts = path_key.split(".")
    cursor: Any = config
    for part in parts:
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise KeyError(f"{path_key} missing in configuration")
        cursor = cursor[part]
    project_root = Path(config.get("project", {}).get("root", "."))
    return (project_root / Path(cursor)).resolve()


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def configure_logging(log_file: Path, console: bool = True) -> logging.Logger:
    """Configure and return a logger that writes to ``log_file``."""
    ensure_directory(log_file.parent)
    logger = logging.getLogger("rabati_pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler(stream=sys.stdout)
        stream.setLevel(logging.INFO)
        stream.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream)

    return logger


def run_command(
    cmd: Iterable[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[MutableMapping[str, str]] = None,
    logger: Optional[logging.Logger] = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """
    Run a subprocess command and optionally log it.
    """
    cmd_list = list(map(str, cmd))
    if logger:
        logger.info("Running: %s", " ".join(cmd_list))

    process = subprocess.run(
        cmd_list,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    if logger:
        if process.stdout:
            logger.debug("stdout:\n%s", process.stdout.strip())
        if process.stderr:
            logger.debug("stderr:\n%s", process.stderr.strip())

    if check and process.returncode != 0:
        raise PipelineError(
            f"Command failed ({process.returncode}): {' '.join(cmd_list)}\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )
    return process


def export_environment(env: Mapping[str, str], output_path: Path) -> None:
    """
    Persist environment key/value pairs as JSON for downstream consumption.
    """
    ensure_directory(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(dict(env), handle, indent=2)


def build_module_load_commands(config: Mapping[str, Any]) -> List[str]:
    """
    Build shell commands for loading modules specified in configuration.
    """
    env_cfg = config.get("environment", {})
    if not env_cfg.get("use_module_system", False):
        return []

    cmds: List[str] = []
    python_module = env_cfg.get("python_module")
    if python_module:
        cmds.append(f"module load {python_module}")

    for module_name in env_cfg.get("extra_modules", []):
        cmds.append(f"module load {module_name}")
    return cmds


def ensure_colmap_database(tree_dir: Path, config: Mapping[str, Any]) -> Path:
    """Return the path to the COLMAP database for a given tree."""
    work_dir = tree_dir / "work_colmap_openmvs"
    ensure_directory(work_dir)
    database_name = config.get("colmap", {}).get("database_name", "database.db")
    return work_dir / database_name


def get_tree_workdir(tree_dir: Path) -> Path:
    """Return the working directory for a tree."""
    work_dir = tree_dir / "work_colmap_openmvs"
    ensure_directory(work_dir)
    return work_dir


def count_images(tree_dir: Path, extensions: Iterable[str]) -> int:
    """Count input images in ``tree_dir`` matching extensions."""
    extensions_lower = {ext.lower() for ext in extensions}
    count = 0
    for item in tree_dir.iterdir():
        if item.is_file() and item.suffix.lower() in extensions_lower:
            count += 1
    return count


@dataclass
class PipelineContext:
    """Convenience bundle for pipeline paths."""

    config_path: Path
    config: Dict[str, Any]
    project_root: Path
    data_root: Path

    @classmethod
    def from_config_path(cls, config_path: str | os.PathLike[str]) -> "PipelineContext":
        config_path = Path(config_path).resolve()
        config = load_config(config_path)
        project_root = Path(config.get("project", {}).get("root", config_path.parent))
        data_root = Path(config.get("project", {}).get("data_root"))
        if not data_root:
            raise PipelineError("project.data_root missing in configuration.")
        return cls(
            config_path=config_path,
            config=config,
            project_root=project_root,
            data_root=data_root,
        )

