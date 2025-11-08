"""
Mesh I/O utilities for PLY and OBJ formats.

Handles loading, saving, and metadata extraction for pottery meshes.
"""

import trimesh
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from .pipeline_utils import PipelineError


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """
    Load mesh from PLY or OBJ file.

    Args:
        mesh_path: Path to mesh file

    Returns:
        trimesh.Trimesh object

    Raises:
        PipelineError if file missing or corrupt
    """
    if not mesh_path.exists():
        raise PipelineError(f"Mesh file not found: {mesh_path}")

    try:
        mesh = trimesh.load(str(mesh_path), force='mesh', process=False)

        if not isinstance(mesh, trimesh.Trimesh):
            raise PipelineError(f"File is not a single mesh: {mesh_path}")

        return mesh

    except Exception as e:
        raise PipelineError(f"Failed to load mesh {mesh_path}: {e}")


def save_mesh(
    mesh: trimesh.Trimesh,
    output_path: Path,
    file_format: Optional[str] = None
) -> None:
    """
    Save mesh to PLY or OBJ file.

    Args:
        mesh: trimesh.Trimesh object
        output_path: Output file path
        file_format: 'ply' or 'obj' (auto-detected from extension if None)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if file_format is None:
        file_format = output_path.suffix.lower().replace('.', '')

    mesh.export(str(output_path), file_type=file_format)


def get_mesh_stats(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """
    Compute basic statistics for a mesh.

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        Dict with vertex_count, face_count, surface_area,
        bbox dimensions (x, y, z), elongation
    """
    bbox = mesh.bounding_box.extents  # [x, y, z] dimensions

    # Elongation: max dimension / min dimension
    # Handle edge case where mesh is completely flat
    min_dim = bbox.min()
    max_dim = bbox.max()
    elongation = max_dim / min_dim if min_dim > 1e-9 else float('inf')

    return {
        'vertex_count': len(mesh.vertices),
        'face_count': len(mesh.faces),
        'surface_area': float(mesh.area),
        'bbox_x': float(bbox[0]),
        'bbox_y': float(bbox[1]),
        'bbox_z': float(bbox[2]),
        'elongation': float(elongation),
        'min_dim': float(min_dim),
        'max_dim': float(max_dim)
    }


def get_edge_lengths(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Compute all unique edge lengths in mesh.

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        1D numpy array of edge lengths
    """
    # Get unique edges
    edges = mesh.edges_unique

    # Compute lengths
    edge_vectors = mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)

    return edge_lengths
