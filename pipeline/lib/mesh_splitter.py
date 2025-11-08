"""
Mesh splitting logic for pottery sherd separation.

Breaks thin bridges, computes connected components, filters noise.
"""

import trimesh
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .mesh_io import get_mesh_stats, get_edge_lengths
from .pipeline_utils import PipelineError

logger = logging.getLogger(__name__)


def break_long_edge_bridges(
    mesh: trimesh.Trimesh,
    quantile: float = 0.995
) -> trimesh.Trimesh:
    """
    Remove faces connected by extremely long edges to break thin bridges.

    Args:
        mesh: Input mesh
        quantile: Keep edges below this quantile (default: 99.5%, drops longest 0.5%)

    Returns:
        New mesh with bridge faces removed
    """
    if quantile >= 1.0 or quantile <= 0.0:
        logger.warning(f"Invalid quantile {quantile}, skipping bridge breaking")
        return mesh

    edge_lengths = get_edge_lengths(mesh)

    if len(edge_lengths) == 0:
        return mesh

    # Compute threshold
    threshold = np.quantile(edge_lengths, quantile)

    logger.info(f"Edge length quantile {quantile:.1%}: {threshold:.6f} units")
    logger.info(f"Median edge length: {np.median(edge_lengths):.6f} units")

    # Find faces that have at least one edge above threshold
    faces_to_remove = []

    for face_idx, face in enumerate(mesh.faces):
        # Check all three edges of triangle
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]

            edge_vec = mesh.vertices[v2] - mesh.vertices[v1]
            edge_len = np.linalg.norm(edge_vec)

            if edge_len > threshold:
                faces_to_remove.append(face_idx)
                break

    if len(faces_to_remove) == 0:
        logger.info("No long-edge bridges found")
        return mesh

    # Create mask for faces to keep
    face_mask = np.ones(len(mesh.faces), dtype=bool)
    face_mask[faces_to_remove] = False

    # Create new mesh with remaining faces
    new_mesh = mesh.copy()
    new_mesh.update_faces(face_mask)
    new_mesh.remove_unreferenced_vertices()

    logger.info(f"Removed {len(faces_to_remove)} faces with long edges")

    return new_mesh


def split_into_components(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
    """
    Split mesh into connected components.

    Args:
        mesh: Input mesh

    Returns:
        List of component meshes, sorted by face count (largest first)
    """
    components = mesh.split(only_watertight=False)

    # trimesh.split returns list or single mesh
    if not isinstance(components, list):
        components = [components]

    # Sort by face count (largest first)
    components.sort(key=lambda m: len(m.faces), reverse=True)

    logger.info(f"Found {len(components)} connected components")

    return components


def filter_noise_components(
    components: List[trimesh.Trimesh],
    min_faces: int = 5000,
    max_elongation: float = 25.0,
    area_elongation_product_threshold: Optional[float] = None
) -> Tuple[List[trimesh.Trimesh], List[Dict]]:
    """
    Filter out noise components (small wisps, rods, clamps).

    Args:
        components: List of component meshes
        min_faces: Minimum face count to keep
        max_elongation: Maximum elongation ratio (max_dim/min_dim)
        area_elongation_product_threshold: If set, drop components where
            (elongation * 1000 / area) > threshold (for rod detection)

    Returns:
        Tuple of (kept_components, all_component_info)
        where all_component_info includes rejection reasons
    """
    kept = []
    all_info = []

    # Compute median area for relative filtering
    areas = [comp.area for comp in components]
    median_area = np.median(areas) if len(areas) > 0 else 0.0

    logger.info(f"Median component area: {median_area:.6f} sq units")

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
            'kept': False,
            'rejection_reason': None
        }

        # Filter 1: Minimum face count
        if stats['face_count'] < min_faces:
            info['rejection_reason'] = f"too_few_faces (< {min_faces})"
            all_info.append(info)
            continue

        # Filter 2: Maximum elongation
        if stats['elongation'] > max_elongation:
            info['rejection_reason'] = f"too_elongated (> {max_elongation:.1f})"
            all_info.append(info)
            continue

        # Filter 3: Rod-like artifacts (high elongation + small area)
        if area_elongation_product_threshold is not None:
            # Metric: elongation / (area / median_area)
            # High value = elongated AND small relative to typical sherd
            if median_area > 0:
                relative_area = stats['surface_area'] / median_area
                rod_metric = stats['elongation'] / relative_area if relative_area > 0 else float('inf')

                if rod_metric > area_elongation_product_threshold:
                    info['rejection_reason'] = f"rod_artifact (metric={rod_metric:.1f})"
                    all_info.append(info)
                    continue

        # Passed all filters
        info['kept'] = True
        kept.append(comp)
        all_info.append(info)

    logger.info(f"Kept {len(kept)}/{len(components)} components after filtering")

    return kept, all_info


def validate_thresholds(
    components: List[trimesh.Trimesh],
    min_faces: int,
    max_elongation: float
) -> None:
    """
    Check if thresholds are pathological (would drop everything).

    Args:
        components: List of components
        min_faces: Minimum face count threshold
        max_elongation: Maximum elongation threshold

    Raises:
        PipelineError if thresholds would drop all components
    """
    if len(components) == 0:
        return

    # Check face count
    face_counts = [len(comp.faces) for comp in components]
    max_faces = max(face_counts)

    if max_faces < min_faces:
        raise PipelineError(
            f"min_faces={min_faces} would drop ALL components. "
            f"Largest component has {max_faces} faces. "
            f"Lower the threshold or check input mesh."
        )

    # Check elongation
    elongations = []
    for comp in components:
        stats = get_mesh_stats(comp)
        elongations.append(stats['elongation'])

    min_elongation = min(elongations)

    if min_elongation > max_elongation:
        raise PipelineError(
            f"max_elongation={max_elongation:.1f} would drop ALL components. "
            f"Least elongated component has {min_elongation:.1f}. "
            f"Raise the threshold or check input mesh."
        )


def write_component_meshes(
    components: List[trimesh.Trimesh],
    component_info: List[Dict],
    output_dir: Path,
    file_format: str = 'ply',
    name_prefix: str = 'sherd'
) -> List[Path]:
    """
    Write kept components to individual mesh files.

    Args:
        components: List of component meshes
        component_info: List of component metadata
        output_dir: Output directory
        file_format: 'ply' or 'obj'
        name_prefix: Prefix for output files

    Returns:
        List of written file paths
    """
    from .mesh_io import save_mesh

    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths = []
    sherd_num = 1

    # Map kept components to their original indices
    kept_indices = [info['component_id'] for info in component_info if info['kept']]

    for comp_idx, comp in zip(kept_indices, components):
        # Filename: sherd_01_comp007.ply
        filename = f"{name_prefix}_{sherd_num:02d}_comp{comp_idx:03d}.{file_format}"
        output_path = output_dir / filename

        save_mesh(comp, output_path, file_format)
        written_paths.append(output_path)

        logger.info(f"Wrote {filename}: {len(comp.vertices)} vertices, {len(comp.faces)} faces")

        sherd_num += 1

    return written_paths
