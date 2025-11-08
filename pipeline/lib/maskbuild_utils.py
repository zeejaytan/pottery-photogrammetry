#!/usr/bin/env python3
"""
Shared utilities for maskbuild system.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from scipy.spatial.distance import pdist


def list_images(image_dir: Path, extensions: List[str] = None) -> List[Path]:
    """
    List all image files in a directory.

    Args:
        image_dir: Directory containing images
        extensions: List of valid extensions (default: ['.jpg', '.jpeg', '.JPG', '.JPEG'])

    Returns:
        List of image paths, sorted
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']

    images = []
    for ext in extensions:
        images.extend(image_dir.glob(f'*{ext}'))

    return sorted(images)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with file_path.open('rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_bundle_hash(sparse_dir: Path) -> str:
    """
    Compute combined hash of COLMAP sparse reconstruction.

    Hashes cameras.bin, images.bin, points3D.bin together.
    """
    sha256 = hashlib.sha256()

    for filename in ['cameras.bin', 'images.bin', 'points3D.bin']:
        file_path = sparse_dir / filename
        if file_path.exists():
            with file_path.open('rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)

    return sha256.hexdigest()


def validate_mesh_frame(
    mesh: trimesh.Trimesh,
    sparse_dir: Path,
    original_mesh_path: Path,
    bbox_tolerance: float = 0.1,
    distance_tolerance: float = 0.05
) -> Dict:
    """
    Validate that edited mesh is in same coordinate frame as original.

    Checks:
    1. Bounding box overlap (>90%)
    2. Reference distances match within 5%
    3. Vertex count reasonable (10% - 150% of original)

    Args:
        mesh: Edited mesh to validate
        sparse_dir: COLMAP sparse directory (for reference)
        original_mesh_path: Path to original coarse model
        bbox_tolerance: Allowed bbox shrinkage (default: 0.1 = 10%)
        distance_tolerance: Allowed distance error (default: 0.05 = 5%)

    Returns:
        {
            "is_valid": bool,
            "message": str,
            "error": str or None,
            "checks": {...}
        }
    """
    checks = {}

    # Load original mesh
    if not original_mesh_path.exists():
        return {
            "is_valid": False,
            "message": "Original mesh not found",
            "error": f"Cannot find {original_mesh_path}",
            "checks": {}
        }

    original = trimesh.load(original_mesh_path)

    # Check 1: Bounding box overlap
    bbox_edited = mesh.bounds
    bbox_original = original.bounds

    # Compute intersection
    bbox_intersection_min = np.maximum(bbox_edited[0], bbox_original[0])
    bbox_intersection_max = np.minimum(bbox_edited[1], bbox_original[1])
    bbox_intersection = bbox_intersection_max - bbox_intersection_min

    # Volume of intersection
    bbox_intersection_volume = np.prod(np.maximum(0, bbox_intersection))

    # Volume of original
    bbox_original_size = bbox_original[1] - bbox_original[0]
    bbox_original_volume = np.prod(bbox_original_size)

    bbox_overlap = bbox_intersection_volume / bbox_original_volume if bbox_original_volume > 0 else 0

    checks["bbox_overlap"] = float(bbox_overlap)
    checks["bbox_threshold"] = 1.0 - bbox_tolerance
    checks["bbox_pass"] = bbox_overlap >= (1.0 - bbox_tolerance)

    # Check 2: Reference distances
    # Sample 10 random points from original, find distances
    np.random.seed(42)
    n_samples = min(10, len(original.vertices))
    sample_indices = np.random.choice(len(original.vertices), n_samples, replace=False)
    sample_points = original.vertices[sample_indices]

    # Compute pairwise distances in original
    if len(sample_points) > 1:
        original_distances = pdist(sample_points)

        # Find closest points in edited mesh
        edited_sample = []
        for pt in sample_points:
            distances_to_pt = np.linalg.norm(mesh.vertices - pt, axis=1)
            closest_idx = np.argmin(distances_to_pt)
            edited_sample.append(mesh.vertices[closest_idx])

        edited_sample = np.array(edited_sample)
        edited_distances = pdist(edited_sample)

        # Compute relative error
        distance_errors = np.abs(edited_distances - original_distances) / (original_distances + 1e-6)
        max_distance_error = float(distance_errors.max())

        checks["max_distance_error"] = max_distance_error
        checks["distance_threshold"] = distance_tolerance
        checks["distance_pass"] = max_distance_error <= distance_tolerance
    else:
        # Not enough points to check
        checks["max_distance_error"] = 0.0
        checks["distance_threshold"] = distance_tolerance
        checks["distance_pass"] = True

    # Check 3: Vertex count
    vertex_ratio = len(mesh.vertices) / len(original.vertices) if len(original.vertices) > 0 else 0
    checks["vertex_ratio"] = float(vertex_ratio)
    checks["vertex_count_edited"] = len(mesh.vertices)
    checks["vertex_count_original"] = len(original.vertices)
    checks["vertex_pass"] = 0.1 <= vertex_ratio <= 1.5

    # Overall validation
    all_pass = checks["bbox_pass"] and checks["distance_pass"] and checks["vertex_pass"]

    if all_pass:
        return {
            "is_valid": True,
            "message": f"Frame validation passed (bbox_overlap={bbox_overlap:.2f}, max_dist_error={checks['max_distance_error']:.3f})",
            "error": None,
            "checks": checks
        }
    else:
        errors = []
        if not checks["bbox_pass"]:
            errors.append(f"Bounding box overlap {bbox_overlap:.2f} < {1.0-bbox_tolerance:.2f}")
        if not checks["distance_pass"]:
            errors.append(f"Distance error {checks['max_distance_error']:.3f} > {distance_tolerance:.3f}")
        if not checks["vertex_pass"]:
            errors.append(f"Vertex ratio {vertex_ratio:.2f} outside [0.1, 1.5]")

        return {
            "is_valid": False,
            "message": "Frame validation failed",
            "error": "; ".join(errors),
            "checks": checks
        }


# =============================================================================
# COLMAP Binary Format Readers
# =============================================================================
# Based on COLMAP's scripts/python/read_write_model.py
# Minimal implementation for reading cameras.bin and images.bin

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.

    Args:
        fid: File handle
        num_bytes: Number of bytes to read
        format_char_sequence: Format characters for struct.unpack
        endian_character: '<' for little-endian, '>' for big-endian
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file: Path) -> Dict:
    """
    Read COLMAP cameras.bin file.

    Returns:
        Dict mapping camera_id to camera dict with:
        {
            'id': int,
            'model': str,
            'width': int,
            'height': int,
            'params': np.ndarray
        }
    """
    cameras = {}
    with path_to_model_file.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8*num_params, format_char_sequence="d"*num_params
            )
            cameras[camera_id] = {
                'id': camera_id,
                'model': CAMERA_MODEL_IDS[model_id].model_name,
                'width': width,
                'height': height,
                'params': np.array(params)
            }
    return cameras


def read_images_binary(path_to_model_file: Path) -> Dict:
    """
    Read COLMAP images.bin file.

    Returns:
        Dict mapping image_id to image dict with:
        {
            'id': int,
            'qvec': np.ndarray (4,),  # Quaternion
            'tvec': np.ndarray (3,),  # Translation
            'camera_id': int,
            'name': str,
            'xys': np.ndarray (N, 2),  # 2D points
            'point3D_ids': np.ndarray (N,)  # Corresponding 3D point IDs
        }
    """
    images = {}
    with path_to_model_file.open("rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24*num_points2D,
                format_char_sequence="ddq"*num_points2D
            )
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))

            images[image_id] = {
                'id': image_id,
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': image_name,
                'xys': xys,
                'point3D_ids': point3D_ids
            }
    return images


# Camera model definitions
from collections import namedtuple

CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])

CAMERA_MODELS = [
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
]

CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {model.model_name: model for model in CAMERA_MODELS}
