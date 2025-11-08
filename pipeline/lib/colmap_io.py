"""
COLMAP sparse model I/O for TXT format.

Reads cameras.txt, images.txt, and points3D.txt to access camera parameters,
poses, and reconstructed points for triangulation.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        qw, qx, qy, qz: Quaternion components (qw is scalar part)

    Returns:
        3x3 rotation matrix
    """
    # Normalize quaternion
    q = np.array([qw, qx, qy, qz])
    q = q / np.linalg.norm(q)

    w, x, y, z = q

    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=float)

    return R


def load_cameras_txt(cameras_file: Path) -> Dict[int, np.ndarray]:
    """
    Load camera intrinsics from cameras.txt.

    Args:
        cameras_file: Path to cameras.txt

    Returns:
        Dict mapping camera_id -> 3x3 intrinsic matrix K
    """
    cameras = {}

    with cameras_file.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            # Build intrinsic matrix K
            if model == "PINHOLE":
                # PINHOLE: fx, fy, cx, cy
                fx, fy, cx, cy = params[:4]
            elif model == "SIMPLE_PINHOLE":
                # SIMPLE_PINHOLE: f, cx, cy
                f, cx, cy = params[:3]
                fx = fy = f
            elif model == "RADIAL":
                # RADIAL: f, cx, cy, k1, k2
                f, cx, cy = params[:3]
                fx = fy = f
            else:
                # Default: assume first param is focal length
                fx = fy = params[0]
                cx = params[1] if len(params) > 1 else width / 2
                cy = params[2] if len(params) > 2 else height / 2

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=float)

            cameras[camera_id] = K

    return cameras


def load_images_txt(images_file: Path) -> Dict[str, Tuple[int, np.ndarray, np.ndarray]]:
    """
    Load image poses from images.txt.

    Args:
        images_file: Path to images.txt

    Returns:
        Dict mapping image_name -> (camera_id, R, C)
        where R is 3x3 rotation matrix and C is 3x1 camera center
    """
    images = {}

    with images_file.open() as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    # images.txt has pairs of lines: image info, then points2D
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue

        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]

        # COLMAP stores world-to-camera transform
        # R rotates world to camera, t is camera position in camera frame
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        t = np.array([[tx], [ty], [tz]], dtype=float)

        # Camera center in world coordinates: C = -R^T * t
        C = -R.T @ t

        images[name] = (camera_id, R, C)

    return images


def export_sparse_to_txt(
    sparse_bin_path: Path,
    sparse_txt_path: Path,
    colmap_exec: str = "colmap"
) -> None:
    """
    Export COLMAP sparse model from binary to TXT format.

    Args:
        sparse_bin_path: Path to sparse/0 (binary format)
        sparse_txt_path: Output path for TXT format
        colmap_exec: COLMAP executable
    """
    import subprocess

    sparse_txt_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        colmap_exec,
        "model_converter",
        "--input_path", str(sparse_bin_path),
        "--output_path", str(sparse_txt_path),
        "--output_type", "TXT"
    ]

    subprocess.run(cmd, check=True, capture_output=True)
