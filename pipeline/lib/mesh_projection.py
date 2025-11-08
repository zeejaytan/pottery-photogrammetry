#!/usr/bin/env python3
"""
3D mesh to 2D mask projection engine.

Projects a 3D mesh into camera views and rasterizes binary masks.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import trimesh


def qvec_to_rotation_matrix(qvec: np.ndarray) -> np.ndarray:
    """
    Convert COLMAP quaternion to 3x3 rotation matrix.

    Args:
        qvec: Quaternion [qw, qx, qy, qz]

    Returns:
        3x3 rotation matrix
    """
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def parse_camera_params(camera: Dict) -> Tuple[float, float, float, float]:
    """
    Extract fx, fy, cx, cy from COLMAP camera.

    Args:
        camera: Camera dict with 'model' and 'params'

    Returns:
        (fx, fy, cx, cy)
    """
    model = camera["model"]
    params = camera["params"]

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        return f, f, cx, cy
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        return fx, fy, cx, cy
    elif model in ["SIMPLE_RADIAL", "RADIAL"]:
        f, cx, cy = params[:3]
        return f, f, cx, cy
    elif model in ["OPENCV", "OPENCV_FISHEYE"]:
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy
    elif model == "SIMPLE_RADIAL_FISHEYE":
        f, cx, cy = params[:3]
        return f, f, cx, cy
    elif model == "RADIAL_FISHEYE":
        f, cx, cy = params[:3]
        return f, f, cx, cy
    else:
        raise ValueError(f"Unsupported camera model: {model}")


def project_mesh_to_mask(
    mesh: trimesh.Trimesh,
    camera: Dict,
    qvec: np.ndarray,
    tvec: np.ndarray,
    image_width: int,
    image_height: int,
    pad_pixels: int = 2,
    erosion: int = 0
) -> np.ndarray:
    """
    Project 3D mesh into camera view and create binary mask.

    Args:
        mesh: Trimesh object in world coordinates
        camera: COLMAP camera dict with 'model', 'params', 'width', 'height'
        qvec: Quaternion rotation (world-to-camera) [qw, qx, qy, qz]
        tvec: Translation vector (world-to-camera) [tx, ty, tz]
        image_width: Output mask width
        image_height: Output mask height
        pad_pixels: Dilation to protect thin edges (default: 2)
        erosion: Erosion iterations to prevent halos (default: 0)

    Returns:
        Binary mask (uint8, 255=keep, 0=discard)
    """
    # 1. Convert quaternion to rotation matrix
    R = qvec_to_rotation_matrix(qvec)

    # 2. Transform mesh vertices to camera frame
    vertices_world = mesh.vertices
    # Apply rotation and translation: v_cam = R @ v_world + t
    vertices_cam = (R @ vertices_world.T).T + tvec

    # 3. Filter faces with all vertices in front of camera (Z > 0)
    valid_faces = []
    for face in mesh.faces:
        v0, v1, v2 = vertices_cam[face]
        # All three vertices must have positive Z
        if v0[2] > 0 and v1[2] > 0 and v2[2] > 0:
            valid_faces.append(face)

    if len(valid_faces) == 0:
        # Mesh not visible in this view - return empty mask
        return np.zeros((image_height, image_width), dtype=np.uint8)

    # 4. Get camera intrinsics
    fx, fy, cx, cy = parse_camera_params(camera)

    # 5. Create empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # 6. Rasterize each visible face
    for face in valid_faces:
        # Get triangle vertices in camera frame
        v0, v1, v2 = vertices_cam[mesh.faces[face]]

        # Project to image plane
        triangle_2d = []
        for v in [v0, v1, v2]:
            x = (fx * v[0] / v[2]) + cx
            y = (fy * v[1] / v[2]) + cy
            triangle_2d.append([int(round(x)), int(round(y))])

        triangle_2d = np.array(triangle_2d, dtype=np.int32)

        # Fill triangle
        cv2.fillConvexPoly(mask, triangle_2d, 255)

    # 7. Apply padding (dilation) to protect thin edges
    if pad_pixels > 0:
        kernel_size = pad_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # 8. Apply erosion (if halos detected)
    if erosion > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=erosion)

    return mask


def project_mesh_to_mask_batch(
    mesh: trimesh.Trimesh,
    cameras: Dict,
    images: Dict,
    pad_pixels: int = 2,
    erosion: int = 0,
    progress_callback=None
) -> Dict[str, np.ndarray]:
    """
    Project mesh to masks for all images in batch.

    Args:
        mesh: Trimesh object in world coordinates
        cameras: Dict of camera_id -> camera dict
        images: Dict of image_id -> image dict
        pad_pixels: Dilation for thin edges
        erosion: Erosion iterations for halos
        progress_callback: Optional callback(current, total, image_name)

    Returns:
        Dict mapping image_name -> mask (np.ndarray)
    """
    masks = {}
    total = len(images)

    for idx, (image_id, image_info) in enumerate(images.items(), 1):
        image_name = image_info["name"]
        camera_id = image_info["camera_id"]
        camera = cameras[camera_id]

        qvec = image_info["qvec"]
        tvec = image_info["tvec"]

        mask = project_mesh_to_mask(
            mesh=mesh,
            camera=camera,
            qvec=qvec,
            tvec=tvec,
            image_width=camera["width"],
            image_height=camera["height"],
            pad_pixels=pad_pixels,
            erosion=erosion
        )

        masks[image_name] = mask

        if progress_callback:
            progress_callback(idx, total, image_name)

    return masks
