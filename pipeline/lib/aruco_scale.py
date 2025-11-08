"""
ArUco marker-based scale measurement with user interaction.

Detects markers in images, triangulates 3D positions from COLMAP sparse,
prompts user for real-world distance, and computes scale factor.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .colmap_io import load_cameras_txt, load_images_txt, export_sparse_to_txt
from .pipeline_utils import PipelineError

logger = logging.getLogger(__name__)


def detect_markers_in_images(
    images_dir: Path,
    dictionary_name: str = "DICT_4X4_50"
) -> Dict[str, List[Tuple[int, Tuple[float, float]]]]:
    """
    Detect ArUco markers in all images.

    Args:
        images_dir: Directory containing images
        dictionary_name: ArUco dictionary name

    Returns:
        Dict mapping image_name -> [(marker_id, (u, v))...]
        where (u, v) is pixel center of marker
    """
    logger.info(f"Detecting markers in {images_dir}...")

    # Load ArUco dictionary
    dict_id = getattr(cv2.aruco, dictionary_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    detections = {}
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.JPG"))

    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        corners, ids, _ = detector.detectMarkers(img)

        if ids is not None and len(ids) > 0:
            markers = []
            for i, marker_id in enumerate(ids.flatten()):
                # Compute center of marker
                center = corners[i][0].mean(axis=0)
                markers.append((int(marker_id), (float(center[0]), float(center[1]))))

            detections[img_path.name] = markers

    logger.info(f"Detected markers in {len(detections)}/{len(image_files)} images")

    return detections


def compute_marker_coverage(
    detections: Dict[str, List[Tuple[int, Tuple[float, float]]]]
) -> Dict[int, int]:
    """
    Count how many images each marker appears in.

    Args:
        detections: Output from detect_markers_in_images

    Returns:
        Dict mapping marker_id -> view_count
    """
    coverage = {}

    for image_name, markers in detections.items():
        for marker_id, _ in markers:
            coverage[marker_id] = coverage.get(marker_id, 0) + 1

    return coverage


def triangulate_marker_center(
    marker_id: int,
    detections: Dict[str, List[Tuple[int, Tuple[float, float]]]],
    cameras: Dict[int, np.ndarray],
    images: Dict[str, Tuple[int, np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, int]:
    """
    Triangulate 3D position of marker center from multiple views.

    Args:
        marker_id: Marker ID to triangulate
        detections: Marker detections per image
        cameras: Camera intrinsics (camera_id -> K)
        images: Image poses (name -> (camera_id, R, C))

    Returns:
        Tuple of (X_world, num_views) where X_world is 3x1 position
    """
    rays = []
    origins = []

    for image_name, markers in detections.items():
        # Find this marker in this image
        marker_pixel = None
        for mid, pixel in markers:
            if mid == marker_id:
                marker_pixel = pixel
                break

        if marker_pixel is None:
            continue

        # Get camera pose
        if image_name not in images:
            continue

        camera_id, R, C = images[image_name]
        K = cameras[camera_id]

        # Unproject pixel to ray direction in world coordinates
        u, v = marker_pixel
        pixel_homo = np.array([u, v, 1.0])

        # Ray direction in camera frame
        ray_camera = np.linalg.inv(K) @ pixel_homo
        ray_camera = ray_camera / np.linalg.norm(ray_camera)

        # Transform to world frame
        ray_world = R.T @ ray_camera.reshape(3, 1)

        rays.append(ray_world)
        origins.append(C)

    if len(rays) < 2:
        raise PipelineError(
            f"Marker {marker_id} only visible in {len(rays)} views. "
            f"Need at least 2 for triangulation."
        )

    # Triangulate using linear least squares
    # Minimize ||P * (X - O)||^2 for each ray, where P = I - d*d^T
    A = np.zeros((3, 3))
    b = np.zeros((3, 1))

    for ray_dir, ray_origin in zip(rays, origins):
        # Projection onto plane perpendicular to ray
        d = ray_dir / np.linalg.norm(ray_dir)
        P = np.eye(3) - d @ d.T

        A += P
        b += P @ ray_origin

    # Solve A * X = b
    X_world = np.linalg.lstsq(A, b, rcond=None)[0]

    logger.info(f"Triangulated marker {marker_id} from {len(rays)} views")

    return X_world, len(rays)


def prompt_marker_ids(coverage: Dict[int, int]) -> Tuple[int, int]:
    """
    Interactively prompt user to choose two marker IDs.

    Args:
        coverage: Marker coverage (marker_id -> view_count)

    Returns:
        Tuple of (idA, idB)
    """
    print("\nDetected marker IDs and view counts:")
    sorted_markers = sorted(coverage.items(), key=lambda x: -x[1])

    for marker_id, count in sorted_markers[:12]:  # Show top 12
        print(f"  ID {marker_id:3d}: {count:3d} views")

    print("\nChoose two markers with good coverage for scale measurement.")

    while True:
        try:
            idA = int(input("Enter marker ID A: ").strip())
            idB = int(input("Enter marker ID B: ").strip())

            if idA == idB:
                print("Error: Must choose two different markers.")
                continue

            if idA not in coverage or idB not in coverage:
                print(f"Error: One or both IDs not detected. Available: {list(coverage.keys())}")
                continue

            if coverage[idA] < 2 or coverage[idB] < 2:
                print(f"Error: Both markers need ≥2 views. A:{coverage[idA]}, B:{coverage[idB]}")
                continue

            return idA, idB

        except ValueError:
            print("Error: Please enter valid integer IDs.")
        except (KeyboardInterrupt, EOFError):
            raise PipelineError("User cancelled marker selection")


def prompt_real_distance() -> float:
    """
    Interactively prompt user for real-world distance.

    Returns:
        Distance in metres
    """
    print("\nEnter the real-world distance between the marker centers.")
    print("This should be the center-to-center distance measured on your hardware.")

    while True:
        try:
            value_str = input("Distance value: ").strip()
            value = float(value_str)

            if value <= 0:
                print("Error: Distance must be positive.")
                continue

            unit = input("Units (mm or m): ").strip().lower()

            if unit not in ['mm', 'm']:
                print("Error: Units must be 'mm' or 'm'.")
                continue

            # Convert to metres
            distance_m = value / 1000.0 if unit == 'mm' else value

            print(f"\nInterpreted as: {distance_m:.6f} m")
            if unit == 'mm':
                print(f"  (from {value:.2f} mm)")

            confirm = input("Is this correct? [y/N]: ").strip().lower()
            if confirm == 'y':
                return distance_m
            else:
                print("Let's try again...")

        except ValueError:
            print("Error: Please enter a valid number.")
        except (KeyboardInterrupt, EOFError):
            raise PipelineError("User cancelled distance input")


def compute_and_validate_scale(
    distance_reconstructed: float,
    distance_real_m: float,
    sanity_min: float = 0.01,
    sanity_max: float = 100.0
) -> float:
    """
    Compute scale factor and validate it's reasonable.

    Args:
        distance_reconstructed: Distance in reconstruction units
        distance_real_m: Real-world distance in metres
        sanity_min: Minimum plausible scale
        sanity_max: Maximum plausible scale

    Returns:
        Scale factor

    Raises:
        PipelineError if scale is implausible
    """
    if distance_reconstructed <= 0:
        raise PipelineError("Reconstructed distance is zero or negative")

    scale = distance_real_m / distance_reconstructed

    logger.info(f"Reconstructed distance: {distance_reconstructed:.6f} units")
    logger.info(f"Real-world distance:    {distance_real_m:.6f} m")
    logger.info(f"Scale factor:           {scale:.9f}")

    if not (sanity_min <= scale <= sanity_max):
        raise PipelineError(
            f"Scale factor {scale:.6f} is outside sane bounds [{sanity_min}, {sanity_max}].\n"
            f"This likely indicates:\n"
            f"  - Wrong marker IDs selected\n"
            f"  - Incorrect distance or units\n"
            f"  - Reconstruction failure\n"
            f"Check your inputs and try again."
        )

    # Warn if scale seems unusual (but don't error)
    if scale < 0.1 or scale > 10.0:
        logger.warning(f"Scale factor {scale:.6f} is unusual. Typical range: [0.1, 10.0]")
        logger.warning("Please verify your marker IDs and distance measurement.")

    return scale


def prompt_confirmation(
    marker_idA: int,
    marker_idB: int,
    distance_real_m: float,
    distance_reconstructed: float,
    scale_factor: float
) -> bool:
    """
    Show summary and ask user to confirm scale application.

    Returns:
        True if user confirms, False if user cancels
    """
    print("\n" + "=" * 70)
    print("SCALE COMPUTATION SUMMARY")
    print("=" * 70)
    print(f"Marker IDs:              {marker_idA}, {marker_idB}")
    print(f"Real-world distance:     {distance_real_m:.6f} m ({distance_real_m*1000:.2f} mm)")
    print(f"Reconstructed distance:  {distance_reconstructed:.6f} units")
    print(f"Scale factor:            {scale_factor:.9f}")
    print("=" * 70)

    while True:
        response = input("\nApply this scale to the sparse model? [y/N]: ").strip().lower()

        if response == 'y':
            return True
        elif response == 'n' or response == '':
            return False
        else:
            print("Please enter 'y' or 'n'.")


def write_aruco_scale_log(
    scale_dir: Path,
    marker_idA: int,
    marker_idB: int,
    distance_real_m: float,
    distance_reconstructed: float,
    scale_factor: float,
    num_views_A: int,
    num_views_B: int
) -> None:
    """
    Write provenance log for ArUco scale measurement.

    Args:
        scale_dir: Directory to write logs
        marker_idA, marker_idB: Chosen marker IDs
        distance_real_m: Real-world distance in metres
        distance_reconstructed: Reconstructed distance
        scale_factor: Computed scale
        num_views_A, num_views_B: Number of views for each marker
    """
    from datetime import datetime

    # Write manifest
    manifest_path = scale_dir / "MANIFEST.txt"
    with manifest_path.open("w") as f:
        f.write(f"method=aruco_scale\n")
        f.write(f"time={datetime.now().isoformat()}\n")
        f.write(f"ids={marker_idA},{marker_idB}\n")
        f.write(f"real_m={distance_real_m:.9f}\n")
        f.write(f"d_rec={distance_reconstructed:.9f}\n")
        f.write(f"scale={scale_factor:.9f}\n")
        f.write(f"views_A={num_views_A}\n")
        f.write(f"views_B={num_views_B}\n")

    # Write scale factor
    scale_file = scale_dir / "SCALE.txt"
    with scale_file.open("w") as f:
        f.write(f"{scale_factor:.9f}\n")

    # Write detailed log
    log_path = scale_dir / "aruco_scale_log.txt"
    with log_path.open("w") as f:
        f.write("=" * 70 + "\n")
        f.write("ARUCO SCALE MEASUREMENT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Application time: {datetime.now().isoformat()}\n\n")

        f.write(f"Marker A: ID {marker_idA}\n")
        f.write(f"  Views used for triangulation: {num_views_A}\n\n")

        f.write(f"Marker B: ID {marker_idB}\n")
        f.write(f"  Views used for triangulation: {num_views_B}\n\n")

        f.write(f"Distance (real-world):    {distance_real_m:.6f} m ({distance_real_m*1000:.2f} mm)\n")
        f.write(f"Distance (reconstructed): {distance_reconstructed:.6f} units\n\n")

        f.write(f"Scale factor: {scale_factor:.9f}\n\n")

        f.write("This scale factor transforms the sparse reconstruction\n")
        f.write("to real-world metric units (metres) based on the known\n")
        f.write("distance between two ArUco markers.\n\n")

        f.write("=" * 70 + "\n")

    logger.info(f"✓ ArUco scale logs written to {scale_dir}")
