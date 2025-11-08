# Implementation Plan: Coded-Target Scale with Explicit User Input

## Overview

This plan implements a coded-target based scale measurement system that:
1. **Detects ArUco markers** in images after COLMAP reconstruction
2. **Triangulates 3D centers** of marker pairs from sparse model
3. **Prompts user** to enter real-world distance between chosen markers
4. **Validates and applies** scale factor to sparse model
5. **Regenerates dense workspace** for metric OpenMVS outputs
6. **Records full provenance** for thesis/lab documentation

This complements the existing coded targets and manual scale workflows by providing an interactive, semi-automated approach with explicit human validation.

---

## Architecture

### Files to Create/Modify

```
pipeline/
├── bin/
│   ├── scale_aruco.py                   # Interactive scale CLI (NEW)
│   └── run_colmap.py                    # Add --scale-aruco flag (MODIFY)
├── lib/
│   ├── aruco_scale.py                   # Triangulation & interactive prompts (NEW)
│   └── colmap_io.py                     # COLMAP TXT format readers (NEW)
├── config/
│   └── pipeline_config.yaml             # Add aruco_scale section (MODIFY)
└── docs/
    └── ARUCO_SCALE_USAGE.md             # User guide (NEW)
```

### Integration Points

1. **After COLMAP Stage 1** (mapper completes)
2. **Before dense reconstruction** (undistortion)
3. **Parallel to** manual scale and coded target alignment
4. **Preference order**: aruco_scale → manual_scale → coded_targets → unscaled

---

## Implementation Details

### 1. COLMAP I/O Library (`lib/colmap_io.py`)

**Purpose**: Read COLMAP sparse models from TXT format for triangulation

```python
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
```

---

### 2. ArUco Scale Library (`lib/aruco_scale.py`)

**Purpose**: Marker detection, triangulation, interactive prompts, scale computation

```python
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
```

---

### 3. CLI Tool (`bin/scale_aruco.py`)

**Purpose**: Interactive/non-interactive CLI for ArUco scale measurement

```python
#!/usr/bin/env python3
"""
ArUco marker-based scale measurement for COLMAP reconstructions.

Detects ArUco markers, triangulates 3D positions, prompts for real-world
distance, and applies scale to sparse model.

Usage:
    # Interactive mode (prompts for IDs and distance)
    scale_aruco.py --images images_jpg --sparse sparse/0

    # Non-interactive mode (fully scripted)
    scale_aruco.py --images images_jpg --sparse sparse/0 \\
        --idA 11 --idB 23 --real-mm 100.0

    # With work directory
    scale_aruco.py --work work_colmap_openmvs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.aruco_scale import (
    detect_markers_in_images,
    compute_marker_coverage,
    triangulate_marker_center,
    prompt_marker_ids,
    prompt_real_distance,
    compute_and_validate_scale,
    prompt_confirmation,
    write_aruco_scale_log
)
from lib.colmap_io import (
    load_cameras_txt,
    load_images_txt,
    export_sparse_to_txt
)
from lib.manual_scale import (
    apply_scale_to_sparse,
    regenerate_dense_workspace,
    append_to_runlog
)
from lib.pipeline_utils import PipelineError, PipelineContext


def main():
    parser = argparse.ArgumentParser(
        description="ArUco marker-based scale measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--work",
        type=Path,
        help="Work directory (auto-detects images and sparse)"
    )

    parser.add_argument(
        "--images",
        type=Path,
        help="Images directory"
    )

    parser.add_argument(
        "--sparse",
        type=Path,
        help="Sparse model directory (e.g., sparse/0)"
    )

    parser.add_argument(
        "--idA",
        type=int,
        help="Marker ID A (non-interactive mode)"
    )

    parser.add_argument(
        "--idB",
        type=int,
        help="Marker ID B (non-interactive mode)"
    )

    parser.add_argument(
        "--real-mm",
        type=float,
        help="Real-world distance in millimetres (non-interactive)"
    )

    parser.add_argument(
        "--real-m",
        type=float,
        help="Real-world distance in metres (non-interactive)"
    )

    parser.add_argument(
        "--dictionary",
        default="DICT_4X4_50",
        help="ArUco dictionary (default: DICT_4X4_50)"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline/config/pipeline_config.yaml"),
        help="Pipeline configuration"
    )

    parser.add_argument(
        "--colmap",
        default="colmap",
        help="COLMAP executable"
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Require all parameters (no prompts)"
    )

    args = parser.parse_args()

    # Determine paths
    if args.work:
        work_dir = args.work
        images_dir = args.images or work_dir.parent
        sparse_dir = args.sparse or work_dir / "sparse" / "0"
    else:
        if not args.images or not args.sparse:
            print("ERROR: Must provide --work OR both --images and --sparse", file=sys.stderr)
            return 1
        images_dir = args.images
        sparse_dir = args.sparse
        work_dir = sparse_dir.parent.parent  # Assume sparse/0 structure

    # Validate paths
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}", file=sys.stderr)
        return 1

    if not sparse_dir.exists():
        print(f"ERROR: Sparse model not found: {sparse_dir}", file=sys.stderr)
        print("Run COLMAP Stage 1 (mapper) first.", file=sys.stderr)
        return 1

    try:
        print("=" * 70)
        print("ARUCO MARKER-BASED SCALE MEASUREMENT")
        print("=" * 70)
        print(f"Images:  {images_dir}")
        print(f"Sparse:  {sparse_dir}")
        print()

        # Step 1: Export sparse to TXT format
        print("Step 1: Exporting sparse model to TXT format...")
        sparse_txt_dir = work_dir / "sparse_txt"
        export_sparse_to_txt(sparse_dir, sparse_txt_dir, args.colmap)
        print(f"✓ Exported to {sparse_txt_dir}")
        print()

        # Step 2: Load camera parameters and poses
        print("Step 2: Loading camera parameters and poses...")
        cameras = load_cameras_txt(sparse_txt_dir / "cameras.txt")
        images = load_images_txt(sparse_txt_dir / "images.txt")
        print(f"✓ Loaded {len(cameras)} cameras, {len(images)} images")
        print()

        # Step 3: Detect markers
        print("Step 3: Detecting ArUco markers...")
        detections = detect_markers_in_images(images_dir, args.dictionary)

        if not detections:
            print("\nERROR: No markers detected in any images.", file=sys.stderr)
            print("Possible issues:", file=sys.stderr)
            print("  - No markers visible in images", file=sys.stderr)
            print("  - Wrong dictionary specified", file=sys.stderr)
            print("  - Markers too small or blurry", file=sys.stderr)
            print("\nYou can fall back to manual PLY measurement instead.", file=sys.stderr)
            return 2

        coverage = compute_marker_coverage(detections)

        # Save detection info
        scale_dir = work_dir / "scale"
        scale_dir.mkdir(exist_ok=True, parents=True)

        import json
        with (scale_dir / "detected_ids.json").open("w") as f:
            json.dump(coverage, f, indent=2)

        print(f"✓ Detected {len(coverage)} unique markers")
        print()

        # Step 4: Choose markers (interactive or from args)
        if args.idA is not None and args.idB is not None:
            marker_idA = args.idA
            marker_idB = args.idB
            print(f"Using markers: {marker_idA}, {marker_idB} (from arguments)")
        else:
            if args.non_interactive:
                print("ERROR: --non-interactive requires --idA and --idB", file=sys.stderr)
                return 1
            marker_idA, marker_idB = prompt_marker_ids(coverage)

        print()

        # Step 5: Triangulate marker centers
        print(f"Step 4: Triangulating marker centers...")
        XA, views_A = triangulate_marker_center(marker_idA, detections, cameras, images)
        XB, views_B = triangulate_marker_center(marker_idB, detections, cameras, images)

        distance_reconstructed = float(np.linalg.norm(XA - XB))
        print(f"✓ Marker {marker_idA}: triangulated from {views_A} views")
        print(f"✓ Marker {marker_idB}: triangulated from {views_B} views")
        print(f"✓ Reconstructed distance: {distance_reconstructed:.6f} units")
        print()

        # Step 6: Get real-world distance
        if args.real_mm is not None:
            distance_real_m = args.real_mm / 1000.0
            print(f"Using distance: {args.real_mm:.2f} mm = {distance_real_m:.6f} m (from arguments)")
        elif args.real_m is not None:
            distance_real_m = args.real_m
            print(f"Using distance: {distance_real_m:.6f} m (from arguments)")
        else:
            if args.non_interactive:
                print("ERROR: --non-interactive requires --real-mm or --real-m", file=sys.stderr)
                return 1
            distance_real_m = prompt_real_distance()

        print()

        # Step 7: Compute scale
        print("Step 5: Computing scale factor...")
        import numpy as np
        scale_factor = compute_and_validate_scale(
            distance_reconstructed,
            distance_real_m
        )
        print()

        # Step 8: Confirm (if interactive)
        if not args.non_interactive:
            if not prompt_confirmation(
                marker_idA, marker_idB,
                distance_real_m, distance_reconstructed,
                scale_factor
            ):
                print("\nScale measurement cancelled by user.")
                return 0
            print()

        # Step 9: Write logs
        print("Step 6: Writing provenance logs...")
        write_aruco_scale_log(
            scale_dir,
            marker_idA, marker_idB,
            distance_real_m, distance_reconstructed,
            scale_factor,
            views_A, views_B
        )
        print()

        # Step 10: Apply scale to sparse
        print("Step 7: Applying scale to sparse model...")
        sparse_scaled = work_dir / "sparse_scaled" / "0"
        apply_scale_to_sparse(
            sparse_dir,
            sparse_scaled,
            scale_factor,
            args.colmap
        )
        print()

        # Step 11: Regenerate dense workspace
        print("Step 8: Regenerating dense workspace...")
        dense_scaled = work_dir / "dense_scaled"

        # Load config for max_image_size
        context = PipelineContext.from_config_path(args.config)
        max_image_size = context.config.get("colmap", {}).get("image_undistorter", {}).get("max_image_size", 6000)

        regenerate_dense_workspace(
            images_dir,
            sparse_scaled,
            dense_scaled,
            max_image_size,
            args.colmap
        )
        print()

        # Step 12: Update runlog
        print("Step 9: Updating pipeline runlog...")
        validation_info = {
            'scale1': scale_factor,
            'scale2': scale_factor,
            'diff_pct': 0.0,
            'd1_real_m': distance_real_m,
            'd1_rec_units': distance_reconstructed,
            'd2_real_m': distance_real_m,
            'd2_rec_units': distance_reconstructed
        }
        append_to_runlog(work_dir, scale_factor, validation_info)
        print()

        print("=" * 70)
        print("ARUCO SCALE MEASUREMENT COMPLETE")
        print("=" * 70)
        print(f"Scaled sparse: {sparse_scaled}")
        print(f"Scaled dense:  {dense_scaled}")
        print(f"Scale factor:  {scale_factor:.9f}")
        print()
        print("Next steps:")
        print("  1. Verify scale in CloudCompare (measure marker distance in dense)")
        print("  2. Continue to OpenMVS using dense_scaled/")
        print(f"     InterfaceCOLMAP -i {dense_scaled} -o scene.mvs")
        print()

        return 0

    except PipelineError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

### 4. Integration with `run_colmap.py`

Add optional `--scale-aruco` flag to run after mapper:

```python
# In parse_args()
parser.add_argument(
    "--scale-aruco",
    action="store_true",
    help="Run ArUco marker-based scale measurement after mapper (interactive).",
)

# After mapper completes (after Step 3.5 sparse export)
if args.scale_aruco:
    logger.info("Running ArUco scale measurement...")
    try:
        scale_aruco_cmd = [
            sys.executable,
            str(Path(__file__).parent / "scale_aruco.py"),
            "--work", str(work_dir),
            "--config", str(args.config),
            "--colmap", colmap_exec
        ]
        run_command(scale_aruco_cmd, logger=logger)
        logger.info("✓ ArUco scale measurement complete")
    except Exception as e:
        logger.warning(f"ArUco scale measurement failed: {e}")
        logger.warning("Continuing without scale")
```

---

### 5. Configuration (`pipeline_config.yaml`)

```yaml
aruco_scale:
  enabled: false                    # Enable ArUco scale workflow
  dictionary: DICT_4X4_50           # ArUco dictionary
  auto_run: false                   # Auto-run after mapper (requires --idA/--idB/--real-m in env)
  sanity_bounds:
    min_scale: 0.01
    max_scale: 100.0
  prefer_aruco_scale: true          # Prefer ArUco scale over manual/coded targets
```

---

### 6. Documentation (`docs/ARUCO_SCALE_USAGE.md`)

Comprehensive guide covering:
- When to use ArUco scale vs manual vs coded targets
- Marker placement guidelines
- Interactive workflow walkthrough
- Non-interactive (scripted) usage
- Troubleshooting marker detection
- Validation and provenance

---

## Success Criteria

1. **Detection**: Detect ≥2 markers with ≥2 views each
2. **Triangulation**: Accurate 3D positions from COLMAP sparse
3. **User interaction**: Clear prompts, validation, confirmation
4. **Scale accuracy**: Reproduced distance within ±0.5% of input
5. **Provenance**: Full logs with IDs, distance, scale, timestamps
6. **Integration**: Seamless with existing pipeline workflows
7. **Non-interactive**: Works fully scripted for batch processing

---

## Workflow Comparison

| Method | Setup | Accuracy | User Effort | Automation | Use Case |
|--------|-------|----------|-------------|------------|----------|
| **Manual Scale** | None | ±0.5% | 4 min | Semi | Retrospective |
| **Coded Targets** | Print board | ±2mm | Minimal | Full | Production |
| **ArUco Scale** | 2 markers | ±0.5% | 2 min | Semi | Flexible |

**ArUco Scale advantages**:
- No PLY export/measurement on laptop
- More accurate than manual (automatic triangulation)
- More flexible than coded targets (just 2 markers)
- Explicit user confirmation (better than fully automatic)
- Works with existing marker placement

---

## Next Steps

1. Implement `lib/colmap_io.py` for TXT reading
2. Implement `lib/aruco_scale.py` for triangulation and interaction
3. Create `bin/scale_aruco.py` CLI tool
4. Integrate with `run_colmap.py`
5. Add configuration section
6. Write documentation
7. Test on sample tree with known marker distance
8. Validate accuracy ±0.5%
