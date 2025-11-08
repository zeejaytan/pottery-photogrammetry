"""
Coded target (ArUco/ChArUco) detection engine.

This module provides functions for detecting ArUco and ChArUco markers in images,
refining corner positions to sub-pixel accuracy, and estimating board poses.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .target_utils import (
    load_config,
    get_aruco_dictionary,
    create_aruco_detector_parameters,
    create_charuco_board,
    list_images,
    configure_logging
)
from .pipeline_utils import PipelineError

logger = logging.getLogger(__name__)


def detect_markers_in_images(
    images_dir: Path,
    work_dir: Path,
    config_path: Path,
    calibration_path: Optional[Path] = None
) -> Dict:
    """
    Detect coded targets in all images.

    Args:
        images_dir: Directory containing images
        work_dir: Work directory
        config_path: Pipeline config
        calibration_path: Optional camera calibration file

    Returns:
        Dict mapping image_name -> detection info:
        {
            "IMG_0001.jpg": {
                "marker_ids": [0, 1, 2],
                "corners": [...],  # Sub-pixel corners
                "board_pose": {    # If calibration available
                    "rvec": [...],
                    "tvec": [...]
                }
            }
        }
    """
    config = load_config(config_path)
    target_cfg = config.get("coded_targets", {})

    # Setup workspace
    targets_dir = work_dir / "coded_targets"
    targets_dir.mkdir(exist_ok=True)
    detections_dir = targets_dir / "detections"
    detections_dir.mkdir(exist_ok=True)

    # Setup logging
    log_file = targets_dir / "detection_log.txt"
    file_logger = configure_logging(log_file, console=True)
    file_logger.info("=" * 70)
    file_logger.info("CODED TARGET DETECTION")
    file_logger.info("=" * 70)

    # Load board configuration
    board_type = target_cfg.get("board_type", "aruco")  # "aruco" or "charuco"
    dictionary_name = target_cfg.get("dictionary", "DICT_4X4_50")
    square_size = target_cfg.get("square_size_mm", 50.0)  # Real-world size

    # Initialize ArUco dictionary
    aruco_dict = get_aruco_dictionary(dictionary_name)
    detector_params = create_aruco_detector_parameters()

    # For ChArUco boards
    charuco_board = None
    if board_type == "charuco":
        board_width = target_cfg.get("board_width", 5)
        board_height = target_cfg.get("board_height", 7)
        square_size_m = square_size / 1000.0  # Convert to meters
        marker_size_m = square_size_m * 0.8  # Marker size (80% of square)

        charuco_board = create_charuco_board(
            board_width,
            board_height,
            square_size_m,
            marker_size_m,
            aruco_dict
        )
        file_logger.info(f"ChArUco board: {board_width}x{board_height}")

    # Load calibration if available
    camera_matrix = None
    dist_coeffs = None
    if calibration_path and calibration_path.exists():
        cal_data = np.load(str(calibration_path))
        camera_matrix = cal_data["camera_matrix"]
        dist_coeffs = cal_data["dist_coeffs"]
        file_logger.info(f"Loaded calibration: {calibration_path}")

    # Get image extensions from config
    image_extensions = config.get("targets", {}).get("extensions", ['.jpg', '.jpeg', '.JPG', '.JPEG'])

    # Detect markers in each image
    image_files = list_images(images_dir, extensions=image_extensions)
    if not image_files:
        raise PipelineError(f"No images found in {images_dir}")

    detections = {}
    total_markers = 0

    file_logger.info(f"Processing {len(image_files)} images...")

    for idx, image_path in enumerate(image_files, 1):
        image_name = image_path.name

        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            file_logger.warning(f"Failed to read image: {image_name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_params
        )

        if ids is not None and len(ids) > 0:
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
            for corner_set in corners:
                cv2.cornerSubPix(gray, corner_set, (5, 5), (-1, -1), criteria)

            marker_ids = ids.flatten().tolist()
            total_markers += len(marker_ids)

            detection_info = {
                "marker_ids": marker_ids,
                "corners": [c.tolist() for c in corners],
                "num_markers": len(marker_ids)
            }

            # Estimate board pose if calibration available
            if camera_matrix is not None:
                if board_type == "charuco" and charuco_board is not None:
                    # ChArUco board pose
                    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        corners, ids, gray, charuco_board
                    )
                    if ret and ret > 4:  # Need at least 4 points
                        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                            charuco_corners, charuco_ids, charuco_board,
                            camera_matrix, dist_coeffs, None, None
                        )
                        if success:
                            detection_info["board_pose"] = {
                                "rvec": rvec.flatten().tolist(),
                                "tvec": tvec.flatten().tolist()
                            }
                else:
                    # Single ArUco marker poses
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, square_size / 1000.0, camera_matrix, dist_coeffs
                    )
                    detection_info["marker_poses"] = [
                        {
                            "id": int(marker_ids[i]),
                            "rvec": rvecs[i].flatten().tolist(),
                            "tvec": tvecs[i].flatten().tolist()
                        }
                        for i in range(len(marker_ids))
                    ]

            detections[image_name] = detection_info

            # Save per-image detection
            detection_file = detections_dir / f"{Path(image_name).stem}.json"
            with detection_file.open("w") as f:
                json.dump(detection_info, f, indent=2)

        if idx % 20 == 0:
            file_logger.info(f"  Processed {idx}/{len(image_files)} images...")

    file_logger.info(f"✓ Detected markers in {len(detections)}/{len(image_files)} images")
    file_logger.info(f"✓ Total markers detected: {total_markers}")

    # Save metadata
    meta = {
        "board_type": board_type,
        "dictionary": dictionary_name,
        "square_size_mm": square_size,
        "total_images": len(image_files),
        "images_with_markers": len(detections),
        "total_markers_detected": total_markers,
        "has_calibration": camera_matrix is not None,
        "has_board_poses": any("board_pose" in d for d in detections.values())
    }

    meta_file = targets_dir / "target_meta.json"
    with meta_file.open("w") as f:
        json.dump(meta, f, indent=2)

    file_logger.info(f"✓ Metadata: {meta_file}")

    return detections
