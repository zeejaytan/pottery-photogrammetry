"""
Utilities for coded target (ArUco/ChArUco) board configuration and management.

This module provides helper functions for loading board configurations,
initializing ArUco dictionaries, and creating board objects.
"""

import cv2
import yaml
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to pipeline_config.yaml

    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f)

    return config


def get_aruco_dictionary(dictionary_name: str):
    """
    Get ArUco dictionary by name.

    Args:
        dictionary_name: Name of the ArUco dictionary (e.g., "DICT_4X4_50")

    Returns:
        cv2.aruco.Dictionary object
    """
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(
            f"Invalid ArUco dictionary: {dictionary_name}. "
            f"Must be one of: DICT_4X4_50, DICT_5X5_50, DICT_6X6_50, etc."
        )

    dict_id = getattr(cv2.aruco, dictionary_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    logger.info(f"Loaded ArUco dictionary: {dictionary_name}")
    return aruco_dict


def create_aruco_detector_parameters() -> cv2.aruco.DetectorParameters:
    """
    Create ArUco detector parameters with optimized settings.

    Returns:
        cv2.aruco.DetectorParameters object
    """
    params = cv2.aruco.DetectorParameters()

    # Optimize for pottery photogrammetry conditions
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.1

    return params


def create_charuco_board(
    board_width: int,
    board_height: int,
    square_size: float,
    marker_size: float,
    aruco_dict
) -> cv2.aruco.CharucoBoard:
    """
    Create a ChArUco board object.

    Args:
        board_width: Number of squares in width
        board_height: Number of squares in height
        square_size: Size of each square in meters
        marker_size: Size of each ArUco marker in meters
        aruco_dict: ArUco dictionary for the markers

    Returns:
        cv2.aruco.CharucoBoard object
    """
    board = cv2.aruco.CharucoBoard(
        (board_width, board_height),
        square_size,
        marker_size,
        aruco_dict
    )

    logger.info(
        f"Created ChArUco board: {board_width}x{board_height}, "
        f"square={square_size*1000:.1f}mm, marker={marker_size*1000:.1f}mm"
    )

    return board


def load_board_config(config_path: Path) -> Tuple[Dict, object, Optional[object]]:
    """
    Load board configuration and create necessary objects.

    Args:
        config_path: Path to pipeline_config.yaml

    Returns:
        Tuple of (target_cfg, aruco_dict, charuco_board or None)
    """
    config = load_config(config_path)
    target_cfg = config.get("coded_targets", {})

    if not target_cfg.get("enabled", False):
        logger.warning("Coded targets not enabled in config")
        return target_cfg, None, None

    # Load ArUco dictionary
    dictionary_name = target_cfg.get("dictionary", "DICT_4X4_50")
    aruco_dict = get_aruco_dictionary(dictionary_name)

    # Create ChArUco board if needed
    charuco_board = None
    board_type = target_cfg.get("board_type", "aruco")

    if board_type == "charuco":
        board_width = target_cfg.get("board_width", 5)
        board_height = target_cfg.get("board_height", 7)
        square_size_mm = target_cfg.get("square_size_mm", 50.0)
        square_size_m = square_size_mm / 1000.0
        marker_size_m = square_size_m * 0.8  # Marker is 80% of square

        charuco_board = create_charuco_board(
            board_width,
            board_height,
            square_size_m,
            marker_size_m,
            aruco_dict
        )

    return target_cfg, aruco_dict, charuco_board


def list_images(images_dir: Path, extensions: Optional[list] = None) -> list:
    """
    List all image files in a directory.

    Args:
        images_dir: Directory containing images
        extensions: List of valid extensions (default: .jpg, .jpeg, .JPG, .JPEG)

    Returns:
        Sorted list of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']

    image_files = []
    for ext in extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))

    image_files = sorted(image_files)
    logger.info(f"Found {len(image_files)} images in {images_dir}")

    return image_files


def configure_logging(log_file: Path, console: bool = True) -> logging.Logger:
    """
    Configure logging for target detection.

    Args:
        log_file: Path to log file
        console: Whether to also log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    return logger
