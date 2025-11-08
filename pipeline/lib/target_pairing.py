"""
Image pair generation from coded target co-visibility.

This module generates COLMAP-compatible image pair lists based on
marker co-visibility and temporal connectivity.
"""

import logging
from pathlib import Path
from typing import Dict, Set, Tuple

from .target_utils import load_config

logger = logging.getLogger(__name__)


def generate_colmap_pairs(
    detections: Dict,
    work_dir: Path,
    config_path: Path
) -> int:
    """
    Generate COLMAP pairs.txt from marker co-visibility.

    Args:
        detections: Dict from detect_markers_in_images
        work_dir: Work directory
        config_path: Pipeline config

    Returns:
        Number of pairs written

    Strategy:
    1. For each pair of images that share ≥1 marker ID, add to pairs
    2. Add temporal neighbors (sliding window) for cross-ring links
    3. Write in COLMAP format: "imageA imageB" per line
    """
    config = load_config(config_path)
    target_cfg = config.get("coded_targets", {}).get("pairing", {})

    min_shared_markers = target_cfg.get("min_shared_markers", 1)
    temporal_window = target_cfg.get("temporal_window", 3)

    # Build co-visibility graph
    image_names = sorted(detections.keys())
    pairs: Set[Tuple[str, str]] = set()

    logger.info(f"Generating pairs for {len(image_names)} images...")

    # 1. Add pairs that share markers
    marker_pairs_count = 0
    for i, img1 in enumerate(image_names):
        ids1 = set(detections[img1]["marker_ids"])

        for j in range(i + 1, len(image_names)):
            img2 = image_names[j]
            ids2 = set(detections[img2]["marker_ids"])

            shared = ids1 & ids2
            if len(shared) >= min_shared_markers:
                pair = tuple(sorted([img1, img2]))
                pairs.add(pair)
                marker_pairs_count += 1

    logger.info(f"  Marker-based pairs: {marker_pairs_count}")

    # 2. Add temporal neighbors (sequential images)
    # This helps ensure connectivity across height rings
    temporal_pairs_count = 0
    all_images = sorted(detections.keys())

    for i in range(len(all_images)):
        for offset in range(1, temporal_window + 1):
            if i + offset < len(all_images):
                img1 = all_images[i]
                img2 = all_images[i + offset]
                pair = tuple(sorted([img1, img2]))

                if pair not in pairs:
                    pairs.add(pair)
                    temporal_pairs_count += 1

    logger.info(f"  Temporal pairs: {temporal_pairs_count}")
    logger.info(f"  Total pairs: {len(pairs)}")

    # 3. Write pairs.txt in COLMAP format
    pairs_file = work_dir / "coded_targets" / "pairs.txt"
    with pairs_file.open("w") as f:
        for img1, img2 in sorted(pairs):
            f.write(f"{img1} {img2}\n")

    logger.info(f"✓ Pairs file: {pairs_file}")

    return len(pairs)


def analyze_pair_coverage(
    detections: Dict,
    pairs_file: Path
) -> Dict:
    """
    Analyze coverage statistics for generated pairs.

    Args:
        detections: Dict from detect_markers_in_images
        pairs_file: Path to pairs.txt

    Returns:
        Statistics dict with coverage metrics
    """
    # Read pairs
    pairs = []
    with pairs_file.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append(tuple(parts))

    # Calculate statistics
    total_images = len(detections)
    total_possible_pairs = total_images * (total_images - 1) // 2
    total_actual_pairs = len(pairs)

    # Count pairs per image
    image_pair_counts = {img: 0 for img in detections.keys()}
    for img1, img2 in pairs:
        if img1 in image_pair_counts:
            image_pair_counts[img1] += 1
        if img2 in image_pair_counts:
            image_pair_counts[img2] += 1

    min_pairs_per_image = min(image_pair_counts.values()) if image_pair_counts else 0
    max_pairs_per_image = max(image_pair_counts.values()) if image_pair_counts else 0
    avg_pairs_per_image = sum(image_pair_counts.values()) / len(image_pair_counts) if image_pair_counts else 0

    stats = {
        "total_images": total_images,
        "total_possible_pairs": total_possible_pairs,
        "total_actual_pairs": total_actual_pairs,
        "pair_reduction": f"{(1 - total_actual_pairs/total_possible_pairs)*100:.1f}%",
        "min_pairs_per_image": min_pairs_per_image,
        "max_pairs_per_image": max_pairs_per_image,
        "avg_pairs_per_image": f"{avg_pairs_per_image:.1f}"
    }

    logger.info("Pair coverage statistics:")
    logger.info(f"  Total images: {stats['total_images']}")
    logger.info(f"  Actual pairs: {stats['total_actual_pairs']} / {stats['total_possible_pairs']}")
    logger.info(f"  Pair reduction: {stats['pair_reduction']}")
    logger.info(f"  Pairs per image: {stats['min_pairs_per_image']} - {stats['max_pairs_per_image']} (avg: {stats['avg_pairs_per_image']})")

    return stats
