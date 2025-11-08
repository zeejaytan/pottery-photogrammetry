# Implementation Plan: Coded-Target Assisted Alignment for COLMAP

## Overview

This plan implements a coded-target (ArUco/ChArUco) assisted workflow that:
1. **Detects markers** in images to identify co-visible image pairs
2. **Restricts COLMAP matching** to target-guided pairs (faster, fewer mismatches)
3. **Aligns sparse models** to real-world scale and axis using known board geometry

The system uses COLMAP's native `matches_importer` (pairs mode) and `model_aligner`/`model_transformer` without patching internals.

---

## Architecture

### New Files to Create

```
pipeline/
├── bin/
│   ├── detect_coded_targets.py         # Marker detection CLI (NEW)
│   └── run_colmap.py                   # Modified for target-guided workflow
├── lib/
│   ├── target_detection.py             # ArUco/ChArUco detection (NEW)
│   ├── target_pairing.py               # Pair generation from markers (NEW)
│   ├── target_alignment.py             # Model alignment/scaling (NEW)
│   └── target_utils.py                 # Board config & utilities (NEW)
├── config/
│   └── pipeline_config.yaml            # Add coded_targets section
└── docs/
    ├── CODED_TARGETS_USAGE.md          # User guide (NEW)
    └── BOARD_PRINTING_GUIDE.md         # Printing & placement (NEW)
```

### Output Directory Structure

```
work_colmap_openmvs/
├── coded_targets/                      # Target detection outputs (NEW)
│   ├── pairs.txt                       # Image pairs for COLMAP
│   ├── target_meta.json                # Detection statistics
│   ├── detections/                     # Per-image detection files
│   │   ├── IMG_0001.json
│   │   ├── IMG_0002.json
│   │   └── ...
│   ├── board_poses.json                # Board poses (if calibration available)
│   └── detection_log.txt               # Marker detection log
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       ├── points3D.bin
│       └── alignment_transform.json    # Similarity transform (NEW)
└── alignment_report.txt                # Scale/alignment summary (NEW)
```

---

## Implementation Details

### 1. Target Detection Script (`pipeline/bin/detect_coded_targets.py`)

**Purpose**: Detect ArUco/ChArUco markers and generate image pair list

**Interface**:
```python
#!/usr/bin/env python3
"""
Detect coded targets (ArUco/ChArUco) in images and generate COLMAP pair list.

Usage:
    detect_coded_targets.py --images <path> --work <path> [--config <path>]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.target_detection import detect_markers_in_images
from lib.target_pairing import generate_colmap_pairs
from lib.pipeline_utils import PipelineError


def main():
    parser = argparse.ArgumentParser(
        description="Detect coded targets and generate COLMAP pair list"
    )
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Path to image directory"
    )
    parser.add_argument(
        "--work",
        required=True,
        type=Path,
        help="Path to work_colmap_openmvs directory"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline/config/pipeline_config.yaml"),
        help="Path to pipeline configuration"
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Optional camera calibration file for pose estimation"
    )

    args = parser.parse_args()

    try:
        # Run marker detection
        detections = detect_markers_in_images(
            images_dir=args.images,
            work_dir=args.work,
            config_path=args.config,
            calibration_path=args.calibration
        )

        # Generate pairs.txt
        pairs_written = generate_colmap_pairs(
            detections=detections,
            work_dir=args.work,
            config_path=args.config
        )

        print(f"✓ Detected markers in {len(detections)} images")
        print(f"✓ Generated {pairs_written} image pairs")
        return 0

    except PipelineError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

### 2. Marker Detection Engine (`lib/target_detection.py`)

**Purpose**: Detect ArUco/ChArUco markers using OpenCV

```python
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

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
    logger = configure_logging(log_file, console=True)
    logger.info("=" * 70)
    logger.info("CODED TARGET DETECTION")
    logger.info("=" * 70)

    # Load board configuration
    board_type = target_cfg.get("board_type", "aruco")  # "aruco" or "charuco"
    dictionary_name = target_cfg.get("dictionary", "DICT_4X4_50")
    square_size = target_cfg.get("square_size_mm", 50.0)  # Real-world size

    # Initialize ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dictionary_name)
    )
    detector_params = cv2.aruco.DetectorParameters()

    # For ChArUco boards
    if board_type == "charuco":
        board_width = target_cfg.get("board_width", 5)
        board_height = target_cfg.get("board_height", 7)
        charuco_board = cv2.aruco.CharucoBoard(
            (board_width, board_height),
            square_size / 1000.0,  # Convert to meters
            (square_size * 0.8) / 1000.0,  # Marker size
            aruco_dict
        )

    # Load calibration if available
    camera_matrix = None
    dist_coeffs = None
    if calibration_path and calibration_path.exists():
        cal_data = np.load(calibration_path)
        camera_matrix = cal_data["camera_matrix"]
        dist_coeffs = cal_data["dist_coeffs"]
        logger.info(f"Loaded calibration: {calibration_path}")

    # Detect markers in each image
    image_files = list_images(images_dir)
    detections = {}
    total_markers = 0

    for idx, image_path in enumerate(image_files, 1):
        image_name = image_path.name

        # Read image
        img = cv2.imread(str(image_path))
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
                if board_type == "charuco":
                    # ChArUco board pose
                    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        corners, ids, gray, charuco_board
                    )
                    if ret > 4:  # Need at least 4 points
                        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                            charuco_corners, charuco_ids, charuco_board,
                            camera_matrix, dist_coeffs, None, None
                        )
                        if ret:
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
            logger.info(f"  Processed {idx}/{len(image_files)} images...")

    logger.info(f"✓ Detected markers in {len(detections)}/{len(image_files)} images")
    logger.info(f"✓ Total markers detected: {total_markers}")

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

    logger.info(f"✓ Metadata: {meta_file}")

    return detections
```

---

### 3. Pair Generation (`lib/target_pairing.py`)

**Purpose**: Generate COLMAP pairs.txt from marker detections

```python
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
    pairs = set()

    # 1. Add pairs that share markers
    for i, img1 in enumerate(image_names):
        ids1 = set(detections[img1]["marker_ids"])

        for j in range(i + 1, len(image_names)):
            img2 = image_names[j]
            ids2 = set(detections[img2]["marker_ids"])

            shared = ids1 & ids2
            if len(shared) >= min_shared_markers:
                pairs.add((img1, img2))

    marker_pairs = len(pairs)

    # 2. Add temporal neighbors (sequential images)
    all_images = sorted(detections.keys())
    for i in range(len(all_images)):
        for offset in range(1, temporal_window + 1):
            if i + offset < len(all_images):
                img1 = all_images[i]
                img2 = all_images[i + offset]
                pairs.add(tuple(sorted([img1, img2])))

    temporal_pairs = len(pairs) - marker_pairs

    # 3. Write pairs.txt
    pairs_file = work_dir / "coded_targets" / "pairs.txt"
    with pairs_file.open("w") as f:
        for img1, img2 in sorted(pairs):
            f.write(f"{img1} {img2}\n")

    logger.info(f"✓ Generated {len(pairs)} pairs:")
    logger.info(f"    Marker-based: {marker_pairs}")
    logger.info(f"    Temporal: {temporal_pairs}")
    logger.info(f"✓ Pairs file: {pairs_file}")

    return len(pairs)
```

---

### 4. Model Alignment (`lib/target_alignment.py`)

**Purpose**: Align sparse model to real-world scale/axis using board

```python
def align_sparse_model(
    sparse_dir: Path,
    work_dir: Path,
    config_path: Path,
    colmap_exec: str = "colmap"
) -> Dict:
    """
    Align sparse model using coded target board.

    Two modes:
    1. Full 7-DoF similarity (if board poses available)
    2. Scale only (if only board size known)

    Args:
        sparse_dir: COLMAP sparse/0 directory
        work_dir: Work directory
        config_path: Pipeline config
        colmap_exec: COLMAP executable

    Returns:
        Transform info dict
    """

    config = load_config(config_path)
    target_cfg = config.get("coded_targets", {})

    targets_dir = work_dir / "coded_targets"
    meta_file = targets_dir / "target_meta.json"

    if not meta_file.exists():
        raise PipelineError("No target metadata found. Run detect_coded_targets first.")

    with meta_file.open() as f:
        meta = json.load(f)

    # Check if we have board poses
    has_poses = meta.get("has_board_poses", False)
    square_size_mm = meta.get("square_size_mm", 50.0)

    if has_poses:
        # Mode 1: Full similarity transform using model_aligner
        return align_with_board_poses(
            sparse_dir, targets_dir, colmap_exec, square_size_mm
        )
    else:
        # Mode 2: Scale only using model_transformer
        return align_with_scale_only(
            sparse_dir, targets_dir, colmap_exec, square_size_mm
        )


def align_with_board_poses(
    sparse_dir: Path,
    targets_dir: Path,
    colmap_exec: str,
    square_size_mm: float
) -> Dict:
    """
    Align using board poses (full 7-DoF similarity).

    Uses COLMAP model_aligner with reference poses.
    """

    # Load board poses
    poses_file = targets_dir / "board_poses.json"
    with poses_file.open() as f:
        board_poses = json.load(f)

    # Convert board poses to COLMAP reference format
    # Create images_ref.txt with known camera centers
    ref_images_file = targets_dir / "images_ref.txt"
    with ref_images_file.open("w") as f:
        for image_name, pose_info in board_poses.items():
            # Convert rvec, tvec to camera center
            rvec = np.array(pose_info["rvec"])
            tvec = np.array(pose_info["tvec"])

            # R = cv2.Rodrigues(rvec)[0]
            # C = -R.T @ tvec  # Camera center in world coords

            # Write in COLMAP format
            # ... (format depends on model_aligner requirements)

    # Run model_aligner
    aligned_dir = sparse_dir.parent / "aligned"
    aligned_dir.mkdir(exist_ok=True)

    align_cmd = [
        colmap_exec,
        "model_aligner",
        "--input_path", str(sparse_dir),
        "--output_path", str(aligned_dir),
        "--ref_images_path", str(ref_images_file),
        "--alignment_type", "custom",  # or "ecef" depending on needs
        "--robust_alignment", "1",
    ]

    run_command(align_cmd, logger=logger)

    # Copy aligned model back to sparse/0
    for fname in ["cameras.bin", "images.bin", "points3D.bin"]:
        shutil.copy(aligned_dir / fname, sparse_dir / fname)

    return {
        "method": "board_poses",
        "alignment_type": "similarity_7dof",
        "square_size_mm": square_size_mm
    }


def align_with_scale_only(
    sparse_dir: Path,
    targets_dir: Path,
    colmap_exec: str,
    square_size_mm: float
) -> Dict:
    """
    Apply scale only using known marker distance.

    Uses COLMAP model_transformer with computed scale factor.
    """

    # Read sparse model
    images = read_images_binary(sparse_dir / "images.bin")
    points = read_points3D_binary(sparse_dir / "points3D.bin")

    # Load detections
    with (targets_dir / "target_meta.json").open() as f:
        meta = json.load(f)

    # Find a pair of images that see the same markers
    # Measure distance between marker corners in reconstruction
    # Compare to known square_size_mm

    # For simplicity: measure distance between two known marker positions
    # This requires loading detections and finding 3D points corresponding to markers

    # ... (implementation details for finding scale)

    scale_factor = compute_scale_factor(
        sparse_dir, targets_dir, square_size_mm
    )

    # Apply scale using model_transformer
    transform_cmd = [
        colmap_exec,
        "model_transformer",
        "--input_path", str(sparse_dir),
        "--output_path", str(sparse_dir),  # Transform in-place
        "--transform_type", "scale",
        "--scale", str(scale_factor),
    ]

    run_command(transform_cmd, logger=logger)

    return {
        "method": "scale_only",
        "alignment_type": "scale_transform",
        "scale_factor": scale_factor,
        "square_size_mm": square_size_mm
    }
```

---

### 5. Modified COLMAP Pipeline (`run_colmap.py` changes)

**Integration points**:

```python
def main() -> int:
    # ... existing setup ...

    # Check for coded targets configuration
    target_cfg = context.config.get("coded_targets", {})
    use_coded_targets = target_cfg.get("enabled", False)

    if use_coded_targets:
        logger.info("Coded targets enabled")

        # Step 0: Detect markers and generate pairs
        targets_dir = work_dir / "coded_targets"
        pairs_file = targets_dir / "pairs.txt"

        if not pairs_file.exists():
            logger.info("Running coded target detection...")
            detect_cmd = [
                "python",
                "pipeline/bin/detect_coded_targets.py",
                "--images", str(tree_dir),
                "--work", str(work_dir),
                "--config", str(args.config)
            ]
            run_command(detect_cmd, logger=logger)

        # Check pairs were generated
        if not pairs_file.exists():
            logger.warning("No pairs.txt generated, falling back to exhaustive matching")
            use_coded_targets = False

    # Step 1: Feature Extraction (unchanged)
    # ... existing feature extraction code ...

    # Step 2: Matching - use pairs or exhaustive
    if use_coded_targets and pairs_file.exists():
        logger.info("Using coded-target guided matching")

        # Import pairs
        import_cmd = [
            colmap_exec,
            "matches_importer",
            "--database_path", str(database_path),
            "--match_list_path", str(pairs_file),
            "--match_type", "pairs",
            "--SiftMatching.use_gpu", "1",
            "--SiftMatching.gpu_index", "0",
        ]
        run_command(import_cmd, logger=logger)

    else:
        # Existing exhaustive matcher code
        # ... (unchanged) ...
        pass

    # Step 3: Mapper (unchanged)
    # ... existing mapper code ...

    # Step 4: Alignment (NEW)
    if use_coded_targets:
        logger.info("Aligning sparse model to real-world scale...")

        from lib.target_alignment import align_sparse_model

        transform_info = align_sparse_model(
            sparse_dir=model_dir,
            work_dir=work_dir,
            config_path=Path(args.config),
            colmap_exec=colmap_exec
        )

        # Save transform info
        transform_file = work_dir / "alignment_transform.json"
        with transform_file.open("w") as f:
            json.dump(transform_info, f, indent=2)

        logger.info(f"✓ Model aligned: {transform_info['method']}")

    # Step 5: Undistortion (unchanged)
    # ... existing undistortion code ...
```

---

### 6. Configuration (`pipeline_config.yaml` additions)

```yaml
coded_targets:
  enabled: false                        # Enable coded target workflow
  board_type: aruco                     # "aruco" or "charuco"
  dictionary: DICT_4X4_50               # ArUco dictionary
  square_size_mm: 50.0                  # Real-world square size (mm)

  # ChArUco board dimensions (if board_type: charuco)
  board_width: 5                        # Squares in width
  board_height: 7                       # Squares in height

  # Pairing parameters
  pairing:
    min_shared_markers: 1               # Minimum shared markers for pair
    temporal_window: 3                  # Sequential neighbor window

  # Alignment parameters
  alignment:
    method: auto                        # "auto", "poses", "scale_only"
    min_board_views: 10                 # Minimum images with board visible
    robust_alignment: true              # Use RANSAC in model_aligner
```

---

### 7. Board Printing Guide (`docs/BOARD_PRINTING_GUIDE.md`)

```markdown
# Coded Target Board Printing and Placement

## Printing the Board

### ArUco Board (Simple)

1. Generate board using OpenCV:
   ```python
   import cv2
   import numpy as np

   aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
   board_img = cv2.aruco.drawPlanarBoard(board, (600, 840), 70, 1)
   cv2.imwrite("aruco_board_4x4.png", board_img)
   ```

2. Print on high-quality paper (matte finish recommended)
3. Measure printed square size with calipers
4. Update `square_size_mm` in config

### ChArUco Board (Recommended)

1. Generate ChArUco board:
   ```python
   charuco_board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.032, aruco_dict)
   board_img = charuco_board.generateImage((800, 1120), marginSize=20)
   cv2.imwrite("charuco_board_5x7.png", board_img)
   ```

2. Print on rigid backing (foam board or acrylic)
3. Ensure flat surface (warping reduces accuracy)

## Placement Guidelines

### Recommended Position
- **Near turntable base** (visible from all rings)
- **Angled 30-45°** toward camera
- **Fixed in place** throughout capture
- **Good lighting** (no glare or shadows)

### Coverage Requirements
- Board should appear in ≥60% of images
- At least one full marker visible per image
- Avoid occlusion by pottery or rig

### Poor Placements (Avoid)
- ✗ Behind pottery (occlusion)
- ✗ Flat on turntable (foreshortening)
- ✗ Too close to rig (clutter)
- ✗ Under direct flash (glare)

## Validation

After capture, verify board appears in enough views:

```bash
python pipeline/bin/detect_coded_targets.py \
  --images /path/to/photos \
  --work /path/to/work_dir

# Check coverage
cat work_dir/coded_targets/target_meta.json
# Look for: "images_with_markers" should be >60% of total
```

## Troubleshooting

### Low Detection Rate (<50%)
- Check board print quality
- Improve lighting (diffuse, even)
- Reduce motion blur (slower shutter)

### Markers Not Detected
- Verify dictionary matches config
- Check for glare/shadows
- Ensure board is flat

### Scale Inaccurate
- Re-measure printed square size
- Check board not warped
- Ensure camera calibration is good
```

---

## Integration Workflow

### Modified Pipeline Execution

```bash
# 1. Run COLMAP with coded targets enabled
python pipeline/bin/run_colmap.py /path/to/tree \
  --config pipeline/config/pipeline_config.yaml

# With coded_targets.enabled: true in config, this will:
#   a. Detect markers
#   b. Generate pairs.txt
#   c. Run matches_importer (instead of exhaustive_matcher)
#   d. Run mapper
#   e. Align model to real-world scale
#   f. Undistort

# 2. Verify alignment
cat work_colmap_openmvs/alignment_transform.json

# 3. Continue with OpenMVS (inherits scaled sparse)
python pipeline/bin/run_openmvs.py work_colmap_openmvs
```

---

## Implementation Roadmap

### Milestone 1: Core Detection (Week 1)
- [ ] Create `target_utils.py` with board config loading
- [ ] Implement ArUco detection in `target_detection.py`
- [ ] Add ChArUco detection support
- [ ] Test on sample images
- [ ] Validate sub-pixel corner refinement

### Milestone 2: Pair Generation (Week 1-2)
- [ ] Implement co-visibility graph in `target_pairing.py`
- [ ] Add temporal neighbor window
- [ ] Write COLMAP pairs.txt format
- [ ] Test pair generation on sample dataset
- [ ] Verify pair quality (coverage vs redundancy)

### Milestone 3: COLMAP Integration (Week 2)
- [ ] Modify `run_colmap.py` to detect coded_targets config
- [ ] Implement `matches_importer` call with pairs mode
- [ ] Test matching with restricted pairs
- [ ] Compare runtime vs exhaustive matcher
- [ ] Verify registered image count

### Milestone 4: Alignment/Scaling (Week 2-3)
- [ ] Implement scale-only transform in `target_alignment.py`
- [ ] Compute scale factor from marker distances
- [ ] Call `model_transformer` with scale
- [ ] Test on known-size board
- [ ] Validate measurements in output

### Milestone 5: Board Pose Alignment (Week 3)
- [ ] Implement board pose estimation
- [ ] Convert poses to COLMAP reference format
- [ ] Call `model_aligner` for 7-DoF similarity
- [ ] Test full alignment workflow
- [ ] Compare accuracy vs scale-only

### Milestone 6: Documentation & Testing (Week 3-4)
- [ ] Write `CODED_TARGETS_USAGE.md`
- [ ] Write `BOARD_PRINTING_GUIDE.md`
- [ ] Create board generation scripts
- [ ] Test with multiple board types
- [ ] Performance benchmarking
- [ ] User acceptance testing

---

## Success Criteria

1. **Detection Rate**: ≥70% of images have ≥1 marker detected
2. **Pairing Efficiency**: Generated pairs are 30-50% of exhaustive pairs
3. **Registration Improvement**: ≥95% of images registered (vs current rate)
4. **Runtime Reduction**: Stage 1 runtime reduced by 20-40%
5. **Scale Accuracy**: Measured sherd sizes within ±2mm of physical sizes
6. **Geometric Quality**: Average inliers per verified pair ≥50

---

## Failure Handling

### Detection Phase
- **No markers found**: Fall back to exhaustive matching, log warning
- **<30% coverage**: Add all temporal pairs, proceed with partial pairing
- **Board type mismatch**: Error early with clear config instructions

### Matching Phase
- **Insufficient pairs**: Add sequential window, retry
- **Low geometric verification**: Log statistics, continue (COLMAP handles this)

### Alignment Phase
- **Scale computation fails**: Skip alignment, log warning, continue
- **model_aligner fails**: Fall back to scale-only
- **Scale-only fails**: Continue without alignment, defer to post-processing

---

## Performance Estimates

### Time Savings (150 images, A100)

| Stage | Exhaustive | Target-Guided | Savings |
|-------|-----------|---------------|---------|
| Feature Extraction | 2 min | 2 min | 0% |
| Marker Detection | - | 30 sec | -30 sec |
| Matching | 5 min | 2 min | 60% |
| Mapper | 3 min | 2.5 min | 17% |
| Alignment | - | 20 sec | -20 sec |
| **Total** | **10 min** | **7.2 min** | **28%** |

### Pair Reduction

- Exhaustive: ~11,000 pairs (150 choose 2)
- Target-guided: ~4,000 pairs (shared markers + temporal)
- **Reduction: 64%**

---

## Open Questions / Design Decisions

1. **Dictionary choice**: DICT_4X4_50 vs DICT_6X6_250?
   - **Decision**: Default to 4X4_50 (smaller, faster), allow config override

2. **Temporal window size**: 3 vs 5 vs 10?
   - **Decision**: Default 3, configurable (balance coverage vs pairs)

3. **Pose estimation**: Always estimate or only with calibration?
   - **Decision**: Only with calibration file (optional)

4. **Alignment method**: Auto-select vs user choice?
   - **Decision**: Auto (try poses, fall back to scale), allow override

5. **Board placement**: Single board vs multiple?
   - **Decision**: Single board (simple), document placement guidelines

6. **Failure behavior**: Strict vs permissive?
   - **Decision**: Permissive (fall back to exhaustive), log warnings

---

## Dependencies

### New Python Dependencies
- `opencv-contrib-python` - For ArUco/ChArUco detection
  ```bash
  pip install opencv-contrib-python>=4.8.0
  ```

### Existing Dependencies (already in pipeline)
- `numpy` - Array operations
- `scipy` - Geometric computations
- `trimesh` - (not needed for targets)

### System Dependencies (already in pipeline)
- COLMAP 3.9+ with `matches_importer` and `model_aligner`
- CUDA for GPU matching

---

## References

- [COLMAP matches_importer](https://colmap.github.io/cli.html#matches-importer)
- [COLMAP model_aligner](https://colmap.github.io/cli.html#model-aligner)
- [OpenCV ArUco Tutorial](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [OpenCV ChArUco Tutorial](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Print test board** and capture sample dataset
3. **Set up development branch** (`feature/coded-targets`)
4. **Begin Milestone 1** (detection implementation)
5. **Test incrementally** after each milestone
