# Coded Target Usage Guide

This guide explains how to use coded targets (ArUco/ChArUco markers) to improve COLMAP reconstruction quality and enable real-world scale alignment.

## Overview

Coded targets are fiducial markers (like QR codes) that can be automatically detected in images. By placing a board with coded targets in your photogrammetry scene, you can:

1. **Improve matching efficiency** - Only match images that share visible markers (30-60% fewer pairs)
2. **Increase registration rate** - Better connectivity across height rings in turntable captures
3. **Enable real-world scaling** - Align reconstruction to known board dimensions
4. **Reduce processing time** - Faster matching with targeted pairs

## Quick Start

### 1. Print and Place a Marker Board

See [BOARD_PRINTING_GUIDE.md](BOARD_PRINTING_GUIDE.md) for detailed instructions on:
- Generating and printing boards
- Measuring physical dimensions
- Optimal placement in your scene

**TL;DR**: Print a board, measure the square size with calipers, place it visible from most camera positions.

### 2. Enable Coded Targets in Configuration

Edit `pipeline/config/pipeline_config.yaml`:

```yaml
coded_targets:
  enabled: true                     # Enable the workflow
  board_type: aruco                 # "aruco" or "charuco"
  dictionary: DICT_4X4_50           # ArUco dictionary
  square_size_mm: 50.0              # MEASURE THIS with calipers!
```

### 3. Run COLMAP with Coded Targets

```bash
python pipeline/bin/run_colmap.py /path/to/tree \
  --config pipeline/config/pipeline_config.yaml
```

The pipeline will automatically:
1. Detect markers in images
2. Generate targeted image pairs
3. Match features only for those pairs
4. Build sparse reconstruction
5. Align model to real-world scale
6. Continue with dense reconstruction

## Configuration Options

### Board Type

**ArUco** (simpler, faster):
```yaml
coded_targets:
  board_type: aruco
  dictionary: DICT_4X4_50           # 50 unique 4x4 markers
  square_size_mm: 50.0
```

**ChArUco** (more accurate pose estimation):
```yaml
coded_targets:
  board_type: charuco
  dictionary: DICT_4X4_50
  square_size_mm: 50.0
  board_width: 5                    # Chessboard squares wide
  board_height: 7                   # Chessboard squares tall
```

### ArUco Dictionary

Common dictionaries:
- `DICT_4X4_50` - 4x4 bit markers, 50 IDs (recommended for small boards)
- `DICT_5X5_50` - 5x5 bit markers, 50 IDs (more robust at distance)
- `DICT_6X6_50` - 6x6 bit markers, 50 IDs (highest robustness)

Use larger dictionaries (6x6) if your camera is far from the board or lighting is poor.

### Pairing Parameters

```yaml
coded_targets:
  pairing:
    min_shared_markers: 1           # Minimum markers in common to pair images
    temporal_window: 3              # Also pair with N sequential neighbors
```

- **min_shared_markers**: Images must share this many markers to be paired. Set to 2-3 for stricter pairing.
- **temporal_window**: Pairs each image with N sequential images (helps cross-ring connectivity). Increase for sparse coverage.

### Alignment Parameters

```yaml
coded_targets:
  alignment:
    method: auto                    # "auto", "poses", "scale_only"
    min_board_views: 10             # Minimum images with board for reliable scale
    robust_alignment: true          # Use RANSAC (recommended)
```

- **method**:
  - `auto` - Automatically choose best method (recommended)
  - `scale_only` - Only scale model (doesn't require camera calibration)
  - `poses` - Full 7-DoF alignment (requires camera calibration)

## Manual Workflow

You can also run detection separately for testing:

```bash
# Detect markers and generate pairs
python pipeline/bin/detect_coded_targets.py \
  --images /path/to/photos \
  --work /path/to/work_colmap_openmvs \
  --config pipeline/config/pipeline_config.yaml

# Analyze pair coverage
python pipeline/bin/detect_coded_targets.py \
  --images /path/to/photos \
  --work /path/to/work_colmap_openmvs \
  --analyze

# With camera calibration for pose estimation
python pipeline/bin/detect_coded_targets.py \
  --images /path/to/photos \
  --work /path/to/work_colmap_openmvs \
  --calibration /path/to/camera_calibration.npz
```

## Output Files

After running with coded targets enabled:

```
work_colmap_openmvs/
├── coded_targets/
│   ├── pairs.txt                   # Image pairs for COLMAP
│   ├── target_meta.json            # Detection statistics
│   ├── detection_log.txt           # Detection log
│   └── detections/                 # Per-image detection files
│       ├── IMG_0001.json
│       ├── IMG_0002.json
│       └── ...
├── alignment_transform.json        # Transform applied to model
└── alignment_report.txt            # Alignment summary
```

### Inspecting Results

**Check detection rate**:
```bash
cat work_colmap_openmvs/coded_targets/target_meta.json
```

Look for:
- `images_with_markers` - Should be >60% of total images
- `total_markers_detected` - More markers = better coverage

**Check pair generation**:
```bash
wc -l work_colmap_openmvs/coded_targets/pairs.txt
```

Expect 30-50% of exhaustive pairs (e.g., 4,000 pairs instead of 11,000 for 150 images).

**Check alignment**:
```bash
cat work_colmap_openmvs/alignment_report.txt
```

Verify scale factor is reasonable (should be close to measured square_size_mm).

## Troubleshooting

### Low Detection Rate (<50% of images)

**Symptoms**: `target_meta.json` shows low `images_with_markers`

**Solutions**:
1. Check board is visible in most images
2. Improve lighting (reduce glare, shadows)
3. Reduce camera motion blur (slower shutter)
4. Verify dictionary matches config
5. Try larger marker dictionary (6x6 instead of 4x4)

### Poor Pair Coverage

**Symptoms**: Few pairs generated, high percentage of unregistered images

**Solutions**:
1. Increase `temporal_window` (e.g., 5 or 10)
2. Decrease `min_shared_markers` to 1
3. Improve board placement to be visible from more rings
4. Add second board at different angle

### Inaccurate Scale

**Symptoms**: Reconstructed object is wrong size

**Solutions**:
1. **Re-measure square_size_mm** with calipers (most common issue!)
2. Check board is flat (warping affects measurement)
3. Ensure camera calibration is accurate (if using pose-based alignment)
4. Increase `min_board_views` for more robust scale estimate

### Fallback to Exhaustive Matching

**Symptoms**: Log shows "Falling back to exhaustive matching"

**Causes**:
- No markers detected in images
- Pairs file is empty
- Detection script failed

**Solutions**:
1. Check detection log: `work_colmap_openmvs/coded_targets/detection_log.txt`
2. Verify OpenCV is installed with ArUco support: `python -c "import cv2; print(cv2.aruco)"`
3. Run detection manually to see errors

## Best Practices

### Board Placement

✓ **DO**:
- Place near turntable base
- Angle 30-45° toward camera
- Keep fixed throughout capture
- Ensure good, even lighting

✗ **DON'T**:
- Place behind pottery (occlusion)
- Lay flat on turntable (foreshortening)
- Move board during capture
- Allow glare or shadows on markers

### Configuration

✓ **DO**:
- Measure square_size_mm precisely with calipers
- Start with `board_type: aruco` (simpler)
- Use `method: auto` for alignment
- Test on small dataset first

✗ **DON'T**:
- Guess square size (measure it!)
- Use ChArUco without camera calibration
- Expect perfect scale without calibration
- Skip board printing guide

### Validation

After first run, check:
1. Detection rate ≥60% of images
2. Pair count is 30-50% of exhaustive
3. Registration rate ≥95% of images
4. Scale factor is sensible (not 0.001 or 1000)

## Performance Impact

Expected performance on 150-image turntable capture:

| Stage | Without Targets | With Targets | Change |
|-------|----------------|--------------|--------|
| Feature Extraction | 2 min | 2 min | 0% |
| Marker Detection | - | 30 sec | +30 sec |
| Matching | 5 min | 2 min | **-60%** |
| Mapper | 3 min | 2.5 min | -17% |
| Alignment | - | 20 sec | +20 sec |
| **Total** | **10 min** | **7.2 min** | **-28%** |

**Benefits**:
- 28% faster overall
- 64% fewer pairs to match
- Better registration across rings
- Real-world scale automatically

## Advanced: Camera Calibration

For pose-based alignment (full 7-DoF similarity transform), you need camera calibration.

### Calibrate Camera

Use OpenCV calibration tools or COLMAP's `image_calibrator`:

```bash
# Using COLMAP
colmap image_calibrator \
  --images /path/to/calibration_images \
  --output /path/to/calibration.json
```

### Use Calibration

```bash
python pipeline/bin/detect_coded_targets.py \
  --images /path/to/photos \
  --work /path/to/work_colmap_openmvs \
  --calibration /path/to/calibration.npz
```

With calibration, the system can:
- Estimate board poses in 3D
- Align reconstruction to board coordinate system
- Apply full similarity transform (rotation, translation, scale)

## FAQ

**Q: Can I use multiple boards?**
A: The current implementation supports single boards. Multiple boards would require code changes.

**Q: Do I need camera calibration?**
A: No, scale-only alignment works without calibration. Calibration enables more accurate pose-based alignment.

**Q: What if markers aren't detected?**
A: The system gracefully falls back to exhaustive matching. Check troubleshooting section above.

**Q: Can I reuse the same board for all captures?**
A: Yes! Just update `square_size_mm` if you print a new one.

**Q: Does this work with handheld captures?**
A: Yes, but board must be visible in enough images. Works best with structured turntable captures.

**Q: Can I disable alignment but keep pair generation?**
A: Not currently. Coded targets enable both pairing and alignment. You can ignore the alignment output.

## References

- [ArUco Marker Detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [ChArUco Boards](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)
- [COLMAP matches_importer](https://colmap.github.io/cli.html#matches-importer)
- [COLMAP model_aligner](https://colmap.github.io/cli.html#model-aligner)

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review detection log: `work_colmap_openmvs/coded_targets/detection_log.txt`
3. Run with `--verbose` flag for detailed output
4. Open an issue with log files and configuration
