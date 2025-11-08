# Coded Target Board Printing and Placement Guide

This guide covers how to generate, print, and place coded target boards for use with the pottery photogrammetry pipeline.

## Table of Contents

1. [Board Generation](#board-generation)
2. [Printing Guidelines](#printing-guidelines)
3. [Measurement](#measurement)
4. [Placement in Scene](#placement-in-scene)
5. [Validation](#validation)
6. [Troubleshooting](#troubleshooting)

## Board Generation

### ArUco Board (Recommended for Beginners)

ArUco boards consist of a grid of square markers. They're simple to generate and detect.

**Python script to generate ArUco board**:

```python
#!/usr/bin/env python3
"""Generate an ArUco marker board for photogrammetry."""

import cv2
import numpy as np

# Configuration
DICTIONARY = cv2.aruco.DICT_4X4_50  # 50 unique 4x4 markers
MARKERS_X = 4                        # Markers in X direction
MARKERS_Y = 6                        # Markers in Y direction
MARKER_SIZE = 50                     # Marker size in mm
MARKER_SEPARATION = 10               # Gap between markers in mm
BORDER_BITS = 1                      # Border width in bits
DPI = 300                            # Print resolution

# Calculate board dimensions
total_marker_mm = MARKER_SIZE
total_sep_mm = MARKER_SEPARATION

# Pixels per mm at target DPI
px_per_mm = DPI / 25.4

# Marker size in pixels
marker_px = int(total_marker_mm * px_per_mm)
sep_px = int(total_sep_mm * px_per_mm)

# Board size in pixels
board_width_px = MARKERS_X * marker_px + (MARKERS_X - 1) * sep_px + 2 * sep_px
board_height_px = MARKERS_Y * marker_px + (MARKERS_Y - 1) * sep_px + 2 * sep_px

# Create dictionary and board
aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
board = cv2.aruco.GridBoard(
    (MARKERS_X, MARKERS_Y),
    total_marker_mm / 1000.0,  # Convert to meters
    total_sep_mm / 1000.0,
    aruco_dict
)

# Generate board image
board_img = board.generateImage((board_width_px, board_height_px), marginSize=sep_px, borderBits=BORDER_BITS)

# Save
output_file = "aruco_board_4x6.png"
cv2.imwrite(output_file, board_img)

print(f"Generated ArUco board: {output_file}")
print(f"  Dictionary: DICT_4X4_50")
print(f"  Grid: {MARKERS_X}x{MARKERS_Y} markers")
print(f"  Marker size: {MARKER_SIZE} mm")
print(f"  Separation: {MARKER_SEPARATION} mm")
print(f"  Image size: {board_width_px}x{board_height_px} px")
print(f"  Board size: {board_width_px/px_per_mm:.1f}x{board_height_px/px_per_mm:.1f} mm")
print()
print("Next steps:")
print(f"  1. Print {output_file} at {DPI} DPI (no scaling)")
print(f"  2. Measure actual printed marker size with calipers")
print(f"  3. Update pipeline_config.yaml with measured size")
```

Save this as `generate_aruco_board.py` and run:

```bash
python generate_aruco_board.py
```

### ChArUco Board (Better Accuracy)

ChArUco boards combine chessboard and ArUco markers. They provide more robust corner detection.

**Python script to generate ChArUco board**:

```python
#!/usr/bin/env python3
"""Generate a ChArUco board for photogrammetry."""

import cv2
import numpy as np

# Configuration
DICTIONARY = cv2.aruco.DICT_4X4_50
SQUARES_X = 5                        # Chessboard squares in X
SQUARES_Y = 7                        # Chessboard squares in Y
SQUARE_SIZE = 40                     # Square size in mm
MARKER_SIZE = 32                     # Marker size in mm (80% of square)
DPI = 300

# Pixels per mm
px_per_mm = DPI / 25.4

# Square and marker sizes in pixels
square_px = int(SQUARE_SIZE * px_per_mm)
marker_px = int(MARKER_SIZE * px_per_mm)

# Board size
margin_px = square_px
board_width_px = SQUARES_X * square_px + 2 * margin_px
board_height_px = SQUARES_Y * square_px + 2 * margin_px

# Create dictionary and board
aruco_dict = cv2.aruco.getPredefinedDictionary(DICTIONARY)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_SIZE / 1000.0,    # Convert to meters
    MARKER_SIZE / 1000.0,
    aruco_dict
)

# Generate image
board_img = board.generateImage((board_width_px, board_height_px), marginSize=margin_px)

# Save
output_file = "charuco_board_5x7.png"
cv2.imwrite(output_file, board_img)

print(f"Generated ChArUco board: {output_file}")
print(f"  Dictionary: DICT_4X4_50")
print(f"  Grid: {SQUARES_X}x{SQUARES_Y} squares")
print(f"  Square size: {SQUARE_SIZE} mm")
print(f"  Marker size: {MARKER_SIZE} mm")
print(f"  Image size: {board_width_px}x{board_height_px} px")
print(f"  Board size: {board_width_px/px_per_mm:.1f}x{board_height_px/px_per_mm:.1f} mm")
print()
print("Next steps:")
print(f"  1. Print {output_file} at {DPI} DPI (no scaling)")
print(f"  2. Measure actual printed square size with calipers")
print(f"  3. Update pipeline_config.yaml with measured size")
```

## Printing Guidelines

### Paper Selection

**Recommended**: Matte photo paper or cardstock
- Reduces glare
- Stays flat
- High contrast black/white

**Not recommended**: Glossy paper
- Can cause reflections
- Makes detection harder

### Printer Settings

1. **Resolution**: 300 DPI minimum (600 DPI better)
2. **Scaling**: NONE - Print at 100% actual size
3. **Color**: Black and white (grayscale)
4. **Quality**: Highest quality setting

### Print Size

**Small boards** (A4 / Letter):
- Good for close-range captures
- Easy to position
- Adequate for turntable work

**Large boards** (A3 / Tabloid):
- Better for distant cameras
- More markers visible per image
- Requires large printer

**Recommendation**: Start with A4 board, upgrade if detection rate is low.

### Mounting

For best results, mount the printed board:

**Option 1: Foam Board**
1. Print board
2. Spray adhesive on foam board
3. Carefully apply print (avoid bubbles)
4. Trim edges

**Option 2: Acrylic/Plexiglass**
1. Print board
2. Sandwich print between acrylic sheets
3. Ensure perfectly flat
4. Use non-reflective acrylic if possible

**Option 3: Rigid Cardstock**
1. Print directly on heavy cardstock (200+ gsm)
2. Keep flat (don't fold or bend)
3. Store under weight when not in use

## Measurement

**CRITICAL**: Measure the actual printed size with calipers. Printers often scale slightly.

### What to Measure

**For ArUco boards**:
- Measure outer edge of one marker square
- Measure in multiple locations
- Average the measurements

**For ChArUco boards**:
- Measure one chessboard square (white or black)
- Measure multiple squares
- Average the measurements

### How to Measure

1. Use digital calipers (±0.1mm accuracy)
2. Measure in good lighting
3. Take 3-5 measurements at different locations
4. Average the results
5. Round to 1 decimal place (e.g., 49.8 mm)

### Update Configuration

Once measured, update `pipeline_config.yaml`:

```yaml
coded_targets:
  square_size_mm: 49.8  # YOUR MEASURED VALUE HERE
```

**DO NOT** use the designed size! Always use the measured size. Even small errors (1-2mm) affect scale accuracy.

## Placement in Scene

### Recommended Position

For turntable pottery photogrammetry:

```
    Camera
      |
      v

  [Pottery]
      |
      |
  .-[Board]-.    <- Angled 30-45° toward camera
 /           \
|  Turntable  |
 \           /
  '---------'
```

**Key points**:
- **Near turntable base** (not on turntable itself!)
- **Angled toward camera** (30-45°) for better marker visibility
- **Fixed in place** throughout entire capture session
- **Good lighting** with no glare or harsh shadows

### Placement Checklist

✓ Board visible in ≥60% of images
✓ At least 1 marker fully visible per image
✓ Board is flat (no warping or bending)
✓ No occlusion by pottery or rig
✓ Even lighting (no glare on markers)
✓ Fixed position (doesn't move during capture)

### Coverage Requirements

**Minimum**: Board visible in 60% of images
**Recommended**: Board visible in 80% of images
**Ideal**: Board visible in 90%+ of images

More coverage = better pair connectivity = higher registration rate

### What to Avoid

✗ **Behind pottery** - Occlusion reduces coverage
✗ **Flat on turntable** - Foreshortening makes detection harder
✗ **Too close to rig** - Can interfere with lighting
✗ **Moving during capture** - Breaks alignment assumptions
✗ **Under direct flash** - Glare prevents marker detection
✗ **In shadows** - Low contrast reduces detection rate

## Validation

### After Placement (Before Capture)

Take a few test photos and verify:

1. Markers are in focus
2. No glare or reflections on board
3. Board is clearly visible
4. Markers are not too small in frame (should be 20+ pixels per marker)

### After Capture (Before Processing)

Run detection to verify coverage:

```bash
python pipeline/bin/detect_coded_targets.py \
  --images /path/to/photos \
  --work /path/to/test_work \
  --config pipeline/config/pipeline_config.yaml \
  --analyze
```

Check the output:

```json
{
  "total_images": 150,
  "images_with_markers": 120,      # Should be >90 (60%)
  "total_markers_detected": 480,
  ...
}
```

**Good**: `images_with_markers` ≥ 90 (for 150 images)
**Acceptable**: `images_with_markers` ≥ 75
**Poor**: `images_with_markers` < 75 - Consider repositioning board

## Troubleshooting

### Low Detection Rate

**Symptoms**: Fewer than 60% of images have detected markers

**Causes & Solutions**:

1. **Board too small / camera too far**
   - Solution: Print larger board or move board closer

2. **Poor lighting**
   - Solution: Add diffuse lighting, eliminate glare
   - Avoid direct flash on board

3. **Motion blur**
   - Solution: Use faster shutter speed or better lighting

4. **Focus issues**
   - Solution: Ensure camera autofocus includes board area
   - Use wider depth of field (higher f-stop)

5. **Wrong dictionary**
   - Solution: Verify config matches generated board

### Markers Not Detected

**Symptoms**: Zero markers detected in images

**Solutions**:

1. Check dictionary in config matches generated board
2. Verify markers are visible and not occluded
3. Test with sample ArUco detection script
4. Ensure OpenCV installed with ArUco module

### Inaccurate Measurements

**Symptoms**: Reconstructed object is wrong size

**Solutions**:

1. **Re-measure with calipers** (most common issue)
2. Measure multiple squares and average
3. Check board is perfectly flat
4. Verify printer didn't scale during printing

### Glare on Markers

**Symptoms**: Intermittent detection failures

**Solutions**:

1. Use matte paper instead of glossy
2. Adjust lighting angle to avoid specular reflections
3. Use polarizing filter on camera (if available)
4. Move board slightly away from direct lights

## Example: Complete Workflow

1. **Generate board**:
   ```bash
   python generate_aruco_board.py
   ```

2. **Print**:
   - Use laser printer at 300 DPI
   - Print on matte cardstock
   - DO NOT scale (100% size)

3. **Mount**:
   - Glue to foam board
   - Let dry completely
   - Trim edges

4. **Measure**:
   - Use calipers to measure marker size
   - Take 5 measurements, average: 49.6 mm

5. **Update config**:
   ```yaml
   coded_targets:
     enabled: true
     square_size_mm: 49.6  # Measured value
   ```

6. **Place in scene**:
   - Position near turntable base
   - Angle 30° toward camera
   - Verify visible in test shots

7. **Validate**:
   ```bash
   # Take 10 test photos
   python pipeline/bin/detect_coded_targets.py \
     --images test_photos/ --work test_work/ --analyze
   # Check: 8-10 images should have markers detected
   ```

8. **Capture dataset**:
   - Board stays fixed
   - Capture full turntable sequence

9. **Process**:
   ```bash
   python pipeline/bin/run_colmap.py /path/to/tree
   ```

## Board Templates

Pre-generated boards available in `pipeline/assets/` (if they exist), or generate your own:

- `aruco_4x4_50mm.png` - 4x6 grid, 50mm markers
- `aruco_4x4_40mm.png` - 4x6 grid, 40mm markers (compact)
- `charuco_5x7_40mm.png` - 5x7 ChArUco, 40mm squares

Always measure actual printed size regardless of filename!

## References

- [OpenCV ArUco Tutorial](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [OpenCV ArUco Marker Generation](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html#tutorial_aruco_board_creation)
- [ChArUco Boards Explained](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html)

## Tips for Success

1. **Measure, don't estimate** - Always measure the printed board
2. **Start simple** - Use ArUco before trying ChArUco
3. **Test early** - Validate detection on a few images before full capture
4. **Keep it flat** - Warped boards = inaccurate measurements
5. **Fix it in place** - Moving board breaks assumptions
6. **Light it well** - Even, diffuse lighting with no glare
7. **Make it visible** - Board should be in 80%+ of images

Following this guide will ensure reliable marker detection and accurate scale alignment in your pottery photogrammetry pipeline!
