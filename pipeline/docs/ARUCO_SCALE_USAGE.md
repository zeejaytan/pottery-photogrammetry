# ArUco Marker-Based Scale Measurement

This guide explains how to use ArUco markers for real-world scale measurement in COLMAP reconstructions.

## Overview

The ArUco scale workflow allows you to:
1. Place two ArUco markers on your pottery (e.g., on the base)
2. Run COLMAP Stage 1 to get a sparse reconstruction
3. Let the system detect markers and triangulate their 3D positions
4. Interactively enter the known distance between markers
5. Pipeline computes and applies scale automatically
6. All downstream products (meshes, textures) are in metric units

**Key Advantages**:
- No need to export PLY and measure on laptop
- More accurate than manual measurement (automatic triangulation)
- More flexible than coded target boards (just 2 markers)
- Interactive confirmation prevents errors
- Works with existing marker placement

## When to Use

**Use ArUco scale when:**
- You have 2+ ArUco markers placed on your pottery
- Markers are visible in ≥2 views each
- You know the real-world distance between markers
- You want semi-automated scale with explicit validation

**Use manual scale when:**
- No markers in scene
- Retrospective scaling of existing datasets
- More than one feature available for validation

**Use coded targets when:**
- Running large-scale production pipeline
- Fully automated workflow preferred
- Markers arranged in calibration board pattern

## Quick Start

### Step 1: Place Markers and Capture

1. Print two ArUco markers from the same dictionary (e.g., DICT_4X4_50)
2. Place markers on pottery base or setup (stable positions)
3. Measure center-to-center distance with calipers (e.g., 100.0 mm)
4. Capture your photos ensuring markers visible in multiple views

### Step 2: Run COLMAP Stage 1

```bash
python pipeline/bin/run_colmap.py /path/to/tree
```

This runs feature extraction, matching, and mapper to get sparse reconstruction.

### Step 3: Run ArUco Scale Measurement (Interactive)

```bash
python pipeline/bin/scale_aruco.py --work /path/to/tree/work_colmap_openmvs
```

**Interactive workflow:**

```
Step 1: Exporting sparse model to TXT format...
✓ Exported to work_colmap_openmvs/sparse_txt

Step 2: Loading camera parameters and poses...
✓ Loaded 1 cameras, 120 images

Step 3: Detecting ArUco markers...
✓ Detected 5 unique markers

Detected marker IDs and view counts:
  ID  11:  45 views
  ID  23:  42 views
  ID   5:  38 views
  ID  17:  12 views
  ID   8:   5 views

Choose two markers with good coverage for scale measurement.
Enter marker ID A: 11
Enter marker ID B: 23

Step 4: Triangulating marker centers...
✓ Marker 11: triangulated from 45 views
✓ Marker 23: triangulated from 42 views
✓ Reconstructed distance: 0.0543271 units

Enter the real-world distance between the marker centers.
This should be the center-to-center distance measured on your hardware.
Distance value: 100
Units (mm or m): mm

Interpreted as: 0.100000 m
  (from 100.00 mm)
Is this correct? [y/N]: y

Step 5: Computing scale factor...

======================================================================
SCALE COMPUTATION SUMMARY
======================================================================
Marker IDs:              11, 23
Real-world distance:     0.100000 m (100.00 mm)
Reconstructed distance:  0.0543271 units
Scale factor:            1.840649285
======================================================================

Apply this scale to the sparse model? [y/N]: y

Step 6: Writing provenance logs...
Step 7: Applying scale to sparse model...
Step 8: Regenerating dense workspace...
Step 9: Updating pipeline runlog...

======================================================================
ARUCO SCALE MEASUREMENT COMPLETE
======================================================================
Scaled sparse: work_colmap_openmvs/sparse_scaled/0
Scaled dense:  work_colmap_openmvs/dense_scaled
Scale factor:  1.840649285

Next steps:
  1. Verify scale in CloudCompare (measure marker distance in dense)
  2. Continue to OpenMVS using dense_scaled/
     InterfaceCOLMAP -i work_colmap_openmvs/dense_scaled -o scene.mvs
```

### Step 4: Continue to OpenMVS

Use the scaled dense workspace:

```bash
InterfaceCOLMAP \
  -i work_colmap_openmvs/dense_scaled \
  -o scene.mvs \
  --image-folder work_colmap_openmvs/dense_scaled/images

DensifyPointCloud -i scene.mvs -o scene_dense.mvs
ReconstructMesh -i scene_dense.mvs -o mesh.mvs
```

All outputs will be in metres.

## Non-Interactive Mode (Scripted)

For batch processing or automation:

```bash
python pipeline/bin/scale_aruco.py \
  --work /path/to/tree/work_colmap_openmvs \
  --idA 11 \
  --idB 23 \
  --real-mm 100.0 \
  --non-interactive
```

Or with metres:

```bash
python pipeline/bin/scale_aruco.py \
  --work /path/to/tree/work_colmap_openmvs \
  --idA 11 \
  --idB 23 \
  --real-m 0.100 \
  --non-interactive
```

**Requirements for non-interactive mode:**
- Must provide `--idA` and `--idB`
- Must provide `--real-mm` OR `--real-m`
- Must provide `--non-interactive` flag

## Integration with Pipeline

### Option 1: Run During COLMAP

```bash
python pipeline/bin/run_colmap.py /path/to/tree --scale-aruco
```

This will:
1. Run COLMAP Stage 1 (feature extraction, matching, mapper)
2. Pause and prompt for marker IDs and distance
3. Apply scale and regenerate dense workspace
4. Continue with dense reconstruction

### Option 2: Run After COLMAP

```bash
# First, run COLMAP normally
python pipeline/bin/run_colmap.py /path/to/tree

# Then, run ArUco scale separately
python pipeline/bin/scale_aruco.py --work /path/to/tree/work_colmap_openmvs
```

This gives you more control and allows you to inspect sparse before scaling.

## Marker Placement Guidelines

### Optimal Placement

**DO:**
- Place markers on stable, rigid surfaces (e.g., pottery base, turntable edge)
- Ensure markers are planar and not curved
- Use high-contrast backgrounds (white marker on dark surface)
- Print markers at sufficient size (≥30mm for typical setups)
- Verify markers visible in ≥10 views each

**DON'T:**
- Place markers on curved surfaces (introduces perspective errors)
- Place markers too close together (<30mm, reduces accuracy)
- Place markers where they might move between shots
- Obscure markers with pottery in later shots

### Recommended Marker Sizes

| Capture Distance | Marker Size | Notes |
|------------------|-------------|-------|
| 30-50 cm | 20-30 mm | Close-up pottery |
| 50-100 cm | 30-50 mm | Standard turntable |
| 100-200 cm | 50-80 mm | Large objects |

**Formula**: Marker should be ≥1% of image width for reliable detection.

### Marker Dictionary

Use `DICT_4X4_50` (default) for most cases:
- 50 unique IDs
- 4x4 bit grid (robust to blur)
- Good balance of capacity and detectability

Other options:
- `DICT_5X5_100`: More IDs, slightly more sensitive to blur
- `DICT_6X6_250`: Maximum IDs, requires sharp images
- `DICT_ARUCO_ORIGINAL`: Legacy compatibility

Generate markers at: https://chev.me/arucogen/

## Troubleshooting

### "No markers detected in any images"

**Possible causes:**
- Markers too small or blurry
- Wrong dictionary specified
- Markers obscured or not in field of view

**Solutions:**
```bash
# Try different dictionary
python pipeline/bin/scale_aruco.py --work work_colmap_openmvs --dictionary DICT_5X5_100

# Check marker visibility manually:
# Open random images and verify markers are visible and sharp
```

### "Marker X only visible in Y views. Need at least 2"

**Cause**: Marker not visible in enough images for triangulation.

**Solution**:
- Choose different marker IDs with better coverage
- Re-capture with markers visible in more views
- Check marker detection JSON: `work_colmap_openmvs/scale/detected_ids.json`

### "Scale factor X is outside sane bounds [0.01, 100.0]"

**Possible causes:**
- Wrong marker IDs selected (measured wrong pair)
- Incorrect distance or units (mm vs m)
- Reconstruction failure for those markers

**Solutions:**
1. Check you entered correct marker IDs
2. Verify distance units (100mm = 0.100m, not 100m)
3. Try different marker pair with better coverage
4. Inspect triangulated positions in CloudCompare:
   ```bash
   # Open sparse_txt/points3D.ply and verify marker positions look reasonable
   ```

### "Scale factor X is unusual. Typical range: [0.1, 10.0]"

**Cause**: Scale is technically valid but outside normal range for pottery.

**Check:**
- Did you swap real and reconstructed values?
- Is distance in correct units?
- Are you measuring the right markers?

Typical scales for pottery photogrammetry:
- 0.5 - 2.0 for turntable setups
- 2.0 - 5.0 for handheld close-up
- 0.1 - 0.5 for large objects

### Markers Detected but Wrong IDs Shown

**Cause**: Different markers detected than you printed.

**Solution:**
1. Check `work_colmap_openmvs/scale/detected_ids.json` for actual detected IDs
2. Print new markers matching detected dictionary
3. Or regenerate markers with correct dictionary

## Verification

After scale application, verify accuracy:

### Method 1: Measure in CloudCompare

```bash
# Open scaled dense point cloud
# Measure distance between markers
# Should match your real-world measurement within ±1%

# Example:
# Real distance: 100.00 mm
# Measured in CloudCompare: 99.8 - 100.2 mm (acceptable)
```

### Method 2: Measure Known Features

```bash
# Measure pottery height, diameter, etc.
# Compare to physical measurements
# Typical accuracy: ±0.5% for well-placed markers
```

## Provenance and Logs

All scale information is logged for thesis documentation:

**Files created:**
- `scale/MANIFEST.txt` - Method, marker IDs, distance, scale, timestamp
- `scale/SCALE.txt` - Computed scale factor (single number)
- `scale/aruco_scale_log.txt` - Detailed log with views, validation
- `scale/detected_ids.json` - All detected markers and coverage
- `pipeline_RUNLOG.txt` - Audit trail entry

**Example `aruco_scale_log.txt`:**
```
======================================================================
ARUCO SCALE MEASUREMENT
======================================================================

Application time: 2025-11-08T12:34:56

Marker A: ID 11
  Views used for triangulation: 45

Marker B: ID 23
  Views used for triangulation: 42

Distance (real-world):    0.100000 m (100.00 mm)
Distance (reconstructed): 0.054327 units

Scale factor: 1.840649285

This scale factor transforms the sparse reconstruction
to real-world metric units (metres) based on the known
distance between two ArUco markers.

======================================================================
```

## Advanced Usage

### Custom Dictionary

```bash
python pipeline/bin/scale_aruco.py \
  --work work_colmap_openmvs \
  --dictionary DICT_6X6_250
```

### Specify Paths Explicitly

```bash
python pipeline/bin/scale_aruco.py \
  --images /path/to/images_jpg \
  --sparse /path/to/sparse/0 \
  --config pipeline/config/pipeline_config.yaml \
  --colmap /custom/path/to/colmap
```

### Reapply Scale

If you made a mistake, simply rerun:

```bash
# Fix your input and rerun
python pipeline/bin/scale_aruco.py --work work_colmap_openmvs
```

The script will:
1. Re-detect markers
2. Re-triangulate positions
3. Overwrite `sparse_scaled/` and `dense_scaled/`

## Comparison: Scale Methods

| Feature | Manual Scale | ArUco Scale | Coded Targets |
|---------|-------------|-------------|---------------|
| **Setup** | None | 2 markers | Print board |
| **User effort** | 4 min (export+measure) | 2 min (enter distance) | Minimal |
| **Accuracy** | ±0.5% (dual validation) | ±0.5% (triangulation) | ±2mm |
| **Error detection** | Automatic (2 measurements) | Sanity bounds | None |
| **Automation** | Semi (laptop needed) | Semi (interactive) | Full |
| **Flexibility** | Any feature | Any 2 markers | Board only |
| **Dependencies** | CloudCompare/MeshLab | OpenCV ArUco | OpenCV ArUco |
| **Use case** | Retrospective | Flexible | Production |

**When to use each:**

1. **ArUco Scale** (this method):
   - You placed 2+ markers in your capture
   - You want faster workflow than manual PLY measurement
   - You want explicit confirmation before applying scale
   - Good for thesis work with documentation requirements

2. **Manual Scale**:
   - No markers in scene
   - Retrospective scaling of old datasets
   - You have multiple known dimensions for validation

3. **Coded Targets**:
   - Large-scale production pipeline
   - Markers arranged in calibration board
   - Fully automated workflow desired

## Configuration

Update `pipeline/config/pipeline_config.yaml`:

```yaml
aruco_scale:
  enabled: false                    # Enable ArUco scale workflow
  dictionary: DICT_4X4_50           # ArUco dictionary
  auto_run: false                   # Auto-run after mapper (requires --idA/--idB/--real-m in env)
  sanity_bounds:
    min_scale: 0.01                 # Minimum allowed scale factor
    max_scale: 100.0                # Maximum allowed scale factor
  prefer_aruco_scale: true          # Prefer ArUco scale over manual/coded targets
```

## FAQ

**Q: How many markers do I need?**
A: Minimum 2, but having 3-5 gives you more pairing options if one has poor coverage.

**Q: Can markers be on different surfaces?**
A: Yes, as long as both are rigid and stationary. Common setup: one on pottery base, one on turntable edge.

**Q: What if markers move between shots?**
A: Movement invalidates scale. Ensure markers are fixed throughout capture.

**Q: Can I use the same markers for multiple trees?**
A: Yes, but you must measure and enter the distance for each tree separately.

**Q: How accurate is triangulation?**
A: Typically ±0.5% when markers are well-covered (≥10 views each). Better than manual PLY measurement.

**Q: What if I enter wrong distance?**
A: Just rerun `scale_aruco.py` with correct distance. It will overwrite previous scale.

**Q: Can I use this with masked images?**
A: Yes, but ensure markers are NOT masked out. Run without masks first, apply scale, then rerun with masks if needed.

## References

- ArUco Marker Generator: https://chev.me/arucogen/
- OpenCV ArUco Module: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- COLMAP Model Transformer: https://colmap.github.io/cli.html#model-transformer

## Support

For issues:
1. Check troubleshooting section above
2. Review `scale/aruco_scale_log.txt` for detailed diagnostics
3. Verify marker detection in `scale/detected_ids.json`
4. Open an issue with log files and marker IDs
