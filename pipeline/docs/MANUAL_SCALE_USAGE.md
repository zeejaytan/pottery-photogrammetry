# Manual Scale Measurement Workflow

This guide explains how to apply real-world scale to COLMAP reconstructions using manual measurement on your laptop, without requiring coded targets in the scene.

## Overview

The manual scale workflow allows you to:
1. Export the sparse model after COLMAP Stage 1
2. Measure TWO known distances on your laptop (e.g., pottery base diameter and height)
3. Feed both measurements back to Spartan for validation
4. Pipeline applies scale to sparse and regenerates dense workspace
5. All downstream products (meshes, textures) are in metric units

**Key Feature**: Dual measurements validate each other - if they don't agree within ±2%, the system catches the error before corrupting your model!

## Quick Start

### Step 1: Run COLMAP Stage 1 with Export

```bash
python pipeline/bin/run_colmap.py /path/to/tree --export-sparse
```

This runs feature extraction, matching, mapper, and then exports:
- `work_colmap_openmvs/sparse_ply/points3D.ply` (for measurement)
- `work_colmap_openmvs/sparse_txt/*.txt` (for provenance)
- `work_colmap_openmvs/scale/MANIFEST.txt` (metadata)

### Step 2: Measure TWO Features on Laptop

Copy `sparse_ply/points3D.ply` to your laptop:

```bash
scp spartan:/path/to/work_colmap_openmvs/sparse_ply/points3D.ply ./
```

Open in **CloudCompare** or **MeshLab** and make TWO independent measurements:

#### Measurement 1 (e.g., base diameter)
1. Select two points on opposite sides of pottery base
2. Use "Point Picking" or "Measure Distance" tool
3. Note **reconstructed distance**: `0.0543271` units (copy full precision!)
4. Know **real-world distance**: `100mm = 0.100m`

#### Measurement 2 (e.g., pottery height)
1. Select two points (bottom to top of pottery)
2. Measure distance
3. Note **reconstructed distance**: `0.0814906` units
4. Know **real-world distance**: `150mm = 0.150m`

**Why two measurements?**
They validate each other. If both give the same scale (±2%), your measurements are correct! This catches:
- Unit errors (forgot to convert mm → m)
- Wrong features (measured diameter vs circumference)
- Typos (swapped real and reconstructed values)

### Step 3: Apply Scale on Spartan

Create measurement file with BOTH measurements:

```bash
cd /path/to/work_colmap_openmvs/scale
cat > measurement.env <<EOF
# Measurement 1: Base diameter
d1_real_m=0.100
d1_rec_units=0.0543271

# Measurement 2: Pottery height
d2_real_m=0.150
d2_rec_units=0.0814906
EOF
```

Apply scale:

```bash
python pipeline/bin/scale_apply.py \
  --work /path/to/work_colmap_openmvs
```

This will:
- Compute scale 1: `0.100 / 0.0543271 = 1.8406...`
- Compute scale 2: `0.150 / 0.0814906 = 1.8407...`
- Check agreement: `0.005% difference` ✓
- Use mean scale: `1.8406...`
- Apply to sparse: `sparse/0` → `sparse_scaled/0`
- Regenerate dense: `dense_scaled/`

**If scales disagree >2%**, the script aborts with an error showing both values.

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

## Detailed Workflow

### Alternative: Manual Export

If you didn't use `--export-sparse` during COLMAP run:

```bash
python pipeline/bin/scale_export.py \
  --work /path/to/work_colmap_openmvs
```

### Verify Scale

After applying scale, verify in CloudCompare:

```bash
# Open fused.ply from dense_scaled
# Measure the same features you measured before
# Should now read ~0.100m and ~0.150m instead of 0.054 and 0.081
```

### Reapply Scale

If measurements disagree or you made a mistake, update `measurement.env` and rerun:

```bash
# Edit measurement.env with corrected values
nano work_colmap_openmvs/scale/measurement.env

# Reapply
python pipeline/bin/scale_apply.py --work /path/to/work_colmap_openmvs
```

The script will regenerate `sparse_scaled/` and `dense_scaled/`.

**Common fixes:**
- **Scales disagree >2%**: Check you measured same features in PLY and real world
- **One scale way off**: Check for unit errors (mm vs m)
- **Both scales too high/low**: Check you read full precision from viewer

## Masking Considerations

### During Scale Measurement

If you're measuring the pottery base, ensure it's visible in the sparse:
- **Option 1**: Don't use masks during Stage 1
- **Option 2**: Use masks that leave a thin band of base exposed

### After Scale Application

Once scale is applied, you can rerun Stage 1 with production masks:

```bash
python pipeline/bin/run_colmap.py /path/to/tree \
  --mask-path work_colmap_openmvs/masks_user \
  --rebuild-from-matching
```

This will use the masked images but you've already captured scale.

## Provenance

All scale information is logged:

- `scale/measurement.env` - Your input measurements
- `scale/SCALE.txt` - Computed scale factor
- `scale/scale_log.txt` - Full calculation details (both measurements, agreement %)
- `scale/MANIFEST.txt` - Export metadata
- `pipeline_RUNLOG.txt` - Audit trail entry

Example `scale_log.txt`:
```
======================================================================
MANUAL SCALE APPLICATION (DUAL-MEASUREMENT VALIDATION)
======================================================================

Application time: 2025-11-08T12:34:56

Measurement 1:
  Real-world distance:    0.100000 m (100.0 mm)
  Reconstructed distance: 0.054327 units
  Scale 1:                1.840649285

Measurement 2:
  Real-world distance:    0.150000 m (150.0 mm)
  Reconstructed distance: 0.081491 units
  Scale 2:                1.840736921

Validation:
  Agreement:              0.005% difference
  Status:                 PASS

Applied scale factor:     1.840693103 (mean of both)
```

## Troubleshooting

### "Scales disagree by X%"

This is the most common error and means your measurements don't match.

**Check:**
- Did you measure the SAME features in the PLY and real world?
  - ✓ Base diameter in both
  - ✗ Base diameter in PLY, base circumference in reality
- Are units consistent?
  - ✓ `d1_real_m=0.100` (100mm converted to metres)
  - ✗ `d1_real_m=100` (forgot to convert mm to m)
- Did you read full precision from viewer?
  - ✓ `d1_rec_units=0.0543271`
  - ✗ `d1_rec_units=0.054` (rounded too much)

**If scales disagree <5%:**
Probably rounding or slightly different measurement points. Usually safe to increase tolerance if you're confident.

**If scales disagree >10%:**
Definitely a unit error or wrong feature. Recheck measurements.

### "Scale factor outside sane bounds"

One or both scales are outside [0.01, 100]:
- Did you use metres for `d_real_m`? (100mm = 0.100, not 100)
- Did you swap real and reconstructed values?
- Is the reconstructed distance reasonable? (Should be 0.01-10 for pottery)

### "Measurement file not found" or "Missing: d2_real_m"

Create the file with all four required values:

```bash
cd work_colmap_openmvs/scale
cat > measurement.env <<EOF
# Measurement 1
d1_real_m=0.100
d1_rec_units=0.0543271

# Measurement 2
d2_real_m=0.150
d2_rec_units=0.0814906
EOF
```

### Dense workspace not regenerated

Check logs in `scale/scale_log.txt` for errors. Common issues:
- Image path not found (use `--image-path`)
- COLMAP not in PATH (use `--colmap /path/to/colmap`)

## Why Scale Before Dense?

Scaling after OpenMVS mesh export is tempting but problematic:
- Camera poses remain in old units
- Depth maps need recomputation
- Texturing may fail
- Hard to re-export with correct scale

By scaling the sparse and regenerating dense, the entire pipeline is coherent.

## Best Practices

### Measurement Selection

✓ **DO**:
- Measure orthogonal features (diameter and height)
- Use clearly defined points (edges, corners)
- Measure at maximum extent (full diameter, not partial)
- Copy full precision from viewer (e.g., 0.0543271, not 0.054)

✗ **DON'T**:
- Measure curved distances (use straight-line only)
- Measure occluded or uncertain points
- Round values (loses accuracy)
- Mix units (all must be metres for real-world)

### Measurement Tools

**CloudCompare**:
1. Tools → Point Picking
2. Click two points
3. Distance shown in console
4. Copy full number (right-click → Copy)

**MeshLab**:
1. Edit → Select Vertices
2. Click two points
3. View → Show Measured Coords
4. Note distance value

### Configuration

Update `pipeline_config.yaml` for custom tolerances:

```yaml
manual_scale:
  enabled: true                     # Enable workflow
  sanity_bounds:
    min_scale: 0.01
    max_scale: 100.0
  agreement_tolerance_pct: 2.0      # Increase if needed (e.g., 5.0)
  prefer_scaled_sparse: true
```

## Advanced Usage

### Custom Image Path

If images aren't auto-detected:

```bash
python pipeline/bin/scale_apply.py \
  --work work_colmap_openmvs \
  --image-path /path/to/images
```

### Custom COLMAP Executable

```bash
python pipeline/bin/scale_apply.py \
  --work work_colmap_openmvs \
  --colmap /custom/path/to/colmap
```

### Batch Processing

For multiple trees:

```bash
for tree in tree1 tree2 tree3; do
  python pipeline/bin/run_colmap.py $tree --export-sparse
done

# Measure all PLYs, create measurement files

for tree in tree1 tree2 tree3; do
  python pipeline/bin/scale_apply.py --work ${tree}/work_colmap_openmvs
done
```

## Comparison: Manual Scale vs Coded Targets

| Feature | Manual Scale | Coded Targets |
|---------|-------------|---------------|
| **Setup** | None (use existing sparse) | Print/place board |
| **User effort** | 4 min (2 measurements) | Minimal after setup |
| **Accuracy** | ±0.5% (validated with 2 measurements) | ±2mm (board-dependent) |
| **Error detection** | Automatic (2 measurements must agree) | None |
| **Automation** | Semi-automated | Fully automated |
| **Flexibility** | Measure any feature | Limited to board |
| **Overhead** | ~3 min per tree | +30 sec per tree |
| **Dependencies** | None (laptop + viewer) | OpenCV ArUco module |
| **Use case** | Retrospective scaling | Production pipeline |

**Recommendation**: Use manual scale for existing datasets or one-off projects. Use coded targets for large-scale production pipelines.

## FAQ

**Q: Can I use just one measurement instead of two?**
A: No, dual measurements are required for validation. This dramatically reduces errors.

**Q: What if I only know one dimension?**
A: Measure it twice in different locations (e.g., diameter at top and bottom).

**Q: Can I measure the same feature twice?**
A: Not recommended. Different features provide better validation.

**Q: How precise should my real-world measurements be?**
A: ±1mm for dimensions 50-200mm is sufficient. The dual-measurement validation catches major errors.

**Q: Can I reuse the scale on other trees of the same pottery?**
A: No, each reconstruction has its own arbitrary scale. Measure each one.

**Q: What if the pottery broke between capture and measurement?**
A: Measure features visible in both the PLY and current state (or use historical measurements).

## References

- COLMAP model_converter: https://colmap.github.io/cli.html#model-converter
- COLMAP model_transformer: https://colmap.github.io/cli.html#model-transformer
- CloudCompare Point Picking: https://www.cloudcompare.org/doc/wiki/index.php?title=Point_picking
- MeshLab Measuring Tool: https://www.meshlab.net/

## Support

For issues:
1. Check troubleshooting section above
2. Review `scale/scale_log.txt` for detailed error messages
3. Verify both measurements are correct and units are in metres
4. Open an issue with log files and measurement.env
