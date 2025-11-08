# User-Edited Model Masking

Generate pixel-perfect binary masks from a user-edited 3D model to exclude turntable rigs, clamps, and background from COLMAP/OpenMVS processing.

## Overview

The maskbuild system provides a two-phase workflow:

1. **Phase 1 (user-init)**: Builds a quick, coarse 3D model for user editing
2. **Phase 2 (user-project)**: Projects the edited model into all camera views to generate masks

The masks ensure that feature detection, matching, and dense reconstruction only process pottery sherd pixels, ignoring stands, clamps, rods, and background.

## Prerequisites

- Completed JPEG photo capture (120-170 images recommended)
- Python 3.10+ with required packages (trimesh, opencv-python, scipy)
- COLMAP 3.9+ with CUDA support
- OpenMVS 2.3+ with CUDA support
- A mesh editing tool (MeshLab, Blender, or CloudCompare)

## Quick Start

### Step 1: Build Coarse Model

```bash
python pipeline/bin/maskbuild.py user-init \
  --images /path/to/pottery_tree/photos_jpg \
  --work /path/to/pottery_tree/work_colmap_openmvs
```

**What this does:**
- Runs fast COLMAP sparse reconstruction (low-res, lenient thresholds)
- Generates low-resolution dense point cloud
- Creates coarse mesh with ~50% simplification
- Exports `coarse_model.ply` for editing
- Writes instructions to `README_EDIT_MESH.txt`

**Time:** ~5-10 minutes on A100 for 150 images

**Output:**
```
work_colmap_openmvs/
└── mask_build_user/
    ├── coarse_model.ply          ← Edit this file
    ├── README_EDIT_MESH.txt      ← Instructions
    ├── sparse/0/                 ← Quick sparse model
    ├── dense/                    ← Low-res images
    └── RUNLOG.txt                ← Phase 1 log
```

### Step 2: Edit the Mesh

1. **Download** `coarse_model.ply` to your workstation

2. **Open** in a mesh editor:
   - **MeshLab** (recommended): File → Import Mesh
   - **Blender**: File → Import → Stanford (.ply)
   - **CloudCompare**: File → Open

3. **Delete** all non-sherd geometry:
   - Turntable surface
   - Support rods and clamps
   - Background walls
   - Reconstruction artifacts
   - Thin floating pieces

4. **Keep** only pottery surfaces:
   - All visible sherd fragments
   - Internal and external surfaces

5. **Save** as `edited_model.ply`
   - Use PLY format (binary or ASCII)
   - **DO NOT** apply any transformations
   - **DO NOT** translate, rotate, or scale
   - Only delete geometry

6. **Upload** to `work_colmap_openmvs/mask_build_user/edited_model.ply`

#### Critical Warnings

⚠️ **DO NOT transform the mesh!**
- No translation, rotation, or scaling
- No coordinate system changes
- Only delete unwanted geometry

The mesh must remain in the same coordinate frame as the sparse reconstruction for mask projection to work correctly.

### Step 3: Generate Masks

```bash
python pipeline/bin/maskbuild.py user-project \
  --work /path/to/pottery_tree/work_colmap_openmvs \
  --mesh work_colmap_openmvs/mask_build_user/edited_model.ply
```

**What this does:**
- Validates edited mesh is in correct coordinate frame
- Loads COLMAP camera parameters from Phase 1
- Projects mesh into each camera view
- Rasterizes silhouette with 2-pixel padding
- Writes one PNG mask per input image
- Computes coverage statistics
- Generates manifest with SHA-256 checksums

**Time:** ~1-2 minutes

**Output:**
```
work_colmap_openmvs/
├── masks_user/                   ← Binary masks (PNG)
│   ├── IMG_0001.png
│   ├── IMG_0002.png
│   └── ...
├── masks_manifest.json           ← Checksums & metadata
└── masks_report.txt              ← Coverage summary
```

### Step 4: Rerun Pipeline with Masks

The main COLMAP pipeline automatically detects and uses `masks_user/`:

```bash
python pipeline/bin/run_colmap.py /path/to/pottery_tree
```

Or explicitly specify the mask path:

```bash
python pipeline/bin/run_colmap.py /path/to/pottery_tree \
  --mask-path work_colmap_openmvs/masks_user
```

**Expected improvements:**
- ✓ More images registered (5-15% increase)
- ✓ Lower reprojection errors (10-20% reduction)
- ✓ Features only on sherd surfaces
- ✓ Cleaner sparse point cloud
- ✓ Better dense reconstruction (no background)

## Advanced Usage

### Custom Padding and Erosion

Control mask edge treatment:

```bash
python pipeline/bin/maskbuild.py user-project \
  --work /path/to/pottery_tree/work_colmap_openmvs \
  --mesh edited_model.ply \
  --pad-pixels 3 \        # More padding for thin rims (default: 2)
  --erosion 1             # Erode 1 pixel to prevent halos (default: 0)
```

**Padding** (dilation):
- Protects thin edges and rims
- Default: 2 pixels
- Increase if rims are being clipped in masks

**Erosion**:
- Shrinks mask to prevent dark halos
- Default: 0 (no erosion)
- Set to 1 if you see dark edge artifacts in reconstruction

### Configuration

Edit `pipeline/config/pipeline_config.yaml`:

```yaml
maskbuild:
  quick_build:
    # Adjust if Phase 1 fails to create single model
    max_num_features: 8000          # Increase for better registration
    mapper_min_inliers: 12          # Lower for difficult captures
    densify_resolution_level: 2     # 0=full res (slower), 2=1/4 res (faster)

  projection:
    # Mask generation parameters
    pad_pixels: 2                   # Default padding
    erosion: 0                      # Default erosion
    # Frame validation thresholds
    bbox_tolerance: 0.1             # Allow 10% bbox shrinkage
    distance_tolerance: 0.05        # Allow 5% distance error
```

## Troubleshooting

### Phase 1: "Multiple models detected"

**Problem:** COLMAP created multiple sparse reconstructions

**Cause:** Insufficient image overlap between photo groups

**Solutions:**
1. Check photo overlap (should be 60-80% between images)
2. Ensure overlap between turntable height rings
3. Increase features in config:
   ```yaml
   maskbuild:
     quick_build:
       max_num_features: 12000  # Increase from 8000
   ```
4. Lower thresholds:
   ```yaml
   maskbuild:
     quick_build:
       mapper_min_inliers: 8    # Lower from 12
   ```

### Phase 2: "Frame validation failed"

**Problem:** Edited mesh appears to have been moved/rotated/scaled

**Error message:**
```
Bounding box overlap 0.45 < 0.90
Distance error 0.15 > 0.05
```

**Cause:** Mesh was transformed during editing

**Solutions:**
1. **Re-edit** the mesh without applying transformations
2. In MeshLab: Don't use Filters → Normals/Curvature → Transform
3. In Blender: Don't move/rotate/scale in Object mode
4. Only **delete** faces/vertices, don't transform

### Phase 2: Coverage outliers

**Problem:** Some masks have <5% or >95% coverage

**Check:** `work_colmap_openmvs/masks_report.txt`

**Causes:**
1. Mesh not visible in some camera views (normal for turntable)
2. Over-aggressive editing (too much geometry deleted)
3. Mesh projection errors

**Solutions:**
1. Review outlier list in report
2. Check if outliers are from similar angles
3. Re-edit mesh to include more pottery surfaces
4. Acceptable if <10% of images are outliers

### Phase 1: Takes too long (>15 minutes)

**Problem:** Quick build is slow

**Solutions:**
1. Reduce resolution in config:
   ```yaml
   maskbuild:
     quick_build:
       max_image_size: 1024         # Lower from 2048
       densify_resolution_level: 3  # 1/8 res instead of 1/4
   ```
2. Fewer features:
   ```yaml
   maskbuild:
     quick_build:
       max_num_features: 6000       # Lower from 8000
   ```

### Masks don't improve reconstruction

**Problem:** Running COLMAP with masks shows minimal improvement

**Diagnostic checklist:**
1. ✓ Masks actually used? Check COLMAP log for `ImageReader.mask_path`
2. ✓ Coverage reasonable? Check median coverage in `masks_report.txt` (should be 20-60%)
3. ✓ Masks match images? Same basename, .png extension?
4. ✓ Background truly excluded? View a few masks to verify

**Solutions:**
1. Increase padding if sherds are being clipped:
   ```bash
   --pad-pixels 4
   ```
2. Check mask quality by overlaying on images
3. Re-edit mesh to better isolate sherds

## File Reference

### Phase 1 Outputs

| File | Description |
|------|-------------|
| `mask_build_user/coarse_model.ply` | Simplified mesh for editing |
| `mask_build_user/edited_model.ply` | User-edited mesh (you create this) |
| `mask_build_user/sparse/0/` | Quick COLMAP sparse reconstruction |
| `mask_build_user/dense/` | Low-res undistorted images |
| `mask_build_user/RUNLOG.txt` | Phase 1 & 2 logs |
| `mask_build_user/README_EDIT_MESH.txt` | Editing instructions |

### Phase 2 Outputs

| File | Description |
|------|-------------|
| `masks_user/*.png` | Binary masks (255=keep, 0=discard) |
| `masks_manifest.json` | Checksums, metadata, coverage stats |
| `masks_report.txt` | Human-readable summary |
| `mask_build_user/frame_validation.json` | Coordinate frame check results |

## Integration with Main Pipeline

### Automatic Detection

The main COLMAP pipeline automatically detects `masks_user/`:

```python
# run_colmap.py automatically checks:
auto_mask_dir = work_dir / "masks_user"
if auto_mask_dir.exists():
    # Use masks
```

### Manual Override

Specify a different mask directory:

```bash
python pipeline/bin/run_colmap.py /path/to/tree \
  --mask-path /custom/mask/directory
```

### Disable Masks

Remove or rename `masks_user/`:

```bash
mv work_colmap_openmvs/masks_user work_colmap_openmvs/masks_user.backup
```

## Best Practices

### Editing Tips

1. **Remove conservatively**: Start by removing obvious non-sherd geometry
2. **Check all angles**: Rotate mesh to see all sides
3. **Clean up artifacts**: Remove thin floating geometry from reconstruction errors
4. **Don't be perfect**: The coarse model is just for masking, not final output
5. **Leave a margin**: Better to include a bit of background than clip sherds

### Performance Tips

1. **Phase 1 is cached**: Rerunning user-init reuses existing reconstruction if present
2. **Edit locally**: Download coarse_model.ply to workstation for faster editing
3. **Batch processing**: Can run Phase 2 on multiple edited meshes sequentially
4. **Resume support**: Phase 1 saves state, can resume if interrupted

### Validation

After generating masks, verify:

```bash
# Check coverage statistics
cat work_colmap_openmvs/masks_report.txt

# Count masks
ls work_colmap_openmvs/masks_user/*.png | wc -l

# View manifest
jq '.statistics.coverage' work_colmap_openmvs/masks_manifest.json
```

Expected coverage:
- **Mean**: 30-50% (pottery sherds are typically 30-50% of frame)
- **Median**: 25-55%
- **Min**: >5% (unless viewing angle issue)
- **Max**: <95% (unless close-up shots)

## Technical Details

### Coordinate Frames

- **World frame**: COLMAP sparse reconstruction coordinates
- **Camera frame**: Per-image coordinate system
- **Validation**: Bounding box overlap >90%, distance errors <5%

### Projection Algorithm

1. Transform mesh vertices from world to camera frame
2. Filter back-facing geometry (Z ≤ 0)
3. Project 3D vertices to 2D image plane using camera intrinsics
4. Rasterize triangles using OpenCV fillConvexPoly
5. Apply morphological dilation (padding)
6. Optionally apply erosion (halo prevention)

### Mask Format

- **Format**: PNG, 8-bit grayscale
- **Values**: 255=keep pixel, 0=discard pixel
- **Size**: Matches input image dimensions exactly
- **Naming**: Same basename as image, .png extension
- **Compression**: Lossless PNG compression

## Examples

### Complete Workflow

```bash
# Step 1: Build coarse model
python pipeline/bin/maskbuild.py user-init \
  --images /data/pottery_tree_001/photos_jpg \
  --work /data/pottery_tree_001/work_colmap_openmvs

# Step 2: Edit mesh (manual)
# - Download coarse_model.ply
# - Edit in MeshLab
# - Save as edited_model.ply
# - Upload to mask_build_user/

# Step 3: Generate masks
python pipeline/bin/maskbuild.py user-project \
  --work /data/pottery_tree_001/work_colmap_openmvs \
  --mesh work_colmap_openmvs/mask_build_user/edited_model.ply

# Step 4: Verify coverage
cat work_colmap_openmvs/masks_report.txt

# Step 5: Rerun COLMAP with masks
python pipeline/bin/run_colmap.py /data/pottery_tree_001
```

### Batch Processing Multiple Trees

```bash
for tree in pottery_tree_*; do
  echo "Processing $tree..."

  # Phase 1
  python pipeline/bin/maskbuild.py user-init \
    --images $tree/photos_jpg \
    --work $tree/work_colmap_openmvs

  echo "Edit $tree/work_colmap_openmvs/mask_build_user/coarse_model.ply"
  echo "Press Enter when done..."
  read

  # Phase 2
  python pipeline/bin/maskbuild.py user-project \
    --work $tree/work_colmap_openmvs \
    --mesh $tree/work_colmap_openmvs/mask_build_user/edited_model.ply

  echo "✓ Masks generated for $tree"
done
```

## References

- [COLMAP Documentation](https://colmap.github.io/)
- [OpenMVS Documentation](https://github.com/cdcseacave/openMVS)
- Pipeline configuration: `pipeline/config/pipeline_config.yaml`
- Implementation plan: `MASKBUILD_IMPLEMENTATION_PLAN.md`

## Support

For issues or questions:
1. Check `masks_report.txt` for diagnostics
2. Review `mask_build_user/RUNLOG.txt` for errors
3. Validate mask coverage statistics
4. Consult troubleshooting section above
