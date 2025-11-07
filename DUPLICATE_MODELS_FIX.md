# Fixing Duplicate 3D Model Issues

## Problem Summary

If you're seeing **two duplicate 3D models side-by-side** in your photogrammetry output, this is caused by **COLMAP failing to register all photos into a single cohesive reconstruction**. This is NOT an intentional feature - it's a symptom of registration failure.

## Root Causes

### 1. Multiple COLMAP Sparse Models (Most Common)
When COLMAP can't connect all images through feature matching, it creates separate reconstructions:
- `sparse/0/` - First model (e.g., images from one photo group)
- `sparse/1/` - Second model (e.g., images from another photo group)

**Why this happens:**
- Insufficient image overlap between photo groups (e.g., between height rings on turntable)
- Low texture on pottery surfaces making feature matching difficult
- Lighting changes between photo sessions
- Too strict matching thresholds

### 2. Disconnected Mesh Components
Even if COLMAP creates a single sparse model, the dense reconstruction might have gaps:
- Some images weren't registered properly
- Dense reconstruction has missing data
- "Holes" between different photo groups result in disconnected mesh pieces

## Fixes Implemented

### Configuration Changes

The following changes have been made to `pipeline/config/pipeline_config.yaml`:

#### 1. Improved Feature Detection
```yaml
colmap:
  feature_extractor:
    max_num_features: 16000  # Increased from 12000
```
**Benefit:** More features per image → better matching between photos

#### 2. More Lenient Mapper Thresholds
```yaml
mapper:
  init_min_tri_angle: 2.0          # Reduced from 4.0
  abs_pose_min_num_inliers: 15     # Reduced from 20
  abs_pose_min_inlier_ratio: 0.18  # Reduced from 0.20
  multiple_models: 0               # NEW: Error if multiple models created
```
**Benefit:**
- Accepts more images with smaller baselines (common in turntable captures)
- Accepts images with fewer good matches
- **multiple_models: 0** forces the pipeline to fail if registration is incomplete (prevents bad output)

#### 3. Better Dense Reconstruction
```yaml
openmvs:
  densify:
    number_views: 8          # Increased from 6
    number_views_fuse: 5     # Increased from 4
```
**Benefit:** Uses more supporting views → better coverage → fewer gaps in mesh

#### 4. Stricter Validation Filtering
```yaml
validation:
  min_component_vertices: 5000  # Increased from 1000
  min_component_area: 10.0      # Increased from 1.0
```
**Benefit:** Filters out small noise artifacts that could be mistaken for separate models

### Code Changes

#### Enhanced Error Detection in `run_colmap.py`
The COLMAP script now:
1. **Detects multiple models** after sparse reconstruction
2. **Errors out** if `multiple_models: 0` (recommended setting)
3. **Provides actionable recommendations** in the error message

Example error message:
```
ERROR: Found 2 separate COLMAP models in work_colmap_openmvs/sparse.
This indicates that not all images were registered into a single reconstruction.
Models found: ['0', '1'].
Consider: (1) lowering init_min_tri_angle, (2) lowering abs_pose_min_num_inliers,
(3) increasing max_num_features, or (4) improving image overlap.

Set 'colmap.mapper.multiple_models: 1' in config to allow multiple models
(not recommended - fix registration instead).
```

## Diagnostic Tool

A new diagnostic script has been added: `pipeline/bin/diagnose_duplicates.py`

### Usage

```bash
# Basic diagnosis
python pipeline/bin/diagnose_duplicates.py /path/to/work_colmap_openmvs

# Detailed output
python pipeline/bin/diagnose_duplicates.py /path/to/work_colmap_openmvs --verbose

# JSON output for parsing
python pipeline/bin/diagnose_duplicates.py /path/to/work_colmap_openmvs --json
```

### What It Checks

1. **COLMAP Models**: Counts sparse models in `sparse/` directory
2. **Validation Report**: Analyzes mesh components for disconnected pieces
3. **Mesh Files**: Checks for exported sherd files
4. **Registration Statistics**: (Future) Parses logs for registration rates

### Example Output

```
🔍 Diagnosing: /data/projects/pottery_001/work_colmap_openmvs

======================================================================
DIAGNOSTIC RESULTS
======================================================================

📊 COLMAP Models: [WARNING]
   Found 2 sparse model(s)
   - Model 0: 15.4 MB
   - Model 1: 8.2 MB

📊 Validation Report: [WARNING]
   Found 12 components, 2 passed validation
   - sherd_001.ply: 3,245,891 vertices (PASS)
   - sherd_002.ply: 2,987,443 vertices (PASS)

📊 Mesh Files: [OK]
   Found 2 exported sherds

======================================================================
RECOMMENDATIONS
======================================================================

⚠️  MULTIPLE COLMAP MODELS DETECTED
   Found 2 separate reconstructions in sparse/ directory.
   This means COLMAP couldn't register all images into one model.

   Fixes:
   1. Ensure guided_matching: 1 in config (enables cross-ring connections)
   2. Lower init_min_tri_angle to 2.0 (currently 4.0)
   3. Lower abs_pose_min_num_inliers to 15 (currently 20)
   4. Increase max_num_features to 16000 (currently 12000)
   5. Check image overlap between photo groups
```

## When to Use Each Solution

### If You're Starting Fresh
1. **Use the updated configuration** (changes already applied)
2. Run your photogrammetry pipeline normally
3. If errors occur about multiple models, see "Troubleshooting" below

### If You Already Have Duplicate Models
1. **Run the diagnostic tool** to understand the issue:
   ```bash
   python pipeline/bin/diagnose_duplicates.py /path/to/work_colmap_openmvs --verbose
   ```

2. **Check the recommendations** provided by the tool

3. **Re-run the pipeline** with updated settings:
   ```bash
   # Delete old sparse reconstruction
   rm -rf /path/to/work_colmap_openmvs/sparse

   # Re-run COLMAP with new settings
   python pipeline/bin/run_colmap.py /path/to/pottery_tree
   ```

## Troubleshooting

### Error: "Found N separate COLMAP models"

This means your images aren't being registered into a single model. Try these in order:

#### Option 1: Further Relax Thresholds
Edit `pipeline/config/pipeline_config.yaml`:
```yaml
mapper:
  init_min_tri_angle: 1.5          # Even more lenient
  abs_pose_min_num_inliers: 12     # Accept more images
```

#### Option 2: Increase Features
```yaml
feature_extractor:
  max_num_features: 20000          # More features = better matching
  max_image_size: 5120             # Larger images = more features
```

#### Option 3: Check Your Photos
- **Image overlap**: Each photo should overlap with neighbors by 60-80%
- **Ring overlap**: If using turntable, ensure vertical overlap between height rings
- **Lighting consistency**: Avoid dramatic lighting changes between sessions
- **Photo quality**: Ensure images are sharp and well-exposed

#### Option 4: Use Sequential Matcher (For Turntable Only)
For ordered turntable captures, sequential matching might work better:
```yaml
# Replace exhaustive_matcher with sequential_matcher
colmap:
  sequential_matcher:
    overlap: 15              # Match each image with 15 neighbors
    loop_detection: 1        # Enable loop closure
    vocab_tree_path: ""      # Optional: path to vocabulary tree
```

Note: This requires modifying `run_colmap.py` to use sequential matcher instead of exhaustive.

### Still Seeing Duplicates After Fixes

If COLMAP reports a single model but you still see duplicates:

1. **Check mesh topology:**
   ```bash
   # Open in MeshLab or CloudCompare to visualize
   meshlab work_colmap_openmvs/scene_dense_mesh_refine.ply
   ```

2. **Check validation report:**
   ```bash
   cat work_colmap_openmvs/validation_report.csv
   ```
   Look for multiple large components (>100k vertices each)

3. **Increase component filtering:**
   ```yaml
   validation:
     min_component_vertices: 10000   # More aggressive filtering
     min_component_area: 20.0        # Filter small artifacts
   ```

## Understanding Registration Success

### Good Registration
- Single sparse model in `sparse/0/`
- 95-100% of images registered
- Dense point cloud with few holes
- Single large mesh component (or expected number for fragmented pottery)

### Poor Registration
- Multiple sparse models (`sparse/0/`, `sparse/1/`, etc.)
- <90% of images registered
- Disconnected components in mesh
- Duplicate or incomplete geometry

## Next Steps

1. **Test with your data:**
   - Run pipeline with updated configuration
   - Use diagnostic tool to verify success

2. **Monitor logs:**
   - Check `pipeline/logs/*colmap*.log` for warnings
   - Look for registration statistics

3. **Iterate if needed:**
   - Use diagnostic recommendations
   - Adjust thresholds based on your specific capture setup

## Technical Details

### Why Exhaustive Matching?
Turntable captures require **exhaustive matching** (not sequential) because:
- Images from different height rings need to match
- Loop closure is essential for full coverage
- Sequential matching might miss cross-ring correspondences

### Why guided_matching: 1?
Guided matching:
- Uses geometric constraints from already-matched images
- Stabilizes matching across height variations
- Reduces false matches on low-texture pottery surfaces

### Why multiple_models: 0?
Setting this to 0 (error on multiple models) is recommended because:
- Forces you to fix the root cause
- Prevents silently producing incomplete reconstructions
- Duplicate models are almost never desired output

If you need to allow multiple models temporarily (not recommended):
```yaml
mapper:
  multiple_models: 1  # Allow but warn
```

## References

- [COLMAP Documentation](https://colmap.github.io/)
- [OpenMVS Documentation](https://github.com/cdcseacave/openMVS)
- Pipeline configuration: `pipeline/config/pipeline_config.yaml`
- COLMAP script: `pipeline/bin/run_colmap.py`
- Diagnostic tool: `pipeline/bin/diagnose_duplicates.py`

## Support

If you continue to experience issues:
1. Run diagnostic tool with `--verbose` flag
2. Check COLMAP logs in `pipeline/logs/`
3. Review photo capture methodology
4. Consider using test data to validate pipeline changes
