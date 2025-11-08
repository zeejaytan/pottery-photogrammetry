# Mesh Sherd Splitter

This guide explains how to split a large pottery-tree mesh into individual sherd meshes for easier review and processing.

## Overview

The mesh splitter takes the refined OpenMVS mesh (typically multi-million faces combining all sherds on the tree) and:
1. **Breaks** thin non-physical bridges between sherds
2. **Separates** into connected components via topology
3. **Filters** noise (small wisps, rods, clamp artifacts)
4. **Writes** one mesh per sherd to `split_sherds/` directory
5. **Records** full provenance in manifest and pipeline log

**Key Benefits**:
- Easier review on laptop (open individual sherds instead of massive combined file)
- Removes scan artifacts (clamps, hardware, thin bridges)
- Non-destructive (preserves original refined mesh)
- Parameterized (rerun with different thresholds without re-meshing)
- Full audit trail (manifest records all decisions)

## When to Use

**Use mesh splitter when:**
- OpenMVS produces a combined mesh with all sherds
- You want to review individual sherds on laptop
- Clamp or stand artifacts survived masking
- You need per-sherd textures or exports

**When NOT needed:**
- Single sherd captures (no splitting necessary)
- Working on HPC only (can handle large files)
- Sherds already separated in earlier workflow

## Quick Start

### Step 1: Run OpenMVS Through RefineMesh

```bash
# After COLMAP completes, run OpenMVS
InterfaceCOLMAP -i work_colmap_openmvs/dense_scaled -o scene.mvs
DensifyPointCloud -i scene.mvs -o scene_dense.mvs
ReconstructMesh -i scene_dense.mvs -o mesh.mvs
RefineMesh -i mesh.mvs -o mesh_refine.mvs

# Export refined mesh
RefineMesh -i mesh_refine.mvs --export-type ply -o scene_dense_mesh_refine.ply
```

This produces the refined mesh: `dense_scaled/scene_dense_mesh_refine.ply`

### Step 2: Split Into Sherds (Interactive)

```bash
cd /data/gpfs/projects/punim2657/Rabati2025/tree_001/work_colmap_openmvs

python3 /path/to/pipeline/bin/split_sherds.py \
    --mesh dense_scaled/scene_dense_mesh_refine.ply
```

**Output:**
```
======================================================================
MESH SHERD SPLITTER
======================================================================
Input mesh: dense_scaled/scene_dense_mesh_refine.ply
Output dir: split_sherds

Step 1: Loading mesh...
✓ Loaded mesh: 1234567 vertices, 2468900 faces
  Surface area: 0.523456 sq units

Step 2: Breaking long-edge bridges (quantile=0.995)...
Edge length quantile 99.5%: 0.012345 units
Median edge length: 0.000234 units
Removed 1234 faces with long edges

Step 3: Computing connected components...
Found 12 connected components

Step 4: Validating thresholds...
✓ Thresholds are valid

Step 5: Filtering noise components...
Median component area: 0.045678 sq units
Kept 8/12 components after filtering

Step 6: Writing 8 sherd meshes...
Wrote sherd_01_comp000.ply: 345678 vertices, 690123 faces
Wrote sherd_02_comp001.ply: 234567 vertices, 468900 faces
...

======================================================================
MESH SPLITTING COMPLETE
======================================================================
Components found:   12
Components kept:    8
Components dropped: 4
Output directory:   split_sherds
Time elapsed:       142.3s
```

### Step 3: Review Sherds on Laptop

```bash
# Pull sherds to laptop
rsync -avz spartan:/path/to/work_colmap_openmvs/split_sherds/ ./local_sherds/

# Open individual sherds in CloudCompare, MeshLab, or Blender
# Much faster than opening the 10M face combined mesh!
```

## Command-Line Options

### Basic Usage

```bash
# Use defaults (recommended for most pottery trees)
python3 split_sherds.py --mesh dense_scaled/scene_dense_mesh_refine.ply

# Custom output directory
python3 split_sherds.py \
    --mesh scene.ply \
    --out my_sherds

# Output as OBJ instead of PLY
python3 split_sherds.py \
    --mesh scene.ply \
    --format obj
```

### Filtering Options

```bash
# Adjust minimum face count (remove smaller wisps)
python3 split_sherds.py \
    --mesh scene.ply \
    --min-faces 8000

# Adjust maximum elongation (remove rod-like artifacts)
python3 split_sherds.py \
    --mesh scene.ply \
    --max-elongation 20.0

# Keep all components (only apply min-faces filter)
python3 split_sherds.py \
    --mesh scene.ply \
    --keep-all

# Keep only top N largest components
python3 split_sherds.py \
    --mesh scene.ply \
    --keep-top 10
```

### Bridge Breaking Options

```bash
# Adjust bridge-breaking threshold (more aggressive)
python3 split_sherds.py \
    --mesh scene.ply \
    --bridge-quantile 0.990  # Remove longest 1% instead of 0.5%

# Skip bridge breaking entirely
python3 split_sherds.py \
    --mesh scene.ply \
    --no-bridge-break
```

### Complete Example

```bash
python3 split_sherds.py \
    --mesh dense_scaled/scene_dense_mesh_refine.ply \
    --out split_sherds \
    --min-faces 5000 \
    --max-elongation 25.0 \
    --bridge-quantile 0.995 \
    --format ply
```

## Output Structure

After splitting, you'll have:

```
work_colmap_openmvs/
├── dense_scaled/
│   └── scene_dense_mesh_refine.ply  # Original (preserved)
└── split_sherds/
    ├── sherd_01_comp000.ply         # Largest component
    ├── sherd_02_comp001.ply         # Second largest
    ├── sherd_03_comp002.ply
    ├── ...
    ├── manifest.csv                  # Human-readable table
    └── manifest.json                 # Machine-readable metadata
```

**File Naming**: `sherd_{N}_comp{ID}.ply`
- `N` = Sequential sherd number (01, 02, 03...)
- `ID` = Original component ID from split (000, 001, 002...)

## Understanding the Filters

### 1. Long-Edge Bridge Breaking

**Purpose**: Sever thin, non-physical connections between sherds

**How it works**:
- Compute all edge lengths in mesh
- Find 99.5th percentile (default)
- Remove faces with edges above threshold

**Example**:
```
Median edge length: 0.000234 units
99.5% quantile:     0.012345 units (53x median)
→ Remove faces with edges > 0.012345 units
```

**When to adjust**:
- **More aggressive** (`--bridge-quantile 0.990`): If sherds still merged
- **Less aggressive** (`--bridge-quantile 0.998`): If real detail removed
- **Skip** (`--no-bridge-break`): If sherds already well-separated

### 2. Minimum Face Count

**Purpose**: Remove small dust, wisps, floating vertices

**Default**: 5000 faces

**Rationale**: Legitimate sherds typically have 50k-500k faces after refinement at default OpenMVS settings

**When to adjust**:
- **Lower** (`--min-faces 2000`): Small but legitimate rim fragments
- **Higher** (`--min-faces 10000`): Very dense meshes, only keep substantial sherds

### 3. Maximum Elongation

**Purpose**: Remove rod-like artifacts (clamps, stand legs)

**Metric**: `elongation = max_bbox_dim / min_bbox_dim`

**Default**: 25.0

**Examples**:
- Bowl sherd: elongation ~2-5
- Tall rim: elongation ~8-12
- Clamp rod: elongation ~30-100

**When to adjust**:
- **Lower** (`--max-elongation 20.0`): Aggressive clamp removal
- **Higher** (`--max-elongation 40.0`): Preserve very tall sherds
- **Disable** (`--keep-all`): No elongation filter

### 4. Keep-All vs Keep-Top

**`--keep-all`**:
- Apply only min-faces filter
- No elongation filter
- Use when you trust your masking

**`--keep-top N`**:
- Keep only N largest components by face count
- Useful for quick previews
- Example: `--keep-top 5` for 5 main sherds only

## Manifest Files

### CSV Manifest (`manifest.csv`)

Human-readable table with one row per component:

```csv
component_id,kept,rejection_reason,vertex_count,face_count,surface_area,bbox_x,bbox_y,bbox_z,elongation
0,True,,345678,690123,0.182345,0.12,0.14,0.08,3.45
1,True,,234567,468900,0.123456,0.10,0.11,0.09,4.12
2,False,too_few_faces (< 5000),456,912,0.000234,0.01,0.03,0.02,2.34
3,False,too_elongated (> 25.0),12345,24680,0.012345,0.02,0.45,0.01,45.2
```

**Fields**:
- `component_id`: Original component number from split
- `kept`: True if written to disk, False if rejected
- `rejection_reason`: Why component was dropped (if applicable)
- `vertex_count`: Number of vertices
- `face_count`: Number of triangles
- `surface_area`: Surface area in squared units (metric if scaled)
- `bbox_x, bbox_y, bbox_z`: Bounding box dimensions
- `elongation`: max_dim / min_dim ratio

### JSON Manifest (`manifest.json`)

Machine-readable with full metadata:

```json
{
  "input_mesh": "dense_scaled/scene_dense_mesh_refine.ply",
  "timestamp": "2025-11-08T15:30:00",
  "parameters": {
    "min_faces": 5000,
    "max_elongation": 25.0,
    "bridge_quantile": 0.995
  },
  "components": [...],
  "summary": {
    "total_components": 12,
    "kept_components": 8,
    "dropped_components": 4,
    "total_faces_kept": 2345678,
    "total_faces_dropped": 123222
  }
}
```

**Use cases**:
- Parse in Python for batch analysis
- Verify face count sums match input
- Track parameters for reproducibility

## Troubleshooting

### "Mesh file not found"

**Cause**: Wrong path to refined mesh

**Solution**:
```bash
# Check file exists
ls -lh dense_scaled/scene_dense_mesh_refine.ply

# Use absolute path if needed
python3 split_sherds.py \
    --mesh /full/path/to/scene_dense_mesh_refine.ply
```

### "min_faces would drop ALL components"

**Cause**: Threshold too high for mesh resolution

**Solution**:
```bash
# Check largest component in manifest
# Lower min-faces threshold
python3 split_sherds.py \
    --mesh scene.ply \
    --min-faces 2000  # Lower from default 5000
```

### Two Sherds Still Merged After Split

**Cause**: No thin bridge to break (sherds fused during reconstruction)

**Options**:

1. **Try more aggressive bridge breaking**:
   ```bash
   python3 split_sherds.py \
       --mesh scene.ply \
       --bridge-quantile 0.990  # Remove longest 1% instead of 0.5%
   ```

2. **Accept as single component**: Manifest will show unusually large area

3. **Manual separation** (if critical):
   - Open merged sherd in MeshLab/Blender
   - Manually select and delete connection
   - Export as two files

### Clamp Artifact Still in Output

**Cause**: Clamp has similar face count and elongation to sherds

**Solutions**:

1. **Lower max-elongation**:
   ```bash
   python3 split_sherds.py \
       --mesh scene.ply \
       --max-elongation 15.0  # Stricter rod filter
   ```

2. **Improve masking**: Re-run COLMAP with better masks to exclude clamp

3. **Manual removal**: Delete clamp mesh file from `split_sherds/`

### Legitimate Sherd Lost to Filtering

**Cause**: Small rim fragment below min-faces or unusually elongated sherd

**Solution**:
```bash
# Use --keep-all to disable elongation filter
python3 split_sherds.py \
    --mesh scene.ply \
    --keep-all \
    --min-faces 2000  # Lower threshold

# Or manually inspect dropped components in manifest.csv
# Look for rejection_reason and assess if legitimate
```

### "File is not a single mesh"

**Cause**: Input is scene with multiple objects or point cloud

**Solution**:
- Ensure input is a refined mesh (not point cloud)
- Check OpenMVS exported correctly to PLY
- Try OBJ format instead: `--format obj`

## Re-Running with Different Thresholds

You can iterate on split parameters without re-running OpenMVS:

```bash
# First attempt
python3 split_sherds.py --mesh scene.ply
# Result: Some sherds still merged

# Second attempt: More aggressive bridge breaking
python3 split_sherds.py \
    --mesh scene.ply \
    --bridge-quantile 0.990

# Third attempt: Keep small fragments too
python3 split_sherds.py \
    --mesh scene.ply \
    --bridge-quantile 0.990 \
    --min-faces 2000

# Fourth attempt: Custom output directory
python3 split_sherds.py \
    --mesh scene.ply \
    --bridge-quantile 0.990 \
    --min-faces 2000 \
    --out split_sherds_v2
```

Each run:
- Takes ~2-5 minutes (vs hours for OpenMVS meshing)
- Writes new manifest
- Preserves original mesh
- Updates pipeline_RUNLOG.txt

## Integration with Texturing

### Option 1: Texture Full Scene (Then Split)

```bash
# Texture combined mesh
TextureMesh -i mesh_refine.mvs -o textured.mvs

# Export textured mesh
TextureMesh -i textured.mvs --export-type ply -o scene_textured.ply

# Then split textured mesh
python3 split_sherds.py --mesh scene_textured.ply --out split_textured
```

**Pros**: Consistent texture atlas across sherds
**Cons**: Large combined texture file

### Option 2: Split First (Then Texture Per-Sherd)

```bash
# Split refined mesh
python3 split_sherds.py --mesh scene_dense_mesh_refine.ply

# For each sherd, texture individually
for sherd in split_sherds/sherd_*.ply; do
    # Import to MVS, texture, export
    # (requires per-sherd MVS project setup)
done
```

**Pros**: Smaller texture files per sherd
**Cons**: More manual work, inconsistent lighting across sherds

### Option 3: Export Sherds as Separate MVS Projects

```bash
# Advanced: Extract camera subsets per sherd
# Then run TextureMesh per sherd with relevant cameras only
# (Outside scope of basic splitter)
```

## Performance

**Time Complexity**: O(F) where F = face count

**Typical Performance** (on Spartan compute node):
- 1M faces: ~30 seconds
- 5M faces: ~2 minutes
- 10M faces: ~5 minutes

**Memory Usage**:
- Peak: ~3× input mesh size
- Typical: 2GB for 5M face mesh

**Disk Usage**:
- Output = sum of kept component sizes
- Typically 80-95% of input (dropped 5-20% noise)

## Provenance and Reproducibility

### Files Created

1. **Sherd meshes**: `split_sherds/sherd_*.ply`
2. **CSV manifest**: `split_sherds/manifest.csv`
3. **JSON manifest**: `split_sherds/manifest.json`
4. **Pipeline log**: `work_dir/pipeline_RUNLOG.txt` (appended)

### Pipeline Log Entry

```
--- MESH SPLIT (2025-11-08T15:30:00) ---
Input:        scene_dense_mesh_refine.ply
Min faces:    5000
Max elong:    25.0
Bridge quant: 0.995
Components:   8/12 kept
Time:         142.3s
```

### Reproducing a Split

Given the manifest.json, you can reproduce the exact split:

```bash
# Read parameters from manifest
MIN_FACES=$(jq -r '.parameters.min_faces' manifest.json)
MAX_ELONG=$(jq -r '.parameters.max_elongation' manifest.json)
BRIDGE_Q=$(jq -r '.parameters.bridge_quantile' manifest.json)

# Rerun with same parameters
python3 split_sherds.py \
    --mesh scene.ply \
    --min-faces $MIN_FACES \
    --max-elongation $MAX_ELONG \
    --bridge-quantile $BRIDGE_Q
```

## Batch Processing (SLURM Integration)

### Manual Batch

```bash
# After OpenMVS RefineMesh in batch script
if [ -f dense_scaled/scene_dense_mesh_refine.ply ]; then
    echo "Splitting mesh into sherds..."
    python3 /path/to/pipeline/bin/split_sherds.py \
        --mesh dense_scaled/scene_dense_mesh_refine.ply \
        --min-faces 5000 \
        --max-elongation 25.0
fi
```

### Automated (Future)

Configuration option:
```yaml
mesh_splitter:
  enabled: true
  auto_run: true  # Run automatically after RefineMesh
```

## Configuration

Edit `pipeline/config/pipeline_config.yaml`:

```yaml
mesh_splitter:
  enabled: false                    # Enable mesh splitting after RefineMesh
  auto_run: false                   # Auto-run after RefineMesh in batch
  min_faces: 5000                   # Minimum face count to keep component
  max_elongation: 25.0              # Maximum elongation ratio (max_dim / min_dim)
  bridge_quantile: 0.995            # Edge length quantile for bridge breaking
  output_dir: split_sherds          # Output directory name
  format: ply                       # Output format: 'ply' or 'obj'
  keep_all: false                   # Keep all components (only min_faces filter)
  keep_top: null                    # Keep only top N largest (null = disabled)
```

## Advanced Usage

### Check Component Stats Without Writing Files

```bash
# Run split and inspect manifest
python3 split_sherds.py --mesh scene.ply

# Read manifest to decide on thresholds
cat split_sherds/manifest.csv

# Rerun with adjusted thresholds if needed
```

### Custom Filtering in Python

```python
import json
from pathlib import Path

# Load manifest
with open('split_sherds/manifest.json') as f:
    manifest = json.load(f)

# Filter components by custom criteria
for comp in manifest['components']:
    if comp['kept'] and comp['surface_area'] > 0.05:
        print(f"Large sherd: comp {comp['component_id']}, "
              f"{comp['face_count']} faces, "
              f"{comp['surface_area']:.6f} sq m")
```

### Export Component Bounds for Thesis Figures

```python
import json
import csv

with open('split_sherds/manifest.json') as f:
    manifest = json.load(f)

with open('sherd_bounds.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Sherd', 'Width (m)', 'Height (m)', 'Depth (m)'])

    for comp in manifest['components']:
        if comp['kept']:
            writer.writerow([
                f"Sherd {comp['component_id']}",
                f"{comp['bbox_x']:.4f}",
                f"{comp['bbox_y']:.4f}",
                f"{comp['bbox_z']:.4f}"
            ])
```

## FAQ

**Q: Does splitting modify the original mesh?**
A: No, the original refined mesh is preserved. All outputs go to `split_sherds/`.

**Q: Can I run splitter multiple times?**
A: Yes, rerunning overwrites `split_sherds/` but preserves the original mesh. Use `--out split_sherds_v2` for separate outputs.

**Q: What if I have 100 sherds on the tree?**
A: The splitter handles any number of components. Use `--keep-top 20` for quick preview, then `--keep-all` for full split.

**Q: How do I know which sherd is which in the photos?**
A: Match bounding box dimensions in manifest to physical measurements, or texture sherds and visually identify.

**Q: Can I split textured meshes?**
A: Yes, but textures may be broken if UVs reference the combined mesh. Best to split before texturing.

**Q: What units are the bbox dimensions in?**
A: Same units as your reconstruction (metres if you applied scale).

**Q: How do I combine sherds back together?**
A: Use MeshLab or Blender to import all sherds and merge. Or keep the original refined mesh.

**Q: Can I use this for non-pottery meshes?**
A: Yes, the splitter works on any mesh where components should be separated topologically.

## Dependencies

**Python Libraries**:
```bash
pip install trimesh numpy
```

**On Spartan**:
```bash
module load GCC/11.3.0 Python/3.10.4
pip install --user trimesh numpy
```

**Optional** (for visualization):
- CloudCompare
- MeshLab
- Blender

## Next Steps After Splitting

1. **Review sherds on laptop**: Open in CloudCompare/MeshLab
2. **Measure dimensions**: Use manifest bbox values
3. **Identify sherds**: Match to field photos
4. **Texture sherds**: Run TextureMesh on selected sherds
5. **Decimate for web**: Reduce poly count for online viewer
6. **Archive**: Keep original + split sherds for thesis

## References

- trimesh documentation: https://trimsh.org/
- OpenMVS RefineMesh: https://github.com/cdcseacave/openMVS/wiki/Usage
- COLMAP model formats: https://colmap.github.io/format.html

## Support

For issues:
1. Check troubleshooting section above
2. Inspect `manifest.csv` for rejection reasons
3. Review `pipeline_RUNLOG.txt` for parameters used
4. Open issue with manifest and component stats
