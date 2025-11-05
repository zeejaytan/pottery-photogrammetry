# Archaeological Pottery Photogrammetry Pipeline

A production-grade, GPU-accelerated photogrammetry pipeline for reconstructing 3D models of archaeological pottery sherds from multi-view turntable captures.

## Overview

This pipeline processes multi-ring turntable photographs (120-170 images per batch) to generate high-quality 3D meshes of pottery sherds with ≥100,000 vertices per sherd. It combines:

- **COLMAP** (Structure-from-Motion) for camera pose estimation and sparse reconstruction
- **OpenMVS** (Multi-View Stereo) for dense point cloud generation and mesh reconstruction
- **Automated validation** for mesh splitting and quality assurance

## Features

- ✅ **Turntable-optimized** COLMAP settings for multi-ring captures with small baselines
- ✅ **GPU-accelerated** processing (CUDA support for COLMAP and OpenMVS)
- ✅ **High-density meshing** targeting 100k+ vertices per sherd
- ✅ **Automated mesh splitting** by connected components
- ✅ **Resume capability** with state tracking
- ✅ **Batch processing** via Slurm array jobs on HPC systems
- ✅ **Comprehensive logging** and validation reports

## Requirements

### Software Dependencies

- Python 3.10+
- COLMAP 3.9+ (with CUDA 11.7+)
- OpenMVS 2.3+ (with CUDA support)
- Slurm (for HPC job scheduling)

### Python Packages

```bash
pip install -r pipeline/requirements.txt
```

Required packages:
- trimesh >= 3.20.0
- numpy >= 1.24.0
- pyyaml >= 6.0
- pandas >= 2.0.0
- scipy >= 1.15.0
- networkx >= 3.4.0

### Hardware

Recommended configuration:
- **GPU**: NVIDIA A100 (80GB VRAM) or V100
- **RAM**: 64GB+
- **CPU**: 8+ cores
- **Storage**: High-speed filesystem for image I/O

## Pipeline Stages

### 1. Feature Extraction (COLMAP)
- GPU-accelerated SIFT feature detection
- 12,000 features per image
- Domain-size pooling for smooth ceramic surfaces
- Max image size: 4096px

### 2. Exhaustive Matching (COLMAP)
- Exhaustive pairwise matching (all combinations)
- Guided matching enabled for cross-ring stability
- GPU-accelerated feature matching

### 3. Sparse Reconstruction (COLMAP)
- Incremental mapper with turntable-optimized thresholds:
  - Initial triangulation angle: 4° (relaxed from 16° default)
  - Min pose inliers: 20 (reduced from 30)
  - Min inlier ratio: 0.20 (reduced from 0.25)
  - Fixed principal point to prevent drift

### 4. Dense Point Cloud (OpenMVS)
- Full-resolution depth-map generation
- 6 supporting views per estimate
- 4 views required for fusion
- Generates 9-10M dense points from ~177 images

### 5. Mesh Reconstruction (OpenMVS)
- Delaunay tetrahedralization
- Graph-cut surface extraction
- Target: 10M faces (no aggressive decimation)

### 6. Mesh Refinement (OpenMVS)
- Multi-scale refinement (2 iterations)
- GPU-accelerated optimization
- Typical output: ~1M vertices, ~2M faces

### 7. Splitting & Validation
- Connected component analysis
- Per-sherd vertex/face counting
- Validation against 100k vertex threshold
- CSV report generation

## Quick Start

### Single Tree Processing

```bash
cd /path/to/Photogrammetry
bash pipeline/bin/pipeline_main.sh <date>/<tree_id>
```

Example:
```bash
bash pipeline/bin/pipeline_main.sh 16062025
```

### Batch Processing (Slurm)

1. **Generate target list**:
```bash
python pipeline/bin/scan_targets.py --config pipeline/config/pipeline_config.yaml
```

2. **Submit array job**:
```bash
bash pipeline/bin/submit_array.sh
```

3. **Monitor progress**:
```bash
squeue -u $USER
tail -f pipeline/logs/rabati_<jobid>_<taskid>.log
```

4. **Aggregate results**:
```bash
python pipeline/bin/aggregate_results.py
```

## Configuration

All pipeline parameters are centralized in `pipeline/config/pipeline_config.yaml`:

### Key COLMAP Settings
```yaml
colmap:
  feature_extractor:
    max_image_size: 4096
    max_num_features: 12000
    domain_size_pooling: 1

  exhaustive_matcher:
    guided_matching: 1

  mapper:
    init_min_tri_angle: 4.0
    abs_pose_min_num_inliers: 20
    abs_pose_min_inlier_ratio: 0.20
    ba_refine_principal_point: 0
```

### Key OpenMVS Settings
```yaml
openmvs:
  densify:
    resolution_level: 0  # Full resolution
    number_views: 6
    number_views_fuse: 4

  reconstruct:
    target_face_num: 10000000

  refine:
    scales: 2
```

## Directory Structure

```
Photogrammetry/
├── pipeline/
│   ├── bin/                    # Executable scripts
│   │   ├── pipeline_main.sh   # Master orchestrator
│   │   ├── run_colmap.py      # COLMAP workflow
│   │   ├── run_openmvs.py     # OpenMVS workflow
│   │   ├── split_and_validate.py
│   │   ├── scan_targets.py
│   │   ├── submit_array.sh    # Slurm array submission
│   │   └── slurm_array_job.sh
│   ├── lib/                    # Python utilities
│   │   ├── pipeline_utils.py
│   │   └── stage_tracker.py
│   ├── config/
│   │   └── pipeline_config.yaml
│   ├── logs/                   # Execution logs
│   └── targets.txt            # Batch processing targets
├── colmap/                     # COLMAP build (not in repo)
├── openmvs/                    # OpenMVS build (not in repo)
└── IMPLEMENTATION_PLAN.md     # Detailed architecture docs
```

## Output Files

For each processed tree (`<tree>/work_colmap_openmvs/`):

```
work_colmap_openmvs/
├── database.db                 # COLMAP feature database
├── sparse/0/                   # Sparse reconstruction
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
├── dense/                      # Undistorted images
│   ├── images/
│   └── sparse/
├── scene_dense.ply            # Dense point cloud (9-10M points)
├── scene_dense_mesh.ply       # Initial mesh (3-4M vertices)
├── scene_refined_mesh.ply     # Refined mesh (1M vertices)
├── sherd_001.ply              # Individual sherds
├── sherd_002.ply
├── ...
├── validation_report.csv      # QA metrics
└── pipeline_*.log             # Execution logs
```

## Troubleshooting

### COLMAP Issues

**Problem**: Multiple sparse models created instead of one unified model

**Solution**: Ensure exhaustive matching with guided matching is enabled:
```yaml
exhaustive_matcher:
  guided_matching: 1
```

**Problem**: Few images registered (low success rate)

**Solution**: Relax mapper thresholds for turntable captures:
```yaml
mapper:
  init_min_tri_angle: 4.0  # Lower for small baselines
  abs_pose_min_num_inliers: 20
```

### OpenMVS Issues

**Problem**: Low mesh density (< 100k vertices per sherd)

**Solution**:
1. Check `resolution_level: 0` (full resolution)
2. Increase `number_views` (try 8-10)
3. Increase `number_views_fuse` (try 5-6)

**Problem**: ReconstructMesh crashes (segmentation fault)

**Solution**: Ensure correct parameter order:
```bash
ReconstructMesh -i scene_dense.mvs -p scene_dense.ply -o output.ply
```

### Memory Issues

**Problem**: Out of memory during densification or meshing

**Solution**:
1. Increase Slurm memory: `--mem=128G`
2. Reduce resolution: `resolution_level: 1` (half res)
3. Reduce views: `number_views: 4`

## Performance Metrics

Typical processing times on A100 GPU (177 images):

| Stage | Time |
|-------|------|
| Feature extraction | 5-10 min |
| Exhaustive matching | 30-60 min |
| Sparse mapper | 1-2 hours |
| Image undistortion | 5-10 min |
| Dense point cloud | 10-15 min |
| Mesh reconstruction | 20-30 min |
| Mesh refinement | 10-15 min |
| Splitting & validation | 5-10 min |
| **Total** | **2-4 hours** |

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{pottery_photogrammetry_pipeline,
  title = {Archaeological Pottery Photogrammetry Pipeline},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/[your-username]/pottery-photogrammetry}
}
```

## License

[Specify your license here]

## Acknowledgments

- COLMAP: https://colmap.github.io/
- OpenMVS: https://github.com/cdcseacave/openMVS
- HPC resources provided by [Your Institution]

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
