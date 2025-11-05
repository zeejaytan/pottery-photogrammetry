# Rabati2025 Photogrammetry Pipeline

This document explains how the automated COLMAP → OpenMVS mesh pipeline is organised in `/data/gpfs/projects/punim2657/Photogrammetry/pipeline` and how to run it on Spartan.

## Overview
- **Input**: `Rabati2025/<date>/<tree>/*.JPG`
- **Stages**:
  1. COLMAP feature extraction, matching, sparse reconstruction, and undistortion.
  2. OpenMVS densification, mesh reconstruction, refinement.
  3. Mesh splitting and quality checks (≥ 100k vertices).
- **Output**: Per-tree working directory `Rabati2025/<date>/<tree>/work_colmap_openmvs` containing intermediate artefacts, individual `sherd_XXX.ply` meshes, `validation_report.csv`, and the pipeline log.

The pipeline is driven by configuration in `pipeline/config/pipeline_config.yaml`. Update this file to change thresholds, resource requests, or binary paths.

## Directory Layout
```
pipeline/
├── bin/
│   ├── pipeline_main.sh        # Run full pipeline for a single tree
│   ├── run_colmap.sh|py        # COLMAP stage driver
│   ├── run_openmvs.sh|py       # OpenMVS stage driver
│   ├── split_and_validate.py   # Mesh splitting + QA
│   ├── scan_targets.py         # Build Slurm target list
│   ├── submit_array.sh         # Submit Slurm array jobs
│   ├── slurm_array_job.sh      # Per-array-task entrypoint
│   └── aggregate_results.py    # Combine validation reports (optional)
├── config/
│   └── pipeline_config.yaml    # Central configuration
├── lib/
│   ├── pipeline_utils.py       # Shared helpers
│   └── stage_tracker.py        # Stage detection + state cache
├── logs/                       # Slurm and manual run logs
└── requirements.txt            # Python dependencies
```

## Prerequisites
1. Spartan modules: defined under `environment` in the config (Python 3.10.4, GCC, OpenMPI, COLMAP, etc.). Adjust if module names change.
2. OpenMVS binaries: default path `/data/gpfs/projects/punim2657/Photogrammetry/openmvs/install/bin/OpenMVS`.
3. Python packages (install once per user environment):
   ```bash
   module load Python/3.10.4
   pip install --user -r pipeline/requirements.txt
   ```

## Configuration Highlights (`config/pipeline_config.yaml`)
- `project`: project root and dataset root.
- `targets`: JPEG extensions, minimum count, output file.
- `environment`: module list, COLMAP executable, OpenMVS directory.
- `colmap` / `openmvs`: runtime parameters for each tool.
- `validation`: vertex and component thresholds for QA.
- `slurm`: default partition, time, memory, CPU/GPU resources, mail options, array chunk size.

Modify these values before running if resource requirements or binary locations differ.

## Running the Pipeline Manually
1. **Scan for targets (optional)**:
   ```bash
   python pipeline/bin/scan_targets.py --config pipeline/config/pipeline_config.yaml --dry-run
   ```
2. **Single tree execution**:
   ```bash
   bash pipeline/bin/pipeline_main.sh 04052025/O01
   ```
   - Creates `/data/.../04052025/O01/work_colmap_openmvs/pipeline_<timestamp>.log`.
   - Writes split meshes and `validation_report.csv`.

3. **Run individual stages** (for troubleshooting):
   ```bash
   bash pipeline/bin/run_colmap.sh 04052025/O01
   bash pipeline/bin/run_openmvs.sh /data/.../04052025/O01/work_colmap_openmvs
   python pipeline/bin/split_and_validate.py --tree 04052025/O01
   ```

### Resuming Partial Runs
- The pipeline records lightweight state in `work_colmap_openmvs/pipeline_state.json`. Stage completion is inferred from existing artefacts.
- The orchestrator runs in resume mode by default; to trigger a full rebuild, add `--no-resume`.
- Typical invocation (resume enabled automatically):
  ```bash
  bash pipeline/bin/pipeline_main.sh 04052025/O01
  ```
  - Skips completed stages (COLMAP, OpenMVS, or split/validate) automatically.
  - Use `--force-colmap`, `--force-openmvs`, or `--force-validate` to rerun specific stages despite existing outputs.
- For manual recovery on a specific stage, run the stage script directly (e.g., rerun only OpenMVS with `run_openmvs.sh --keep-existing`).

## Slurm Array Workflow
1. Populate `targets.txt` automatically and submit (defaults request the `gpu-a100` partition with `gpu:A100:1`):
   ```bash
   bash pipeline/bin/submit_array.sh
   ```
   - Uses `pipeline/bin/scan_targets.py` to regenerate the targets list.
   - Submits `slurm_array_job.sh` with resource settings from the config.

2. Dry-run submission to inspect the `sbatch` command without executing:
   ```bash
   bash pipeline/bin/submit_array.sh --dry-run
   ```

3. Array tasks read `targets.txt` and invoke `pipeline_main.sh` for each tree. Logs go to `pipeline/logs/rabati_<jobid>_<taskid>.log` as well as the per-tree working directories.

## Results Aggregation
- After runs complete, collate QA reports:
  ```bash
  python pipeline/bin/aggregate_results.py
  ```
  - Produces `pipeline/summary_report.csv` with per-tree status lines.

## Troubleshooting Tips
- **Missing modules**: ensure `module load` commands in `pipeline_main.sh` succeed. Adjust `environment` in the config if module names change.
- **OpenMVS library errors**: confirm `environment.openmvs_bin_dir` is correct; pipeline scripts extend `LD_LIBRARY_PATH` automatically.
- **Resuming after failure**: rerun `pipeline_main.sh --resume <tree>` to continue from the last successful stage; check `pipeline_state.json` and the latest `pipeline_*.log` for context.
- **Low vertex counts**: tweak `openmvs.densify.number_views` and `openmvs.reconstruct.min_point_distance` in the config, then rerun the affected tree.
- **Slurm timeouts**: increase `slurm.time_hours` or lower OpenMVS resolution settings.
- **Validation failures**: inspect `<tree>/work_colmap_openmvs/validation_report.csv` for specific sherd metrics.

## Cleaning Up
- Intermediate files live under each tree's `work_colmap_openmvs` directory. Remove entire folders only after verifying outputs are archived. Do **not** delete input JPEGs.

## Versioning
- Implementation plan: `docs/IMPLEMENTATION_PLAN.md`
- Pipeline scripts version: v1.0 (2025-10-22). Update this README whenever parameters or layout change.
