# AGENTS.md — pottery-photogrammetry (project)

Follow the workspace root **`../AGENTS.md`** (laptop ↔ GitHub ↔ Spartan) for all shared
rules. This file adds only the paths and domain notes specific to this repo.

## What this repo is for

The **COLMAP → OpenMVS** route from turntable photographs of Rabati pottery to per-sherd
meshes. This is the primary reconstruction route for the project.

> **Status, 2026-09-02.** The conservator has chosen this route over MILo (3D Gaussian
> Splatting) for geometric accuracy. The evidence is `../MILo/docs/notes/A02_MESH_METHOD_COMPARISON.md`
> — one capture (A02), seven sherds, four methods. OpenMVS refined won on every surface
> measure, usually by a factor: noise on known-flat geometry **0.186 mm vs MILo's 0.485 mm**,
> zero loose debris against MILo's 17 pieces per sherd, zero interior holes against 10 mm.
>
> **The one thing to watch, and it is the thing this project needs most.** On sherd SH5 of
> the seven, OpenMVS's refinement *smoothed the fracture edge out of existence* — 8 mm of
> steep fold where Poisson resolved 1,298 mm. Fracture surfaces are exactly what GARF and
> TORA match on. The median across sherds (141 mm) looks healthy and hides the failure
> completely. **Check break edges sherd by sherd before feeding any mesh downstream.**
> If fracture fidelity becomes the binding constraint, `refine.scales` is the parameter to
> attack.

## Paths

| What | Where |
|---|---|
| Spartan checkout (`REMOTE_ROOT`) | `/data/gpfs/projects/punim2657/Photogrammetry` |
| Photographs (`data_root`) | `/data/gpfs/projects/punim2657/Rabati2025/<date>/<group>` |
| OpenMVS binaries (built, not in repo) | `$REMOTE_ROOT/openmvs/install/bin/OpenMVS` |
| COLMAP | module `COLMAP/3.9-CUDA-11.7.0`, under `GCC/11.3.0` |
| Per-group outputs | `<data_root>/<date>/<group>/work_colmap_openmvs/` |
| Small fetched logs/renders | `./artifacts/` (gitignored) |

Groups reconstructed so far: `17062025/A02`, `A03`, `A04`, `03072025/N01`.

## Running it

The repo's own entry point works again as of 2026-09-02:

```bash
./scripts/remote/pull_and_sbatch.sh pipeline/bin/submit_single.sh 17062025/A02
```

Four fixes were folded in from what was, until now, a wrapper living in the MILo project
(`../MILo/scripts/run_photogrammetry.sh`, now superseded — do not use it). All four were
environment drift since November 2025; none was a fault in the pipeline's logic. The
reasoning for each sits next to the change: `pipeline/bin/pipeline_python.sh`, the comments
in `pipeline/config/pipeline_config.yaml`, and README → *Troubleshooting → Environment*.

1. The config read that discovers which modules to load ran on `python3`, which is now the
   `graphify` conda env with no PyYAML. The interpreter is named in the config instead.
2. `Python/3.10.4` loaded before `GCC/11.3.0` cannot resolve under Lmod. Module order is
   now compiler-first, and that Python module is dropped as redundant.
3. `validation.min_vertices: 100000` discarded every real sherd (18k-82k vertices) and kept
   the clamp rig and backdrop. Now 2000.
4. 64 G was OOM-killed at 67 G on one 5568x3712 image. Now 256 G in both submit paths.

### The interpreter is borrowed

`environment.python_interpreter` points at **the MILo project's conda environment**
(`/data/gpfs/projects/punim2657/MILo/envs/milo/bin/python`, Python 3.9.23), because it is
the one environment on Spartan that already has all six dependencies — PyYAML, trimesh,
numpy, pandas, scipy, networkx.

**This is a known piece of fragility, not a design.** This repo should not depend on
another project's environment surviving; if MILo's env is rebuilt or removed, this pipeline
stops. The fix is a dedicated venv under `$REMOTE_ROOT/envs/`, built from
`pipeline/requirements.txt` — a login-node job of a few minutes. Until that exists, the
borrowing is at least explicit, in one config key, overridable with `$PIPELINE_PYTHON`.

## Untracked working scripts on Spartan

`$REMOTE_ROOT` carries nine untracked ad-hoc scripts (`test_turntable_colmap.sh`,
`force_rebuild_colmap.sh`, `validate_existing_model.py`, …) from the November 2025 tuning.
Tracked files there are clean and level with `origin/main`. Ask before committing any of
them — see workspace rule 3.
