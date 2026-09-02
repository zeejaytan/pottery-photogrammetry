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

### Which parameter to attack (corrected 2026-09-02)

This file previously named `refine.scales`. That was wrong, and measurement says so.
RefineMesh does not build the mesh — on A02 it added **302 vertices out of 1.93M (0.016%)**
and then moved the existing ones. `scales` controls how many *smoothing* iterations run, so
raising it makes SH5 worse and lowering it discards the photometric fit that gives the
0.186 mm surface.

Vertex density is also not the constraint. Vertices sit **0.461 mm apart**; the surface
noise floor is **0.186 mm**. Poisson recovered SH5's edge with spacing of 0.436 mm — 5%
finer. Density is not what separated them.

The two knobs that do control smoothing, both now exposed in `pipeline_config.yaml` and
both currently left at the OpenMVS default:

| knob | default | effect |
|---|---|---|
| `openmvs.refine.regularity_weight` | 0.2 | photo-consistency vs smoothness prior. **Lower trusts the photographs more.** First thing to try. |
| `openmvs.reconstruct.smooth` | 2 | smoothing passes applied before refinement even starts |

And if density ever *is* wanted: `openmvs.reconstruct.min_point_distance` (default 1.5 px)
is the only lever that adds vertices from real measurements — A02 fed 9.80M dense points in
and kept 3.24M. `target_face_num` is a decimation ceiling and has never fired.

**The cheap test.** A02's `dense_masked/scene_dense.mvs` + `scene_dense.ply` are already on
disk, so ReconstructMesh + RefineMesh can be re-run without re-densifying (~20-30 min on a
GPU node vs hours). Rerun with `smooth: 0`, `regularity_weight: 0.05`, then re-measure
SH5's steep-fold length **and render the break edge** — the numeric median across sherds
(141 mm) hid this failure once already.

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
