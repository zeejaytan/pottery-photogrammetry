# 01: Its own Python environment; the 9 ad-hoc scripts committed or archived

**What to build:** The pipeline's interpreter is a venv under `$REMOTE_ROOT/envs/` built from `pipeline/requirements.txt`, not MILo's conda environment; and the 9 untracked November 2025 scripts at the checkout are either committed (useful) or archived (not).

**Answers:** none (workbench routine).

**Blocked by:** None (can start immediately).

**Status:** resolved

Amendments 3.4 and 2.7. `AGENTS.md` itself calls the borrowed interpreter fragile: a rebuild of MILo's environment silently changes this pipeline's runs. The conservator agreed the scripts' fate is decided here (2026-10-24).

- [x] venv built in a short login-node job; `environment.python_interpreter` points at it (built 2026-10-24 19:57 AEDT on the login node: Python 3.9.25 from `/usr/bin/python3`, RHEL 9.8; `envs/pipeline/FREEZE.txt`)
- [x] One stage re-run with it gives the same output as before. The sherd-splitting stage on A02's refined mesh (`dense_masked/scene_refined_mesh.ply`), run under both interpreters, gives byte-identical output. Two runs:
  - job 33745432, run `20261024-1957_bebfc9c_env-parity`, shipped config: identical, but only the report was compared. `min_component_area: 10.0` skipped all 16 components, so no mesh was exported (split exit 3).
  - job 33748450, run `20261024-2316_8b5c4ab_env-parity-area0`, `MIN_AREA=0` in a copy of the config: **IDENTICAL, 3 files**: the report and both exported meshes, one of them 1,925,664 vertices / 3,851,289 faces. This is the parity evidence.
- [x] Each of the 9 scripts committed or archived, listed here (all 9 archived, proved and removed 2026-10-24; checkout now shows only the ignored `envs/ runs/ attic/`)

## What was found before building (2026-10-24, read-only on Spartan)

Raw listings: project files `photogrammetry_hygiene_{checkout_state,milo_env,base_python,a02_layout}.txt`.

- **The venv cannot stand on a module Python.** `GCC/11.3.0 Python/3.10.4` gives a
  `python3.10` linked to the module's `libpython3.10.so`; after `module purge` it exits 127.
  `pipeline_main.sh` runs the interpreter *before* loading modules, so Spartan's usual recipe
  (load the Python module, then make a venv: https://dashboard.hpc.unimelb.edu.au/software/python/)
  would break the pipeline's first line. **Chosen: `/usr/bin/python3` (3.9.25, RHEL 9.8)**,
  which needs nothing loaded and is the same 3.9 the MILo environment runs (3.9.23), so the
  same wheels install. Risk stated in `build_env.sh`: an operating-system upgrade means a rebuild.
- **`requirements.txt` was incomplete.** It listed 4 packages; the pipeline imports 7 (scipy
  and networkx for splitting the mesh into sherds, OpenCV with ArUco for the scale step). Now
  all 7, pinned to the MILo environment's versions so the move changes nothing computed.
- **There was no earlier split of A02 to compare against** (`work_colmap_openmvs` holds no
  `sherds/` or report). So the "same output" check runs the split twice on the same mesh,
  once per interpreter (`scripts/experiments/env_parity.slurm`), and compares byte for byte.

## The 9 scripts: all archived, none committed

All dated 2025-10-26, all hard-wired to tree `16062025`, all from one day of COLMAP tuning.

| script | what it did | verdict |
|---|---|---|
| `force_rebuild_colmap.sh` | `rm -rf` the whole 16062025 work dir, rerun the pipeline | archive: destructive one-off |
| `rebuild_colmap_matching.sh` | delete the pipeline state, rerun `run_colmap.py --rebuild-from-matching` with the old broken module order | archive: one-off, superseded module order |
| `submit_standard_gpu.sh` | runs `test_ultra_permissive.sh`, despite its name submits nothing | archive: misnamed wrapper |
| `targets_single_16062025.txt` | one line, `16062025` | archive: input of `test_single.sh` |
| `test_permissive_colmap.sh` | `rm -rf` sparse/, rerun with `--force-colmap` | archive: destructive one-off |
| `test_single.sh` | resume the pipeline on that one tree | archive: `pipeline_main.sh --resume` does it |
| `test_turntable_colmap.sh` | the turntable COLMAP settings (guided matching, min triangulation angle 2°, 15 inliers) run by hand; its registration count reads `database.db` as if it were `images.bin` | archive: the settings are already in `pipeline_config.yaml`; its check is wrong |
| `test_ultra_permissive.sh` | `rm -rf` sparse/, rerun with `--force-colmap` | archive: destructive one-off |
| `validate_existing_model.py` | point count of each sparse model via `colmap model_analyzer` | archive: `colmap model_analyzer` does it directly |

Record: `$REMOTE_ROOT/attic/untracked-scripts-20261024.tar.gz` + `-INDEX.txt` +
`-deleted-manifest-20261024.tsv.gz`. Proved before removal: `gzip -t` ok, 9 members, `cmp` ok
for all 9 (conservator's OK on the decision card, 2026-10-24 08:50 UTC). Index row
`attic-untracked-scripts-20261024` in `$REMOTE_ROOT/runs/index.tsv`. Log: project file
`photogrammetry_hygiene_step2.txt`.

Side note: the venv prints `RuntimeWarning: networkx backend defined more than once` on
import. On RHEL a venv's `lib64` is a symlink to `lib`, so the same package appears twice on
the path and networkx counts its entry point twice. It is a warning only (the MILo env shows
it too); it changes no result.

## Seen in passing, not settled here

On A02 the split stage does not separate the sherds. 16 connected pieces: one of 1.93M
vertices (area 2.85 in mesh units), one of 6,915, and 14 specks of 83 vertices or fewer. The
large piece is most likely several sherds and/or the base still joined. That reading comes
from the vertex counts and **has not been checked by rendering it**. The shipped area filter
(10.0) would have dropped every piece anyway. This is the pipeline's own split behaving as it
always has (identical under both interpreters), not a change made by this ticket. The config
already says to judge components by eye and with `MILo/scripts/extract_sherds.py`. If the
split stage is to be trusted, it needs its own ticket that starts with a render.
Reports: project files `photogrammetry_parity1_report.csv`, `photogrammetry_parity2_report.csv`.
