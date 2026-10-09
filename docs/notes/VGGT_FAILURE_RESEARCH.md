# VGGT failure research — why poses collapsed, what VGGT likes, fast mesh options (2026-09-07)

> Status: VGGT track retired. No further spend. This is a literature + issue-tracker read-out, not a re-run.

## Plain-language result

**What happened is a genuine method failure on this material, not a broken ruler.**

VGGT guessed that all the photographs were taken from nearly the same spot, so there was no baseline to build 3D from. The refinement step then refused to build rather than building something wrong. That matches how VGGT is known to fail elsewhere.

In practical terms:
- Small pottery on a turntable photographed with a fixed camera is outside what VGGT was trained for. It learned from a moving camera walking around static rooms, streets and objects with background visible.
- A fixed camera + rotating object looks identical to no motion if the background does not move (COLMAP maintainer note, colmap/colmap#2576). The software has to choose between "nothing moved" (background) and "object turned" (sherd). The cheap average is "nothing moved" — collapsed poses.
- Small object in a big frame makes it worse: VGGT decides everything at 518×518 pixels. If the sherd fills a small fraction, the table and wall dominate the decision.
- Featureless, symmetric, shiny sherds make it worse again: every N degrees looks the same, so "same place" is again the cheapest answer.

Weight this can bear: medium. It rests on the VGGT paper, the official issue tracker, and controlled follow-up tests — not on re-running local captures (retired by decision). It explains collapse in both photo modes without needing a new local test.

What it means for the work: keep COLMAP→OpenMVS as the measuring route. Do not re-spend on VGGT for turntable sherds. If speed ever matters more than measured accuracy, use a generative route knowing it invents the unseen side.

---

## 1. Why collapse + BA-refuse happens

Source: Wang et al. VGGT, arXiv:2503.11651 (CVPR'25 Best Paper); repo facebookresearch/vggt; project page vgg-t.github.io.

- VGGT predicts cameras as quaternion + translation + field-of-view in one forward pass, all in frame-1 coordinates, principal point assumed at image centre. Scale is learned, non-metric, needs alignment. Training: 2–24 frames per batch at 518 px, moving-camera static-scene data (Co3Dv2, MegaDepth, ScanNet, DL3DV, Objaverse-like synthetic, etc.). Limitations section names: no fisheye/panorama, degrades under extreme rotation.
- Raw poses are coarse by design. Author in facebookresearch/vggt#254: ~2° rotation error on easy object-centric (Co3D-like) material. In facebookresearch/vggt#79: VGGT poses into 3D splatting lose 1–2 dB image quality vs COLMAP unless jointly optimised; author recommends bundle adjustment (quotes ~2–3 dB gain from VGGSfM experience).
- Known failure vocabulary in tracker (not "collapse" as a single string):
  - #436 duplicated/shifted geometry even with 2 views; author: known feed-forward alignment limit, "did you try BA?"
  - #424 long-orbit drift (210 images on sphere); halves reconstruct better than whole; author suggests doppelgangers/drift.
  - #184 BA gate all-or-nothing: `Not enough inliers per frame, skip BA` → `ValueError: No reconstruction can be built with BA`. Author: defaults strict; try raising `max_reproj_error`, `query_frame_num`, `max_query_pts`, lowering `vis_thresh`. Same pattern #188 (large scenes), #257 (sparse LLFF horns/trex → empty .ply because `conf_thres_value=5.0` too high; fix ~1).
- Triggers from follow-ups and curation practice:
  - Full symmetry / repetitive texture with symmetric light is fatal by theory; VGGT uses shadows to break symmetry but fails when nothing disambiguates.
  - Low overlap (<10% pairwise / sparse scenes) degrades silently into plausible-but-inconsistent output.
  - Curators filter exactly this case out (VGGT-Omega curation): reject insufficient-parallax / repetitive-texture clips, FoV outside 30–120°, distortion >0.1.
  - Turntable prior conflict: static-camera + rotating object is ill-posed for any moving-camera SfM without textured background; generic COLMAP on turntable orbits yields partial arcs/drift without a single-axis prior (CVPRW'26 turntable note).

Top hypotheses for this track (both photo modes collapsed):
1. Static camera + static background vs rotating object — background votes "no motion", object votes "orbit" → near-identity average. Most likely.
2. Background-dominant confidence — small object in 518² frame, depth/confidence pools to table/wall plane.
3. Symmetry/textureless/specular aliasing → canonical-pose collapse or doppelganger cluster.
4. Intrinsics/centre/crop entanglement — close-up + square pad-and-resize + centre assumption; per-image FoV (`shared_camera=False`) lets each view pick a different scale.
5. No rescue: weak tracks → `demo_colmap.py --use_ba` throws instead of partial result; without BA, shifted duplicates remain.

One discriminating test (not to be run — track retired): object filling >60% frame on asymmetric textured mat, 8–12 views at 30–40° spacing, fixed shared intrinsics, feed-forward vs `--use_ba --query_frame_num 24`. Opening into orbit only after masking = (1)+(2); staying clustered despite texture = (3)+(4).

## 2. What VGGT usually works on

- Needs no EXIF/intrinsics; predicts field-of-view. Pinhole only. Network runs at 518×518 (`run_VGGT()` in `demo_colmap.py`); official COLMAP export loads at 1024, runs at 518, rescales K, tracks at 1024. Feeding 4K does not add accuracy.
- Needs some co-visibility but far less than COLMAP. Explicit wins claimed on non-overlapping / repeating / uniform material. Community repro: 10 sparse views → COLMAP registers 2/10, VGGT 10/10; 25 dense views both succeed.
- Best case is object-centric with translation between views and context kept (Co3D AUC 88.2 in paper). Keep background/context; avoid framing only the symmetric object; mask sky/water/movers to 0/1 or boxes (README, issue #47). Fully symmetric with no shadow cue still breaks poses (author confirms issue #87).
- Standard refinement: VGGT init + VGGSfM tracker (ALIKED+SP coarse-to-fine), filter vis>0.2/reproj<8px, pycolmap bundle adjustment. Paper gains e.g. IMC AUC@10 71.3→84.9 in ~1.8 s vs 10–20 s. Code: `demo_colmap.py --use_ba`. Long-sequence: VGGT-Long (chunks 30–60 + confidence-weighted Sim3 chain + loop closure), VGGT-SLAM (submaps + factor graph).
- VGGT beats COLMAP on few/sparse/wide-baseline, low overlap, uncalibrated in-the-wild, speed and completeness. COLMAP wins on dense highly-overlapping captures at pixel accuracy, metric scale, fisheye/pano, extreme rotations, thousand-frame scale without chunking.

## 3. Speed-oriented mesh developments to Sept 2026

None of the feed-forward geometry methods outputs a clean watertight mesh directly — they output poses + depth/pointmaps. Mesh comes next.

Feed-forward geometry (seconds):
- VGGT-Omega (May 2026): VGGT successor, single dense head, ~30% memory, more data, longer context (1000+ frames on 80 GB, ~300 frames on 32 GB). Backported memory fix gives 2–3× more frames.
- Fast3R (CVPR'25): N-view generalisation, no pair selection; 2 views 0.065 s, 32 views 0.5 s on A100; 1000+ images with parallelism. Best when views scale to 100s–1000s.
- MV-DUSt3R / MV-DUSt3R+ (CVPR'25 Oral): multi-view decoder, explicit sparse-view object/scene; 12 views ~0.9 s; +Gaussian heads. Best for 4–12 sparse views.
- Spann3R, CUT3R: incremental/streaming pointmaps; video-oriented, not tens-of-photos object.
- StreamVGGT / OVGGT / VGGT-Long / SwiftVGGT: video / km-scale adaptations; not relevant to tens of photos.
- MonST3R, Geo4D: dynamic extensions; slower (diffusion + alignment).

Splat-to-mesh (faithful, minutes) — current best practice for bounded small object:
- 2DGS + bounded TSDF: standard fast-object choice; DTU Chamfer ~0.8; ~100× faster than NeuS/Neuralangelo class.
- GOF (opacity field + marching tetrahedra, no Poisson/TSDF) then SOF (>3× total vs GOF, meshing 6–10× faster).

Large reconstruction models (direct mesh in seconds, invents unseen side — not a measurement):
- SPAR3D / TripoSR / Stable Fast 3D: sub-second preview, vertex colours or simple texture; rough backsides.
- TRELLIS (Microsoft, MIT core): fidelity leader, full PBR; TRELLIS.2 needs ≥24 GB. Best when 1–4 views or watertight PBR asset needed and hallucination acceptable.
- Hunyuan3D-2.x (Tencent, community licence with territory/MAU limits): practical pick on 8–16 GB; -mv takes multi-view; Turbo halves steps. 2.5/3.x hosted-only.

Turntable note (applies to all): move the camera around the object, not the object before a fixed camera. If turntable fixed, keep turntable + markers in frame so rotation can lock; masking to object-only removes the reference.

Ranked for "fast mesh from tens of photos of a small object":
1. VGGT-Omega/VGGT → 2DGS or SOF. Best fidelity per minute, direct COLMAP export, single-GPU minutes. Default if conservation-grade geometry needed fast.
2. MV-DUSt3R+ for sparse 4–12 views (~1–2 s to point cloud).
3. Fast3R when views scale to 100s–1000s.
4. TRELLIS.2 (or Hunyuan3D-2mv/Turbo if VRAM <16 GB) when 1–4 views or hallucination acceptable.
5. SPAR3D/TripoSR for sub-second preview only.

Licensing (check repo before any commercial use): VGGT code now commercial-friendly excl. military but only VGGT-1B-Commercial checkpoint (gated) cleared for commercial; TRELLIS MIT core (submodules own licences); Hunyuan3D-2.x Tencent Community Licence (excludes EU/UK/Korea, >1M MAU needs approval); SPAR3D Stability Community Licence (free incl. commercial under ~$1M revenue); TripoSR MIT cleanest; SOF NOASSERTION — clarify before product use.

## Sources

- VGGT paper arXiv:2503.11651; code github.com/facebookresearch/vggt; page vgg-t.github.io
- Tracker: #79 (poses vs COLMAP, BA +2–3 dB), #87 (symmetric breakage), #184/#188 (BA refuses), #257 (conf threshold), #254 (~2° easy-case accuracy), #424 (orbit drift), #436 (duplication)
- COLMAP static-camera turntable note colmap/colmap#2576
- VGGT-Long arXiv:2507.16443; Fast3R github.com/facebookresearch/fast3r; MV-DUSt3R; CUT3R; Spann3R; 2DGS github.com/hbb1/2d-gaussian-splatting; GOF/SOF; TRELLIS github.com/microsoft/TRELLIS; Hunyuan3D-2 github.com/Tencent-Hunyuan/Hunyuan3D-2; SPAR3D spar3d.github.io
