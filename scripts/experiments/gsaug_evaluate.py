"""Evaluate the A02_gsaug re-solve against the A02 baseline (cheap test, step 2b).

Login-node, seconds. Answers in order:
  1. registration — how many of 177 registered, one model or several, renders in?
  2. turntable arc (imports check_turntable's measure, not its threshold sermon)
  3. similarity old->new frame (Umeyama on the 162 shared camera centres) —
     carries the SH5 box and the millimetre scale into the new frame
  4. SH5 steep-fold coherence + longest run on the new refined mesh, in mm

Usage:
  python gsaug_evaluate.py --old-sparse <A02 dense_masked/sparse>
      --new-sparse <A02_gsaug sparse/0> --boxes <A02_sherd_boxes.json>
      --sherd SH5 --mesh <gsaug scene_refined_mesh.ply> [--crop-out SH5_gsaug.ply]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_images(model_dir: Path):
    names, centres = [], []
    with open(model_dir / "images.bin", "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n):
            _, qw, qx, qy, qz, tx, ty, tz, _c = struct.unpack("<idddddddi", fh.read(64))
            nm = b""
            while (ch := fh.read(1)) != b"\x00":
                nm += ch
            (p,) = struct.unpack("<Q", fh.read(8))
            fh.seek(24 * p, 1)
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
            names.append(nm.decode())
            centres.append(-R.T @ np.array([tx, ty, tz]))
    return names, np.array(centres)


def umeyama(src: np.ndarray, dst: np.ndarray):
    """src -> dst similarity: dst ≈ s*R*src + t. Returns s, R, t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    H = (S.T @ D) / len(src)
    U, w, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    s = float(len(src) * w.sum() / ((S ** 2).sum()))
    return s, R, mu_d - s * R @ mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-sparse", required=True)
    ap.add_argument("--new-sparse", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--sherd", default="SH5")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--crop-out", default=None)
    a = ap.parse_args()

    on, oc = read_images(Path(a.old_sparse))
    nn, nc = read_images(Path(a.new_sparse))
    print(f"old: {len(on)} registered; new: {len(nn)} registered (177 input)")
    renders = [n for n in nn if n.startswith("RENDER")]
    print(f"renders registered in new solve: {len(renders)}/15")
    shared = sorted(set(on) & set(nn))
    print(f"shared originals: {len(shared)}/162")

    Cold = np.array([oc[on.index(n)] for n in shared])
    Cnew = np.array([nc[nn.index(n)] for n in shared])
    s, R, t = umeyama(Cnew, Cold)
    resid = np.linalg.norm((s * (R @ Cnew.T).T + t) - Cold, axis=1)
    print(f"Umeyama new->old: scale {s:.6f}, median residual {np.median(resid):.5f} old-units "
          f"(p95 {np.percentile(resid, 95):.5f})")

    d = json.loads(Path(a.boxes).read_text())
    mm_per_old = float(d["mm_per_unit"])
    box = next(b for b in d["boxes"] if b["id"] == a.sherd)
    lo_mm = np.array(box["min_mm"], float) / mm_per_old
    hi_mm = np.array(box["max_mm"], float) / mm_per_old
    # old-unit box corners -> new frame: X_new = R' X_old + t'
    Ri = R.T
    ti = -Ri @ t / s
    corners = np.array([[x, y, z] for x in (lo_mm[0], hi_mm[0])
                        for y in (lo_mm[1], hi_mm[1]) for z in (lo_mm[2], hi_mm[2])])
    cn = corners @ Ri.T / s + ti
    nlo, nhi = cn.min(0), cn.max(0)
    mm_per_new = s * mm_per_old
    print(f"mm per new-unit: {mm_per_new:.3f}")

    sys.path.insert(0, str(Path("scripts/experiments").resolve()))
    from measure_fold import crop, fold_mm, load_mesh
    m = load_mesh(Path(a.mesh))
    print(f"new refined mesh: {len(m.vertices):,} vertices")
    s_mesh = crop(m, nlo, nhi)
    assert s_mesh is not None and len(s_mesh.vertices) > 100, "SH5 box empty in new frame"
    f = fold_mm(s_mesh)
    conv = mm_per_new
    print(f"{a.sherd} in gsaug: 15-30 {f['15-30']*conv:7.0f} | 30-45 {f['30-45']*conv:6.0f} | "
          f"45-60 {f['45-60']*conv:6.0f} | >60 {f['60-90']*conv:7.0f} mm | "
          f"coherent {f['coherent_mm']*conv:7.0f} mm ({100*f['coherent_frac']:4.0f}%) "
          f"longest {f['longest_chain_mm']*conv:6.0f} mm | boundary {f['open_boundary']*conv:5.0f} mm")
    if a.crop_out:
        s_mesh.vertices = np.asarray(s_mesh.vertices) * conv
        s_mesh.export(a.crop_out)
        print(f"crop (mm) -> {a.crop_out}")


if __name__ == "__main__":
    main()
