"""Rank A02 views by how many sparse points fall inside SH5's box.

Login-node step, seconds. No GPU. Decides where synthetic close-ups should go:
the test only spends GPU where the hole is.

Usage:
    python sh5_rank_views.py --sparse <sparse_dir> --boxes <A02_sherd_boxes.json> \
        --sherd SH5 [--top 8]

Box JSON is in millimetres; divide by mm_per_unit to reach the cameras' frame.
Counts COLMAP track observations per image for points inside the box, and prints
camera centres for the top views (needed for the dolly-in pose step).
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np


def read_images(model_dir: Path):
    ids, names, centres = [], [], []
    with open(model_dir / "images.bin", "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n):
            img_id, qw, qx, qy, qz, tx, ty, tz, _cam = struct.unpack("<idddddddi", fh.read(64))
            nm = b""
            while (ch := fh.read(1)) != b"\x00":
                nm += ch
            (p,) = struct.unpack("<Q", fh.read(8))
            fh.seek(24 * p, 1)
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
            centres.append(-R.T @ np.array([tx, ty, tz]))
            ids.append(img_id)
            names.append(nm.decode())
    return ids, names, np.array(centres)


def rank(model_dir: Path, lo: np.ndarray, hi: np.ndarray):
    ids, names, _ = read_images(model_dir)
    id2name = dict(zip(ids, names))
    cnt = Counter()
    n_inside = 0
    with open(model_dir / "points3D.bin", "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n):
            _pid, x, y, z, _r, _g, _b, _e = struct.unpack("<QdddBBBd", fh.read(43))
            (t,) = struct.unpack("<Q", fh.read(8))
            obs = fh.read(8 * t)
            if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and lo[2] <= z <= hi[2]:
                n_inside += 1
                for k in range(t):
                    (img_id,) = struct.unpack_from("<I", obs, 8 * k)
                    cnt[img_id] += 1
    return n_inside, [(id2name.get(i, str(i)), c) for i, c in cnt.most_common()], id2name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--sherd", default="SH5")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()

    d = json.loads(Path(a.boxes).read_text())
    scale = float(d.get("mm_per_unit", 1.0))
    box = next(b for b in d["boxes"] if b["id"] == a.sherd)
    lo = np.array(box["min_mm"], float) / scale
    hi = np.array(box["max_mm"], float) / scale
    print(f"{a.sherd} box (camera units): lo={lo.round(3)} hi={hi.round(3)}")

    _, names_top, _ = rank(Path(a.sparse), lo, hi)
    # re-read ids/names/centres for centre print of top views
    ids, names, centres = read_images(Path(a.sparse))
    name2centre = dict(zip(names, centres))
    # n_inside recompute message from rank: sum over views is obs count; print top list
    print(f"top {a.top} views by SH5 observations:")
    for nm, c in names_top[:a.top]:
        cc = name2centre.get(nm, (float("nan"),) * 3)
        print(f"  {nm:18s} {c:6d}  centre=({cc[0]:+.3f},{cc[1]:+.3f},{cc[2]:+.3f})")


if __name__ == "__main__":
    main()
