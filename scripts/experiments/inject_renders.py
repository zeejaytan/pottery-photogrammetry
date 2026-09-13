"""Inject the 15 GS renders into a copy of the dense model with KNOWN poses.

Bypasses COLMAP intake (which silently skipped every synthetic view in the
A02_gsaug run). The poses come from Step 2a's poses.json, carried from the old
(A02) frame into the gsaug frame with the Umeyama fit over the 162 shared
camera centres — the same fit gsaug_evaluate.py uses, residuals ~1 mm.

Renders are 3200x2133; dense views are 5567x3711 (same aspect). They enter
under a SECOND camera with scaled intrinsics — no resampling blur, no fake
pixels. Projection-equivalent poses (uniform scale cancels in x/z).

Usage (login node, seconds):
  python inject_renders.py --work <work_inject> --poses <poses.json>
      --renders <gsaug_renders dir> --old-sparse <A02 dense_masked/sparse>
      --new-sparse <gsaug sparse/0> --boxes <A02_sherd_boxes.json>
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
import shutil

import numpy as np


def read_images(model_dir: Path):
    out = {}
    with open(model_dir / "images.bin", "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n):
            img_id, qw, qx, qy, qz, tx, ty, tz, cam = struct.unpack("<idddddddi", fh.read(64))
            nm = b""
            while (ch := fh.read(1)) != b"\x00":
                nm += ch
            (p,) = struct.unpack("<Q", fh.read(8))
            fh.seek(24 * p, 1)
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
            out[nm.decode()] = (img_id, R, np.array([tx, ty, tz]), cam)
    return out


def read_cameras(model_dir: Path):
    cams = {}
    with open(model_dir / "cameras.bin", "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<iiQQ", fh.read(24))
            nparams = {0: 4, 1: 4, 2: 5, 3: 5, 4: 8, 5: 8, 6: 12}[mid]
            params = struct.unpack("<" + "d" * nparams, fh.read(8 * nparams))
            cams[cid] = (mid, w, h, np.array(params))
    return cams


def umeyama(src: np.ndarray, dst: np.ndarray):
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


def r_to_quat(R: np.ndarray):
    t = np.trace(R)
    if t > 0:
        w = np.sqrt(1 + t) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        x = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2
        w = (R[2, 1] - R[1, 2]) / (4 * x)
        y = (R[0, 1] + R[1, 0]) / (4 * x)
        z = (R[0, 2] + R[2, 0]) / (4 * x)
    elif R[1, 1] > R[2, 2]:
        y = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) / 2
        w = (R[0, 2] - R[2, 0]) / (4 * y)
        x = (R[0, 1] + R[1, 0]) / (4 * y)
        z = (R[1, 2] + R[2, 1]) / (4 * y)
    else:
        z = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) / 2
        w = (R[1, 0] - R[0, 1]) / (4 * z)
        x = (R[0, 2] + R[2, 0]) / (4 * z)
        y = (R[1, 2] + R[2, 1]) / (4 * z)
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="work_inject dir (copy of work_colmap_openmvs)")
    ap.add_argument("--poses", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--old-sparse", required=True)
    ap.add_argument("--new-sparse", required=True)
    ap.add_argument("--boxes", required=True)
    a = ap.parse_args()

    old = read_images(Path(a.old_sparse))
    new = read_images(Path(a.new_sparse))
    shared = sorted(set(old) & set(new))
    assert len(shared) == 162, len(shared)
    Cold = np.array([(-old[n][1].T @ old[n][2]) for n in shared])
    Cnew = np.array([(-new[n][1].T @ new[n][2]) for n in shared])
    s, R, t = umeyama(Cnew, Cold)
    print(f"frame carry: scale {s:.6f}")

    d = json.loads(Path(a.boxes).read_text())
    mm_per_old = float(d["mm_per_unit"])
    spec = json.loads(Path(a.poses).read_text())
    renders = Path(a.renders)
    dense = Path(a.work) / "dense"
    sparse = dense / "sparse"
    cams = read_cameras(sparse)
    assert list(cams) == [1] and cams[1][0] == 1, cams
    _, W1, H1, p1 = cams[1]
    from PIL import Image
    w2, h2 = Image.open(renders / spec[0]["file"]).size
    sx, sy = w2 / W1, h2 / H1
    assert abs(sx / sy - 1) < 1e-3, (sx, sy)
    fx, fy, cx, cy = p1[:4]
    # sanity: carried SH5 centre should project near-frame-centre in a mid render
    box = next(b for b in d["boxes"] if b["id"] == "SH5")
    B_old = (np.array(box["min_mm"]) + np.array(box["max_mm"])) / 2 / mm_per_old

    max_id = max(v[0] for v in new.values())
    new_rows = []
    for i, e in enumerate(spec):
        Rr = np.array(e["R"])
        Tr = np.array(e["T"])
        Rn = Rr @ R.T
        Tn = Rr @ t + Tr
        det = np.linalg.det(Rn)
        assert abs(det - 1) < 1e-6, det
        # orthonormal safety (poses.json came from orthonormalized matrices)
        U, _, Vt = np.linalg.svd(Rn)
        Rn = U @ Vt
        q = r_to_quat(Rn)
        # projection sanity for the dolly/mid views
        C = -Rn.T @ Tn
        B_new = R.T @ (B_old - t) / s
        cam = Rn @ (B_new - C)
        assert cam[2] > 0, f"{e['file']}: SH5 behind camera"
        px = fx * sx * cam[0] / cam[2] + cx * sx
        py = fy * sy * cam[1] / cam[2] + cy * sy
        new_rows.append((max_id + 1 + i, q, Tn, e["file"], px, py, cam[2]))
        dst = dense / "images" / e["file"]
        if not dst.exists():
            shutil.copy(renders / e["file"], dst)

    with open(sparse / "cameras.bin", "r+b") as fh:
        fh.seek(0)
        fh.write(struct.pack("<Q", 2))
        fh.seek(0, 2)
        fh.write(struct.pack("<iiQQdddd", 2, 1, w2, h2, fx * sx, fy * sy, cx * sx, cy * sy))
    with open(sparse / "images.bin", "r+b") as fh:
        raw = bytearray(fh.read())
    off = 8
    (n0,) = struct.unpack_from("<Q", raw, 0)
    # walk to end to append (entries have variable names; re-parse)
    import io
    buf = io.BytesIO(raw[8:])
    for _ in range(n0):
        struct.unpack("<idddddddi", buf.read(64))
        while buf.read(1) != b"\x00":
            pass
        (p,) = struct.unpack("<Q", buf.read(8))
        buf.seek(24 * p, 1)
    end = 8 + buf.tell()
    assert end == len(raw), (end, len(raw))
    out = bytearray()
    out += struct.pack("<Q", n0 + len(new_rows))
    out += raw[8:end]
    for img_id, q, Tn, name, px, py, z in new_rows:
        nm = name.encode()
        out += struct.pack("<idddddddi", img_id, *q, *Tn, 2)
        out += nm + b"\x00"
        out += struct.pack("<Q", 0)
        print(f"  {name}: id {img_id} SH5 at ({px:.0f},{py:.0f}) of {w2}x{h2}, depth {z:.3f}")
    (sparse / "images.bin").write_bytes(bytes(out))
    print(f"dense model now {n0}+{len(new_rows)} images under 2 cameras")


if __name__ == "__main__":
    main()
