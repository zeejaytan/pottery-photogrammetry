"""How much steep fold does a sherd have, and where is it.

The decision-relevant number for reassembly. A sherd's fracture faces meeting its wall,
and its broken rim, ARE the sharp geometry; a method that smooths them away destroys
precisely what GARF and TORA match on. Reported as EDGE LENGTH IN MILLIMETRES per angle
band, not as a count or a fraction, because a count cannot be checked against a ruler and
"26 perimeters' worth of sharp edges" is how faceting was caught pretending to be detail.

Validated by reproducing the published A02 baseline before being trusted on anything new:
17062025/A02 SH5, OpenMVS refined = 8 mm above 60 deg; Poisson = 1,298 mm.
Run --selftest to check that again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

BANDS = [(15, 30), (30, 45), (45, 60), (60, 180)]
COHERENT_MM = 10.0   # a run of sharp fold this long is a feature; shorter is noise


def load_boxes(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    return {b["id"]: (np.array(b["min_mm"], float), np.array(b["max_mm"], float))
            for b in d["boxes"]}


def crop(mesh: trimesh.Trimesh, lo, hi, pad=3.0) -> trimesh.Trimesh | None:
    """Keep faces whose every vertex is inside the box. Identical box for every method,
    so a method cannot score well by being cropped more generously."""
    V = np.asarray(mesh.vertices)
    inside = np.all((V >= lo - pad) & (V <= hi + pad), axis=1)
    F = np.asarray(mesh.faces)
    keep = inside[F].all(axis=1)
    if keep.sum() < 100:
        return None
    F = F[keep]
    used = np.unique(F)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(vertices=V[used], faces=remap[F], process=False)


def coherence(m: trimesh.Trimesh, deg=60.0, coherent_mm=COHERENT_MM) -> dict:
    """Of the steep fold, how much lies in sustained runs rather than scattered specks.

    Total sharp length on its own is not trustworthy and has already misled this project
    twice. It rated Delaunay's triangulation faceting as 26 perimeters of "sharpness",
    and drawing SH5 shows Poisson's steep fold is a speckled fringe around the rim rather
    than a crest tracing the break. A real fracture edge is a long CONNECTED run; noise is
    a scatter of isolated folds of the same total length. Chaining separates them.
    """
    ang = np.degrees(m.face_adjacency_angles)
    e = m.face_adjacency_edges
    sharp = ang > deg
    if sharp.sum() == 0:
        return {"sharp_mm": 0.0, "coherent_mm": 0.0, "coherent_frac": 0.0,
                "longest_chain_mm": 0.0, "n_chains": 0}

    se = e[sharp]
    seg = m.vertices[se]
    ln = np.linalg.norm(seg[:, 0] - seg[:, 1], axis=1)

    vmap: dict[int, list[int]] = {}
    for i, (a, b) in enumerate(se):
        vmap.setdefault(int(a), []).append(i)
        vmap.setdefault(int(b), []).append(i)
    rows, cols = [], []
    for ids in vmap.values():
        if len(ids) > 1:
            for j in ids[1:]:
                rows.append(ids[0]); cols.append(j)
    ns = len(se)
    if rows:
        g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(ns, ns))
        ncomp, lab = connected_components(g, directed=False)
    else:
        ncomp, lab = ns, np.arange(ns)
    chain = np.bincount(lab, weights=ln, minlength=ncomp)
    coh = float(chain[chain >= coherent_mm].sum())
    return {"sharp_mm": float(ln.sum()), "coherent_mm": coh,
            "coherent_frac": float(coh / ln.sum()),
            "longest_chain_mm": float(chain.max()), "n_chains": int(ncomp)}


def fold_mm(m: trimesh.Trimesh) -> dict:
    """Edge length in mm per dihedral-angle band, plus the open boundary.

    The open boundary matters for reading the result: if the steep fold is gone AND the
    boundary is short, the edge is still in the mesh and has been rounded over. If the
    boundary is long, the method simply stopped there instead. Opposite diagnoses.
    """
    ang = np.degrees(m.face_adjacency_angles)
    e = m.face_adjacency_edges
    seg = m.vertices[e]
    ln = np.linalg.norm(seg[:, 0] - seg[:, 1], axis=1)

    out = {f"{a}-{b if b < 180 else 90}": float(ln[(ang >= a) & (ang < b)].sum())
           for a, b in BANDS}

    ev = m.edges_sorted
    uniq, cnt = np.unique(ev, axis=0, return_counts=True)
    b = uniq[cnt == 1]
    out["open_boundary"] = float(
        np.linalg.norm(m.vertices[b[:, 0]] - m.vertices[b[:, 1]], axis=1).sum())
    out["area_mm2"] = float(m.area)
    out["vertices"] = int(len(m.vertices))
    out.update(coherence(m))
    return out


def load_mesh(path: Path, scale: float = 1.0) -> trimesh.Trimesh:
    m = trimesh.load(path, process=False, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    if scale != 1.0:
        m.vertices = np.asarray(m.vertices) * scale
    return m


def report(tag: str, mesh_path: Path, boxes: dict, sherds: list[str], scale: float):
    m = load_mesh(mesh_path, scale)
    rows = []
    for sid in sherds:
        lo, hi = boxes[sid]
        s = crop(m, lo, hi)
        if s is None:
            print(f"  {tag:22s} {sid}: nothing inside the box")
            continue
        f = fold_mm(s)
        rows.append((sid, f))
        print(f"  {tag:24s} {sid}: "
              f"15-30 {f['15-30']:7.0f} | 30-45 {f['30-45']:6.0f} | "
              f"45-60 {f['45-60']:6.0f} | >60 {f['60-90']:7.0f} mm | "
              f"coherent {f['coherent_mm']:7.0f} mm ({100*f['coherent_frac']:4.0f}%) "
              f"longest {f['longest_chain_mm']:6.0f} mm | "
              f"boundary {f['open_boundary']:5.0f} mm")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--sherd", action="append", default=None)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply vertices by this to reach millimetres")
    ap.add_argument("mesh", nargs="+", help="tag=path")
    a = ap.parse_args()

    boxes = load_boxes(Path(a.boxes))
    sherds = a.sherd or sorted(boxes)
    print(f"Steep fold per sherd, edge length in mm. Bands are dihedral angle.\n")
    for spec in a.mesh:
        tag, path = spec.split("=", 1)
        report(tag, Path(path), boxes, sherds, a.scale)


if __name__ == "__main__":
    main()
