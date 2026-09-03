"""Draw the measured quantity itself: every vertex coloured by how sharply it folds.

NOT a shaded render of the sherd. A shaded view of a rounded-over break edge looks like a
shaded view of a sharp one from most angles, which is how four successive views of the
wear simulation each answered the wrong question convincingly (docs/lessons.md). So this
paints the dihedral angle straight onto the surface, unbinned and unprojected: grey where
the surface is flat, hot where it folds past 60 degrees. A fracture edge appears as a
continuous hot line running around the sherd. A smoothed-away one does not appear at all.

Four fixed viewpoints, identical for every method, so the comparison is not a viewpoint
argument. Also writes the cropped sherd as PLY with the colours baked in, so it can be
opened in CloudCompare and turned by hand rather than taken on trust.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from measure_fold import BANDS, crop, load_boxes, load_mesh  # noqa: E402


def vertex_fold_deg(m: trimesh.Trimesh) -> np.ndarray:
    """Per vertex: the steepest fold on any edge touching it. Per-vertex and unbinned —
    no averaging that could dilute a one-triangle-wide crest into the wall around it."""
    ang = np.degrees(m.face_adjacency_angles)
    e = m.face_adjacency_edges
    out = np.zeros(len(m.vertices))
    for col in (0, 1):
        np.maximum.at(out, e[:, col], ang)
    return out


def colourise(deg: np.ndarray) -> np.ndarray:
    """Grey below 15 deg, then blue -> yellow -> red to 90. The 60 deg threshold the
    measurement uses sits in the orange, so 'is there a hot line' matches 'is there
    steep fold in the table'."""
    t = np.clip((deg - 15.0) / 75.0, 0.0, 1.0)
    c = np.zeros((len(deg), 4), np.uint8)
    c[:, 3] = 255
    flat = deg < 15.0
    c[flat, :3] = 190
    x = t[~flat]
    c[~flat, 0] = (255 * np.clip(x * 2, 0, 1)).astype(np.uint8)
    c[~flat, 1] = (255 * np.clip(1.6 * x * (1 - x) * 2.2, 0, 1)).astype(np.uint8)
    c[~flat, 2] = (255 * np.clip(1 - x * 2, 0, 1)).astype(np.uint8)
    return c


VIEWS = {"front": (12, 0), "back": (12, 180), "left": (12, 90), "top": (78, 0)}


def render(mesh: trimesh.Trimesh, colours, deg, stem: str, out: Path) -> bool:
    """Scatter every vertex at its own position, coloured by its own fold angle.

    A scatter of the actual vertices, not a shaded surface: shading is exactly what makes
    a rounded edge and a sharp one look alike. Hot vertices are drawn last and larger so a
    thin real crest cannot be buried under the wall it sits in — the failure mode is a
    MISSING edge, so the drawing must be biased towards showing one if it is there.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V = np.asarray(mesh.vertices)
    c = colours[:, :3] / 255.0
    hot = deg > 60.0
    order = np.argsort(hot.astype(int))       # flat first, hot on top

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), subplot_kw={"projection": "3d"})
    for ax, (name, (elev, azim)) in zip(axes, VIEWS.items()):
        ax.scatter(V[order, 0], V[order, 1], V[order, 2], c=c[order],
                   s=np.where(hot[order], 6.0, 0.7), marker=".", linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{name}", fontsize=11)
        ax.set_box_aspect(np.ptp(V, axis=0))
        ax.set_axis_off()
    n = int(hot.sum())
    fig.suptitle(f"{stem}   |   {n:,} of {len(deg):,} vertices fold past 60 deg "
                 f"({100*n/len(deg):.2f}%)   |   grey = flat, red = 90 deg",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--sherd", default="SH5")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("mesh", nargs="+", help="tag=path")
    a = ap.parse_args()

    boxes = load_boxes(Path(a.boxes))
    lo, hi = boxes[a.sherd]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for spec in a.mesh:
        tag, path = spec.split("=", 1)
        s = crop(load_mesh(Path(path), a.scale), lo, hi)
        if s is None:
            print(f"{tag}: nothing inside {a.sherd}")
            continue
        deg = vertex_fold_deg(s)
        s.visual.vertex_colors = colourise(deg)
        stem = f"{a.sherd}_{tag.replace(' ', '_')}"
        s.export(outdir / f"{stem}.ply")
        hot = deg[deg > 60]
        print(f"{tag:22s} {a.sherd}: {len(hot):,} of {len(deg):,} vertices "
              f"fold past 60 deg ({100*len(hot)/len(deg):.2f}%)  -> {stem}.ply")
        render(s, s.visual.vertex_colors, deg, stem, outdir / f"{stem}.png")
        print(f"  -> {stem}.png")


if __name__ == "__main__":
    main()
