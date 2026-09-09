"""Render every session mapping on top of its own session image.

The calibration matrix is what takes a raw camera frame into the coordinate
space the tiles live in, so each session image is warped with its calibration
before the node/edge tiles are drawn on it. A correct mapping has every tile
sitting on the maze structure underneath it.

    python render_all_mappings.py                 # all sessions
    python render_all_mappings.py --only NPC4     # just one subject
    python render_all_mappings.py --contact-sheet # one overview page as well

Output goes to resources/mapping_qc/<mapping name>_overlay.png.
"""

import argparse
import pickle
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "resources"
OUT_DIR = RESOURCES / "mapping_qc"


def matrix_for(session_date, matrices):
    """Pick the calibration for this date: exact match, else nearest earlier."""
    exact = matrices.get(session_date)
    if exact is not None:
        return session_date, exact
    earlier = sorted(d for d in matrices if d <= session_date)
    if earlier:
        return earlier[-1], matrices[earlier[-1]]
    first = sorted(matrices)[0]
    return first, matrices[first]


def load_matrices():
    """{YYYY_MM_DD: path} for every calibration on disk.

    Older files use DD_MM_YYYY; normalise so dates sort chronologically.
    """
    out = {}
    for path in (RESOURCES / "transform_matrix").glob("*.npy"):
        stem = path.stem
        if re.fullmatch(r"\d{4}_\d{2}_\d{2}", stem):
            out[stem] = path
        elif re.fullmatch(r"\d{2}_\d{2}_\d{4}", stem):
            dd, mm, yyyy = stem.split("_")
            out[f"{yyyy}_{mm}_{dd}"] = path
    return out


def tile_sizes(nodes):
    sizes = np.array([np.asarray(v[0], float).max(0) - np.asarray(v[0], float).min(0)
                      for v in nodes.values()])
    return sizes


def render(mapping_path, matrices, ax=None, out_png=None):
    name = mapping_path.stem
    image_path = RESOURCES / "maze_images" / f"{name}.png"
    if not image_path.exists():
        print(f"  skip {name}: no matching session image")
        return None

    session_date = name.split("_", 1)[1]
    used_date, matrix_path = matrix_for(session_date, matrices)
    matrix = np.load(matrix_path)

    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    warped = cv2.warpPerspective(image, matrix, (width, height))

    data = pickle.load(open(mapping_path, "rb"))
    nodes = data["mappings"]["nodes"]
    edges = data["mappings"]["edges"]
    sizes = tile_sizes(nodes)
    resized = int((np.abs(sizes - sizes[0]).max(axis=1) > 0.5).sum())

    # Crop to the tiles so the warped border does not dominate the picture.
    pts = np.vstack([np.asarray(v[0], float) for v in nodes.values()])
    pad = 40
    x0, x1 = max(pts[:, 0].min() - pad, 0), min(pts[:, 0].max() + pad, width)
    y0, y1 = max(pts[:, 1].min() - pad, 0), min(pts[:, 1].max() + pad, height)

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(14, 14 * (y1 - y0) / (x1 - x0)), dpi=110)
    ax.imshow(warped)

    for polys in edges.values():
        for poly in polys:
            p = np.asarray(poly, float)
            if len(p) >= 3:
                ax.add_patch(MplPoly(p, closed=True, fill=False,
                                     edgecolor="#4da6ff", lw=0.6, alpha=0.5))
    for node, polys in nodes.items():
        p = np.asarray(polys[0], float)
        # Flag tiles that were resized away from the default.
        span = p.max(0) - p.min(0)
        odd = abs(span - sizes[0]).max() > 0.5
        ax.add_patch(MplPoly(p, closed=True, fill=False,
                             edgecolor="#ffd60a" if odd else "#00e5ff",
                             lw=1.6 if odd else 0.9, alpha=0.95))

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    ax.set_title(f"{name}   ·   calibration {matrix_path.name}"
                 f"{'' if used_date == session_date else '  (no matching date!)'}\n"
                 f"{len(nodes)} nodes, {len(edges)} edges, {resized} resized tiles",
                 fontsize=10)

    if own:
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  wrote {out_png.name}   (calibration {matrix_path.name}, "
              f"{resized} resized tiles)")
    return {"name": name, "matrix": matrix_path.name, "exact": used_date == session_date,
            "resized": resized, "n_nodes": len(nodes)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="substring filter on the mapping name, e.g. NPC4")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="also write one page with every session side by side")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrices = load_matrices()
    print(f"calibrations available: {', '.join(sorted(matrices))}\n")

    mappings = sorted(p for p in (RESOURCES / "maze_mappings").glob("*.pkl")
                      if not args.only or args.only in p.stem)
    rows = []
    for path in mappings:
        row = render(path, matrices, out_png=OUT_DIR / f"{path.stem}_overlay.png")
        if row:
            rows.append(row)

    if args.contact_sheet and rows:
        n = len(rows)
        cols = 4
        sheet_rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(sheet_rows, cols, figsize=(6 * cols, 5 * sheet_rows), dpi=95)
        for ax, path in zip(np.atleast_1d(axes).ravel(),
                            [p for p in mappings
                             if (RESOURCES / "maze_images" / f"{p.stem}.png").exists()]):
            render(path, matrices, ax=ax)
        for ax in np.atleast_1d(axes).ravel()[n:]:
            ax.axis("off")
        fig.suptitle("Session mappings on their own calibrated images "
                     "(yellow = resized tile)", fontsize=15)
        fig.tight_layout()
        sheet = OUT_DIR / "all_sessions_contact_sheet.png"
        fig.savefig(sheet, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"\nwrote {sheet}")

    print(f"\n{'mapping':26s} {'calibration':16s} {'exact date':>11s} {'resized tiles':>14s}")
    for r in rows:
        print(f"  {r['name']:24s} {r['matrix']:16s} {str(r['exact']):>11s} "
              f"{r['resized']:>10d}/{r['n_nodes']}")


if __name__ == "__main__":
    main()
