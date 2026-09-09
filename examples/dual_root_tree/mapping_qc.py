"""Validate a session's maze mapping + calibration against the actual video.

Two checks, both anchored on ground truth from BPOD:

  1. Visual  — warps a video frame into mapping space with the calibration
     matrix and draws the node polygons on top, so you can see whether the
     graph sits on the right pixels.
  2. Numeric — for every reward event, takes the mouse's tracked position at
     that moment, maps it to a node, and asks whether it is the port BPOD
     says was rewarded. A good calibration puts nearly every reward inside
     its own port; a bad one scatters them into parents, edges, or nowhere.

Run it for a new mouse before trusting any port-based analysis:

    python mapping_qc.py --subject NPC4 --session 10/05/2026
    python mapping_qc.py --subject NPC1 --session 10/09/2025   # known-good reference

`--all-matrices` scores every calibration matrix on disk, which is how you
find out whether a session is using the wrong one.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
RESOURCES = HERE / "resources"
BPOD_ROOT = Path("/var/home/almogmeir/Documents/M.Sc/Project/BPODData")

# Port layout per subject, mirroring PORT_LAYOUTS in align_frames_trials.ipynb:
# BPOD port number -> leaf node.
PORT_MAPS = {
    "NPC1": {1: "R522", 2: "R526", 3: "L56", 4: "L510",
             5: "L521", 6: "L525", 7: "R59", 8: "R55"},
    "NPC4": {1: "R522", 2: "R526", 3: "L510", 4: "L56",
             5: "L521", 6: "L525", 7: "R59", 8: "R55"},
}


def session_parts(session):
    """'DD/MM/YYYY' -> (DD_MM_YYYY, YYYY_MM_DD, YYYYMMDD)."""
    dd, mm, yyyy = session[:2], session[3:5], session[6:10]
    return f"{dd}_{mm}_{yyyy}", f"{yyyy}_{mm}_{dd}", f"{yyyy}{mm}{dd}"


def load_nodes(mapping_name):
    data = pickle.load(open(RESOURCES / "maze_mappings" / f"{mapping_name}.pkl", "rb"))
    return data["mappings"]["nodes"]


def check_calibration_match(mapping_name, matrix_path):
    """Warn if the mapping was aligned against a different calibration.

    The GUI positions tiles on the session image *after* warping it with
    setup.calibration_matrix; the analysis projects tracking with whatever the
    calibration plugin loads. If those two differ, every node is displaced by
    the difference between the matrices, which looks like a mapping error but
    is not one. Mappings saved before this stamp existed report 'unknown'.
    """
    import hashlib

    data = pickle.load(open(RESOURCES / "maze_mappings" / f"{mapping_name}.pkl", "rb"))
    stamp = (data.get("setup_mode") or {}).get("calibration")
    if not stamp:
        print(f"  calibration stamp: none recorded (mapping predates the stamp) — "
              f"cannot verify it matches {Path(matrix_path).name}")
        return None

    used = np.load(matrix_path).astype(np.float64)
    used_md5 = hashlib.md5(used.tobytes()).hexdigest()
    if stamp.get("md5") == used_md5:
        print(f"  calibration stamp: matches {Path(matrix_path).name} ✓")
        return True

    stamped = np.asarray(stamp.get("matrix"), dtype=np.float64) if stamp.get("matrix") else None
    print(f"  calibration stamp: MISMATCH")
    print(f"     tiles aligned with : {stamp.get('path')}")
    print(f"     analysis will use  : {matrix_path}")
    if stamped is not None and stamped.shape == (3, 3):
        corners = np.array([[x, y, 1.0] for x in (0, 600, 1200) for y in (0, 500, 1000)])
        a = corners @ used.T; a = a[:, :2] / a[:, 2:]
        b = corners @ stamped.T; b = b[:, :2] / b[:, 2:]
        d = a - b
        print(f"     every node is displaced by dx {d[:, 0].mean():+.1f}, "
              f"dy {d[:, 1].mean():+.1f} px on average")
    return False


def reward_positions(subject, session):
    """[(port_number, x, y)] — tracked headstage position at each reward."""
    dd_mm_yyyy, _, yyyymmdd = session_parts(session)
    summary = pd.read_pickle(RESOURCES / "outputs" / subject / dd_mm_yyyy / "summary_df.pkl")
    summary.columns = [c[0] if c[1] == "" else "_".join(c) for c in summary.columns]

    reward_csv = (BPOD_ROOT / subject / "maze_pairs_fix" / "Session Data" / "output"
                  / yyyymmdd / "reward.csv")
    rewards = pd.read_csv(reward_csv)
    # Column was renamed between subjects.
    rel_col = "reward_relative_time" if "reward_relative_time" in rewards else "relative_time"

    points = []
    for row in rewards.itertuples(index=False):
        row = row._asdict()
        trial = summary[summary["trial_idx"] == row["trial"] - 1]
        if trial.empty:
            continue
        idx = (trial["frame_time_in_trial"] - row[rel_col]).abs().idxmin()
        x, y = summary.loc[idx, "headstage_x"], summary.loc[idx, "headstage_y"]
        if np.isfinite(x) and np.isfinite(y):
            points.append((row["port_number"], x, y))
    return points, len(rewards)


def node_lookup(nodes):
    paths = {name: MplPath(np.asarray(polys[0], float)) for name, polys in nodes.items()}

    def at(x, y):
        for name, path in paths.items():
            if path.contains_point((x, y)):
                return name
        return None

    return at


def transform(matrix, x, y):
    v = matrix @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def score(points, matrix, nodes, port_map, shift=(0.0, 0.0)):
    """-> (n_correct, n_in_any_node, Counter of wrong nodes)."""
    at = node_lookup(nodes)
    correct = in_node = 0
    wrong = []
    for port, x, y in points:
        tx, ty = transform(matrix, x, y)
        name = at(tx + shift[0], ty + shift[1])
        in_node += name is not None
        if name == port_map.get(port):
            correct += 1
        elif name is not None:
            wrong.append(name)
    return correct, in_node, pd.Series(wrong).value_counts()


def port_offsets(points, matrix, nodes, port_map):
    """Mean offset of each reward position from its own port tile's centre.

    A correct registration leaves only a radial spread — the mouse licks at the
    spout, which is at the tip of the leaf arm, not at the tile centre — and
    those radial offsets cancel across the eight ports. A non-zero overall mean
    is a translation error between the camera calibration and the tile mapping.
    """
    per_port = {}
    for port, x, y in points:
        node = port_map.get(port)
        if node is None:
            continue
        centre = np.asarray(nodes[node][0], float).mean(axis=0)
        per_port.setdefault(node, []).append(np.array(transform(matrix, x, y)) - centre)
    return {k: np.array(v) for k, v in per_port.items()}


def suggest_shift(points, matrix, nodes, port_map, span=40):
    """Brute-force the translation that puts the most rewards in their port."""
    best = (-1, 0, 0)
    for dx in range(-span, span + 1, 2):
        for dy in range(-span, span + 1, 2):
            correct, _, _ = score(points, matrix, nodes, port_map, (dx, dy))
            if correct > best[0]:
                best = (correct, dx, dy)
    return best


def render(subject, session, mapping_name, matrix_path, points, out_png, frame_idx=10000):
    _, yyyy_mm_dd, _ = session_parts(session)
    video = RESOURCES / "maze_videos" / f"{subject}_{yyyy_mm_dd}.mp4"
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx} from {video}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    matrix = np.load(matrix_path)
    height, width = frame.shape[:2]
    warped = cv2.warpPerspective(frame, matrix, (width, height))

    nodes = load_nodes(mapping_name)
    port_map = PORT_MAPS[subject]
    ports = set(port_map.values())
    at = node_lookup(nodes)

    fig, ax = plt.subplots(figsize=(15, 15 * height / width), dpi=110)
    ax.imshow(warped)
    for name, polys in nodes.items():
        pts = np.asarray(polys[0], float)
        is_port = name in ports
        ax.add_patch(MplPoly(pts, closed=True, fill=False,
                             edgecolor="#ff2d55" if is_port else "#00e5ff",
                             lw=2.2 if is_port else 0.7,
                             alpha=1.0 if is_port else 0.45))
        if is_port:
            ax.text(pts[:, 0].mean(), pts[:, 1].mean(), name, color="white",
                    fontsize=7, ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="#ff2d55", ec="none", alpha=0.85))

    correct = 0
    for port, x, y in points:
        tx, ty = transform(matrix, x, y)
        good = at(tx, ty) == port_map.get(port)
        correct += good
        ax.plot(tx, ty, "o", ms=7, zorder=5, mec="black", mew=0.8,
                mfc="#00ff85" if good else "#ff9500")

    pct = 100 * correct / len(points) if points else 0
    ax.set_title(f"{subject}  {session}   ·   mapping {mapping_name}   ·   "
                 f"calibration {Path(matrix_path).name}\n"
                 f"{correct}/{len(points)} rewards inside their own port ({pct:.0f}%)   "
                 f"·   red = reward ports, green dot = reward in the right port",
                 fontsize=11)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="e.g. NPC4")
    ap.add_argument("--session", required=True, help="DD/MM/YYYY, e.g. 10/05/2026")
    ap.add_argument("--mapping", help="maze_mappings stem; default <SUBJECT>_<YYYY_MM_DD>")
    ap.add_argument("--matrix", default=str(RESOURCES / "transform_matrix.npy"),
                    help="calibration .npy to render with (default: the one the "
                         "analysis pipeline actually loads)")
    ap.add_argument("--all-matrices", action="store_true",
                    help="score every calibration matrix on disk")
    ap.add_argument("--suggest-shift", action="store_true",
                    help="report per-port offsets and search for the translation "
                         "that best registers tracking to the tiles")
    ap.add_argument("--out", help="output PNG path")
    args = ap.parse_args()

    _, yyyy_mm_dd, _ = session_parts(args.session)
    mapping_name = args.mapping or f"{args.subject}_{yyyy_mm_dd}"
    port_map = PORT_MAPS[args.subject]

    points, n_rewards = reward_positions(args.subject, args.session)
    print(f"{args.subject} {args.session}: {len(points)} of {n_rewards} rewards "
          f"have valid tracking\n")

    nodes = load_nodes(mapping_name)
    check_calibration_match(mapping_name, args.matrix)
    print()
    candidates = [Path(args.matrix)]
    if args.all_matrices:
        candidates = ([RESOURCES / "transform_matrix.npy"]
                      + sorted((RESOURCES / "transform_matrix").glob("*.npy")))

    print(f"{'calibration matrix':26s} {'in a node':>11s} {'correct port':>14s}  most common wrong node")
    for path in candidates:
        correct, in_node, wrong = score(points, np.load(path), nodes, port_map)
        pct = 100 * correct / len(points) if points else 0
        top = ", ".join(f"{k}×{v}" for k, v in wrong.head(3).items()) or "-"
        print(f"  {path.name:24s} {in_node:6d}/{len(points):<4d} "
              f"{correct:7d}/{len(points):<4d} ({pct:3.0f}%)  {top}")

    if args.suggest_shift:
        matrix = np.load(args.matrix)
        print(f"\nper-port offset from tile centre (radial spread is normal; "
              f"a non-zero overall mean is a registration error)")
        offs = port_offsets(points, matrix, nodes, port_map)
        for node, arr in sorted(offs.items()):
            print(f"  {node:6s} n={len(arr):3d}  mean=({arr[:, 0].mean():6.1f},{arr[:, 1].mean():6.1f})")
        allo = np.vstack(list(offs.values()))
        print(f"  {'ALL':6s} n={len(allo):3d}  mean=({allo[:, 0].mean():6.1f},{allo[:, 1].mean():6.1f})")
        correct, dx, dy = suggest_shift(points, matrix, nodes, port_map)
        base, _, _ = score(points, matrix, nodes, port_map)
        print(f"\nbest translation: dx={dx:+d}, dy={dy:+d}  ->  {correct}/{len(points)} correct "
              f"({100*correct/len(points):.0f}%), up from {base}/{len(points)} "
              f"({100*base/len(points):.0f}%)")
        if abs(dx) > 5 or abs(dy) > 5:
            print("  A shift this large means the calibration for this session is "
                  "mis-registered, or the per-session tile adjustment was not applied.")

    out = args.out or str(HERE / "resources" / "mapping_qc"
                          / f"{args.subject}_{yyyy_mm_dd}_mapping_qc.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    render(args.subject, args.session, mapping_name, args.matrix, points, out)


if __name__ == "__main__":
    main()
