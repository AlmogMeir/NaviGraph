"""Behavioral-state features and anchor-based validation for HMM state inference.

This module turns the per-node-visit traversal table (``node_df``) + frame-level
``summary_df`` into a tidy per-node-visit feature frame suitable for fitting
unsupervised HMMs that infer a latent behavioral state (e.g. *explore*,
*exploit / goal-directed*, *change-of-mind*) at the **graph-location level**
(one state per node-visit, not per video frame).

Because there is no ground-truth state label, the module also attaches *weak
labels* ("anchors") encoding behaviorally-motivated expectations, and provides
:func:`validate_states` to score an inferred state sequence against them:

- ``anchor_exploit``          direct alternation within a reward patch  → exploit
- ``anchor_change_of_mind``   the path-commitment apex node             → change-of-mind
- ``anchor_explore``          a node on a long indirect (detour) path   → explore

The anchor definitions mirror the notebooks:
``direct_path_analysis.ipynb`` (``target_group`` alternation-vs-switch) and
``change_of_mind.ipynb`` (long-indirect ``excess_nodes >= 4`` and the
``direct_start_node`` commitment apex).

Design goals
------------
- **Dependency-light**: only pandas / numpy / networkx / scipy, so it imports
  without hmmlearn / IOHMM / ssm. The modelling lives in the notebook.
- **Reuses** :func:`navigraph.analysis.speed_analysis.compute_path_speed` for the
  per-frame speed, aggregated to node-visit windows.
- **Turn type is computed directly from the graph** (:func:`classify_turn`,
  the same geometry as ``turn_analysis.ipynb``'s ``classify_turn``), walked
  continuously across the *whole session* rather than restarted per trial —
  this needs no precomputed ``turn_df.csv`` (most sessions don't have one) and
  avoids the spurious ``NONE``s a per-trial walk produces at every trial
  (reward) boundary, even though the mouse's path is physically continuous
  there.

Typical usage
-------------
    import networkx as nx, pandas as pd
    from navigraph.analysis.state_features import (
        build_graph_from_full_maze, build_state_feature_frame,
        add_weak_labels, build_design_matrix, validate_states,
    )

    G = build_graph_from_full_maze('FullMazeGrid.pkl')
    feat = build_state_feature_frame(node_df, summary_df, G, fps=40)
    feat = add_weak_labels(feat, direct_opportunities_df)
    X, cols = build_design_matrix(feat)          # z-scored covariate matrix
    # ... fit HMM -> state_seq (len == len(feat)) ...
    report = validate_states(state_seq, feat)
"""

from __future__ import annotations

import ast
import pickle
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:  # networkx is a hard dep of navigraph, but keep the import defensive
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

from .speed_analysis import compute_path_speed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FPS: float = 40.0
#: Ordered turn categories; ``NONE`` covers node-visits with no defined turn
#: (reward ports, leaves, tracking-violation triples).
TURN_CATEGORIES: List[str] = ["LEFT", "RIGHT", "STRAIGHT", "UTURN", "NONE"]

#: Continuous covariates that get z-scored in :func:`build_design_matrix`.
CONTINUOUS_FEATURES: List[str] = [
    "speed_mean",
    "log_dwell",
    "node_depth",
    "dist_to_goal",
    "path_excess",
    "log_frames_since_reward",
    "local_reward_rate",
    "abs_graph_angle",
]
#: Binary covariates passed through as 0/1.
BINARY_FEATURES: List[str] = [
    "is_port_node",
    "reward_during_node",
    "is_indirect",
    "is_alternation",
]

_DEPTH_RE = re.compile(r"^[LR](\d)")


# ---------------------------------------------------------------------------
# Graph geometry
# ---------------------------------------------------------------------------

def build_graph_from_full_maze(
    full_maze_grid_path: str = "FullMazeGrid.pkl",
) -> "nx.Graph":
    """Build an undirected maze graph from ``FullMazeGrid.pkl``.

    The pickle stores ``mappings['nodes']`` (node -> list of polygon rings) and
    ``mappings['edges']`` (keys ``'A_B'``). We add every node and a bidirectional
    edge for each ``A_B`` key. Undirected is sufficient for the tree-distance
    (``dist_to_goal``) used downstream. Node centroids are stored as the
    ``'pos'`` attribute for convenience.
    """
    if nx is None:  # pragma: no cover
        raise ImportError("networkx is required for build_graph_from_full_maze")
    with open(full_maze_grid_path, "rb") as fh:
        data = pickle.load(fh)
    mappings = data["mappings"] if "mappings" in data else data
    G = nx.Graph()
    for node, rings in mappings["nodes"].items():
        poly = np.asarray(rings[0], dtype=float)
        G.add_node(node, pos=tuple(poly.mean(axis=0)))
    for edge_key in mappings["edges"]:
        if "_" in str(edge_key):
            a, b = str(edge_key).split("_", 1)
            if a in G and b in G:
                G.add_edge(a, b)
    return G


def node_depth(node_id: object) -> Optional[int]:
    """Depth from the root encoded in the node name (``L31`` -> 3). ``None`` if unparseable."""
    m = _DEPTH_RE.match(str(node_id))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Turn classification (mirrors turn_analysis.ipynb's classify_turn exactly)
# ---------------------------------------------------------------------------

def classify_turn(
    prev_id: object, curr_id: object, next_id: object, node_pos: Dict[object, Tuple[float, float]],
) -> Tuple[str, float]:
    """Signed turn angle from fixed graph-node positions, at ``curr_id``.

    Same geometry as ``turn_analysis.ipynb``'s ``classify_turn``: y increases
    downward (image coordinates), so ``cross_z > 0`` -> RIGHT (clockwise from
    above), ``cross_z < 0`` -> LEFT. ``|angle| < 20`` -> STRAIGHT,
    ``|angle| > 160`` -> UTURN.
    """
    px, py = node_pos[prev_id]
    cx, cy = node_pos[curr_id]
    nx_, ny = node_pos[next_id]

    v1 = np.array([cx - px, cy - py], dtype=float)
    v2 = np.array([nx_ - cx, ny - cy], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return "STRAIGHT", 0.0

    v1n, v2n = v1 / n1, v2 / n2
    cross_z = v1n[0] * v2n[1] - v1n[1] * v2n[0]
    dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    angle = float(np.degrees(np.arctan2(cross_z, dot)))

    if abs(angle) < 20:
        return "STRAIGHT", angle
    if abs(angle) > 160:
        return "UTURN", angle
    return ("RIGHT" if cross_z > 0 else "LEFT"), angle


def turns_from_graph(node_names: Sequence[object], G: "nx.Graph") -> Tuple[List[str], List[float]]:
    """Turn type + graph angle for every node-visit in a **session-continuous**
    node sequence (not restarted per trial — the mouse's path is physically
    continuous across trial/reward boundaries).

    ``NONE`` only for the first/last visit of the sequence or a triple where
    either hop is not a real graph edge (traversal violation / bounce
    artifact), matching ``turn_analysis.ipynb``'s "drop traversal violations"
    rule.
    """
    pos = nx.get_node_attributes(G, "pos")
    n = len(node_names)
    turn_type = ["NONE"] * n
    angle = [np.nan] * n
    for i in range(1, n - 1):
        prev_n, curr_n, next_n = node_names[i - 1], node_names[i], node_names[i + 1]
        if not all(nd in pos for nd in (prev_n, curr_n, next_n)):
            continue
        if not (G.has_edge(prev_n, curr_n) and G.has_edge(curr_n, next_n)):
            continue
        tt, ang = classify_turn(prev_n, curr_n, next_n, pos)
        turn_type[i] = tt
        angle[i] = ang
    return turn_type, angle


# ---------------------------------------------------------------------------
# Column resolution (summary_df may be flat or MultiIndex)
# ---------------------------------------------------------------------------

def _resolve(df: pd.DataFrame, wanted: Sequence[str]) -> Union[str, tuple]:
    """Resolve a summary_df column from candidate level-substrings.

    Returns the first column whose (any-level) name set contains all *wanted*
    substrings (case-insensitive). Raises KeyError if none match.
    """
    wanted_l = [w.lower() for w in wanted]
    for col in df.columns:
        parts = [str(v).strip().lower() for v in (col if isinstance(col, tuple) else (col,))]
        if all(any(w == p or w in p for p in parts) for w in wanted_l):
            return col
    raise KeyError(f"No summary_df column matching {wanted}; have e.g. {list(df.columns[:6])}")


# ---------------------------------------------------------------------------
# Speed: one pass over the session, binned to node-visit windows
# ---------------------------------------------------------------------------

def _node_visit_speed(
    node_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fps: float,
    body_part: str = "headstage",
    min_likelihood: Optional[float] = 0.5,
) -> np.ndarray:
    """Mean frame-by-frame speed (px/s) attributed to each node-visit.

    Computes per-frame speed once over the whole session via
    :func:`compute_path_speed`, then averages the samples falling inside each
    node-visit's window (fast, via ``searchsorted`` on sorted mid-frames).

    The window for row *i* is ``[end_frame[i-1], end_frame[i]]`` — i.e. it
    spans the edge traversed *into* this node plus the node dwell itself, not
    just ``[start_frame[i], end_frame[i]]``. Many node-visits last a single
    frame (``start_frame == end_frame``), which would otherwise leave no
    frame pair to difference and yield ``NaN`` speed for ~29% of rows; dwell
    time (``duration_frames`` / ``log_dwell``) is unaffected and still comes
    strictly from the node's own window.
    """
    x_col = _resolve(summary_df, [body_part, "x"])
    y_col = _resolve(summary_df, [body_part, "y"])
    frame_col = _resolve(summary_df, ["frame_idx_global"])
    like_col = None
    try:
        like_col = _resolve(summary_df, [body_part, "likelihood"])
    except KeyError:
        pass

    fmin = int(node_df["start_frame"].min())
    fmax = int(node_df["end_frame"].max())

    # Re-slice identically to compute_path_speed to recover the mid-frame axis.
    fc = frame_col
    sub = summary_df[(summary_df[fc] >= fmin) & (summary_df[fc] <= fmax)].sort_values(fc)
    frames = pd.to_numeric(sub[fc], errors="coerce").values.astype(float)
    if len(frames) < 2:
        return np.full(len(node_df), np.nan)
    mid_frames = (frames[:-1] + frames[1:]) / 2.0

    _, speed, _ = compute_path_speed(
        summary_df, fmin, fmax, fps=fps,
        x_col=x_col, y_col=y_col, frame_col=frame_col,
        min_likelihood=min_likelihood,
        likelihood_col=like_col if like_col is not None else "headstage_likelihood",
    )
    # speed aligns with mid_frames (both length N-1)
    n = min(len(speed), len(mid_frames))
    mid_frames, speed = mid_frames[:n], speed[:n]

    out = np.full(len(node_df), np.nan)
    ends = node_df["end_frame"].values.astype(float)
    # window i = [end_frame[i-1], end_frame[i]]; first row falls back to its own start_frame
    starts = np.empty(len(node_df))
    starts[0] = node_df["start_frame"].values[0]
    starts[1:] = ends[:-1]
    lo = np.searchsorted(mid_frames, starts, side="left")
    hi = np.searchsorted(mid_frames, ends, side="right")
    for i in range(len(node_df)):
        if hi[i] > lo[i]:
            seg = speed[lo[i]:hi[i]]
            if np.any(~np.isnan(seg)):
                out[i] = np.nanmean(seg)
    return out


# ---------------------------------------------------------------------------
# Segment-level helpers (destination port + alternation) via path_pair_label runs
# ---------------------------------------------------------------------------

def _segment_id(path_pair_label: pd.Series) -> np.ndarray:
    """Integer id per contiguous run of identical (non-null) ``path_pair_label``.

    ``-1`` for rows with no path label (off-segment wandering / port dwell).
    """
    lab = path_pair_label.astype(object).where(path_pair_label.notna(), None).values
    seg = np.full(len(lab), -1, dtype=int)
    cur = -1
    prev = None
    for i, v in enumerate(lab):
        if v is None:
            prev = None
            continue
        if v != prev:
            cur += 1
        seg[i] = cur
        prev = v
    return seg


def _target_group(label: object) -> Optional[int]:
    """Reward-pair group index for a ``'TargetN'`` label (ports [1,2]->0, [3,4]->1, ...)."""
    m = re.search(r"\d+", str(label))
    return (int(m.group()) - 1) // 2 if m else None


def _parse_pair(label: object) -> Tuple[Optional[str], Optional[str]]:
    try:
        t = ast.literal_eval(label)
        return (t[0], t[1]) if isinstance(t, (tuple, list)) and len(t) >= 2 else (None, None)
    except (ValueError, SyntaxError, TypeError):
        return (None, None)


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_state_feature_frame(
    node_df: pd.DataFrame,
    summary_df: Optional[pd.DataFrame],
    G: "nx.Graph",
    fps: float = FPS,
    reward_window: int = 5,
) -> pd.DataFrame:
    """Build the per-node-visit feature frame (one row per ``node_df`` row).

    Parameters
    ----------
    node_df:
        Node-visit table (``build_node_df`` output / ``node_df.csv``). Must be in
        chronological order; sorted by ``node_visit_idx`` internally.
    summary_df:
        Frame-level tracking table for speed. May be ``None`` (``speed_mean`` NaN).
    G:
        Maze graph for ``dist_to_goal`` and ``turn_type`` (from
        :func:`build_graph_from_full_maze`).
    fps:
        Video frame rate.
    reward_window:
        Number of recent port-visits over which ``local_reward_rate`` is computed.

    Returns
    -------
    DataFrame with one row per node-visit: identity columns
    (``node_visit_idx, node_name, trial_idx, start_frame, end_frame``), the
    covariate columns in :data:`CONTINUOUS_FEATURES` + :data:`BINARY_FEATURES`,
    ``turn_type`` (the categorical emission target, computed from ``G`` via
    :func:`turns_from_graph`) and ``prev_turn_type``.
    """
    df = node_df.sort_values("node_visit_idx").reset_index(drop=True).copy()

    # --- dwell ---
    df["log_dwell"] = np.log1p(df["duration_frames"].astype(float))

    # --- node depth (0 root .. 5 port/leaf) ---
    df["node_depth"] = df["node_name"].map(node_depth).astype(float)

    # --- path detour ---
    df["path_excess"] = (df["path_seq_length_unique"] - df["path_shortest_length"]).astype(float)
    df["is_indirect"] = (df["port_to_port_path_type"] == "indirect").astype(int)

    # --- segment structure: destination node + alternation ---
    seg = _segment_id(df["path_pair_label"])
    df["_seg"] = seg
    dest_of_seg: Dict[int, object] = {}
    grp_of_seg: Dict[int, Optional[int]] = {}
    for s, sub in df.groupby("_seg"):
        if s < 0:
            continue
        dest_of_seg[s] = sub.iloc[-1]["node_name"]            # arrival port node
        src_l, dst_l = _parse_pair(sub.iloc[0]["path_pair_label"])
        gs, gd = _target_group(src_l), _target_group(dst_l)
        grp_of_seg[s] = int(gs == gd) if (gs is not None and gd is not None) else None

    def _dist_to_goal(row) -> float:
        s = row["_seg"]
        if s < 0 or s not in dest_of_seg:
            return np.nan
        dest = dest_of_seg[s]
        try:
            return float(nx.shortest_path_length(G, row["node_name"], dest))
        except Exception:
            return np.nan

    df["dist_to_goal"] = df.apply(_dist_to_goal, axis=1)
    df["is_alternation"] = df["_seg"].map(
        lambda s: grp_of_seg.get(s) if s in grp_of_seg else None
    ).astype("float")  # 1 alternation, 0 switch, NaN off-segment

    # --- reward history ---
    rewarded = df["reward_during_node"].fillna(False).astype(bool).values
    frames_since = np.full(len(df), np.nan)
    last_reward_frame: Optional[float] = None
    starts = df["start_frame"].values.astype(float)
    for i in range(len(df)):
        if last_reward_frame is not None:
            frames_since[i] = starts[i] - last_reward_frame
        if rewarded[i]:
            last_reward_frame = df["end_frame"].values[i]
    df["frames_since_reward"] = frames_since
    df["log_frames_since_reward"] = np.log1p(df["frames_since_reward"].clip(lower=0))

    # local reward rate over the last `reward_window` PORT visits, ffilled to all rows
    port_mask = df["is_port_node"].fillna(False).astype(bool)
    port_rewarded = pd.Series(np.where(port_mask, rewarded, np.nan), index=df.index)
    port_rate = port_rewarded.dropna().rolling(reward_window, min_periods=1).mean()
    df["local_reward_rate"] = port_rate.reindex(df.index).ffill().fillna(0.0)

    # --- turn type: computed directly from G, walked continuously across the
    # whole session (not restarted per trial -- the path is physically
    # continuous across trial/reward boundaries). NONE only for the first/last
    # visit of the session or a non-edge triple (bounce/traversal violation).
    turn_type, graph_angle = turns_from_graph(df["node_name"].tolist(), G)
    df["turn_type"] = turn_type
    df["graph_angle_deg"] = graph_angle
    df["turn_type"] = pd.Categorical(df["turn_type"], categories=TURN_CATEGORIES)
    df["abs_graph_angle"] = df["graph_angle_deg"].abs()
    df["prev_turn_type"] = df["turn_type"].shift(1).astype(object).fillna("NONE")

    # --- speed ---
    if summary_df is not None:
        df["speed_mean"] = _node_visit_speed(df, summary_df, fps)
    else:
        df["speed_mean"] = np.nan

    # --- binary casts ---
    df["is_port_node"] = df["is_port_node"].fillna(False).astype(int)
    df["reward_during_node"] = df["reward_during_node"].fillna(False).astype(int)
    df["is_alternation"] = df["is_alternation"].fillna(0).astype(int)  # off-segment -> 0

    keep_cols = (
        ["node_visit_idx", "node_name", "trial_idx", "start_frame", "end_frame",
         "port_to_port_path_type", "path_pair_label", "_seg"]
        + CONTINUOUS_FEATURES + BINARY_FEATURES
        + ["turn_type", "prev_turn_type", "graph_angle_deg"]
    )
    return df[keep_cols].copy()


# ---------------------------------------------------------------------------
# Weak labels (anchors)
# ---------------------------------------------------------------------------

def add_weak_labels(
    feat_df: pd.DataFrame,
    direct_opportunities_df: Optional[pd.DataFrame],
    excess_threshold: int = 4,
    commitment_window: int = 0,
) -> pd.DataFrame:
    """Attach boolean anchor columns used by :func:`validate_states`.

    - ``anchor_exploit``        direct path within one reward patch (alternation).
    - ``anchor_explore``        node on a long indirect path (``path_excess >= excess_threshold``).
    - ``anchor_change_of_mind`` the commitment apex node (``direct_start_node_visit_idx``
      from ``direct_opportunities_unique_paths.csv``), optionally widened by
      ``commitment_window`` node-visits on each side.
    """
    df = feat_df.copy()
    df["anchor_exploit"] = (
        (df["port_to_port_path_type"] == "direct") & (df["is_alternation"] == 1)
    ).astype(bool)
    df["anchor_explore"] = (
        (df["port_to_port_path_type"] == "indirect")
        & (df["path_excess"] >= excess_threshold)
    ).astype(bool)

    df["anchor_change_of_mind"] = False
    if direct_opportunities_df is not None and "direct_start_node_visit_idx" in direct_opportunities_df:
        apex = direct_opportunities_df["direct_start_node_visit_idx"].dropna().astype(int).tolist()
        apex_set = set()
        for a in apex:
            for w in range(-commitment_window, commitment_window + 1):
                apex_set.add(a + w)
        df["anchor_change_of_mind"] = df["node_visit_idx"].isin(apex_set)
    return df


# ---------------------------------------------------------------------------
# Design matrix for HMM covariates
# ---------------------------------------------------------------------------

def build_design_matrix(
    feat_df: pd.DataFrame,
    continuous: Optional[List[str]] = None,
    binary: Optional[List[str]] = None,
    add_intercept: bool = False,
    winsorize_pct: float = 1.0,
) -> Tuple[np.ndarray, List[str]]:
    """Assemble a numeric covariate matrix: z-scored continuous + 0/1 binary.

    NaNs in continuous features are median-imputed and values are winsorized to
    the ``[winsorize_pct, 100-winsorize_pct]`` percentile range (guards against
    tracking-jump speed outliers) before z-scoring, so the HMM fitters never see
    missing or extreme values. Returns ``(X, column_names)``.
    """
    continuous = CONTINUOUS_FEATURES if continuous is None else continuous
    binary = BINARY_FEATURES if binary is None else binary

    cols: List[str] = []
    mats: List[np.ndarray] = []
    for c in continuous:
        v = pd.to_numeric(feat_df[c], errors="coerce").values.astype(float)
        med = np.nanmedian(v)
        v = np.where(np.isnan(v), med if np.isfinite(med) else 0.0, v)
        if winsorize_pct and winsorize_pct > 0:
            lo, hi = np.percentile(v, [winsorize_pct, 100 - winsorize_pct])
            if hi > lo:
                v = np.clip(v, lo, hi)
        sd = v.std()
        v = (v - v.mean()) / sd if sd > 1e-9 else v - v.mean()
        mats.append(v[:, None]); cols.append(c)
    for b in binary:
        v = pd.to_numeric(feat_df[b], errors="coerce").fillna(0).values.astype(float)
        mats.append(v[:, None]); cols.append(b)
    X = np.hstack(mats) if mats else np.empty((len(feat_df), 0))
    if add_intercept:
        X = np.hstack([np.ones((len(X), 1)), X]); cols = ["intercept"] + cols
    return X, cols


# ---------------------------------------------------------------------------
# Validation without ground truth
# ---------------------------------------------------------------------------

def _enrichment(state_seq: np.ndarray, anchor: np.ndarray, state: int) -> Dict[str, float]:
    """2x2 enrichment of `state` at `anchor` nodes: odds ratio + Fisher p."""
    from scipy.stats import fisher_exact

    in_state = state_seq == state
    a = int(np.sum(in_state & anchor))       # state & anchor
    b = int(np.sum(in_state & ~anchor))      # state & ~anchor
    c = int(np.sum(~in_state & anchor))      # ~state & anchor
    d = int(np.sum(~in_state & ~anchor))     # ~state & ~anchor
    table = [[a, b], [c, d]]
    try:
        odds, p = fisher_exact(table, alternative="greater")
    except Exception:
        odds, p = np.nan, np.nan
    frac_anchor_in_state = a / (a + c) if (a + c) else np.nan
    return {"odds_ratio": float(odds), "p": float(p),
            "n_anchor_in_state": a, "frac_anchor_in_state": float(frac_anchor_in_state)}


def validate_states(
    state_seq: Sequence[int],
    feat_df: pd.DataFrame,
    anchors: Sequence[str] = ("anchor_exploit", "anchor_change_of_mind", "anchor_explore"),
) -> Dict[str, object]:
    """Score an inferred state sequence against weak-label anchors (no ground truth).

    Returns a dict with:

    - ``enrichment``: for each anchor, the best (max odds-ratio) state and the
      full per-state odds-ratio / Fisher-p table. A good model has a distinct
      state enriched for each anchor.
    - ``state_feature_means``: per-state mean of the continuous features
      (face-validity check).
    - ``stickiness``: per-state self-transition probability and mean dwell length
      (consecutive same-state run).
    - ``occupancy``: fraction of node-visits in each state.
    """
    state_seq = np.asarray(state_seq)
    states = sorted(np.unique(state_seq).tolist())

    # --- anchor enrichment ---
    enrichment: Dict[str, object] = {}
    for anc in anchors:
        if anc not in feat_df:
            continue
        av = feat_df[anc].fillna(False).astype(bool).values
        per_state = {s: _enrichment(state_seq, av, s) for s in states}
        best = max(per_state.items(), key=lambda kv: (kv[1]["odds_ratio"]
                                                       if np.isfinite(kv[1]["odds_ratio"]) else -1))
        enrichment[anc] = {"best_state": int(best[0]),
                           "best_odds_ratio": best[1]["odds_ratio"],
                           "best_p": best[1]["p"],
                           "n_anchor": int(av.sum()),
                           "per_state": per_state}

    # --- state-conditioned feature means ---
    fm = feat_df.copy()
    fm["_state"] = state_seq
    means_cols = [c for c in CONTINUOUS_FEATURES if c in fm]
    state_feature_means = fm.groupby("_state")[means_cols].mean()

    # --- stickiness / dwell ---
    stickiness: Dict[int, Dict[str, float]] = {}
    for s in states:
        mask = state_seq == s
        # self-transition prob among steps currently in s
        cur = state_seq[:-1] == s
        stay = (state_seq[:-1] == s) & (state_seq[1:] == s)
        p_self = float(stay.sum() / cur.sum()) if cur.sum() else np.nan
        # mean run length
        runs, run = [], 0
        for v in mask:
            if v:
                run += 1
            elif run:
                runs.append(run); run = 0
        if run:
            runs.append(run)
        stickiness[int(s)] = {"p_self": p_self,
                              "mean_dwell": float(np.mean(runs)) if runs else 0.0,
                              "n_runs": len(runs)}

    occupancy = {int(s): float(np.mean(state_seq == s)) for s in states}
    return {"enrichment": enrichment,
            "state_feature_means": state_feature_means,
            "stickiness": stickiness,
            "occupancy": occupancy,
            "n_states": len(states)}


__all__ = [
    "FPS", "TURN_CATEGORIES", "CONTINUOUS_FEATURES", "BINARY_FEATURES",
    "build_graph_from_full_maze", "node_depth", "classify_turn", "turns_from_graph",
    "build_state_feature_frame", "add_weak_labels",
    "build_design_matrix", "validate_states",
]
