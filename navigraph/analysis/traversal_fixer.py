"""Reconstruct a validated node-edge traversal sequence from raw tracking.

`ImprovedMazeTraversalFixer` turns the frame-by-frame ``summary_df`` produced by the
alignment step into one row per location visit, smoothing tracking noise, interpolating
implausible jumps and inferring the nodes and edges a run must have passed through. Its
output is the ``traversal_df`` that :mod:`navigraph.analysis.traversal_builder` consumes.

The class was developed in ``examples/dual_root_tree/traversal_pipeline.ipynb`` and is
kept here so the single-session notebook and the multi-session batch notebook share one
implementation rather than each carrying a copy that can drift.

Note that the fixer encodes dual-root-tree maze assumptions: node names are ``L``/``R``
prefixed with the level in the second character, level 4 to level 5 links are treated as
virtual edges, and degree-1 nodes are leaves reached through such an edge. It is not a
general-purpose graph utility.
"""

import ast
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "ImprovedMazeTraversalFixer",
    "load_maze_graph",
    "make_improved_traversal",
    "build_traversal_df",
]


def load_maze_graph(graph_pkl: Union[str, Path]) -> nx.Graph:
    """Load the maze mapping and return its **undirected** graph.

    Args:
        graph_pkl: Path to the mapping pickle (e.g. ``FullMazeGrid.pkl``), whose
            ``mappings['edges']`` keys are ``"<node_a>_<node_b>"`` strings.

    Returns:
        An undirected graph whose nodes are the node-name strings.

    The graph must be undirected: the fixer identifies leaves by ``degree(n) == 1``,
    which a DiGraph would report as 2 for the same node. Notebooks that build a
    DiGraph for drawing must not pass it here.
    """
    with open(graph_pkl, "rb") as f:
        maze_data = pickle.load(f)

    G = nx.Graph()
    for edge_key in maze_data["mappings"]["edges"].keys():
        a, b = edge_key.split("_")
        G.add_edge(a, b)
    return G


class ImprovedMazeTraversalFixer:
    """
    Reconstructs a validated node-edge traversal sequence from raw summary_df.

    Changes vs original:
      1. is_inferred tracked per run; carried into output DataFrame.
      2. Bidirectional noise smoothing: [A, X(short), A] → smooth X, not A.
      3. Long-jump noise: if G_aug path > max_plausible_hops AND the shorter of the two
         runs is ≤ noise_threshold, treat as tracking artifact. If both runs are long
         (genuine port-to-port traversal with a mid-path tracking gap), fall through to
         interpolation instead.
      4. Invalid edge states (not present in G_aug) returned as NaN from _get_state so
         they are absorbed into adjacent runs rather than leaving ghost edges in the
         traversal.
      5. Validation allows same-state consecutive runs (no self-loop needed).
    """

    def __init__(self, G, noise_threshold: int = 5, max_plausible_hops: int = 8):
        self.noise_threshold    = noise_threshold
        self.max_plausible_hops = max_plausible_hops
        self.G_aug = self._build_augmented_graph(G)
        self.diagnostics = {
            'trials_processed': 0,
            'virtual_edges_ignored': 0,
            'noise_smoothed_forward': 0,    # runs[i+1] smoothed to runs[i]
            'noise_smoothed_backward': 0,   # runs[i] smoothed (sandwiched by same state)
            'noise_smoothed_long_jump': 0,  # path > max_plausible_hops, short run → artifact
            'jumps_interpolated': 0,
            'nodes_edges_inferred': 0,
            'invalid_unresolved': 0,
        }

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _is_virtual_edge(self, u_str: str, v_str: str) -> bool:
        """True for L4↔L5 / R4↔R5 pairs (no physical maze corridor)."""
        def level(s):
            return s[1] if len(s) >= 2 and s[0] in ('L', 'R') else None
        l_u, l_v = level(u_str), level(v_str)
        return bool(l_u and l_v and {l_u, l_v} == {'4', '5'})

    def _build_augmented_graph(self, G) -> nx.Graph:
        """Augmented state graph: physical edges become intermediate tuple-nodes.
        Virtual edges (leaf↔parent) remain direct node-node connections."""
        G_aug = nx.Graph()
        for u in G.nodes():
            G_aug.add_node(str(u), state_type='node')
        for u, v in G.edges():
            u_s, v_s = str(u), str(v)
            if G.degree(u) == 1 or G.degree(v) == 1 or self._is_virtual_edge(u_s, v_s):
                G_aug.add_edge(u_s, v_s)
            else:
                edge_state = tuple(sorted((u_s, v_s)))
                G_aug.add_node(edge_state, state_type='edge')
                G_aug.add_edge(u_s, edge_state)
                G_aug.add_edge(v_s, edge_state)
        return G_aug

    # ------------------------------------------------------------------
    # Row accessors (handle MultiIndex summary_df columns)
    # ------------------------------------------------------------------

    def _is_missing(self, val) -> bool:
        if val is None:
            return True
        if isinstance(val, float) and np.isnan(val):
            return True
        if pd.api.types.is_scalar(val) and pd.isna(val):
            return True
        return False

    def _scalar(self, row, col):
        v = row.get(col)
        if isinstance(v, pd.Series):
            v = v.dropna()
            return v.iloc[0] if not v.empty else np.nan
        return v

    def _get_state(self, row):
        node = row.get('headstage_graph_node')
        if isinstance(node, pd.Series):
            node = node.dropna().iloc[0] if not node.dropna().empty else np.nan
        if not self._is_missing(node):
            if isinstance(node, float) and node.is_integer():
                return str(int(node))
            return str(node)

        edge = row.get('headstage_graph_edge')
        if isinstance(edge, pd.Series):
            edge = edge.dropna().iloc[0] if not edge.dropna().empty else np.nan
        if not self._is_missing(edge):
            if isinstance(edge, str):
                try:
                    edge = ast.literal_eval(edge)
                except (ValueError, SyntaxError):
                    clean = edge.strip('() ')
                    if ',' in clean:
                        parts = [p.strip(" '\"") for p in clean.split(',')]
                        if len(parts) == 2:
                            edge = parts
            if isinstance(edge, (list, tuple)) and len(edge) == 2:
                u_s, v_s = str(edge[0]), str(edge[1])
                if self._is_virtual_edge(u_s, v_s):
                    self.diagnostics['virtual_edges_ignored'] += 1
                    return np.nan
                edge_state = tuple(sorted((u_s, v_s)))
                # FIX: invalid edge (nodes not adjacent in maze) → treat as unknown frame
                if edge_state not in self.G_aug:
                    return np.nan
                return edge_state
        return np.nan

    # ------------------------------------------------------------------
    # Run compression
    # ------------------------------------------------------------------

    def _compress_to_runs(self, df_sorted: pd.DataFrame) -> list:
        """Collapse frame-by-frame rows into (state, frame-range) runs.
        NaN frames are absorbed into the adjacent run if no global-frame gap."""
        runs = []
        cur = None

        for _, row in df_sorted.iterrows():
            state   = self._get_state(row)
            f_glob  = self._scalar(row, 'frame_idx_global')
            f_trial = self._scalar(row, 'frame_idx_in_trial')
            t_idx   = self._scalar(row, 'trial_idx')

            if self._is_missing(state):
                if cur is not None and (f_glob - cur['end_frame_global']) == 1:
                    cur['end_frame_global']   = f_glob
                    cur['end_frame_in_trial'] = f_trial
                    cur['duration']          += 1
                continue

            is_gap = cur is not None and (f_glob - cur['end_frame_global']) > 1

            if cur is None or cur['state'] != state or cur['trial_idx'] != t_idx or is_gap:
                if cur is not None:
                    runs.append(cur)
                cur = {
                    'state':                state,
                    'start_frame_global':   f_glob,
                    'end_frame_global':     f_glob,
                    'start_frame_in_trial': f_trial,
                    'end_frame_in_trial':   f_trial,
                    'trial_idx':            t_idx,
                    'duration':             1,
                    'is_inferred':          False,
                }
            else:
                cur['end_frame_global']   = f_glob
                cur['end_frame_in_trial'] = f_trial
                cur['duration']          += 1

        if cur is not None:
            runs.append(cur)
        return runs

    # ------------------------------------------------------------------
    # Run fixing
    # ------------------------------------------------------------------

    def _fix_runs(self, runs: list) -> list:
        i = 0
        while i < len(runs) - 1:
            s1 = runs[i]['state']
            s2 = runs[i + 1]['state']

            if s1 not in self.G_aug or s2 not in self.G_aug:
                self.diagnostics['invalid_unresolved'] += 1
                i += 1
                continue

            if s1 == s2 or self.G_aug.has_edge(s1, s2):
                i += 1
                continue

            try:
                path = nx.shortest_path(self.G_aug, s1, s2)
            except nx.NetworkXNoPath:
                self.diagnostics['invalid_unresolved'] += 1
                i += 1
                continue

            # Long-jump check: only treat as noise artifact when the SHORTER of the two
            # runs is also short (≤ noise_threshold). A genuine port-to-port traversal
            # where tracking dropped out mid-path will have two long runs separated by a
            # large G_aug gap — that must fall through to interpolation, not be smoothed.
            if len(path) - 1 > self.max_plausible_hops:
                shorter_dur = min(runs[i]['duration'], runs[i + 1]['duration'])
                if shorter_dur <= self.noise_threshold:
                    # Short run near long jump → tracking artifact, smooth it away
                    if runs[i + 1]['duration'] <= runs[i]['duration']:
                        runs[i + 1]['state']       = s1
                        runs[i + 1]['is_inferred'] = True
                    else:
                        runs[i]['state']       = s2
                        runs[i]['is_inferred'] = True
                    self.diagnostics['noise_smoothed_long_jump'] += 1
                    continue  # re-check pair; now they match → will increment i
                # Both runs are long → genuine traversal with mid-path tracking gap
                # Fall through to interpolation below

            # Bidirectional noise: run[i] is sandwiched between the same state.
            # Pattern: [... s2, s1(short), s2, ...] → s1 is noise, smooth to s2.
            if (i > 0
                    and runs[i]['duration'] <= self.noise_threshold
                    and runs[i - 1]['state'] == s2):
                runs[i]['state']       = s2
                runs[i]['is_inferred'] = True
                self.diagnostics['noise_smoothed_backward'] += 1
                continue

            # Forward noise: run[i+1] is short → absorb into run[i].
            if runs[i + 1]['duration'] <= self.noise_threshold:
                runs[i + 1]['state']       = s1
                runs[i + 1]['is_inferred'] = True
                self.diagnostics['noise_smoothed_forward'] += 1
                continue

            # Jump interpolation: steal frames and inject intermediate states.
            missing = path[1:-1]
            n_miss  = len(missing)

            avail_left  = max(0, runs[i]['duration'] - 1)
            avail_right = max(0, runs[i + 1]['duration'] - 1)

            if avail_left + avail_right < n_miss:
                self.diagnostics['invalid_unresolved'] += 1
                i += 1
                continue

            take_left  = min(n_miss, avail_left)
            take_right = n_miss - take_left

            orig_end_G   = runs[i]['end_frame_global']
            orig_end_T   = runs[i]['end_frame_in_trial']
            orig_start_G = runs[i + 1]['start_frame_global']
            orig_start_T = runs[i + 1]['start_frame_in_trial']

            if take_left > 0:
                runs[i]['end_frame_global']   -= take_left
                runs[i]['end_frame_in_trial'] -= take_left
                runs[i]['duration']           -= take_left
            if take_right > 0:
                runs[i + 1]['start_frame_global']   += take_right
                runs[i + 1]['start_frame_in_trial'] += take_right
                runs[i + 1]['duration']             -= take_right

            insert_idx = i + 1
            left_done  = 0
            right_done = 0

            for state in missing:
                if left_done < take_left:
                    offset = take_left - left_done - 1
                    fg, ft = orig_end_G - offset, orig_end_T - offset
                    t = runs[i]['trial_idx']
                    left_done += 1
                else:
                    fg, ft = orig_start_G + right_done, orig_start_T + right_done
                    t = runs[i + 1]['trial_idx']
                    right_done += 1

                runs.insert(insert_idx, {
                    'state':                state,
                    'start_frame_global':   fg,
                    'end_frame_global':     fg,
                    'start_frame_in_trial': ft,
                    'end_frame_in_trial':   ft,
                    'trial_idx':            t,
                    'duration':             1,
                    'is_inferred':          True,
                })
                insert_idx += 1

            self.diagnostics['jumps_interpolated']   += 1
            self.diagnostics['nodes_edges_inferred'] += n_miss
            i = insert_idx

        return runs

    # ------------------------------------------------------------------
    # Validation (same-state allowed)
    # ------------------------------------------------------------------

    def validate(self, runs: list) -> tuple:
        for i in range(len(runs) - 1):
            s1, s2 = runs[i]['state'], runs[i + 1]['state']
            if s1 == s2:
                continue  # same-state runs with a frame gap are valid
            if not self.G_aug.has_edge(s1, s2):
                return False, f'Invalid transition @ index {i}: {s1} → {s2}'
        return True, 'Traversal valid.'

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_session(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        t_col = summary_df['trial_idx']
        self.diagnostics['trials_processed'] = (
            len(t_col.unique()) if not isinstance(t_col, pd.DataFrame)
            else len(t_col.iloc[:, 0].unique())
        )

        df = summary_df.loc[:, ~summary_df.columns.duplicated()].copy()
        df = df.sort_values('frame_idx_global')

        runs = self._compress_to_runs(df)
        runs = self._fix_runs(runs)

        # Final merge: collapse consecutive same-state runs with no frame gap
        merged = []
        for r in runs:
            if (merged
                    and merged[-1]['state']     == r['state']
                    and merged[-1]['trial_idx'] == r['trial_idx']
                    and r['start_frame_global'] - merged[-1]['end_frame_global'] <= 1):
                merged[-1]['end_frame_global']   = r['end_frame_global']
                merged[-1]['end_frame_in_trial'] = r['end_frame_in_trial']
                merged[-1]['duration']          += r['duration']
                merged[-1]['is_inferred'] = merged[-1]['is_inferred'] or r['is_inferred']
            else:
                merged.append(r)

        ok, msg = self.validate(merged)
        if not ok:
            print(f'  Warning: {msg}')

        out = pd.DataFrame(merged)
        out['location_type'] = out['state'].apply(
            lambda x: 'edge' if isinstance(x, tuple) else 'node'
        )
        out['headstage_graph_node'] = out.apply(
            lambda r: r['state'] if r['location_type'] == 'node' else np.nan, axis=1
        )
        out['headstage_graph_edge'] = out.apply(
            lambda r: r['state'] if r['location_type'] == 'edge' else np.nan, axis=1
        )

        cols = ['headstage_graph_node', 'headstage_graph_edge', 'location_type',
                'start_frame_global', 'end_frame_global',
                'start_frame_in_trial', 'end_frame_in_trial',
                'trial_idx', 'duration', 'is_inferred']
        out = out[cols]
        for c in ['start_frame_global', 'end_frame_global', 'start_frame_in_trial',
                  'end_frame_in_trial', 'duration']:
            out[c] = pd.to_numeric(out[c], errors='coerce')
        return out

    def print_diagnostics(self):
        d = self.diagnostics
        print(f'  Trials:               {d["trials_processed"]}')
        print(f'  Virtual edges ignored:{d["virtual_edges_ignored"]:5d} frames')
        print(f'  Noise (forward):      {d["noise_smoothed_forward"]:5d}   (i+1 smoothed to i)')
        print(f'  Noise (backward):     {d["noise_smoothed_backward"]:5d}   (i smoothed; sandwiched)')
        if d['noise_smoothed_long_jump']:
            print(f'  Noise (long-jump):    {d["noise_smoothed_long_jump"]:5d}   (path > {self.max_plausible_hops} hops, short run → artifact)')
        print(f'  Jumps interpolated:   {d["jumps_interpolated"]:5d}')
        print(f'  States inferred:      {d["nodes_edges_inferred"]:5d}')
        if d['invalid_unresolved']:
            print(f'  ⚠ Unresolved:         {d["invalid_unresolved"]:5d}')


def make_improved_traversal(trav: pd.DataFrame, frame_dt: float) -> pd.DataFrame:
    """Produce improved_traversal_df.csv-compatible DataFrame from new fixer output."""
    out = trav.copy()
    out['location'] = out.apply(
        lambda r: r['headstage_graph_node'] if r['location_type'] == 'node'
                  else str(r['headstage_graph_edge']),
        axis=1,
    )
    out = out.rename(columns={
        'start_frame_global': 'start_frame',
        'end_frame_global':   'end_frame',
        'duration':           'duration_frames',
    })
    out['start_time']    = out['start_frame'] * frame_dt
    out['end_time']      = out['end_frame']   * frame_dt
    out['duration_time'] = out['duration_frames'] * frame_dt
    out['inference_reason'] = out['is_inferred'].map(
        {True: 'noise_smoothed_or_interpolated', False: None}
    )
    keep = ['location', 'location_type', 'start_frame', 'end_frame',
            'duration_frames', 'start_time', 'end_time', 'duration_time',
            'is_inferred', 'inference_reason']
    return out[[c for c in keep if c in out.columns]]


def build_traversal_df(
    summary_df: pd.DataFrame,
    G: nx.Graph,
    *,
    fps: float = 40.0,
    noise_threshold: int = 5,
    max_plausible_hops: int = 8,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build both traversal DataFrames for one session.

    Args:
        summary_df: Frame-by-frame tracking with ``headstage_graph_node`` /
            ``headstage_graph_edge``, ``trial_idx`` and ``frame_idx_global``.
        G: Undirected maze graph, e.g. from :func:`load_maze_graph`.
        fps: Frame rate (Hz); used to convert frame counts to seconds.
        noise_threshold: Runs of at most this many frames are noise candidates.
        max_plausible_hops: A single step needing more hops than this in the
            augmented graph is treated as a tracking artefact.
        verbose: Print the fixer's diagnostics after processing.

    Returns:
        ``(traversal_df, improved_traversal_df, diagnostics)`` — the raw fixer output,
        the ``improved_traversal_df.csv``-compatible view of it, and the diagnostics
        counters.
    """
    fixer = ImprovedMazeTraversalFixer(
        G,
        noise_threshold=noise_threshold,
        max_plausible_hops=max_plausible_hops,
    )
    traversal_df = fixer.process_session(summary_df)
    improved_df = make_improved_traversal(traversal_df, 1.0 / fps)

    if verbose:
        fixer.print_diagnostics()

    return traversal_df, improved_df, dict(fixer.diagnostics)
