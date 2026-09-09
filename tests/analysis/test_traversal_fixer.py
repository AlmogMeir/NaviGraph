#!/usr/bin/env python3
"""Tests for navigraph.analysis.traversal_fixer.

The fixer was lifted out of examples/dual_root_tree/traversal_pipeline.ipynb so that
the single-session notebook and the batch notebook share one implementation. These
tests pin the output contract that navigraph.analysis.traversal_builder and the
notebooks depend on.
"""

import networkx as nx
import numpy as np
import pandas as pd

from navigraph.analysis.traversal_fixer import (
    ImprovedMazeTraversalFixer,
    build_traversal_df,
    make_improved_traversal,
)

TRAVERSAL_COLUMNS = [
    'headstage_graph_node', 'headstage_graph_edge', 'location_type',
    'start_frame_global', 'end_frame_global', 'start_frame_in_trial',
    'end_frame_in_trial', 'trial_idx', 'duration', 'is_inferred',
]

IMPROVED_COLUMNS = [
    'location', 'location_type', 'start_frame', 'end_frame', 'duration_frames',
    'start_time', 'end_time', 'duration_time', 'is_inferred', 'inference_reason',
]

DIAGNOSTIC_KEYS = {
    'trials_processed', 'virtual_edges_ignored', 'noise_smoothed_forward',
    'noise_smoothed_backward', 'noise_smoothed_long_jump', 'jumps_interpolated',
    'nodes_edges_inferred', 'invalid_unresolved',
}


def _toy_graph():
    """A small dual-root-style tree: L0 root, one branch down to leaves L50/L51."""
    G = nx.Graph()
    G.add_edges_from([
        ('L0', 'L10'), ('L10', 'L20'), ('L20', 'L30'),
        ('L30', 'L40'), ('L40', 'L50'), ('L40', 'L51'),
        ('L0', 'R0'), ('R0', 'R10'),
    ])
    return G


def _summary(states, trial_idx=0, start=0):
    """Frame-by-frame summary_df from a per-frame list of node names (None = gap)."""
    rows = []
    for offset, node in enumerate(states):
        rows.append({
            'headstage_graph_node': node,
            'headstage_graph_edge': None,
            'trial_idx': trial_idx,
            'frame_idx_global': start + offset,
            'frame_idx_in_trial': offset,
        })
    return pd.DataFrame(rows)


def test_output_contract():
    """process_session returns the exact columns traversal_builder expects."""
    G = _toy_graph()
    summary = _summary(['L0'] * 10 + ['L10'] * 10 + ['L20'] * 10)

    fixer = ImprovedMazeTraversalFixer(G, noise_threshold=5, max_plausible_hops=8)
    trav = fixer.process_session(summary)

    assert list(trav.columns) == TRAVERSAL_COLUMNS, list(trav.columns)
    assert len(trav) > 0
    assert set(trav['location_type'].unique()) <= {'node', 'edge'}
    assert trav['is_inferred'].dtype == bool
    assert set(fixer.diagnostics) == DIAGNOSTIC_KEYS, set(fixer.diagnostics)
    assert fixer.diagnostics['trials_processed'] == 1
    print(f"  output contract: {len(trav)} rows, columns and diagnostics as expected")


def test_short_run_is_smoothed_and_flagged():
    """A 2-frame blip inside a long stay is smoothed away and marked inferred."""
    G = _toy_graph()
    clean = _summary(['L20'] * 40)
    noisy = _summary(['L20'] * 19 + ['L50'] * 2 + ['L20'] * 19)

    fixer_clean = ImprovedMazeTraversalFixer(G, noise_threshold=5)
    fixer_noisy = ImprovedMazeTraversalFixer(G, noise_threshold=5)
    trav_clean = fixer_clean.process_session(clean)
    trav_noisy = fixer_noisy.process_session(noisy)

    # The blip must not survive as its own visit.
    assert 'L50' not in set(trav_noisy['headstage_graph_node'].dropna())
    # And the smoothing must be visible in the diagnostics.
    smoothed = sum(v for k, v in fixer_noisy.diagnostics.items() if 'noise_smoothed' in k)
    assert smoothed >= 1, fixer_noisy.diagnostics
    assert len(trav_noisy) == len(trav_clean)
    print(f"  noise smoothing: blip removed, {smoothed} smoothing event(s) recorded")


def test_make_improved_traversal_schema():
    """The improved view renames/derives exactly the columns the CSV consumers read."""
    G = _toy_graph()
    trav = ImprovedMazeTraversalFixer(G).process_session(
        _summary(['L0'] * 10 + ['L10'] * 10))

    improved = make_improved_traversal(trav, frame_dt=1.0 / 40.0)

    assert list(improved.columns) == IMPROVED_COLUMNS, list(improved.columns)
    assert len(improved) == len(trav)
    # Times are frame counts scaled by the frame interval.
    assert np.allclose(improved['start_time'], trav['start_frame_global'] / 40.0)
    assert np.allclose(improved['duration_time'], trav['duration'] / 40.0)
    # Node rows carry the node name in `location`.
    node_rows = improved[improved['location_type'] == 'node']
    assert node_rows['location'].notna().all()
    print(f"  improved schema: {len(improved)} rows, times scaled by 1/40 s")


def test_build_traversal_df_wrapper():
    """The convenience wrapper returns both frames plus a diagnostics copy."""
    G = _toy_graph()
    summary = _summary(['L0'] * 10 + ['L10'] * 10 + ['L20'] * 10)

    trav, improved, diagnostics = build_traversal_df(
        summary, G, fps=40.0, noise_threshold=5, max_plausible_hops=8, verbose=False)

    assert list(trav.columns) == TRAVERSAL_COLUMNS
    assert list(improved.columns) == IMPROVED_COLUMNS
    assert set(diagnostics) == DIAGNOSTIC_KEYS
    # diagnostics is a copy, not a live reference into the fixer
    diagnostics['trials_processed'] = -1
    _, _, fresh = build_traversal_df(summary, G, verbose=False)
    assert fresh['trials_processed'] == 1
    print(f"  wrapper: {len(trav)} traversal rows, {len(improved)} improved rows")


def test_multiple_trials_are_independent():
    """Runs are not merged across a trial boundary."""
    G = _toy_graph()
    summary = pd.concat([
        _summary(['L20'] * 10, trial_idx=0, start=0),
        _summary(['L20'] * 10, trial_idx=1, start=10),
    ], ignore_index=True)

    fixer = ImprovedMazeTraversalFixer(G)
    trav = fixer.process_session(summary)

    assert fixer.diagnostics['trials_processed'] == 2
    assert sorted(trav['trial_idx'].unique()) == [0, 1]
    print(f"  trial isolation: {len(trav)} rows across 2 trials")


if __name__ == "__main__":
    print("Testing navigraph.analysis.traversal_fixer...\n")

    test_output_contract()
    test_short_run_is_smoothed_and_flagged()
    test_make_improved_traversal_schema()
    test_build_traversal_df_wrapper()
    test_multiple_trials_are_independent()

    print("\n✅ All tests passed!")
