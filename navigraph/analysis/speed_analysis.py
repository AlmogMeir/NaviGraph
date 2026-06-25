"""Speed analysis utilities for NaviGraph behavioral data.

Provides functions to compute and plot mouse movement speed from coordinate
tracking data, with support for:
- Smoothing over frame bins to suppress tracking noise
- Secondary node axis overlay (from improved_node_summary_df)
- PSTH overlay on a twin y-axis for neural comparison
- Mean ± SEM speed across multiple path instances (time-normalized)

Typical usage
-------------
Single path::

    from navigraph.analysis.speed_analysis import compute_path_speed, plot_speed

    times, speed, raw = compute_path_speed(summary_df, start_frame=1000, end_frame=1400)
    fig = plot_speed(summary_df, 1000, 1400, node_rows=node_rows_for_path)

Mean across paths::

    from navigraph.analysis.speed_analysis import compute_mean_path_speed, plot_mean_speed

    stats = compute_mean_path_speed(direct_instances, summary_df)
    fig = plot_mean_speed(stats, psth_mean=mean_rate, psth_sem=sem_rate,
                          node_labels=canon_seq, node_fracs=canon_fracs)

Column resolution
-----------------
``summary_df`` may have flat or MultiIndex columns.  The helpers try an exact
match first, then a case-insensitive partial match against MultiIndex levels.
Pass MultiIndex tuples directly (e.g. ``x_col=('headstage_x', '')``) or flat
strings (e.g. ``x_col='headstage_x'``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

VIDEO_FPS: float = 40.0
SMOOTH_WINDOW_FRAMES: int = 5
N_GRID: int = 100

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_col(
    df: pd.DataFrame,
    candidates: List[Union[str, tuple]],
) -> Optional[Union[str, tuple]]:
    """Return the first matching column name from *candidates*.

    Handles both flat and MultiIndex DataFrames. For MultiIndex columns the
    search checks whether the candidate string appears as any level value
    (case-insensitive), so ``'headstage_x'`` matches ``('headstage_x', '')``.
    """
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        # Try exact tuple match first
        for cand in candidates:
            if cand in cols:
                return cand
        # Partial string match on any level
        low = [str(c).lower() for c in candidates]
        for col in cols:
            parts = [str(v).strip().lower() for v in col]
            for cand in low:
                if cand in parts:
                    return col
        return None
    # Flat columns
    for cand in candidates:
        if cand in cols:
            return cand
    lower_map = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if str(cand).lower() in lower_map:
            return lower_map[str(cand).lower()]
    return None


def _require_col(
    df: pd.DataFrame,
    name: str,
    candidates: List[Union[str, tuple]],
) -> Union[str, tuple]:
    col = _resolve_col(df, candidates)
    if col is None:
        raise KeyError(
            f"Column '{name}' not found. Tried: {candidates}. "
            f"Available: {list(df.columns[:10])}..."
        )
    return col


def _frame_slice(
    summary_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    frame_col: Union[str, tuple],
) -> pd.DataFrame:
    """Return rows where frame_col is in [start_frame, end_frame], sorted."""
    fc = _require_col(summary_df, 'frame_col', [frame_col])
    mask = (summary_df[fc] >= start_frame) & (summary_df[fc] <= end_frame)
    return summary_df[mask].sort_values(fc).copy()


def _nan_aware_uniform_smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Box-filter mean that skips NaN samples (tracking drop-outs)."""
    nan_mask = np.isnan(arr)
    fill = np.where(nan_mask, 0.0, arr)
    wt = np.where(nan_mask, 0.0, 1.0)
    s_fill = uniform_filter1d(fill, size=window, mode='nearest')
    s_wt = uniform_filter1d(wt, size=window, mode='nearest')
    s_wt_safe = np.where(s_wt > 1e-9, s_wt, 1.0)
    return np.where(s_wt > 1e-9, s_fill / s_wt_safe, np.nan)


def _resample_to_grid(values: np.ndarray, n_grid: int) -> np.ndarray:
    """Linearly interpolate *values* onto *n_grid* uniform points in [0, 1]."""
    values = np.asarray(values, dtype=float)
    L = len(values)
    if L == 0:
        return np.full(n_grid, np.nan)
    if L == 1:
        return np.full(n_grid, values[0])
    return np.interp(
        np.linspace(0, 1, n_grid),
        np.linspace(0, 1, L),
        values,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_path_speed(
    summary_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    fps: float = VIDEO_FPS,
    x_col: Union[str, tuple] = 'headstage_x',
    y_col: Union[str, tuple] = 'headstage_y',
    frame_col: Union[str, tuple] = 'frame_idx_global',
    smooth_window_frames: int = SMOOTH_WINDOW_FRAMES,
    pixel_to_unit: Optional[float] = None,
    min_likelihood: Optional[float] = None,
    likelihood_col: Union[str, tuple] = 'headstage_likelihood',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame speed along a path segment from coordinate tracking data.

    Speed is the Euclidean pixel displacement between consecutive frames,
    multiplied by *fps* to give pixels per second. Frames with NaN coordinates
    or below *min_likelihood* produce NaN speed; smoothing skips these samples.

    Parameters
    ----------
    summary_df:
        Frame-by-frame tracking DataFrame. Columns may be flat strings or
        MultiIndex (the level containing the name is matched automatically).
    start_frame, end_frame:
        ``frame_idx_global`` bounds (inclusive).
    fps:
        Video frame rate (default 40 Hz).
    x_col, y_col:
        Coordinate column names. Pass ``'headstage_x'`` (flat) or
        ``('headstage_x', '')`` (MultiIndex tuple) — both are resolved.
        For map-calibrated coordinates use ``'headstage_map_x'``.
    frame_col:
        Frame index column.
    smooth_window_frames:
        Number of frames for the uniform (box) smoothing filter.
        Set to 1 to disable smoothing.
    pixel_to_unit:
        Optional scale factor applied to speed (e.g. cm/px conversion).
    min_likelihood:
        If given, frames below this DLC confidence threshold are treated as NaN.
    likelihood_col:
        Column holding per-frame DLC likelihood values.

    Returns
    -------
    times_s:
        (N-1,) array. Time in seconds relative to *start_frame*, at the midpoint
        between each consecutive pair of frames.
    speed:
        (N-1,) array. Smoothed speed in px/s (or unit/s if *pixel_to_unit* given).
    raw_speed:
        (N-1,) array. Unsmoothed speed, same units.
    """
    sub = _frame_slice(summary_df, start_frame, end_frame, frame_col)
    fc = _require_col(sub, 'frame_col', [frame_col])
    xc = _require_col(sub, 'x_col', [x_col])
    yc = _require_col(sub, 'y_col', [y_col])

    frames = sub[fc].values.astype(float)
    x = pd.to_numeric(sub[xc], errors='coerce').values.astype(float)
    y = pd.to_numeric(sub[yc], errors='coerce').values.astype(float)

    # Mask low-confidence frames
    if min_likelihood is not None:
        lc = _resolve_col(sub, [likelihood_col])
        if lc is not None:
            lk = pd.to_numeric(sub[lc], errors='coerce').values.astype(float)
            bad = lk < min_likelihood
            x[bad] = np.nan
            y[bad] = np.nan

    if len(frames) < 2:
        return np.array([]), np.array([]), np.array([])

    dx = np.diff(x)
    dy = np.diff(y)
    mid_frames = (frames[:-1] + frames[1:]) / 2.0

    raw_speed = np.sqrt(dx**2 + dy**2) * fps
    if pixel_to_unit is not None:
        raw_speed = raw_speed * pixel_to_unit

    speed = _nan_aware_uniform_smooth(raw_speed, smooth_window_frames)
    times_s = (mid_frames - float(start_frame)) / fps
    return times_s, speed, raw_speed


def compute_mean_path_speed(
    path_instances: List[Dict],
    summary_df: pd.DataFrame,
    n_grid: int = N_GRID,
    fps: float = VIDEO_FPS,
    x_col: Union[str, tuple] = 'headstage_x',
    y_col: Union[str, tuple] = 'headstage_y',
    frame_col: Union[str, tuple] = 'frame_idx_global',
    smooth_window_frames: int = SMOOTH_WINDOW_FRAMES,
    pixel_to_unit: Optional[float] = None,
    min_likelihood: Optional[float] = None,
    likelihood_col: Union[str, tuple] = 'headstage_likelihood',
) -> Dict:
    """Compute mean ± SEM speed across multiple path instances on a normalized grid.

    Each path is resampled onto *n_grid* uniformly spaced points spanning the
    full path duration (0 = start, 1 = end), then averaged.  This mirrors the
    continuous-PSTH normalization used in ``direct_path_analysis.ipynb``.

    Parameters
    ----------
    path_instances:
        List of dicts. Each must supply the path's frame range via either:

        - ``'start_frame'`` + ``'end_frame'`` (frame_idx_global integers), or
        - ``'path_start_abs'`` + ``'path_end_abs'`` (absolute seconds, as in the
          existing instance dicts produced by ``direct_path_analysis.ipynb``).

    n_grid:
        Number of points on the normalized time axis.
    All other parameters:
        Same as :func:`compute_path_speed`.

    Returns
    -------
    dict with keys:

    ``frac_grid``
        ndarray (n_grid,) — normalized path time 0 → 1.
    ``mean``
        ndarray (n_grid,) — mean speed.
    ``sem``
        ndarray (n_grid,) — standard error of the mean.
    ``n``
        Number of instances successfully processed.
    ``mean_duration_s``
        Mean path duration in seconds.
    ``all_resampled``
        ndarray (n, n_grid) — per-instance resampled speed traces.
    """
    resampled: List[np.ndarray] = []
    durations: List[float] = []

    for inst in path_instances:
        if 'start_frame' in inst and 'end_frame' in inst:
            sf = int(inst['start_frame'])
            ef = int(inst['end_frame'])
        elif 'path_start_abs' in inst and 'path_end_abs' in inst:
            sf = round(inst['path_start_abs'] * fps)
            ef = round(inst['path_end_abs'] * fps) - 1
        else:
            continue

        _, speed, _ = compute_path_speed(
            summary_df, sf, ef, fps=fps,
            x_col=x_col, y_col=y_col, frame_col=frame_col,
            smooth_window_frames=smooth_window_frames,
            pixel_to_unit=pixel_to_unit,
            min_likelihood=min_likelihood,
            likelihood_col=likelihood_col,
        )
        if len(speed) == 0:
            continue
        resampled.append(_resample_to_grid(speed, n_grid))
        durations.append((ef - sf + 1) / fps)

    frac = np.linspace(0, 1, n_grid)
    if not resampled:
        nan = np.full(n_grid, np.nan)
        return {
            'frac_grid': frac, 'mean': nan, 'sem': nan,
            'n': 0, 'mean_duration_s': np.nan,
            'all_resampled': np.empty((0, n_grid)),
        }

    arr = np.array(resampled)
    n_valid = np.sum(~np.isnan(arr), axis=0).clip(min=1)
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_valid)
    return {
        'frac_grid': frac,
        'mean': mean,
        'sem': sem,
        'n': len(resampled),
        'mean_duration_s': float(np.mean(durations)),
        'all_resampled': arr,
    }


def plot_speed(
    summary_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    fps: float = VIDEO_FPS,
    x_col: Union[str, tuple] = 'headstage_x',
    y_col: Union[str, tuple] = 'headstage_y',
    frame_col: Union[str, tuple] = 'frame_idx_global',
    smooth_window_frames: int = SMOOTH_WINDOW_FRAMES,
    pixel_to_unit: Optional[float] = None,
    unit_label: str = 'px/s',
    min_likelihood: Optional[float] = None,
    likelihood_col: Union[str, tuple] = 'headstage_likelihood',
    node_rows: Optional[pd.DataFrame] = None,
    psth_times: Optional[np.ndarray] = None,
    psth_rate: Optional[np.ndarray] = None,
    psth_sem: Optional[np.ndarray] = None,
    psth_label: str = 'Firing rate (Hz)',
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    speed_color: str = 'steelblue',
    psth_color: str = 'darkorange',
    show_raw: bool = False,
) -> plt.Figure:
    """Plot speed for a single path with optional node axis and PSTH overlay.

    The primary (bottom) x-axis shows time in seconds from path start.
    If *node_rows* is provided, vertical dashed lines mark node entry times and
    a secondary (top) x-axis labels each node.
    If *psth_times* + *psth_rate* are provided, the firing rate is overlaid on
    a twin right y-axis in a contrasting colour.

    Parameters
    ----------
    summary_df, start_frame, end_frame, fps, x_col, y_col, frame_col,
    smooth_window_frames, pixel_to_unit, min_likelihood, likelihood_col:
        Forwarded to :func:`compute_path_speed`.
    unit_label:
        Y-axis label suffix, e.g. ``'px/s'`` or ``'cm/s'``.
    node_rows:
        Rows from ``improved_node_summary_df`` for this path (already filtered).
        Must contain ``'start_frame'`` and ``'node_name'`` columns.
    psth_times:
        1-D array — time in seconds relative to *start_frame* for PSTH values.
    psth_rate:
        1-D array — firing rate (Hz) aligned to *psth_times*.
    psth_sem:
        Optional 1-D SEM array for shaded PSTH band.
    psth_label:
        Right y-axis label for the PSTH.
    ax:
        Existing :class:`~matplotlib.axes.Axes` to draw into (creates a new
        figure if ``None``).
    show_raw:
        Also plot the unsmoothed speed as a faint dashed line.

    Returns
    -------
    matplotlib Figure
    """
    times_s, speed, raw_speed = compute_path_speed(
        summary_df, start_frame, end_frame, fps=fps,
        x_col=x_col, y_col=y_col, frame_col=frame_col,
        smooth_window_frames=smooth_window_frames,
        pixel_to_unit=pixel_to_unit,
        min_likelihood=min_likelihood, likelihood_col=likelihood_col,
    )
    path_duration_s = (end_frame - start_frame + 1) / fps

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5))
    else:
        fig = ax.figure

    # --- speed curve ---
    ax.plot(times_s, speed, color=speed_color, linewidth=1.5, label='Speed (smoothed)')
    if show_raw:
        ax.plot(times_s, raw_speed, color=speed_color, linewidth=0.6,
                alpha=0.3, linestyle='--', label='Speed (raw)')
    ax.set_xlabel('Time from path start (s)')
    ax.set_ylabel(f'Speed ({unit_label})', color=speed_color)
    ax.tick_params(axis='y', labelcolor=speed_color)
    ax.set_xlim(0, path_duration_s)

    # --- node boundaries + secondary top x-axis ---
    node_tick_pos: List[float] = []
    node_tick_lbl: List[str] = []
    if node_rows is not None and len(node_rows) > 0:
        for _, nr in node_rows.iterrows():
            t_node = (float(nr['start_frame']) - float(start_frame)) / fps
            if 0 <= t_node <= path_duration_s:
                ax.axvline(t_node, color='gray', linewidth=0.7,
                           linestyle='--', alpha=0.55, zorder=0)
                node_tick_pos.append(t_node)
                node_tick_lbl.append(str(nr['node_name']))

    if node_tick_pos:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(node_tick_pos)
        ax_top.set_xticklabels(node_tick_lbl, fontsize=7, rotation=45, ha='left')
        ax_top.set_xlabel('Node', fontsize=8)

    # --- PSTH on twin right y-axis ---
    if psth_times is not None and psth_rate is not None:
        ax_r = ax.twinx()
        ax_r.plot(psth_times, psth_rate, color=psth_color,
                  linewidth=1.5, label=psth_label)
        if psth_sem is not None:
            ax_r.fill_between(psth_times, psth_rate - psth_sem,
                               psth_rate + psth_sem, color=psth_color, alpha=0.25)
        ax_r.set_ylabel(psth_label, color=psth_color)
        ax_r.tick_params(axis='y', labelcolor=psth_color)

    ax.legend(fontsize=8, loc='upper left')
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_mean_speed(
    speed_stats: Dict,
    psth_mean: Optional[np.ndarray] = None,
    psth_sem: Optional[np.ndarray] = None,
    psth_label: str = 'Firing rate (Hz)',
    node_labels: Optional[List[str]] = None,
    node_fracs: Optional[List[float]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    speed_color: str = 'steelblue',
    psth_color: str = 'darkorange',
    unit_label: str = 'px/s',
    show_individual: bool = False,
) -> plt.Figure:
    """Plot mean ± SEM speed on normalized path time with optional PSTH overlay.

    The x-axis is fractional path time (0 = path start, 1 = end).  This is
    directly comparable to the continuous-PSTH plots from
    ``direct_path_analysis.ipynb``.

    Parameters
    ----------
    speed_stats:
        Output of :func:`compute_mean_path_speed`.
    psth_mean:
        Firing rate mean array of length ``n_grid``, on the same frac_grid.
    psth_sem:
        Optional SEM for shaded PSTH band.
    node_labels:
        Canonical node names for the secondary top x-axis.
    node_fracs:
        Fractional positions (0–1) for *node_labels*.  Compute as the mean of
        ``(node_start_frame - path_start_frame) / path_length_frames`` across
        all instances for each canonical node position.
    show_individual:
        Overlay each instance's resampled speed trace (semi-transparent).

    Returns
    -------
    matplotlib Figure

    Notes
    -----
    To compute *node_fracs* from a list of instances where each has
    ``starts_abs`` (per-node start times) and ``path_start_abs`` /
    ``path_end_abs``::

        canon_n = len(canonical_seq)
        frac_matrix = np.zeros((len(instances), canon_n))
        for i, inst in enumerate(instances):
            dur = inst['path_end_abs'] - inst['path_start_abs']
            frac_matrix[i] = (inst['starts_abs'] - inst['path_start_abs']) / dur
        node_fracs = frac_matrix.mean(axis=0).tolist()
    """
    frac = speed_stats['frac_grid']
    mean = speed_stats['mean']
    sem = speed_stats['sem']
    n = speed_stats['n']
    dur = speed_stats['mean_duration_s']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.5))
    else:
        fig = ax.figure

    if show_individual:
        for row in speed_stats['all_resampled']:
            ax.plot(frac, row, color=speed_color, linewidth=0.5, alpha=0.2)

    ax.plot(frac, mean, color=speed_color, linewidth=1.8,
            label=f'Mean speed (n={n})')
    ax.fill_between(frac, mean - sem, mean + sem,
                    color=speed_color, alpha=0.25, label='±SEM')
    ax.set_xlabel(
        f'Normalized path time  (0 = start, 1 = end)  '
        f'[mean duration {dur:.2f} s]'
    )
    ax.set_ylabel(f'Speed ({unit_label})', color=speed_color)
    ax.tick_params(axis='y', labelcolor=speed_color)
    ax.set_xlim(0, 1)

    # --- node secondary top x-axis ---
    if node_labels and node_fracs:
        for nf in node_fracs:
            ax.axvline(nf, color='gray', linewidth=0.7, linestyle='--', alpha=0.5, zorder=0)
        ax_top = ax.twiny()
        ax_top.set_xlim(0, 1)
        ax_top.set_xticks(node_fracs)
        ax_top.set_xticklabels(node_labels, fontsize=7, rotation=45, ha='left')
        ax_top.set_xlabel('Node (canonical)', fontsize=8)

    # --- PSTH on twin right y-axis ---
    if psth_mean is not None:
        ax_r = ax.twinx()
        ax_r.plot(frac, psth_mean, color=psth_color, linewidth=1.8, label=psth_label)
        if psth_sem is not None:
            ax_r.fill_between(frac, psth_mean - psth_sem, psth_mean + psth_sem,
                               color=psth_color, alpha=0.25)
        ax_r.set_ylabel(psth_label, color=psth_color)
        ax_r.tick_params(axis='y', labelcolor=psth_color)

    ax.legend(fontsize=8, loc='upper left')
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


__all__ = [
    'compute_path_speed',
    'compute_mean_path_speed',
    'plot_speed',
    'plot_mean_speed',
]
