"""Analysis functions for NaviGraph.

Contains session-level and cross-session analysis functions.
All functions are auto-discovered and registered via decorators.
"""

try:
    from . import metrics  # Import to register metrics
    from .analyzer import Analyzer
except ImportError:
    pass  # omegaconf / hydra not installed; traversal_builder is still importable
from .speed_analysis import (
    compute_path_speed,
    compute_mean_path_speed,
    plot_speed,
    plot_mean_speed,
)
from .traversal_fixer import (
    ImprovedMazeTraversalFixer,
    load_maze_graph,
    make_improved_traversal,
    build_traversal_df,
)
from .traversal_builder import (
    fix_edge_bounces,
    build_node_df,
    build_node_edge_df,
)

__all__ = [
    'Analyzer',
    'compute_path_speed',
    'compute_mean_path_speed',
    'plot_speed',
    'plot_mean_speed',
    'ImprovedMazeTraversalFixer',
    'load_maze_graph',
    'make_improved_traversal',
    'build_traversal_df',
    'fix_edge_bounces',
    'build_node_df',
    'build_node_edge_df',
]