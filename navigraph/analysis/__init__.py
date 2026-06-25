"""Analysis functions for NaviGraph.

Contains session-level and cross-session analysis functions.
All functions are auto-discovered and registered via decorators.
"""

from . import metrics  # Import to register metrics
from .analyzer import Analyzer
from .speed_analysis import (
    compute_path_speed,
    compute_mean_path_speed,
    plot_speed,
    plot_mean_speed,
)

__all__ = [
    'Analyzer',
    'compute_path_speed',
    'compute_mean_path_speed',
    'plot_speed',
    'plot_mean_speed',
]