"""Analyzer plugins for NaviGraph.

This package contains all analyzer plugins that perform behavioral and neural
analysis on session data. Analyzers have full access to integrated DataFrames,
graph structures, and shared resources.

Available analyzers:
- spatial_metrics: Time and velocity measurements between spatial locations
- navigation_metrics: Path analysis and graph-based navigation measurements  
- exploration_metrics: Exploration patterns and node visit statistics
"""

# Import all analyzer plugins to trigger registration
from . import spatial_metrics
from . import navigation_metrics
from . import exploration_metrics

__all__ = ['spatial_metrics', 'navigation_metrics', 'exploration_metrics']