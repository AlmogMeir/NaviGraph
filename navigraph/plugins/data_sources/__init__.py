"""Data source plugins for NaviGraph.

This package contains all data source plugins that can integrate data into
NaviGraph sessions. Data sources are executed in configuration order and
each adds its columns to the accumulating session DataFrame.

Available data sources:
- deeplabcut: DeepLabCut pose estimation data
- map_integration: Spatial map coordinate transformation and tile detection
- graph_integration: Graph-based spatial navigation analysis with pathfinding
"""

# Import all data source plugins to trigger registration
from . import deeplabcut
from . import map_integration
from . import graph_integration

__all__ = ['deeplabcut', 'map_integration', 'graph_integration']