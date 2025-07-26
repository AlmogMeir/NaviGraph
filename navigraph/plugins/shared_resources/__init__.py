"""Shared resource plugins for NaviGraph.

This package contains all shared resource plugins that provide resources
used by data sources and analyzers. Shared resources are initialized once
per session and made available to all plugins that need them.

Available shared resources:
- map_provider: Provides maze map and spatial configuration
- graph_provider: Provides graph instance and navigation utilities
- calibration_provider: Provides camera calibration and coordinate transformation
"""

# Import all shared resource plugins to trigger registration
from . import map_provider
from . import graph_provider  
from . import calibration_provider

__all__ = ['map_provider', 'graph_provider', 'calibration_provider']