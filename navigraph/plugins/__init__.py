"""Plugin system for NaviGraph.

This module provides extensible plugins for data sources, analyzers, 
and shared resources.
"""

# Import all plugins to trigger registration
from . import data_sources
from . import shared_resources
from . import analyzers
from . import visualizers

__all__ = ["data_sources", "shared_resources", "analyzers", "visualizers"]