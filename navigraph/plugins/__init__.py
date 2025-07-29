"""Plugin system for NaviGraph.

This module provides extensible plugins for data sources, analyzers, 
and shared resources.
"""

# Plugins are now loaded lazily when imported explicitly
# This prevents automatic registration during CLI help

__all__ = ["data_sources", "shared_resources", "analyzers", "visualizers"]