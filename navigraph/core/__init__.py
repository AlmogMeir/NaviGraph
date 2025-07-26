"""Core functionality for NaviGraph framework.

This module contains the essential interfaces, registry system, and orchestration
classes that form the foundation of the plugin-based architecture.
"""

from .interfaces import IDataSource, ISharedResource, IAnalyzer, IVisualizer, IGraphProvider
from .registry import PluginRegistry, registry

__all__ = [
    "IDataSource",
    "ISharedResource", 
    "IAnalyzer",
    "IVisualizer",
    "IGraphProvider",
    "PluginRegistry",
    "registry",
]