"""Core components for NaviGraph."""

# Core interfaces and classes
from .interfaces import IDataSource, ISharedResource, IAnalyzer, IVisualizer
from .base_plugin import BasePlugin
from .session import Session
from .experiment_runner import ExperimentRunner

# Type definitions
from .types import AnalysisResult, AnalysisMetadata

# Exceptions
from .exceptions import (
    NavigraphError, DataSourceError, AnalysisError, ConfigurationError
)

# Utilities
from .utils import resolve_path, ensure_directory, compute_configuration_hash

__all__ = [
    # Core interfaces and classes
    "IDataSource",
    "ISharedResource", 
    "IAnalyzer",
    "IVisualizer",
    "BasePlugin",
    "Session",
    "ExperimentRunner",
    
    # Types
    "AnalysisResult",
    "AnalysisMetadata",
    
    # Exceptions
    "NavigraphError",
    "DataSourceError", 
    "AnalysisError",
    "ConfigurationError",
    
    # Utilities
    "resolve_path",
    "ensure_directory", 
    "compute_configuration_hash"
]