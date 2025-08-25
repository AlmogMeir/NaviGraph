"""Core components for NaviGraph."""

# Core classes
from .navigraph_plugin import NaviGraphPlugin
from .session import Session
from .session_analyzer import SessionAnalyzer
from .session_visualizer import SessionVisualizer
from .experiment_runner import ExperimentRunner

# Registry and decorators
from .registry import registry, register_plugin, register_analysis, register_visualizer

# Type definitions
from .types import AnalysisResult, AnalysisMetadata

# Exceptions
from .exceptions import (
    NavigraphError, DataSourceError, AnalysisError, ConfigurationError
)

# Utilities
from .utils import resolve_path, ensure_directory, compute_configuration_hash
from .conversion_utils import quaternions_to_euler, wrap_angle, degrees_to_radians, radians_to_degrees

__all__ = [
    # Core classes
    "NaviGraphPlugin",
    "Session",
    "SessionAnalyzer",
    "SessionVisualizer", 
    "ExperimentRunner",
    
    # Registry and decorators
    "registry",
    "register_plugin",
    "register_analysis", 
    "register_visualizer",
    
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
    "compute_configuration_hash",
    "quaternions_to_euler",
    "wrap_angle",
    "degrees_to_radians",
    "radians_to_degrees"
]