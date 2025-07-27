"""Core components for NaviGraph.

This module provides the foundational classes and interfaces for the
NaviGraph plugin system, including type definitions, constants, utilities,
and standardized exceptions.
"""

# Core interfaces and classes
from .interfaces import IDataSource, ISharedResource, IAnalyzer, IVisualizer, IGraphProvider
from .base_plugin import BasePlugin
from .registry import PluginRegistry, registry
from .session import Session
from .file_discovery import FileDiscoveryEngine
from .experiment_runner import ExperimentRunner
from .visualization_pipeline import VisualizationPipeline
from .visualization_config import (
    VisualizationConfig, OutputFormat, ColorTheme, ColorPalette,
    FontSettings, PlotSettings, create_default_configs
)

# Type definitions
from .types import (
    # Basic types
    SessionID, TileID, NodeID, PluginName, FilePath,
    Coordinate, RGBColor, NumericValue,
    
    # Configuration types
    ExperimentConfig, DataSourceConfig, AnalyzerConfig, VisualizationPluginConfig,
    
    # Data structures
    SessionMetadata, AnalysisResult, AnalysisMetadata, VisualizationResult,
    SessionSummary, StatisticalResult,
    
    # Analysis types
    MetricValue, MetricDict, CrossSessionMetrics,
    
    # Utility types
    OperationResult, ValidationResult
)

# Constants
from .constants import (
    DataColumns, ConfigKeys, FileExtensions, DefaultValues,
    ErrorMessages, LogMessages, VisualizationDefaults
)

# Exceptions
from .exceptions import (
    NavigraphError, PluginError, ConfigurationError, DataSourceError,
    SessionError, AnalysisError, VisualizationError, FileSystemError
)

# Utilities
from .utils import (
    validate_file_path, ensure_directory, validate_dataframe_columns,
    safe_eval_config_value, TimingContext, format_duration
)

__all__ = [
    # Core interfaces and classes
    "IDataSource",
    "ISharedResource", 
    "IAnalyzer",
    "IVisualizer",
    "IGraphProvider",
    "BasePlugin",
    "PluginRegistry",
    "registry",
    "Session",
    "FileDiscoveryEngine",
    "ExperimentRunner",
    "VisualizationPipeline",
    
    # Visualization configuration
    "VisualizationConfig",
    "OutputFormat",
    "ColorTheme",
    "ColorPalette",
    "FontSettings",
    "PlotSettings",
    "create_default_configs",
    
    # Types
    "SessionID", "TileID", "NodeID", "PluginName", "FilePath",
    "Coordinate", "RGBColor", "NumericValue",
    "ExperimentConfig", "DataSourceConfig", "AnalyzerConfig", "VisualizationPluginConfig",
    "SessionMetadata", "AnalysisResult", "AnalysisMetadata", "VisualizationResult",
    "SessionSummary", "StatisticalResult",
    "MetricValue", "MetricDict", "CrossSessionMetrics",
    "OperationResult", "ValidationResult",
    
    # Constants
    "DataColumns", "ConfigKeys", "FileExtensions", "DefaultValues",
    "ErrorMessages", "LogMessages", "VisualizationDefaults",
    
    # Exceptions
    "NavigraphError", "PluginError", "ConfigurationError", "DataSourceError",
    "SessionError", "AnalysisError", "VisualizationError", "FileSystemError",
    
    # Utilities
    "validate_file_path", "ensure_directory", "validate_dataframe_columns",
    "safe_eval_config_value", "TimingContext", "format_duration"
]