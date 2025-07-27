"""Type definitions for NaviGraph.

This module provides comprehensive type definitions, type aliases, and
typed data structures used throughout the NaviGraph codebase.
"""

from typing import (
    Dict, List, Tuple, Union, Optional, Any, Callable, TypeVar, Generic,
    NamedTuple, Protocol, TYPE_CHECKING
)
from typing_extensions import TypedDict, Literal
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from .interfaces import IDataSource, IAnalyzer, IVisualizer, ISharedResource


# =============================================================================
# BASIC TYPE ALIASES
# =============================================================================

# Coordinate types
Coordinate = Tuple[float, float]
Coordinate3D = Tuple[float, float, float]
CoordinateList = List[Coordinate]

# Color types (RGB)
RGBColor = Tuple[int, int, int]
RGBAColor = Tuple[int, int, int, int]
ColorValue = Union[RGBColor, RGBAColor, str]

# File path types
FilePath = Union[str, Path]
FilePathList = List[FilePath]

# Numeric types
NumericValue = Union[int, float]
NumericArray = Union[List[NumericValue], np.ndarray]

# Time types
TimeStamp = Union[float, datetime]
Duration = float  # Duration in seconds

# ID types
SessionID = str
TileID = int
NodeID = int
PluginName = str


# =============================================================================
# CONFIGURATION TYPES
# =============================================================================

class PluginConfig(TypedDict, total=False):
    """Configuration for a single plugin."""
    plugin_name: str
    enabled: bool
    config: Dict[str, Any]


class DataSourceConfig(PluginConfig, total=False):
    """Configuration for data source plugins."""
    type: str
    file_pattern: str
    required: bool
    bodypart: str
    likelihood_threshold: float


class SharedResourceConfig(PluginConfig, total=False):
    """Configuration for shared resource plugins."""
    type: str
    map_path: str
    map_settings: Dict[str, Any]


class AnalyzerConfig(PluginConfig, total=False):
    """Configuration for analyzer plugins."""
    type: str
    metrics: Dict[str, Dict[str, Any]]


class VisualizationPluginConfig(PluginConfig, total=False):
    """Configuration for visualization plugins."""
    type: str
    output_formats: List[str]
    plot_types: List[str]


class ExperimentConfig(TypedDict, total=False):
    """Complete experiment configuration."""
    data_sources: List[DataSourceConfig]
    shared_resources: List[SharedResourceConfig]
    analyzers: List[AnalyzerConfig]
    visualizations: Dict[str, VisualizationPluginConfig]
    output_path: str
    session_settings: Dict[str, Any]


# =============================================================================
# DATA STRUCTURE TYPES
# =============================================================================

@dataclass
class SessionMetadata:
    """Metadata for a session."""
    session_id: SessionID
    timestamp: datetime
    duration: Optional[Duration] = None
    frame_count: Optional[int] = None
    fps: Optional[float] = None
    video_path: Optional[FilePath] = None
    config_hash: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class AnalysisMetadata:
    """Metadata for analysis results."""
    analyzer_name: str
    version: str
    timestamp: datetime
    computation_time: Duration
    config_hash: str
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Structured analysis result with metadata."""
    session_id: SessionID
    analyzer_name: str
    metrics: Dict[str, Any]
    metadata: AnalysisMetadata
    
    def get_metric(self, name: str, default: Any = None) -> Any:
        """Get metric value by name."""
        return self.metrics.get(name, default)
    
    def has_metric(self, name: str) -> bool:
        """Check if metric exists."""
        return name in self.metrics


@dataclass
class VisualizationResult:
    """Result of visualization creation."""
    visualizer_name: str
    output_files: List[FilePath]
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class SessionSummary:
    """Summary statistics for a session."""
    session_id: SessionID
    metadata: SessionMetadata
    data_sources: List[str]
    total_frames: int
    total_columns: int
    analysis_results: Dict[str, AnalysisResult] = field(default_factory=dict)
    visualizations: Dict[str, VisualizationResult] = field(default_factory=dict)


# =============================================================================
# PLUGIN INTERFACE TYPES
# =============================================================================

PluginFactory = Callable[[Dict[str, Any], Any], Any]
PluginValidator = Callable[[str, type], None]

# Plugin registry mappings
DataSourceRegistry = Dict[str, type["IDataSource"]]
AnalyzerRegistry = Dict[str, type["IAnalyzer"]]
VisualizerRegistry = Dict[str, type["IVisualizer"]]
SharedResourceRegistry = Dict[str, type["ISharedResource"]]


# =============================================================================
# ANALYSIS TYPES
# =============================================================================

MetricValue = Union[float, int, List[float], List[int], np.ndarray]
MetricDict = Dict[str, MetricValue]
CrossSessionMetrics = Dict[SessionID, MetricDict]

# Statistical types
ConfidenceInterval = Tuple[float, float]
PValue = float
EffectSize = float

@dataclass
class StatisticalResult:
    """Result of statistical analysis."""
    test_name: str
    p_value: PValue
    effect_size: Optional[EffectSize] = None
    confidence_interval: Optional[ConfidenceInterval] = None
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# VISUALIZATION TYPES
# =============================================================================

# Plot types
PlotType = Literal[
    "learning_curve", "distribution", "comparison", 
    "correlation", "timeline", "summary", "heatmap"
]

# Visualization settings
class VisualizationSettings(TypedDict, total=False):
    """Settings for visualization creation."""
    plot_type: PlotType
    title: str
    xlabel: str
    ylabel: str
    colors: List[ColorValue]
    figure_size: Tuple[int, int]
    dpi: int
    output_formats: List[str]


# Video overlay settings
class OverlaySettings(TypedDict, total=False):
    """Settings for video overlay visualization."""
    position: Literal["top_left", "top_right", "bottom_left", "bottom_right"]
    size: float  # Fraction of frame size
    opacity: float
    colors: Dict[str, ColorValue]


# =============================================================================
# SPATIAL ANALYSIS TYPES
# =============================================================================

BoundingBox = Tuple[int, int, int, int]  # x, y, width, height
GridCoordinate = Tuple[int, int]  # row, col
MapCoordinate = Tuple[float, float]  # x, y in map space

@dataclass
class SpatialRegion:
    """Definition of a spatial region."""
    name: str
    bounds: BoundingBox
    tile_ids: List[TileID]
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectorySegment:
    """A segment of trajectory data."""
    start_frame: int
    end_frame: int
    coordinates: CoordinateList
    tile_sequence: List[TileID]
    duration: Duration
    distance: float


# =============================================================================
# GRAPH ANALYSIS TYPES
# =============================================================================

GraphNode = Union[int, str]
GraphEdge = Tuple[GraphNode, GraphNode]
GraphPath = List[GraphNode]

@dataclass
class GraphStructure:
    """Graph structure representation."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    node_properties: Dict[GraphNode, Dict[str, Any]] = field(default_factory=dict)
    edge_properties: Dict[GraphEdge, Dict[str, Any]] = field(default_factory=dict)


# =============================================================================
# PROTOCOL TYPES (for structural typing)
# =============================================================================

class Drawable(Protocol):
    """Protocol for objects that can be drawn/visualized."""
    
    def draw(self, ax: Any, **kwargs) -> None:
        """Draw the object on matplotlib axes."""
        ...


class Analyzable(Protocol):
    """Protocol for objects that can be analyzed."""
    
    def get_data(self) -> pd.DataFrame:
        """Get data for analysis."""
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for analysis."""
        ...


class Configurable(Protocol):
    """Protocol for configurable objects."""
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the object with given settings."""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        ...


# =============================================================================
# CALLBACK TYPES
# =============================================================================

ProgressCallback = Callable[[int, int, str], None]  # current, total, message
ErrorCallback = Callable[[Exception, str], None]  # error, context
LogCallback = Callable[[str, str], None]  # level, message

# Event handlers
EventHandler = Callable[[str, Dict[str, Any]], None]  # event_name, data
SessionEventHandler = Callable[[SessionID, str, Any], None]  # session_id, event, data


# =============================================================================
# UTILITY TYPES
# =============================================================================

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class LazyProperty(Generic[T]):
    """Type for lazy-loaded properties."""
    
    def __init__(self, func: Callable[[], T]):
        self.func = func
        self._value = None
        self._loaded = False
    
    def __get__(self, obj, objtype=None) -> T:
        if not self._loaded:
            self._value = self.func()
            self._loaded = True
        return self._value


# Result types for operations
class OperationResult(Generic[T]):
    """Generic result type for operations that may succeed or fail."""
    
    def __init__(self, success: bool, value: Optional[T] = None, error: Optional[str] = None):
        self.success = success
        self.value = value
        self.error = error
    
    @classmethod
    def success_result(cls, value: T) -> "OperationResult[T]":
        """Create successful result."""
        return cls(True, value)
    
    @classmethod
    def error_result(cls, error: str) -> "OperationResult[T]":
        """Create error result."""
        return cls(False, None, error)
    
    def is_success(self) -> bool:
        """Check if operation was successful."""
        return self.success
    
    def is_error(self) -> bool:
        """Check if operation failed."""
        return not self.success
    
    def unwrap(self) -> T:
        """Get value or raise exception if error."""
        if self.success:
            return self.value
        else:
            raise ValueError(f"Operation failed: {self.error}")
    
    def unwrap_or(self, default: T) -> T:
        """Get value or default if error."""
        return self.value if self.success else default


# =============================================================================
# VALIDATION TYPES
# =============================================================================

ValidationRule = Callable[[Any], bool]
ValidationError = NamedTuple("ValidationError", [("field", str), ("message", str)])

class ValidationResult:
    """Result of validation operation."""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
    
    def add_error(self, field: str, message: str) -> None:
        """Add validation error."""
        self.errors.append(ValidationError(field, message))
    
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return len(self.errors) == 0
    
    def get_error_messages(self) -> List[str]:
        """Get formatted error messages."""
        return [f"{error.field}: {error.message}" for error in self.errors]