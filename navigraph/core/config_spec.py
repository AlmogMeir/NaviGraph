"""Configuration specifications using Pydantic for validation."""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator, ValidationError
import os


class DataSourceSpec(BaseModel):
    """Specification for a data source."""
    name: str = Field(..., description="Name identifier for this data source")
    type: str = Field(..., description="Type of data source plugin to use")
    file_pattern: Optional[str] = Field(None, description="Regex pattern to match files")
    shared: bool = Field(False, description="Whether this is a shared resource")
    config: Dict[str, Any] = Field(default_factory=dict, description="Plugin-specific configuration")


class AnalyzeMetricSpec(BaseModel):
    """Specification for an analysis metric."""
    func_name: str = Field(..., description="Function name to call")
    mode: Optional[str] = Field(None, description="Mode for the function")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the function")
    
    # Common metric parameters
    a: Optional[Union[int, str]] = Field(None, description="Start location/tile")
    b: Optional[Union[int, str]] = Field(None, description="End location/tile")
    min_nodes_on_path: Optional[int] = Field(None, description="Minimum nodes on path")


class AnalyzeConfig(BaseModel):
    """Configuration for analysis phase."""
    save_as_csv: bool = Field(True, description="Save results as CSV")
    save_as_pkl: bool = Field(True, description="Save results as pickle")
    save_raw_data_as_pkl: bool = Field(False, description="Save raw session data")
    metrics: Dict[str, AnalyzeMetricSpec] = Field(
        default_factory=dict, 
        description="Metrics to compute"
    )


class VisualizationSpec(BaseModel):
    """Specification for a visualization."""
    name: str = Field(..., description="Visualization name")
    type: str = Field(..., description="Type of visualizer plugin")
    config: Dict[str, Any] = Field(default_factory=dict, description="Visualizer configuration")


class MapSettings(BaseModel):
    """Map configuration settings."""
    segment_length: int = Field(..., description="Pixels per maze segment")
    origin: Union[tuple[int, int], str] = Field(..., description="Top-left corner of maze in pixels")
    grid_size: Union[tuple[int, int], str] = Field(..., description="Maze grid dimensions")
    pixel_to_meter: float = Field(1.0, description="Conversion factor")
    
    @validator('origin', pre=True)
    def parse_origin(cls, v):
        """Parse origin from string format like '(47, 40)'."""
        if isinstance(v, str):
            # Remove parentheses and split
            v = v.strip('()')
            return tuple(int(x.strip()) for x in v.split(','))
        return v
    
    @validator('grid_size', pre=True)
    def parse_grid_size(cls, v):
        """Parse grid_size from string format like '(17, 17)'."""
        if isinstance(v, str):
            # Remove parentheses and split
            v = v.strip('()')
            return tuple(int(x.strip()) for x in v.split(','))
        return v


class LocationSettings(BaseModel):
    """Location tracking settings."""
    bodypart: str = Field("Nose", description="Primary bodypart for tracking")
    likelihood: float = Field(0.3, description="Confidence threshold")
    bodyparts: Union[str, List[str]] = Field("all", description="Bodyparts to track")


class ExperimentConfig(BaseModel):
    """Main experiment configuration with validation."""
    # Required fields
    experiment_path: Path = Field(..., description="Path to experiment data")
    
    # Optional fields with defaults
    experiment_output_path: str = Field(
        "{PROJECT_ROOT}/output", 
        description="Output path for results"
    )
    system_running_mode: str = Field(
        "analyze",
        description="System mode: calibrate, test, analyze, visualize"
    )
    
    # Component configurations
    data_sources: List[DataSourceSpec] = Field(
        default_factory=list,
        description="Data source specifications"
    )
    analyze: AnalyzeConfig = Field(
        default_factory=AnalyzeConfig,
        description="Analysis configuration"
    )
    visualizations: Union[List[VisualizationSpec], Dict[str, Any]] = Field(
        default_factory=list,
        description="Visualization specifications"
    )
    
    # Settings
    map_settings: Optional[MapSettings] = Field(None, description="Map configuration")
    location_settings: Optional[LocationSettings] = Field(None, description="Location settings")
    
    # Legacy fields (to be moved to appropriate plugins)
    reward_tile_id: Optional[int] = Field(None, description="Reward location tile ID")
    map_path: Optional[Path] = Field(None, description="Path to map file")
    
    # Logging
    verbose: bool = Field(False, description="Enable verbose logging")
    
    @validator('experiment_path')
    def validate_experiment_path(cls, v, values):
        """Ensure experiment path exists."""
        if not v.exists():
            raise ValueError(f"Experiment path does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Experiment path must be a directory: {v}")
        return v
    
    @validator('system_running_mode')
    def validate_running_mode(cls, v):
        """Validate running mode string."""
        valid_modes = {'calibrate', 'test', 'analyze', 'visualize'}
        modes = set(mode.strip() for mode in v.lower().split('&'))
        invalid_modes = modes - valid_modes
        if invalid_modes:
            raise ValueError(f"Invalid running modes: {invalid_modes}")
        return v
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for backward compatibility


def validate_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Validate configuration dictionary and return typed config.
    
    Args:
        config_dict: Raw configuration dictionary
        
    Returns:
        Validated ExperimentConfig
        
    Raises:
        ValidationError: If configuration is invalid
    """
    try:
        return ExperimentConfig(**config_dict)
    except ValidationError as e:
        # Enhance error messages for better user experience
        error_messages = []
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            error_messages.append(f"{field_path}: {error['msg']}")
        
        raise ValueError(
            f"Configuration validation failed:\n" + 
            "\n".join(f"  - {msg}" for msg in error_messages)
        ) from e