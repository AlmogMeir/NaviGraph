"""Centralized visualization configuration for NaviGraph.

This module provides a structured configuration system for visualization
settings, themes, and output formats.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from .types import RGBColor, ColorValue, PlotType, VisualizationSettings
from .constants import VisualizationDefaults
from .exceptions import ConfigurationError


class OutputFormat(Enum):
    """Supported output formats for visualizations."""
    PNG = "png"
    JPG = "jpg"
    SVG = "svg"
    PDF = "pdf"
    MP4 = "mp4"
    GIF = "gif"
    HTML = "html"
    

class ColorTheme(Enum):
    """Pre-defined color themes."""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    HIGH_CONTRAST = "high_contrast"
    COLORBLIND_SAFE = "colorblind_safe"
    PUBLICATION = "publication"


@dataclass
class FontSettings:
    """Font configuration for visualizations."""
    family: str = "Arial"
    size: int = 12
    title_size: int = 16
    label_size: int = 10
    weight: str = "normal"


@dataclass
class ColorPalette:
    """Color palette configuration."""
    background: ColorValue = (255, 255, 255)
    foreground: ColorValue = (0, 0, 0)
    primary: ColorValue = (31, 119, 180)
    secondary: ColorValue = (255, 127, 14)
    success: ColorValue = (44, 160, 44)
    warning: ColorValue = (255, 127, 14)
    error: ColorValue = (214, 39, 40)
    grid: ColorValue = (200, 200, 200)
    trajectory: List[ColorValue] = field(default_factory=lambda: [
        (31, 119, 180),   # Blue
        (255, 127, 14),   # Orange
        (44, 160, 44),    # Green
        (214, 39, 40),    # Red
        (148, 103, 189),  # Purple
        (140, 86, 75),    # Brown
        (227, 119, 194),  # Pink
        (127, 127, 127),  # Gray
    ])
    
    @classmethod
    def from_theme(cls, theme: ColorTheme) -> "ColorPalette":
        """Create color palette from predefined theme."""
        if theme == ColorTheme.DARK:
            return cls(
                background=(30, 30, 30),
                foreground=(220, 220, 220),
                primary=(100, 180, 255),
                secondary=(255, 180, 100),
                grid=(60, 60, 60)
            )
        elif theme == ColorTheme.HIGH_CONTRAST:
            return cls(
                background=(0, 0, 0),
                foreground=(255, 255, 255),
                primary=(0, 255, 0),
                secondary=(255, 255, 0),
                grid=(128, 128, 128)
            )
        elif theme == ColorTheme.COLORBLIND_SAFE:
            return cls(
                trajectory=[
                    (0, 73, 73),      # Dark teal
                    (0, 146, 146),    # Teal
                    (255, 109, 182),  # Pink
                    (255, 182, 119),  # Light orange
                    (73, 0, 146),     # Dark purple
                    (0, 109, 219),    # Blue
                    (182, 109, 255),  # Light purple
                    (109, 182, 255),  # Light blue
                ]
            )
        elif theme == ColorTheme.PUBLICATION:
            return cls(
                background=(255, 255, 255),
                foreground=(0, 0, 0),
                primary=(0, 0, 0),
                secondary=(128, 128, 128),
                grid=(200, 200, 200),
                trajectory=[
                    (0, 0, 0),        # Black
                    (128, 128, 128),  # Gray
                    (0, 0, 255),      # Blue
                    (255, 0, 0),      # Red
                ]
            )
        else:  # DEFAULT or LIGHT
            return cls()


@dataclass
class PlotSettings:
    """General plot configuration."""
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 100
    tight_layout: bool = True
    show_grid: bool = True
    show_legend: bool = True
    show_title: bool = True
    show_axes_labels: bool = True
    
    # Animation settings
    animation_fps: int = 30
    animation_bitrate: int = 5000
    
    # Marker settings
    marker_size: int = 50
    marker_alpha: float = 0.8
    line_width: float = 2.0
    line_alpha: float = 0.6


@dataclass
class VisualizationConfig:
    """Complete visualization configuration."""
    # Output settings
    output_formats: List[OutputFormat] = field(default_factory=lambda: [OutputFormat.PNG])
    output_path: Optional[str] = None
    filename_pattern: str = "{session_id}_{visualizer}_{timestamp}"
    
    # Visual settings
    theme: ColorTheme = ColorTheme.DEFAULT
    colors: ColorPalette = field(default_factory=ColorPalette)
    fonts: FontSettings = field(default_factory=FontSettings)
    plot_settings: PlotSettings = field(default_factory=PlotSettings)
    
    # Visualizer-specific settings
    trajectory_settings: Dict[str, Any] = field(default_factory=lambda: {
        "trail_length": 100,
        "show_confidence": True,
        "confidence_threshold": 0.9,
        "interpolate": True
    })
    
    map_settings: Dict[str, Any] = field(default_factory=lambda: {
        "show_tile_ids": True,
        "show_grid": True,
        "highlight_current": True,
        "overlay_alpha": 0.7
    })
    
    tree_settings: Dict[str, Any] = field(default_factory=lambda: {
        "node_size": 300,
        "edge_width": 2,
        "show_labels": True,
        "layout": "hierarchical",
        "highlight_path": True
    })
    
    metrics_settings: Dict[str, Any] = field(default_factory=lambda: {
        "plot_types": ["line", "bar", "scatter"],
        "show_statistics": True,
        "show_confidence_intervals": True,
        "group_by": "session"
    })
    
    def __post_init__(self):
        """Apply theme if specified."""
        if self.theme != ColorTheme.DEFAULT:
            self.colors = ColorPalette.from_theme(self.theme)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VisualizationConfig":
        """Create configuration from dictionary."""
        # Handle output formats
        if "output_formats" in config_dict:
            formats = config_dict["output_formats"]
            if isinstance(formats, list):
                config_dict["output_formats"] = [
                    OutputFormat(fmt) if isinstance(fmt, str) else fmt 
                    for fmt in formats
                ]
        
        # Handle theme
        if "theme" in config_dict:
            theme = config_dict["theme"]
            if isinstance(theme, str):
                config_dict["theme"] = ColorTheme(theme)
        
        # Handle nested dataclasses
        if "colors" in config_dict and isinstance(config_dict["colors"], dict):
            config_dict["colors"] = ColorPalette(**config_dict["colors"])
        
        if "fonts" in config_dict and isinstance(config_dict["fonts"], dict):
            config_dict["fonts"] = FontSettings(**config_dict["fonts"])
        
        if "plot_settings" in config_dict and isinstance(config_dict["plot_settings"], dict):
            config_dict["plot_settings"] = PlotSettings(**config_dict["plot_settings"])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {
            "output_formats": [fmt.value for fmt in self.output_formats],
            "output_path": self.output_path,
            "filename_pattern": self.filename_pattern,
            "theme": self.theme.value,
            "colors": {
                "background": self.colors.background,
                "foreground": self.colors.foreground,
                "primary": self.colors.primary,
                "secondary": self.colors.secondary,
                "success": self.colors.success,
                "warning": self.colors.warning,
                "error": self.colors.error,
                "grid": self.colors.grid,
                "trajectory": self.colors.trajectory
            },
            "fonts": {
                "family": self.fonts.family,
                "size": self.fonts.size,
                "title_size": self.fonts.title_size,
                "label_size": self.fonts.label_size,
                "weight": self.fonts.weight
            },
            "plot_settings": {
                "figure_size": self.plot_settings.figure_size,
                "dpi": self.plot_settings.dpi,
                "tight_layout": self.plot_settings.tight_layout,
                "show_grid": self.plot_settings.show_grid,
                "show_legend": self.plot_settings.show_legend,
                "show_title": self.plot_settings.show_title,
                "show_axes_labels": self.plot_settings.show_axes_labels,
                "animation_fps": self.plot_settings.animation_fps,
                "animation_bitrate": self.plot_settings.animation_bitrate,
                "marker_size": self.plot_settings.marker_size,
                "marker_alpha": self.plot_settings.marker_alpha,
                "line_width": self.plot_settings.line_width,
                "line_alpha": self.plot_settings.line_alpha
            },
            "trajectory_settings": self.trajectory_settings,
            "map_settings": self.map_settings,
            "tree_settings": self.tree_settings,
            "metrics_settings": self.metrics_settings
        }
        return result
    
    def save(self, filepath: Path) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> "VisualizationConfig":
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_visualizer_config(self, visualizer_name: str) -> Dict[str, Any]:
        """Get configuration for specific visualizer."""
        # Map visualizer names to settings
        visualizer_configs = {
            "trajectory_visualizer": self.trajectory_settings,
            "map_visualizer": self.map_settings,
            "tree_visualizer": self.tree_settings,
            "metrics_visualizer": self.metrics_settings
        }
        
        base_config = {
            "colors": self.colors,
            "fonts": self.fonts,
            "plot_settings": self.plot_settings,
            "output_formats": self.output_formats
        }
        
        # Merge base config with visualizer-specific settings
        if visualizer_name in visualizer_configs:
            base_config.update(visualizer_configs[visualizer_name])
        
        return base_config


def create_default_configs() -> Dict[str, VisualizationConfig]:
    """Create a set of default configuration presets."""
    presets = {
        "default": VisualizationConfig(),
        
        "publication": VisualizationConfig(
            theme=ColorTheme.PUBLICATION,
            output_formats=[OutputFormat.PDF, OutputFormat.SVG],
            plot_settings=PlotSettings(
                figure_size=(8, 6),
                dpi=300,
                show_grid=False
            )
        ),
        
        "presentation": VisualizationConfig(
            theme=ColorTheme.HIGH_CONTRAST,
            output_formats=[OutputFormat.PNG],
            plot_settings=PlotSettings(
                figure_size=(16, 9),
                dpi=150,
                marker_size=100,
                line_width=3.0
            ),
            fonts=FontSettings(
                size=16,
                title_size=24,
                label_size=14,
                weight="bold"
            )
        ),
        
        "web": VisualizationConfig(
            output_formats=[OutputFormat.HTML, OutputFormat.SVG],
            plot_settings=PlotSettings(
                figure_size=(12, 8),
                dpi=100
            )
        ),
        
        "animation": VisualizationConfig(
            output_formats=[OutputFormat.MP4, OutputFormat.GIF],
            plot_settings=PlotSettings(
                animation_fps=60,
                animation_bitrate=8000
            )
        )
    }
    
    return presets