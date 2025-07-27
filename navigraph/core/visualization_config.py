"""Simple visualization configuration for NaviGraph."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from .exceptions import ConfigurationError

# Simple type aliases  
ColorValue = Tuple[int, int, int]


class OutputFormat(Enum):
    """Supported output formats."""
    PNG = "png"
    SVG = "svg" 
    PDF = "pdf"


@dataclass
class VisualizationConfig:
    """Simple visualization configuration."""
    # Output settings
    output_formats: List[OutputFormat] = field(default_factory=lambda: [OutputFormat.PNG])
    output_path: Optional[str] = None
    
    # Basic settings
    figure_size: Tuple[int, int] = (10, 8)
    dpi: int = 100
    
    # Colors
    background_color: ColorValue = (255, 255, 255)
    primary_color: ColorValue = (31, 119, 180)
    
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
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "output_formats": [fmt.value for fmt in self.output_formats],
            "output_path": self.output_path,
            "figure_size": self.figure_size,
            "dpi": self.dpi,
            "background_color": self.background_color,
            "primary_color": self.primary_color
        }
    
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