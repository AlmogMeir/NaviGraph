"""Enumerations for NaviGraph system."""

from enum import Enum, auto
from typing import Set


class SystemMode(Enum):
    """System running modes for experiments."""
    CALIBRATE = "calibrate"
    TEST = "test"
    ANALYZE = "analyze"
    VISUALIZE = "visualize"
    
    @classmethod
    def from_string(cls, mode_string: str) -> Set['SystemMode']:
        """Parse mode string which may contain multiple modes separated by &.
        
        Args:
            mode_string: Mode string (e.g., 'analyze', 'visualize&analyze')
            
        Returns:
            Set of SystemMode enums
        """
        modes = set()
        for mode_part in mode_string.lower().split('&'):
            mode_part = mode_part.strip()
            for mode in cls:
                if mode.value == mode_part:
                    modes.add(mode)
                    break
            else:
                raise ValueError(f"Invalid mode: {mode_part}")
        return modes
    
    @classmethod
    def default(cls) -> 'SystemMode':
        """Get default system mode."""
        return cls.ANALYZE


class FileType(Enum):
    """Supported file types for data sources."""
    VIDEO = auto()
    H5 = auto()
    CSV = auto()
    NPY = auto()
    PNG = auto()
    ZARR = auto()
    
    @classmethod
    def get_patterns(cls, file_type: 'FileType') -> list[str]:
        """Get file patterns for a given file type."""
        patterns = {
            cls.VIDEO: ['*.mp4', '*.avi'],
            cls.H5: ['*.h5'],
            cls.CSV: ['*.csv'],
            cls.NPY: ['*.npy'],
            cls.PNG: ['*.png'],
            cls.ZARR: ['*.zarr']
        }
        return patterns.get(file_type, [])


class PluginType(Enum):
    """Types of plugins in the system."""
    DATA_SOURCE = "data_source"
    SHARED_RESOURCE = "shared_resource"
    ANALYZER = "analyzer"
    VISUALIZER = "visualizer"
    CROSS_SESSION_ANALYZER = "cross_session_analyzer"