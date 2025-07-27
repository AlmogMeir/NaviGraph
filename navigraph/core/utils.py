"""Utility functions for NaviGraph.

This module provides common utility functions used across multiple
NaviGraph components to reduce code duplication and improve consistency.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, TYPE_CHECKING
import pandas as pd
import numpy as np
from loguru import logger

from .constants import FileExtensions, DefaultValues, ConfigKeys
from .exceptions import (
    FileSystemError, ConfigurationValidationError, 
    DataFormatError, NavigraphError
)

if TYPE_CHECKING:
    from .interfaces import Logger


def validate_file_path(
    file_path: Union[str, Path], 
    required_extensions: Optional[List[str]] = None,
    must_exist: bool = True,
    context: Optional[str] = None
) -> Path:
    """Validate and normalize file path.
    
    Args:
        file_path: Path to validate
        required_extensions: List of allowed file extensions (e.g., ['.h5', '.csv'])
        must_exist: Whether the file must exist
        context: Optional context for error messages
        
    Returns:
        Validated Path object
        
    Raises:
        FileSystemError: If validation fails
    """
    try:
        path = Path(file_path)
        
        # Check if file exists when required
        if must_exist and not path.exists():
            raise FileSystemError(
                f"File not found: {path}",
                {"file_path": str(path), "context": context}
            )
        
        # Check file extension if specified
        if required_extensions and path.suffix not in required_extensions:
            raise FileSystemError(
                f"Invalid file extension for {path}: expected one of {required_extensions}",
                {"file_path": str(path), "extensions": required_extensions}
            )
        
        return path
        
    except Exception as e:
        if isinstance(e, FileSystemError):
            raise
        raise FileSystemError(f"Invalid file path '{file_path}': {str(e)}")


def ensure_directory(dir_path: Union[str, Path], parents: bool = True) -> Path:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        dir_path: Directory path to ensure
        parents: Whether to create parent directories
        
    Returns:
        Path object for the directory
        
    Raises:
        FileSystemError: If directory creation fails
    """
    try:
        path = Path(dir_path)
        path.mkdir(parents=parents, exist_ok=True)
        return path
    except Exception as e:
        raise FileSystemError(f"Failed to create directory '{dir_path}': {str(e)}")


def validate_dataframe_columns(
    df: pd.DataFrame, 
    required_columns: List[str],
    context: Optional[str] = None
) -> None:
    """Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        context: Optional context for error messages
        
    Raises:
        DataFormatError: If required columns are missing
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        available_columns = list(df.columns)
        message = f"Missing required columns: {missing_columns}"
        if context:
            message = f"{context}: {message}"
        
        raise DataFormatError("DataFrame", "required columns", {
            "missing_columns": missing_columns,
            "available_columns": available_columns,
            "context": context
        })


def safe_eval_config_value(value: Any, allowed_types: Tuple = (int, float, str, list, tuple)) -> Any:
    """Safely evaluate configuration values that might be strings.
    
    This function handles the common pattern of configuration values
    that can be either the actual value or a string representation.
    
    Args:
        value: Value to evaluate
        allowed_types: Tuple of allowed result types
        
    Returns:
        Evaluated value
        
    Raises:
        ConfigurationValidationError: If evaluation fails or result type is invalid
    """
    if not isinstance(value, str):
        if isinstance(value, allowed_types):
            return value
        else:
            raise ConfigurationValidationError(
                "config_value", value, f"one of {allowed_types}"
            )
    
    try:
        # Use eval safely for known patterns
        if value.startswith('(') and value.endswith(')'):
            # Tuple pattern like "(47, 40)"
            result = eval(value)
        elif value.startswith('[') and value.endswith(']'):
            # List pattern like "[255, 0, 0]"
            result = eval(value)
        elif value.replace('.', '').replace('-', '').isdigit():
            # Numeric values
            result = float(value) if '.' in value else int(value)
        else:
            # String values
            result = value
        
        if isinstance(result, allowed_types):
            return result
        else:
            raise ConfigurationValidationError(
                "config_value", result, f"one of {allowed_types}"
            )
            
    except Exception as e:
        raise ConfigurationValidationError(
            "config_value", value, f"valid configuration value: {str(e)}"
        )


def compute_configuration_hash(config: Dict[str, Any]) -> str:
    """Compute hash of configuration for reproducibility tracking.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Hexadecimal hash string
    """
    # Convert config to sorted string representation for consistent hashing
    config_str = str(sorted(config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()


def merge_configurations(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries with deep merging.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    if not configs:
        return {}
    
    result = configs[0].copy()
    for config in configs[1:]:
        result = deep_merge(result, config)
    
    return result


def extract_session_id_from_path(file_path: Union[str, Path]) -> str:
    """Extract session ID from file path using common naming patterns.
    
    Args:
        file_path: Path to extract session ID from
        
    Returns:
        Extracted session ID
    """
    path = Path(file_path)
    filename = path.stem
    
    # Common patterns for session IDs in filenames
    patterns = [
        r'session_(\w+)',  # session_12345
        r'(\w+)_session',  # abc_session
        r'sess(\d+)',      # sess001
        r'(\w+)_\d{6}',    # prefix_123456
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback: use filename without extension
    return filename


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes}m"


def format_bytes(bytes_value: int) -> str:
    """Format byte count to human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted byte string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """Sanitize filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        replacement: Character to replace invalid characters with
        
    Returns:
        Sanitized filename
    """
    # Replace invalid filename characters
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, replacement, filename)
    
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    
    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized


def validate_coordinate_bounds(
    x: float, y: float, 
    width: int, height: int,
    margin: int = 0
) -> bool:
    """Validate that coordinates are within image bounds.
    
    Args:
        x: X coordinate
        y: Y coordinate
        width: Image width
        height: Image height
        margin: Optional margin for bounds checking
        
    Returns:
        True if coordinates are valid
    """
    return (margin <= x <= width - margin and 
            margin <= y <= height - margin)


def filter_dataframe_by_confidence(
    df: pd.DataFrame,
    confidence_column: str,
    threshold: float = DefaultValues.DEFAULT_LIKELIHOOD_THRESHOLD
) -> pd.DataFrame:
    """Filter DataFrame rows based on confidence threshold.
    
    Args:
        df: DataFrame to filter
        confidence_column: Name of confidence column
        threshold: Minimum confidence threshold
        
    Returns:
        Filtered DataFrame
    """
    if confidence_column not in df.columns:
        logger.warning(f"Confidence column '{confidence_column}' not found in DataFrame")
        return df
    
    return df[df[confidence_column] >= threshold].copy()


def calculate_distance(
    x1: float, y1: float, 
    x2: float, y2: float
) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        x1, y1: Coordinates of first point
        x2, y2: Coordinates of second point
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def smooth_trajectory(
    coordinates: np.ndarray, 
    window_size: int = 5,
    method: str = 'moving_average'
) -> np.ndarray:
    """Smooth trajectory data to reduce noise.
    
    Args:
        coordinates: Array of coordinates (N x 2 for x,y)
        window_size: Size of smoothing window
        method: Smoothing method ('moving_average', 'gaussian')
        
    Returns:
        Smoothed coordinates
    """
    if len(coordinates) < window_size:
        return coordinates
    
    if method == 'moving_average':
        # Use pandas rolling window for convenience
        df = pd.DataFrame(coordinates, columns=['x', 'y'])
        smoothed = df.rolling(window=window_size, center=True, min_periods=1).mean()
        return smoothed.values
    
    elif method == 'gaussian':
        from scipy import ndimage
        sigma = window_size / 3.0  # Convert window size to sigma
        smoothed_x = ndimage.gaussian_filter1d(coordinates[:, 0], sigma)
        smoothed_y = ndimage.gaussian_filter1d(coordinates[:, 1], sigma)
        return np.column_stack([smoothed_x, smoothed_y])
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a simple text progress bar.
    
    Args:
        current: Current progress value
        total: Total progress value
        width: Width of progress bar in characters
        
    Returns:
        Progress bar string
    """
    if total == 0:
        return "[" + "=" * width + "] 100%"
    
    progress = current / total
    filled = int(width * progress)
    bar = "=" * filled + "-" * (width - filled)
    percentage = int(progress * 100)
    
    return f"[{bar}] {percentage}%"


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, logger_instance: Optional["Logger"] = None):
        """Initialize timing context.
        
        Args:
            operation_name: Name of operation being timed
            logger_instance: Logger to use for output
        """
        self.operation_name = operation_name
        self.logger = logger_instance or logger
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is not None:
            self.logger.error(f"Failed {self.operation_name} after {format_duration(duration)}")
        else:
            self.logger.info(f"Completed {self.operation_name} in {format_duration(duration)}")
    
    @property
    def duration(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None