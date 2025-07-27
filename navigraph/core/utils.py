"""Essential utility functions for NaviGraph."""

import os
import hashlib
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any


def resolve_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve relative paths to absolute paths.
    
    Args:
        path: Path to resolve
        base_dir: Base directory for relative paths
        
    Returns:
        Absolute Path object
    """
    path = Path(path)
    
    if path.is_absolute():
        return path
    
    if base_dir:
        return Path(base_dir) / path
    
    return path.resolve()


def ensure_directory(dir_path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.
    
    Args:
        dir_path: Directory path to create
        
    Returns:
        Path object for the directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_configuration_hash(config: Dict[str, Any]) -> str:
    """Compute hash of configuration for caching/comparison.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MD5 hash string
    """
    try:
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    except Exception:
        return ""