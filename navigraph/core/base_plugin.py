"""Base plugin class for NaviGraph plugin system."""

import os
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar, List, TYPE_CHECKING
from pathlib import Path
from loguru import logger

if TYPE_CHECKING:
    from .models import PluginValidationResult as ValidationResult
else:
    # Define locally to avoid circular imports
    from dataclasses import dataclass, field
    
    @dataclass
    class ValidationResult:
        """Result of plugin data validation."""
        plugin_name: str
        plugin_type: str
        is_valid: bool
        found_count: int
        message: str
        found_files: List[Path] = field(default_factory=list)

# Type alias for logger
Logger = type(logger)
T = TypeVar('T', bound='BasePlugin')


class BasePlugin(ABC):
    """Base class for all NaviGraph plugins."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger_instance: Optional[Logger] = None):
        """Initialize plugin with configuration and logger."""
        self.config = config or {}
        self.logger = logger_instance or logger
        self._initialized = False
        
        # Log initialization
        self.logger.debug(f"Initializing {self.__class__.__name__} with config: {self.config}")
    
    def initialize(self) -> None:
        """Initialize the plugin after construction."""
        if self._initialized:
            self.logger.warning(f"{self.__class__.__name__} already initialized")
            return
            
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self._validate_config()
        self._initialized = True
    
    def _validate_config(self) -> None:
        """Validate plugin configuration. Override in subclasses if needed."""
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls: Type[T], config: Dict[str, Any], logger_instance: Optional[Logger] = None) -> T:
        """Factory method to create plugin instance from configuration."""
        pass
    
    def discover_files(self, session_path: Path, pattern: str = None, is_shared: bool = False) -> List[Path]:
        """Discover files matching pattern in appropriate location.
        
        Args:
            session_path: Path to session directory
            pattern: Regex pattern to match files (uses config if not provided)
            is_shared: If True, look in resources folder instead of session
            
        Returns:
            List of matching file paths
        """
        # Get pattern from config if not provided
        if pattern is None:
            pattern = self.config.get('file_pattern', '')
        
        if not pattern:
            # No pattern = no files needed
            return []
        
        # Determine search paths based on shared flag
        if is_shared:
            # Look in resources folder
            search_paths = [
                session_path.parent / 'resources',  # ../resources
                session_path.parent.parent / 'resources',  # ../../resources  
            ]
        else:
            # Look in session folder
            search_paths = [session_path]
        
        # Search for files matching pattern
        matching_files = []
        for search_path in search_paths:
            if search_path.exists():
                self.logger.debug(f"Searching in {search_path} for pattern: {pattern}")
                for file_path in search_path.rglob('*'):
                    if file_path.is_file():
                        # Match against relative path from search root
                        relative_path = str(file_path.relative_to(search_path))
                        if re.match(pattern, relative_path):
                            matching_files.append(file_path)
                            self.logger.debug(f"Found matching file: {file_path}")
        
        return matching_files
    
    def validate_data_availability(self, session_path: Path) -> 'ValidationResult':
        """Validate that required data is available for this plugin.
        
        Args:
            session_path: Path to session directory
            
        Returns:
            ValidationResult with validation status
        """
        pattern = self.config.get('file_pattern', '')
        is_shared = self.config.get('shared', False)
        
        if not pattern:
            # No pattern = no file requirements
            return ValidationResult(
                plugin_name=self.__class__.__name__,
                plugin_type=getattr(self, 'plugin_type', 'unknown'),
                is_valid=True,
                found_count=0,
                message="No file requirements"
            )
        
        # Discover files
        found_files = self.discover_files(session_path, pattern, is_shared)
        
        # Determine location description
        location = "resources" if is_shared else "session"
        
        # Create validation result
        return ValidationResult(
            plugin_name=self.__class__.__name__,
            plugin_type=getattr(self, 'plugin_type', 'unknown'),
            is_valid=len(found_files) > 0,
            found_count=len(found_files),
            found_files=found_files,
            message=f"Found {len(found_files)} files in {location}" if found_files 
                    else f"No files matching '{pattern}' in {location}"
        )