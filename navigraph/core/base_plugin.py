"""Base plugin class for NaviGraph plugin system."""

import os
import glob
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar
from pathlib import Path
from loguru import logger

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
    
    def get_file_requirements(self, session_path: str) -> Dict[str, str]:
        """Get file requirements using config regex or custom logic.
        
        Default implementation uses 'file_requirements' from config and performs
        regex-based file discovery. Plugins can override for custom logic.
        
        Args:
            session_path: Path to session directory for file discovery
            
        Returns:
            Dict mapping requirement_name -> discovered_file_path
            Empty dict if no requirements or files not found
        """
        # Get file requirements from config
        requirements = self.config.get('file_requirements', {})
        if not requirements:
            return {}
        
        # Use regex-based discovery
        return self._discover_files_by_regex(session_path, requirements)
    
    def _discover_files_by_regex(self, session_path: str, requirements: Dict[str, str]) -> Dict[str, str]:
        """Discover files using regex patterns.
        
        Args:
            session_path: Path to session directory
            requirements: Dict mapping requirement_name -> regex_pattern
            
        Returns:
            Dict mapping requirement_name -> discovered_file_path
        """
        discovered_files = {}
        session_dir = Path(session_path)
        
        if not session_dir.exists():
            self.logger.warning(f"Session directory does not exist: {session_path}")
            return discovered_files
        
        for req_name, pattern in requirements.items():
            try:
                # Find all files in session directory
                all_files = []
                for file_path in session_dir.rglob('*'):
                    if file_path.is_file():
                        all_files.append(str(file_path))
                
                # Filter by regex pattern
                import re
                regex = re.compile(pattern)
                matching_files = [f for f in all_files if regex.search(os.path.basename(f))]
                
                self.logger.debug(f"Searching for '{req_name}' with pattern '{pattern}' in {len(all_files)} files")
                
                if matching_files:
                    # Take first match (could be enhanced to handle multiple matches)
                    discovered_files[req_name] = matching_files[0]
                    self.logger.debug(f"Found {req_name}: {matching_files[0]}")
                else:
                    self.logger.debug(f"No file found matching pattern '{pattern}' for requirement '{req_name}'")
                    
            except Exception as e:
                self.logger.error(f"Error discovering files for requirement '{req_name}': {str(e)}")
        
        return discovered_files