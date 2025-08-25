"""Unified plugin architecture for NaviGraph.

Single base class that replaces the old data_sources/shared_resources/analyzers/visualizers split.
Implements two-phase execution: provide() then augment_data().
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
from loguru import logger

from .file_discovery import FileDiscoveryEngine


class NaviGraphPlugin(ABC):
    """Unified base class for all NaviGraph plugins.
    
    Plugins can:
    1. provide() objects to shared resources
    2. augment_data() to add columns to the session DataFrame  
    3. Both (provide objects AND augment data)
    4. Neither (pure analyzers that just consume data)
    
    Two-phase execution:
    1. All plugins run provide() first (in config order)
    2. All plugins run augment_data() second (in config order)
    
    Plugins can access any shared resources gathered so far and fail loudly 
    if they need something that's missing.
    """
    
    def __init__(self, config: Dict[str, Any], session_path: Path, experiment_path: Path):
        """Initialize plugin with configuration and paths.
        
        Args:
            config: Plugin configuration including file_pattern, file_path, shared
            session_path: Path to the specific session directory
            experiment_path: Path to the root experiment directory
        """
        self.config = config
        self.session_path = session_path
        self.experiment_path = experiment_path
        self.logger = logger
        
        # Store discovered files for this plugin
        self.discovered_files: List[Path] = []
        
        # Automatically setup file discovery based on config
        self._setup_file_discovery()
    
    def _setup_file_discovery(self) -> None:
        """Setup file discovery based on configuration.
        
        Supports:
        - file_pattern: regex pattern to match files/folders
        - file_path: direct file path (absolute or relative to experiment_path)
        - shared: if True, look in resources folder; if False, look in session folder
        """
        file_pattern = self.config.get('file_pattern')
        file_path = self.config.get('file_path') 
        shared = self.config.get('shared', False)
        
        # Create temporary file discovery engine for this plugin
        file_discovery_engine = FileDiscoveryEngine(self.experiment_path, self.logger)
        
        if file_pattern:
            # Use regex pattern to discover files/folders
            if shared:
                # Look in resources folder
                search_path = self.experiment_path / 'resources'
            else:
                # Look in session folder
                search_path = self.session_path
            
            if search_path.exists():
                self.discovered_files = file_discovery_engine.discover_files_by_pattern(
                    search_path, file_pattern
                )
                self.logger.debug(f"Plugin {self.__class__.__name__} discovered {len(self.discovered_files)} files with pattern '{file_pattern}' in {search_path}")
                    
        elif file_path:
            # Direct file path specified
            if Path(file_path).is_absolute():
                # Absolute path - use as-is
                resolved_path = Path(file_path)
            else:
                # Relative path - resolve from experiment directory
                resolved_path = self.experiment_path / file_path
                
            if resolved_path.exists():
                self.discovered_files = [resolved_path]
                self.logger.debug(f"Plugin {self.__class__.__name__} using direct file path: {resolved_path}")
            else:
                self.logger.warning(f"Plugin {self.__class__.__name__} file path not found: {resolved_path}")
                
        # If neither file_pattern nor file_path specified, no file discovery needed
        self.logger.debug(f"Plugin {self.__class__.__name__} file discovery complete: {len(self.discovered_files)} files")
    
    def _load_discovered_files(self) -> Any:
        """Load data from discovered files.
        
        Override this method in subclasses to implement specific loading logic.
        
        Returns:
            Loaded data in plugin-specific format
        """
        if not self.discovered_files:
            return None
            
        # Default implementation - return file paths
        return self.discovered_files
    
    # ============= Execution Methods (with defaults) =============
    
    def provide(self, shared_resources: Dict[str, Any]) -> None:
        """Provide objects to shared resources (Phase 1).
        
        Override this method if your plugin provides shared objects.
        Objects should be stored directly in the shared_resources dict.
        
        Args:
            shared_resources: Dictionary to store shared objects
        """
        # Default: provide nothing
        pass
    
    def augment_data(self, dataframe: pd.DataFrame, shared_resources: Dict[str, Any]) -> pd.DataFrame:
        """Add columns to the session DataFrame (Phase 2).
        
        Override this method if your plugin adds data columns.
        
        Args:
            dataframe: Current session DataFrame
            shared_resources: Available shared resources from provide() phase
            
        Returns:
            DataFrame with additional columns (or same DataFrame if no changes)
        """
        # Default: no data augmentation
        return dataframe
    
    @abstractmethod
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information for debugging and validation.
        
        Returns:
            Dictionary with plugin metadata
        """
        pass
    
    def validate(self) -> bool:
        """Validate plugin configuration and requirements.
        
        Returns:
            True if plugin is valid and ready to use
        """
        # Default validation: check if required files were found
        file_pattern = self.config.get('file_pattern')
        file_path = self.config.get('file_path')
        
        if file_pattern or file_path:
            # Plugin requires files - check if any were discovered
            if not self.discovered_files:
                self.logger.error(f"Plugin {self.__class__.__name__} requires files but none were found")
                return False
        
        return True