"""File discovery engine for NaviGraph experiments.

This module provides regex-based file discovery capabilities for automatically
finding session folders and matching data files within each session. It handles
the complex task of mapping file patterns to actual files while providing
comprehensive error reporting and logging.

Key features:
- Regex-based pattern matching for flexible file naming
- Comprehensive error handling and user guidance  
- Support for optional vs required files
- Clear logging of discovery results
- Validation of experiment folder structure
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from loguru import logger

# Type alias for logger
Logger = type(logger)

from .exceptions import NavigraphError


class SessionDiscoveryError(NavigraphError):
    """Raised when session discovery fails."""
    pass


class FileDiscoveryError(NavigraphError):
    """Raised when file discovery encounters issues."""
    pass


class FileDiscoveryEngine:
    """Handles regex-based file discovery for experimental sessions."""
    
    def __init__(self, experiment_root_path: str, logger_instance: Logger):
        """Initialize file discovery engine."""
        self.experiment_path = Path(experiment_root_path).resolve()
        self.logger = logger_instance
        
        self._validate_experiment_path()
        
        # Cache for discovered sessions to avoid repeated filesystem operations
        self._discovered_sessions: Optional[List[str]] = None
        
        self.logger.debug(f"Initialized FileDiscoveryEngine for: {self.experiment_path}")
    
    def discover_session_folders(self, force_refresh: bool = False) -> List[str]:
        """Discover all session folders in experiment directory."""
        if self._discovered_sessions is not None and not force_refresh:
            return self._discovered_sessions
        
        try:
            # Folders to exclude from session discovery
            excluded_folders = {
                'shared_resources', 'shared', 'resources', 
                'output', 'results', 'analysis',
                '.git', '__pycache__', '.mypy_cache'
            }
            
            # Find all directories except common resource/system folders
            session_folders = [
                folder.name for folder in self.experiment_path.iterdir()
                if (folder.is_dir() and 
                    folder.name not in excluded_folders and 
                    not folder.name.startswith('.'))
            ]
            
            if not session_folders:
                raise SessionDiscoveryError(
                    f"No session folders found in experiment directory: {self.experiment_path}. "
                    f"Make sure your experiment contains session folders (directories with session data)."
                )
            
            # Sort naturally (session_1, session_2, session_10)
            session_folders.sort(key=self._natural_sort_key)
            
            self._discovered_sessions = session_folders
            
            self.logger.info(
                f"Discovered {len(session_folders)} session folders: "
                f"{', '.join(session_folders[:5])}{'...' if len(session_folders) > 5 else ''}"
            )
            
            return session_folders
            
        except PermissionError as e:
            raise SessionDiscoveryError(
                f"Permission denied accessing experiment directory: {self.experiment_path}"
            ) from e
        except OSError as e:
            raise SessionDiscoveryError(
                f"Filesystem error accessing experiment directory: {self.experiment_path} - {e}"
            ) from e
    
    
    def discover_files_by_pattern(self, search_path: Path, pattern: str) -> List[Path]:
        """Discover files and directories matching a regex pattern in a directory.
        
        Args:
            search_path: Directory to search in
            pattern: Regex pattern to match files/directories
            
        Returns:
            List of matching file/directory paths
        """
        if not search_path.exists():
            self.logger.warning(f"Search path does not exist: {search_path}")
            return []
        
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            matches = []
            
            for item in search_path.iterdir():
                # Match both files and directories
                if (item.is_file() or item.is_dir()) and compiled_pattern.search(item.name):
                    matches.append(item)
            
            self.logger.debug(f"Found {len(matches)} items matching pattern '{pattern}' in {search_path}")
            return matches
            
        except re.error as e:
            self.logger.error(f"Invalid regex pattern '{pattern}': {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error discovering files: {e}")
            return []

    def get_shared_resources_path(self) -> Path:
        """Get path to shared resources folder."""
        return self.experiment_path / 'shared_resources'
    
    def _validate_experiment_path(self) -> None:
        """Validate that experiment path exists and is accessible."""
        if not self.experiment_path.exists():
            raise SessionDiscoveryError(
                f"Experiment directory does not exist: {self.experiment_path}. "
                f"Make sure the path is correct and accessible."
            )
        
        if not self.experiment_path.is_dir():
            raise SessionDiscoveryError(
                f"Experiment path is not a directory: {self.experiment_path}. "
                f"Provide a path to a directory containing session folders."
            )
        
        # Test read access
        try:
            list(self.experiment_path.iterdir())
        except PermissionError:
            raise SessionDiscoveryError(
                f"Permission denied reading experiment directory: {self.experiment_path}. "
                f"Make sure you have read access to this directory."
            )
    
    
    def _natural_sort_key(self, session_name: str) -> List:
        """Generate sort key for natural sorting (session_1, session_2, session_10)."""
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', session_name)]