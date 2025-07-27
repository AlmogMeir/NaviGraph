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
            # Find all directories except shared_resources
            session_folders = [
                folder.name for folder in self.experiment_path.iterdir()
                if folder.is_dir() and folder.name != 'shared_resources'
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
    
    def match_files_in_session(
        self, 
        session_folder_name: str, 
        file_pattern_mapping: Dict[str, str],
        validate_session_exists: bool = True
    ) -> Dict[str, Optional[str]]:
        """Match files in session folder using regex patterns."""
        session_path = self.experiment_path / session_folder_name
        
        if validate_session_exists and not session_path.exists():
            self.logger.error(f"Session folder not found: {session_path}")
            return {name: None for name in file_pattern_mapping.keys()}
        
        # Get all files in session folder efficiently
        try:
            available_files = [f.name for f in session_path.iterdir() if f.is_file()]
        except PermissionError:
            self.logger.error(f"Permission denied accessing session folder: {session_path}")
            return {name: None for name in file_pattern_mapping.keys()}
        except OSError as e:
            self.logger.error(f"Error reading session folder {session_path}: {e}")
            return {name: None for name in file_pattern_mapping.keys()}
        
        discovered_files = {}
        pattern_match_stats = []
        
        for data_source_name, regex_pattern in file_pattern_mapping.items():
            matched_file_path = self._find_matching_file(
                available_files, regex_pattern, session_path, data_source_name
            )
            discovered_files[data_source_name] = matched_file_path
            
            # Track pattern matching statistics
            if matched_file_path:
                pattern_match_stats.append(f"✓ {data_source_name}")
            else:
                pattern_match_stats.append(f"✗ {data_source_name}")
        
        self.logger.debug(
            f"File matching for {session_folder_name}: {', '.join(pattern_match_stats)}"
        )
        
        return discovered_files
    
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
    
    def _find_matching_file(
        self, 
        available_files: List[str], 
        regex_pattern: str, 
        session_path: Path,
        data_source_name: str
    ) -> Optional[str]:
        """Find single matching file using regex pattern with comprehensive logging."""
        try:
            compiled_pattern = re.compile(regex_pattern, re.IGNORECASE)
        except re.error as e:
            self.logger.error(
                f"Invalid regex pattern for {data_source_name}: '{regex_pattern}' - {e}"
            )
            return None
        
        # Find all matching files
        matching_files = [
            filename for filename in available_files 
            if compiled_pattern.match(filename)
        ]
        
        if not matching_files:
            self.logger.debug(
                f"No files found for {data_source_name} in {session_path.name} "
                f"(pattern: {regex_pattern}). "
                f"Available files: {', '.join(available_files[:5])}{'...' if len(available_files) > 5 else ''}"
            )
            return None
        
        if len(matching_files) == 1:
            selected_file = matching_files[0]
            full_path = str(session_path / selected_file)
            self.logger.debug(f"Found {data_source_name} file: {selected_file}")
            return full_path
        
        # Multiple matches - use first and warn
        selected_file = matching_files[0]
        full_path = str(session_path / selected_file)
        
        self.logger.warning(
            f"Multiple files match pattern for {data_source_name} in {session_path.name}: "
            f"{', '.join(matching_files)}. Using: {selected_file}. "
            f"Consider making your regex pattern more specific."
        )
        
        return full_path
    
    def _natural_sort_key(self, session_name: str) -> List:
        """Generate sort key for natural sorting (session_1, session_2, session_10)."""
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', session_name)]