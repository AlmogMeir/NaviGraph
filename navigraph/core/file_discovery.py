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


#: Session folders are named session_<subject>_<DD>_<MM>_<YYYY>.
SESSION_FOLDER_PATTERN = re.compile(
    r"^session_(?P<subject>.+)_(?P<day>\d{2})_(?P<month>\d{2})_(?P<year>\d{4})$"
)


def normalize_session_date(date: str) -> str:
    """Normalize a session date to the DD_MM_YYYY used by session folders.

    Accepts the forms that appear across the configs and resource names:
    DD_MM_YYYY, YYYY_MM_DD, DD/MM/YYYY, YYYY-MM-DD.

    Args:
        date: Date in any of the accepted forms

    Returns:
        The date as DD_MM_YYYY

    Raises:
        SessionDiscoveryError: If the date cannot be understood
    """
    text = str(date).strip().replace('/', '_').replace('-', '_').replace('.', '_')

    # YAML reads an unquoted 2026_06_18 as the integer 20260618, so accept the
    # digits-only form as well rather than making the quoting a trap.
    if text.isdigit() and len(text) == 8:
        if 1900 <= int(text[:4]) <= 2999:
            text = f"{text[:4]}_{text[4:6]}_{text[6:]}"      # YYYYMMDD
        elif 1900 <= int(text[4:]) <= 2999:
            text = f"{text[:2]}_{text[2:4]}_{text[4:]}"      # DDMMYYYY

    parts = text.split('_')

    if len(parts) == 3 and all(p.isdigit() for p in parts):
        if len(parts[0]) == 4:                       # YYYY_MM_DD
            year, month, day = parts
        elif len(parts[2]) == 4:                     # DD_MM_YYYY
            day, month, year = parts
        else:
            year = month = day = None
        if year and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            return f"{int(day):02d}_{int(month):02d}_{int(year):04d}"

    raise SessionDiscoveryError(
        f"Could not read session date {date!r}. Use DD_MM_YYYY, YYYY_MM_DD, "
        f"DD/MM/YYYY or YYYY-MM-DD."
    )


def parse_session_folder(folder_name: str) -> Optional[Tuple[str, str]]:
    """Split a session folder name into (subject, DD_MM_YYYY), or None."""
    match = SESSION_FOLDER_PATTERN.match(folder_name)
    if not match:
        return None
    return (match.group('subject'),
            f"{match.group('day')}_{match.group('month')}_{match.group('year')}")


def find_session_folder(sessions_dir: Path, date: str,
                        subject: Optional[str] = None) -> Path:
    """Find the one session folder for a date, or fail saying what is there.

    Picking the session by date rather than by whatever folder happens to sit
    in the experiment directory is what keeps a run from silently analysing a
    different day's recording than the one its mapping and calibration are for.

    Args:
        sessions_dir: Directory holding session_<subject>_<DD>_<MM>_<YYYY> folders
        date: Session date, in any form normalize_session_date() accepts
        subject: Optional subject, required only when a date has several

    Returns:
        Path to the matching session folder

    Raises:
        SessionDiscoveryError: If the directory, the date, or a unique match is missing
    """
    sessions_dir = Path(sessions_dir)
    wanted = normalize_session_date(date)

    if not sessions_dir.is_dir():
        raise SessionDiscoveryError(
            f"Session directory does not exist: {sessions_dir}. "
            f"Set session.sessions_dir to the folder holding the session_* folders."
        )

    available = {}
    for folder in sorted(sessions_dir.iterdir()):
        if not folder.is_dir():
            continue
        parsed = parse_session_folder(folder.name)
        if parsed:
            available.setdefault(parsed[1], []).append((parsed[0], folder))

    matches = available.get(wanted, [])
    if subject:
        matches = [(subj, folder) for subj, folder in matches if subj == subject]

    if len(matches) == 1:
        return matches[0][1]

    if not matches:
        known = ", ".join(sorted(available)) or "none"
        for_subject = f" for subject {subject}" if subject else ""
        raise SessionDiscoveryError(
            f"No session folder{for_subject} for {wanted} in {sessions_dir}. "
            f"Expected session_<subject>_{wanted}. Dates present: {known}"
        )

    subjects = ", ".join(sorted(subj for subj, _ in matches))
    raise SessionDiscoveryError(
        f"{len(matches)} session folders match {wanted} in {sessions_dir} "
        f"(subjects: {subjects}). Set session.subject to choose one."
    )


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
        self.logger.debug(f"[DEBUG] discover_files_by_pattern called with search_path: {search_path} and pattern: {pattern}")
        if not search_path.exists():
            self.logger.warning(f"Search path does not exist: {search_path}")
            return []
        try:
            all_files = list(search_path.iterdir())
            self.logger.debug(f"[DEBUG] All files in {search_path}: {[f.name for f in all_files]}")
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            matches = []
            for item in all_files:
                # Match both files and directories
                if (item.is_file() or item.is_dir()) and compiled_pattern.search(item.name):
                    matches.append(item)
            self.logger.debug(f"[DEBUG] Found {len(matches)} items matching pattern '{pattern}' in {search_path}: {[m.name for m in matches]}")
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