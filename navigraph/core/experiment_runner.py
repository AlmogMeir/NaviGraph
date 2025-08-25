"""Experiment runner for NaviGraph.

Orchestrates experiment execution: session discovery, data integration, 
analysis, and result output.
"""

import os
from typing import List, Dict, Optional, Set
from datetime import datetime
from pathlib import Path
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from .session import Session
from .enums import SystemMode
from .config_spec import ExperimentConfig, validate_config
from .models import SessionInfo, SessionValidation, ValidationReport, PluginValidationResult, CrossSessionResults
from .constants import (
    ConfigKeys, Directories, FileNames, ColumnNames, 
    LogFormats, FilePatterns, Defaults
)


class ExperimentRunner:
    """Orchestrates experiment execution: discovery, integration, analysis, output."""
    
    def __init__(self, config: DictConfig):
        """Initialize experiment runner with configuration.
        
        Args:
            config: Experiment configuration (OmegaConf DictConfig)
        """
        # Validate configuration (optional for backward compatibility)
        try:
            self.validated_config = validate_config(dict(config))
            logger.debug(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.INIT)} Configuration validation passed")
        except ValueError as e:
            logger.warning(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.WARNING)} Configuration validation failed: {e}")
            logger.warning(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.WARNING)} Continuing with legacy configuration format")
            self.validated_config = None
        
        self.config = config  # Keep original for backward compatibility
        self.system_modes = SystemMode.from_string(
            config.get(ConfigKeys.SYSTEM_MODE, Defaults.RUNNING_MODE)
        )
        
        # Resolve relative paths using config directory
        config_dir = getattr(config, '_config_dir', None)
        self.config_dir = config_dir
        
        # Create timestamped experiment folder with flexible path resolution
        base_output_path = self._resolve_output_path(
            config.get(ConfigKeys.OUTPUT_PATH, Defaults.OUTPUT_PATH), 
            config_dir
        )
        
        # Create experiment folder with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_folder = Path(base_output_path) / f"experiment_{timestamp}"
        self.experiment_folder.mkdir(parents=True, exist_ok=True)
        
        # Save configuration to experiment folder for reproducibility
        config_copy_path = self.experiment_folder / FileNames.CONFIG_YAML
        OmegaConf.save(config, str(config_copy_path))
        
        # Configure loguru with file output in experiment folder
        log_file = self.experiment_folder / FileNames.EXPERIMENT_LOG
        logger.add(str(log_file), format="{time} | {level} | {message}", level="DEBUG")
        
        if config.get(ConfigKeys.VERBOSE, False):
            logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.INIT)} Verbose logging enabled")
        
        # Validate and resolve experiment path
        experiment_path = config.get(ConfigKeys.EXPERIMENT_PATH)
        if not experiment_path:
            raise ValueError("experiment_path is required in configuration")
        
        if config_dir and not os.path.isabs(experiment_path):
            experiment_path = os.path.join(config_dir, experiment_path)
        
        self.experiment_path = Path(experiment_path)
        if not self.experiment_path.is_dir():
            raise ValueError(f"Experiment directory does not exist: {self.experiment_path}")
            
        self.sessions: List[Session] = []
        self.session_results: Dict[str, Dict[str, float]] = {}
        self.cross_session_results: CrossSessionResults = CrossSessionResults()
        
        modes_str = ', '.join(mode.value for mode in self.system_modes)
        logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.INIT)} Experiment runner initialized with modes: {modes_str}")
    
    def _resolve_output_path(self, output_path: str, config_dir: Optional[str]) -> str:
        """Resolve output path with flexible options.
        
        Supports:
        - Absolute paths: /absolute/path
        - Home directory: ~/path
        - Project root: {PROJECT_ROOT}/path
        - Config relative: ./path (default behavior)
        
        Args:
            output_path: Path from configuration
            config_dir: Directory containing the config file
            
        Returns:
            Resolved absolute path
        """
        # Handle absolute paths
        if os.path.isabs(output_path):
            return output_path
        
        # Handle home directory paths
        if output_path.startswith('~/'):
            return os.path.expanduser(output_path)
        
        # Handle project root token
        if '{PROJECT_ROOT}' in output_path:
            # Find project root (directory containing pyproject.toml)
            project_root = self._find_project_root()
            return output_path.replace('{PROJECT_ROOT}', project_root)
        
        # Default: relative to config directory (backwards compatibility)
        if config_dir and not os.path.isabs(output_path):
            return os.path.join(config_dir, output_path)
        
        return output_path
    
    def _find_project_root(self) -> str:
        """Find the project root directory by looking for pyproject.toml."""
        current_dir = os.path.abspath(os.getcwd())
        
        # Start from current directory and walk up
        while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
            if os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        
        # If we can't find pyproject.toml, use the directory containing navigraph package
        try:
            import navigraph
            navigraph_path = os.path.dirname(os.path.dirname(navigraph.__file__))
            return navigraph_path
        except:
            # Fallback to current working directory
            return os.getcwd()
    
    def discover_sessions(self) -> List[SessionInfo]:
        """Discover session directories in the experiment.
        
        Returns:
            List of SessionInfo objects
        """
        logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.DISCOVERY)} Discovering sessions")
        
        try:
            sessions = []
            
            # Look for session subdirectories
            for item in self.experiment_path.iterdir():
                if (item.is_dir() and 
                    not item.name.startswith('.') and 
                    item.name != Directories.RESOURCES):
                    sessions.append(SessionInfo(name=item.name, path=item))
            
            if sessions:
                logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.DISCOVERY)} {LogFormats.FOUND} {len(sessions)} sessions")
            else:
                # Fallback: Check if experiment_path itself is a session
                if self._is_session_directory(self.experiment_path):
                    sessions.append(SessionInfo(
                        name=self.experiment_path.name,
                        path=self.experiment_path
                    ))
                    logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.DISCOVERY)} {LogFormats.FOUND} direct session in root")
                else:
                    logger.warning(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.WARNING)} No sessions found in {self.experiment_path}")
            
            return sessions
            
        except Exception as e:
            logger.error(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.ERROR)} Session discovery {LogFormats.FAILED}: {str(e)}")
            return []
    
    def _is_session_directory(self, path: Path) -> bool:
        """Check if directory could be a session (has files or data subdirs)."""
        if not path.exists() or not path.is_dir():
            return False
            
        # Check for any files
        has_files = any(p.is_file() for p in path.iterdir())
        
        # Check for data directories (non-hidden, non-resources)
        has_data_dirs = any(
            p.is_dir() and not p.name.startswith('.') and p.name != Directories.RESOURCES
            for p in path.iterdir()
        )
        
        return has_files or has_data_dirs
    
    
    def create_sessions(self, sessions: List[SessionInfo]) -> None:
        """Create Session objects for discovered sessions.
        
        Args:
            sessions: List of session information
        """
        logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.SESSION)} {LogFormats.CREATING} {len(sessions)} sessions")
        
        try:
            for i, session_info in enumerate(sessions):
                try:
                    # Pass entire config to session - let it extract what it needs
                    session_config = dict(self.config)
                    session_config[ConfigKeys.SESSION_ID] = session_info.name
                    session_config[ConfigKeys.EXPERIMENT_PATH] = str(self.experiment_path)
                    
                    # Transform data_sources to data_source_specifications for Session compatibility
                    if ConfigKeys.DATA_SOURCES in session_config:
                        session_config['data_source_specifications'] = session_config[ConfigKeys.DATA_SOURCES]
                    
                    # Create session with full configuration
                    session = Session(session_config, logger)
                    self.sessions.append(session)
                    
                    progress = LogFormats.PROGRESS.format(current=i+1, total=len(sessions))
                    logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.SESSION)} Created session {progress}: {session_info.name}")
                    
                except Exception as e:
                    logger.error(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.ERROR)} Failed to create session {session_info.name}: {str(e)}")
                    continue
            
            logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.SESSION)} {LogFormats.SUCCESS} - created {len(self.sessions)} sessions")
            
        except Exception as e:
            logger.error(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.ERROR)} Session creation {LogFormats.FAILED}: {str(e)}")
            raise
    
    def run_analysis(self) -> None:
        """Run analysis on all sessions using analyzer plugins."""
        logger.warning("Analysis system is being migrated to new architecture")
        return
    
    def validate_sessions(self, sessions: List[SessionInfo]) -> ValidationReport:
        """Validate sessions using configured data source plugins.
        
        Args:
            sessions: List of session directories
            
        Returns:
            ValidationReport with results from all plugins
        """
        logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.VALIDATION)} Validating sessions")
        
        # Load plugins to ensure they're available
        from .registry import registry
        
        all_validations = []
        
        for session_info in sessions:
            session_validation = SessionValidation(session_id=session_info.name)
            
            # Validate each configured data source
            for ds_spec in self.config.get(ConfigKeys.DATA_SOURCES, []):
                try:
                    plugin_type = ds_spec['type']
                    plugin_name = ds_spec.get('name', plugin_type)
                    
                    # Get plugin class
                    plugin_class = registry.get_data_source_plugin(plugin_type)
                    
                    # Create plugin config including file_pattern and shared flag
                    plugin_config = {
                        'file_pattern': ds_spec.get('file_pattern', ''),
                        'shared': ds_spec.get('shared', False),
                        **ds_spec.get('config', {})
                    }
                    
                    # Create plugin instance
                    plugin = plugin_class()
                    plugin.config = plugin_config
                    plugin.plugin_type = 'data_source'
                    
                    # Validate using base class method
                    result = plugin.validate_data_availability(session_info.path)
                    result.plugin_name = plugin_name  # Use configured name
                    session_validation.add_result(result)
                    
                except Exception as e:
                    # Create error result
                    error_result = PluginValidationResult(
                        plugin_name=plugin_name,
                        plugin_type='data_source',
                        is_valid=False,
                        found_count=0,
                        message=f"Validation error: {str(e)}"
                    )
                    session_validation.add_result(error_result)
            
            all_validations.append(session_validation)
        
        report = ValidationReport(session_validations=all_validations)
        logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.VALIDATION)} {LogFormats.SUCCESS} - {report.validation_rate:.1f}% valid")
        
        return report
    
    def run_calibration(self, sessions: List[SessionInfo]) -> None:
        """Run calibration mode using calibration plugin.
        
        Args:
            sessions: List of session directories
        """
        # Load plugins for calibration
        from .registry import registry
        
        logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.INIT)} Starting calibration mode")
        
        try:
            if not sessions:
                raise ValueError("No sessions found for calibration")
            
            # Get calibration plugin to find video
            calibration_provider_class = registry.get_shared_resource_plugin('calibration_provider')
            calibrator = calibration_provider_class()
            
            # Find first session with video file
            video_path = None
            
            # Look for video data source in config
            video_pattern = None
            for ds in self.config.get(ConfigKeys.DATA_SOURCES, []):
                if ds.get('type') == 'deeplab_cut':  # Assuming video comes from DLC
                    video_pattern = ds.get('video_pattern', r'.*\.mp4$')
                    break
            
            if not video_pattern:
                video_pattern = r'.*\.(mp4|avi|mov)$'  # Default video pattern
            
            # Find video in sessions
            for session_info in sessions:
                files = calibrator.discover_files(session_info.path, video_pattern)
                if files:
                    video_path = files[0]
                    break
            
            if not video_path:
                raise ValueError("No video file found in any session for calibration")
            
            # Initialize with basic config for calibration mode
            basic_config = {
                'pre_calculated_transform_matrix_path': None,  # We're creating new calibration
                'transform_matrix': None
            }
            calibrator.initialize_resource(basic_config, logger)
            
            # Run calibration using plugin
            transform_matrix = calibrator.find_transform_matrix(
                str(video_path), 
                str(self.config.get(ConfigKeys.MAP_PATH, '')),
                dict(self.config)  # Pass full config for calibrator_parameters
            )
            
            # Update configuration with calibration results
            matrix_path = calibrator.get_transform_matrix_path()
            if matrix_path:
                self.config.calibrator_parameters.pre_calculated_transform_matrix_path = matrix_path
            
            logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.INIT)} Calibration {LogFormats.SUCCESS}")
            
            # Test calibration if requested
            if SystemMode.TEST in self.system_modes:
                logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.INIT)} Testing calibration")
                calibrator.test_calibration(
                    str(video_path),
                    str(self.config.get(ConfigKeys.MAP_PATH, '')),
                    transform_matrix,
                    dict(self.config)  # Pass full config for test parameters
                )
                logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.INIT)} Calibration test {LogFormats.SUCCESS}")
            
        except Exception as e:
            logger.error(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.ERROR)} Calibration {LogFormats.FAILED}: {str(e)}")
            raise
    
    def run_experiment(self) -> None:
        """Run complete experiment pipeline."""
        modes_str = ', '.join(mode.value for mode in self.system_modes)
        logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.INIT)} Starting experiment with modes: {modes_str}")
        
        try:
            # Discover sessions
            discovered_sessions = self.discover_sessions()
            
            if not discovered_sessions:
                logger.warning(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.WARNING)} No sessions discovered - experiment terminated")
                return
            
            # Validate sessions
            validation_report = self.validate_sessions(discovered_sessions)
            if validation_report.invalid_sessions > 0:
                logger.warning(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.WARNING)} {validation_report.invalid_sessions} invalid sessions")
                logger.info(validation_report.print_report())
            
            # Handle calibration mode
            if SystemMode.CALIBRATE in self.system_modes:
                self.run_calibration(discovered_sessions)
            
            # Handle test-only mode
            if SystemMode.TEST in self.system_modes and SystemMode.CALIBRATE not in self.system_modes:
                self._test_saved_calibration(discovered_sessions)
            
            # Create sessions for analysis/visualization
            if SystemMode.ANALYZE in self.system_modes or SystemMode.VISUALIZE in self.system_modes:
                self.create_sessions(discovered_sessions)
            
            # Run analysis
            if SystemMode.ANALYZE in self.system_modes:
                self.run_analysis()
            
            # Handle visualization mode
            if SystemMode.VISUALIZE in self.system_modes:
                self._run_visualization()
            
            logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.INIT)} Experiment {LogFormats.SUCCESS}")
            
        except Exception as e:
            logger.error(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.ERROR)} Experiment {LogFormats.FAILED}: {str(e)}")
            raise
    
    
    def _save_analysis_results(self) -> None:
        """Save analysis results to configured outputs."""
        logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.SAVE)} Saving analysis results")
        
        try:
            analyze_config = self.config.get(ConfigKeys.ANALYZE, {})
            
            # Save session results
            if self.session_results:
                session_df = pd.DataFrame(self.session_results)
                
                # Save per-session data
                for session in self.sessions:
                    # Create session directories
                    session_dir = self.experiment_folder / Directories.SESSIONS / session.session_id
                    metrics_dir = session_dir / Directories.METRICS
                    raw_data_dir = session_dir / Directories.RAW_DATA
                    metrics_dir.mkdir(parents=True, exist_ok=True)
                    raw_data_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save session-specific metrics
                    if analyze_config.get('save_as_csv', False):
                        session_metrics = session_df[[session.session_id]]
                        session_csv_path = metrics_dir / FileNames.SESSION_METRICS_CSV
                        session_metrics.to_csv(str(session_csv_path))
                    
                    if analyze_config.get('save_as_pkl', False):
                        session_metrics = session_df[[session.session_id]]
                        session_pkl_path = metrics_dir / FileNames.SESSION_METRICS_PKL
                        session_metrics.to_pickle(str(session_pkl_path))
                    
                    # Save raw data
                    if analyze_config.get('save_raw_data_as_pkl', False):
                        raw_df = session.get_integrated_dataframe()
                        raw_path = raw_data_dir / FileNames.SESSION_RAW_DATA_PKL
                        raw_df.to_pickle(str(raw_path))
                
                logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.SAVE)} Session data saved for {len(self.sessions)} sessions")
            
            # Save cross-session results only if they exist
            if self.cross_session_results.metrics:
                cross_session_dir = self.experiment_folder / Directories.CROSS_SESSION / Directories.METRICS
                cross_session_dir.mkdir(parents=True, exist_ok=True)
                
                cross_df = pd.DataFrame([self.cross_session_results.metrics])
                cross_df.index = ['cross_session_stats']
                
                if analyze_config.get('save_as_csv', False):
                    csv_path = cross_session_dir / FileNames.ANALYSIS_RESULTS_CSV
                    cross_df.to_csv(str(csv_path))
                    logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.SAVE)} Cross-session CSV: {csv_path}")
                
                if analyze_config.get('save_as_pkl', False):
                    pkl_path = cross_session_dir / FileNames.ANALYSIS_RESULTS_PKL
                    cross_df.to_pickle(str(pkl_path))
                    logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.SAVE)} Cross-session PKL: {pkl_path}")
        except Exception as e:
            logger.warning(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.WARNING)} Failed to save some results: {str(e)}")
    
    def _test_saved_calibration(self, sessions: List[SessionInfo]) -> None:
        """Test saved calibration using calibration plugin."""
        # Load plugins for calibration testing
        from .registry import registry
        
        logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.INIT)} Testing saved calibration")
        
        try:
            # Create calibration provider plugin
            calibration_provider_class = registry.get_shared_resource_plugin('calibration_provider')
            calibrator = calibration_provider_class()
            
            # Find first session with video file
            video_path = None
            
            # Look for video data source in config
            video_pattern = None
            for ds in self.config.get(ConfigKeys.DATA_SOURCES, []):
                if ds.get('type') == 'deeplab_cut':  # Assuming video comes from DLC
                    video_pattern = ds.get('video_pattern', r'.*\.mp4$')
                    break
            
            if not video_pattern:
                video_pattern = r'.*\.(mp4|avi|mov)$'  # Default video pattern
            
            # Find video in sessions
            for session_info in sessions:
                files = calibrator.discover_files(session_info.path, video_pattern)
                if files:
                    video_path = files[0]
                    break
            
            if not video_path:
                raise ValueError("No video file found in any session for calibration test")
            
            # Initialize with saved calibration matrix
            calibration_config = {
                'pre_calculated_transform_matrix_path': str(self.config.calibrator_parameters.pre_calculated_transform_matrix_path),
                'transform_matrix': None
            }
            calibrator.initialize_resource(calibration_config, logger)
            
            # Test calibration using plugin
            calibrator.test_calibration(
                str(video_path),
                str(self.config.get(ConfigKeys.MAP_PATH, '')),
                str(self.config.calibrator_parameters.pre_calculated_transform_matrix_path),
                dict(self.config)  # Pass full config for test parameters
            )
            
            logger.info(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.INIT)} Calibration test {LogFormats.SUCCESS}")
            
        except Exception as e:
            logger.error(f"{LogFormats.PHASE_PREFIX.format(phase=LogFormats.ERROR)} Calibration test {LogFormats.FAILED}: {str(e)}")
            raise
    
    def _run_visualization(self) -> None:
        """Run visualization mode using plugin system."""
        logger.warning("Visualization pipeline is being migrated to new architecture")
        return
