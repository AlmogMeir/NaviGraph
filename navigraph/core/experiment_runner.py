"""Experiment runner for NaviGraph.

Orchestrates experiment execution: session discovery, data integration, 
analysis, and result output.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import shutil
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from .session import Session
from .file_discovery import FileDiscoveryEngine
from .visualization_pipeline import VisualizationPipeline


# Configuration constants
DEFAULT_RUNNING_MODE = 'analyze'
EXPERIMENT_PATH = 'experiment_path'
OUTPUT_PATH = 'experiment_output_path'

# System Modes
SYSTEM_RUNNING_MODE_KEY = 'system_running_mode'
CALIBRATE_MODE = 'calibrate'
TEST_MODE = 'test'
VISUALIZE_MODE = 'visualize'
ANALYZE_MODE = 'analyze'
SUPPORTED_VIDEO_FORMATS = ['*.mp4', '*.avi']


class ExperimentRunner:
    """Orchestrates experiment execution: discovery, integration, analysis, output."""
    
    def __init__(self, config: DictConfig):
        """Initialize experiment runner with configuration.
        
        Args:
            config: Experiment configuration (OmegaConf DictConfig)
        """
        self.config = config
        self.system_mode = config.get(SYSTEM_RUNNING_MODE_KEY, DEFAULT_RUNNING_MODE)
        
        # Resolve relative paths using config directory
        config_dir = getattr(config, '_config_dir', None)
        self.config_dir = config_dir
        
        # Create timestamped experiment folder with flexible path resolution
        base_output_path = self._resolve_output_path(config.get(OUTPUT_PATH, '.'), config_dir)
        
        # Create experiment folder with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_folder = os.path.join(base_output_path, f"experiment_{timestamp}")
        os.makedirs(self.experiment_folder, exist_ok=True)
        
        # Save configuration to experiment folder for reproducibility
        config_copy_path = os.path.join(self.experiment_folder, 'config.yaml')
        OmegaConf.save(config, config_copy_path)
        
        # Configure loguru with file output in experiment folder
        log_file = os.path.join(self.experiment_folder, 'experiment.log')
        logger.add(log_file, format="{time} | {level} | {message}", level="DEBUG")
        
        if config.get('verbose', False):
            logger.info("Verbose logging enabled")
        
        # Validate and resolve experiment path
        experiment_path = config.get(EXPERIMENT_PATH)
        if not experiment_path:
            raise ValueError("experiment_path is required in configuration")
        
        if config_dir and not os.path.isabs(experiment_path):
            experiment_path = os.path.join(config_dir, experiment_path)
        
        if not os.path.isdir(experiment_path):
            raise ValueError(f"Experiment directory does not exist: {experiment_path}")
            
        self.experiment_path = experiment_path
        self.sessions: List[Session] = []
        self.analysis_results: Optional[pd.DataFrame] = None
        
        logger.info(f"Experiment runner initialized with mode: {self.system_mode}")
    
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
    
    def discover_session_folders(self) -> List[str]:
        """Discover session folders in the experiment directory.
        
        Returns:
            List of session folder names
        """
        logger.info("Discovering session folders")
        
        try:
            session_folders = []
            
            # Look for session subdirectories (normal case)
            for item in os.listdir(self.experiment_path):
                item_path = os.path.join(self.experiment_path, item)
                if os.path.isdir(item_path) and not item.startswith('.') and item != 'resources':
                    session_folders.append(item)
            
            if session_folders:
                logger.info(f"Found {len(session_folders)} session folders: {session_folders}")
            else:
                # Fallback: Check if experiment_path itself is a session (has data files directly)
                has_session_files = any(
                    os.path.isfile(os.path.join(self.experiment_path, f)) and 
                    (f.endswith('.h5') or f.endswith('.csv') or f.endswith('.avi') or f.endswith('.mp4'))
                    for f in os.listdir(self.experiment_path)
                )
                
                if has_session_files:
                    # Experiment path is itself a session folder - use "." as folder name
                    session_folders.append(".")
                    logger.info(f"Found direct session in experiment root")
                else:
                    logger.warning(f"No session folders or session files found in {self.experiment_path}")
            
            return session_folders
            
        except Exception as e:
            logger.error(f"Session folder discovery failed: {str(e)}")
            return []
    
    
    def create_sessions(self, session_folders: List[str]) -> None:
        """Create Session objects for discovered session folders.
        
        Args:
            session_folders: List of session folder names
        """
        logger.info(f"Creating {len(session_folders)} sessions")
        
        try:
            for i, session_folder in enumerate(session_folders):
                try:
                    # Handle direct session (.) vs named session folder
                    if session_folder == ".":
                        session_id = os.path.basename(self.experiment_path)
                    else:
                        session_id = session_folder
                    
                    # Create session configuration with data source specifications
                    session_config = {
                        'session_id': session_id,
                        'experiment_path': self.experiment_path,
                        'data_source_specifications': self.config.get('data_sources', []),
                        'session_settings': dict(self.config.get('location_settings', {})),
                        'analyze': dict(self.config.get('analyze', {})),
                        'reward_tile_id': self.config.get('reward_tile_id'),
                        'map_settings': dict(self.config.get('map_settings', {})),
                        'graph': dict(self.config.get('graph', {})),
                        'map_path': str(self.config.get('map_path', ''))
                    }
                    
                    # Create session - it will handle its own resource creation
                    session = Session(session_config, logger)
                    self.sessions.append(session)
                    
                    logger.info(f"✓ Created session {i+1}/{len(session_folders)}: {session_folder}")
                    
                except Exception as e:
                    logger.error(f"Failed to create session {session_folder}: {str(e)}")
                    continue
            
            logger.info(f"Successfully created {len(self.sessions)} sessions")
            
        except Exception as e:
            logger.error(f"Session creation failed: {str(e)}")
            raise
    
    def run_analysis(self) -> pd.DataFrame:
        """Run analysis on all sessions using analyzer plugins.
        
        Returns:
            DataFrame with analysis results
        """
        # Load plugins for analysis
        from ..plugins import data_sources, shared_resources, analyzers, visualizers
        from .registry import registry
        
        logger.info("Starting analysis phase")
        
        try:
            # Get available analyzer plugins
            available_analyzers = registry.list_all_plugins()['analyzers']
            logger.info(f"Available analyzers: {available_analyzers}")
            
            # Run single-session analysis
            session_results = {}  # For DataFrame creation
            analysis_results = {}  # For cross-session analysis (AnalysisResult objects)
            
            for session in self.sessions:
                logger.info(f"Analyzing session: {session.session_id}")
                session_metrics = {}
                session_analysis_results = {}
                
                # Run each analyzer plugin
                for analyzer_name in available_analyzers:
                    try:
                        # Get analyzer class and instantiate using factory pattern
                        analyzer_class = registry.get_analyzer_plugin(analyzer_name)
                        analyzer = analyzer_class.from_config({}, logger)
                        
                        # Run analysis
                        result = analyzer.analyze_session(session)
                        # Store both the result object and extracted metrics
                        session_analysis_results[analyzer_name] = result
                        session_metrics.update(result.metrics)
                        
                        logger.debug(f"✓ {analyzer_name}: {len(result.metrics)} metrics computed")
                        
                    except Exception as e:
                        logger.error(f"Analyzer {analyzer_name} failed for {session.session_id}: {str(e)}")
                        continue
                
                session_results[session.session_id] = session_metrics
                analysis_results[session.session_id] = session_analysis_results
                logger.info(f"✓ Session {session.session_id}: {len(session_metrics)} total metrics")
            
            # Run cross-session analysis
            logger.info("Running cross-session analysis")
            cross_session_results = {}
            
            for analyzer_name in available_analyzers:
                try:
                    analyzer_class = registry.get_analyzer_plugin(analyzer_name)
                    analyzer = analyzer_class()
                    
                    # Collect AnalysisResult objects for this analyzer across sessions
                    analyzer_results = {}
                    for session_id, session_analyzers in analysis_results.items():
                        if analyzer_name in session_analyzers:
                            analyzer_results[session_id] = session_analyzers[analyzer_name]
                    
                    if analyzer_results:
                        cross_metrics = analyzer.analyze_cross_session(self.sessions, analyzer_results)
                        cross_session_results.update(cross_metrics)
                        logger.debug(f"✓ {analyzer_name}: {len(cross_metrics)} cross-session metrics")
                    
                except Exception as e:
                    logger.error(f"Cross-session analysis failed for {analyzer_name}: {str(e)}")
                    continue
            
            # Create results DataFrame
            results_df = pd.DataFrame(session_results)
            
            # Add cross-session results as additional rows
            if cross_session_results:
                cross_session_df = pd.DataFrame([cross_session_results])
                cross_session_df.index = ['cross_session_stats']
                # Note: This is a simplified combination - more sophisticated merging may be needed
            
            self.analysis_results = results_df
            
            # Save results
            self._save_analysis_results(results_df)
            
            logger.info(f"Analysis complete: {len(results_df.columns)} sessions, {len(results_df.index)} metrics")
            return results_df
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def run_calibration(self, discovered_sessions: List[Dict[str, str]]) -> None:
        """Run calibration mode using calibration plugin.
        
        Args:
            discovered_sessions: List of discovered session files
        """
        # Load plugins for calibration
        from ..plugins import data_sources, shared_resources, analyzers, visualizers
        from .registry import registry
        
        logger.info("Starting calibration mode")
        
        try:
            if not discovered_sessions:
                raise ValueError("No sessions found for calibration")
            
            # Use first session for calibration (preserving original logic)
            first_session = discovered_sessions[0]
            calibration_video_path = first_session.get('video_file')
            
            if not calibration_video_path:
                raise ValueError("No video file found for calibration")
            
            # Create calibration provider plugin
            calibration_provider_class = registry.get_shared_resource_plugin('calibration_provider')
            calibrator = calibration_provider_class()
            
            # Initialize with basic config for calibration mode
            basic_config = {
                'pre_calculated_transform_matrix_path': None,  # We're creating new calibration
                'transform_matrix': None
            }
            calibrator.initialize_resource(basic_config, logger)
            
            # Run calibration using plugin
            transform_matrix = calibrator.find_transform_matrix(
                calibration_video_path, 
                str(self.config.map_path),
                dict(self.config)  # Pass full config for calibrator_parameters
            )
            
            # Update configuration with calibration results
            matrix_path = calibrator.get_transform_matrix_path()
            if matrix_path:
                self.config.calibrator_parameters.pre_calculated_transform_matrix_path = matrix_path
            
            logger.info("✓ Calibration completed successfully")
            
            # Test calibration if requested
            if TEST_MODE in self.system_mode:
                logger.info("Testing calibration")
                calibrator.test_calibration(
                    calibration_video_path,
                    str(self.config.map_path),
                    transform_matrix,
                    dict(self.config)  # Pass full config for test parameters
                )
                logger.info("✓ Calibration test completed")
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            raise
    
    def run_experiment(self) -> Optional[pd.DataFrame]:
        """Run complete experiment pipeline.
        
        Returns:
            Analysis results DataFrame if analysis was performed
        """
        logger.info(f"Starting experiment with mode: {self.system_mode}")
        
        try:
            # Discover session folders
            session_folders = self.discover_session_folders()
            
            if not session_folders:
                logger.warning("No session folders discovered - experiment terminated")
                return None
            
            # TODO: Update calibration and test modes for new architecture
            if CALIBRATE_MODE in self.system_mode:
                logger.warning("Calibration mode not yet updated for new architecture")
            
            if TEST_MODE in self.system_mode and CALIBRATE_MODE not in self.system_mode:
                logger.warning("Test mode not yet updated for new architecture")
            
            # Create sessions for analysis/visualization
            if ANALYZE_MODE in self.system_mode or VISUALIZE_MODE in self.system_mode:
                self.create_sessions(session_folders)
            
            # Run analysis
            results = None
            if ANALYZE_MODE in self.system_mode:
                results = self.run_analysis()
            
            # Handle visualization mode
            if VISUALIZE_MODE in self.system_mode:
                self._run_visualization()
            
            logger.info("✓ Experiment completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise
    
    
    def _save_analysis_results(self, results_df: pd.DataFrame) -> None:
        """Save analysis results to configured outputs."""
        try:
            analyze_config = self.config.get('analyze', {})
            
            # Create cross_session metrics directory
            cross_session_dir = os.path.join(self.experiment_folder, 'cross_session', 'metrics')
            os.makedirs(cross_session_dir, exist_ok=True)
            
            # Save cross-session results
            if analyze_config.get('save_as_csv', False):
                csv_path = os.path.join(cross_session_dir, 'analysis_results.csv')
                results_df.to_csv(csv_path)
                logger.info(f"✓ Cross-session results saved to CSV: {csv_path}")
            
            if analyze_config.get('save_as_pkl', False):
                pkl_path = os.path.join(cross_session_dir, 'analysis_results.pkl')
                results_df.to_pickle(pkl_path)
                logger.info(f"✓ Cross-session results saved to PKL: {pkl_path}")
            
            # Save per-session data
            for session in self.sessions:
                # Create session directories
                session_dir = os.path.join(self.experiment_folder, session.session_id)
                metrics_dir = os.path.join(session_dir, 'metrics')
                raw_data_dir = os.path.join(session_dir, 'raw_data')
                os.makedirs(metrics_dir, exist_ok=True)
                os.makedirs(raw_data_dir, exist_ok=True)
                
                # Save session-specific metrics
                if analyze_config.get('save_as_csv', False):
                    session_metrics = results_df[[session.session_id]]
                    session_csv_path = os.path.join(metrics_dir, 'session_metrics.csv')
                    session_metrics.to_csv(session_csv_path)
                
                if analyze_config.get('save_as_pkl', False):
                    session_metrics = results_df[[session.session_id]]
                    session_pkl_path = os.path.join(metrics_dir, 'session_metrics.pkl')
                    session_metrics.to_pickle(session_pkl_path)
                
                # Save raw data
                if analyze_config.get('save_raw_data_as_pkl', False):
                    raw_df = session.get_integrated_dataframe()
                    raw_path = os.path.join(raw_data_dir, 'session_raw_data.pkl')
                    raw_df.to_pickle(raw_path)
            
            logger.info(f"✓ Session data saved for {len(self.sessions)} sessions")
                
        except Exception as e:
            logger.warning(f"Failed to save some results: {str(e)}")
    
    def _test_saved_calibration(self, discovered_sessions: List[Dict[str, str]]) -> None:
        """Test saved calibration using calibration plugin."""
        # Load plugins for calibration testing
        from ..plugins import data_sources, shared_resources, analyzers, visualizers
        from .registry import registry
        
        logger.info("Testing saved calibration")
        
        try:
            first_session = discovered_sessions[0]
            calibration_video_path = first_session.get('video_file')
            
            # Create calibration provider plugin
            calibration_provider_class = registry.get_shared_resource_plugin('calibration_provider')
            calibrator = calibration_provider_class()
            
            # Initialize with saved calibration matrix
            calibration_config = {
                'pre_calculated_transform_matrix_path': str(self.config.calibrator_parameters.pre_calculated_transform_matrix_path),
                'transform_matrix': None
            }
            calibrator.initialize_resource(calibration_config, logger)
            
            # Test calibration using plugin
            calibrator.test_calibration(
                calibration_video_path,
                str(self.config.map_path),
                str(self.config.calibrator_parameters.pre_calculated_transform_matrix_path),
                dict(self.config)  # Pass full config for test parameters
            )
            
            logger.info("✓ Calibration test completed")
            
        except Exception as e:
            logger.error(f"Calibration test failed: {str(e)}")
            raise
    
    def _run_visualization(self) -> None:
        """Run visualization mode using plugin system."""
        # Load plugins for visualization
        from ..plugins import data_sources, shared_resources, analyzers, visualizers
        from .registry import registry
        
        logger.info("Starting visualization phase")
        
        try:
            # Check if we have sessions
            if not self.sessions:
                logger.warning("No sessions available for visualization")
                return
            
            # Check if visualization config exists
            if 'visualizations' not in self.config:
                logger.warning("No visualization configuration found")
                return
            
            # Create visualization pipeline
            viz_pipeline = VisualizationPipeline(self.config, registry, logger)
            
            # Process each session
            all_results = {}
            for session in self.sessions:
                logger.info(f"Creating visualizations for session: {session.session_id}")
                
                # Create session output directory
                session_output_path = os.path.join(self.experiment_folder, session.session_id)
                
                # Use experiment path as session path for file discovery
                session_path = self.experiment_path
                
                if not session_path:
                    logger.warning(f"No session path available for session {session.session_id}")
                    continue
                
                logger.info(f"Using session path: {session_path}")
                
                # Create visualizations
                session_results = viz_pipeline.create_session_visualizations(
                    session,
                    output_path=session_output_path,
                    session_path=session_path
                )
                
                all_results[session.session_id] = session_results
                
                # Log results
                for viz_name, outputs in session_results.items():
                    if outputs:
                        logger.info(f"✓ {viz_name} created {len(outputs)} output(s)")
                    else:
                        logger.warning(f"✗ {viz_name} produced no output")
            
            logger.info(f"Visualization complete: {len(all_results)} sessions processed")
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            raise