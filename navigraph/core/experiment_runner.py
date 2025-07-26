"""Experiment runner for NaviGraph plugin system.

This module provides the main orchestration for running experiments with the new
plugin architecture. It replaces the original Manager class while preserving
all existing functionality and adding modern features like structured logging,
progress tracking, and comprehensive error handling.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from .session import Session
from .file_discovery import FileDiscoveryEngine
from .registry import registry
from ..plugins import data_sources, shared_resources, analyzers

# Import original components for backward compatibility (calibrator now migrated)
# from calibrator.maze_calibrator import MazeCalibrator - MIGRATED TO PLUGIN

# Constants from original Manager
DEFAULT_RUNNING_MODE = 'analyze'
STREAM_PATH = 'stream_path'
DETECTION_PATH = 'keypoint_detection_file_path'
OUTPUT_PATH = 'experiment_output_path'

# System Modes
SYSTEM_RUNNING_MODE_KEY = 'system_running_mode'
CALIBRATE_MODE = 'calibrate'
TEST_MODE = 'test'
VISUALIZE_MODE = 'visualize'
ANALYZE_MODE = 'analyze'
SUPPORTED_VIDEO_FORMATS = ['*.mp4', '*.avi']


class ExperimentRunner:
    """Modern experiment runner with plugin architecture support.
    
    This class orchestrates the entire experiment pipeline:
    1. File discovery and session detection
    2. Shared resource initialization
    3. Session data integration via plugins
    4. Analysis execution via analyzer plugins
    5. Result aggregation and output
    """
    
    def __init__(self, config: DictConfig):
        """Initialize experiment runner with configuration.
        
        Args:
            config: Experiment configuration (OmegaConf DictConfig)
        """
        self.config = config
        self.system_mode = config.get(SYSTEM_RUNNING_MODE_KEY, DEFAULT_RUNNING_MODE)
        
        # Setup logging
        output_path = config.get(OUTPUT_PATH, '.')
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            
        # Configure loguru with file output
        log_file = os.path.join(output_path, 'navigraph_experiment.log')
        logger.add(log_file, format="{time} | {level} | {message}", level="DEBUG")
        
        if config.get('verbose', False):
            logger.info("Verbose logging enabled")
        
        # Initialize components
        stream_path = config.get(STREAM_PATH, '.')
        self.file_discovery = FileDiscoveryEngine(stream_path, logger)
        self.sessions: List[Session] = []
        self.shared_resources: Dict[str, Any] = {}
        self.analysis_results: Optional[pd.DataFrame] = None
        
        logger.info(f"Experiment runner initialized with mode: {self.system_mode}")
    
    def discover_sessions(self) -> List[Dict[str, str]]:
        """Discover available sessions using file discovery (preserving original Manager logic).
        
        Returns:
            List of session dictionaries with discovered file paths
        """
        logger.info("Starting session discovery")
        
        try:
            # Get paths from configuration
            stream_path = self.config.get(STREAM_PATH)
            detection_path = self.config.get(DETECTION_PATH)
            
            if not stream_path or not detection_path:
                raise ValueError("stream_path and keypoint_detection_file_path required in configuration")
            
            # Validate paths exist
            if not os.path.isdir(stream_path):
                raise ValueError(f"Stream path does not exist: {stream_path}")
            if not os.path.isdir(detection_path):
                raise ValueError(f"Detection path does not exist: {detection_path}")
            
            # Find video files (preserving original logic)
            input_streams = []
            for supported_type in SUPPORTED_VIDEO_FORMATS:
                input_streams.extend(glob.glob(os.path.join(stream_path, supported_type), recursive=True))
            
            # Find H5 files (preserving original logic)
            detection_paths = glob.glob(os.path.join(stream_path, "*.h5"), recursive=True)
            
            if len(input_streams) == 0:
                logger.warning(f"No video files found in {stream_path}")
                return []
            
            if len(detection_paths) == 0:
                logger.warning(f"No H5 files found in {stream_path}")
                return []
            
            if len(input_streams) != len(detection_paths):
                logger.warning(f"Mismatch: {len(input_streams)} videos vs {len(detection_paths)} H5 files")
            
            # Match detection files to stream files by name (preserving original logic)
            sorted_detection_paths = []
            matched_streams = []
            
            for input_stream in input_streams:
                for detection_path in detection_paths:
                    stream_basename = os.path.basename(input_stream).split('.')[0]
                    detection_basename = os.path.basename(detection_path)
                    if stream_basename in detection_basename:
                        sorted_detection_paths.append(detection_path)
                        matched_streams.append(input_stream)
                        break
            
            # Create session dictionaries
            discovered_sessions = []
            for video_file, h5_file in zip(matched_streams, sorted_detection_paths):
                session_dict = {
                    'video_file': video_file,
                    'h5_file': h5_file,
                    'session_name': os.path.basename(video_file).split('.')[0]
                }
                discovered_sessions.append(session_dict)
            
            logger.info(f"Discovered {len(discovered_sessions)} sessions")
            return discovered_sessions
            
        except Exception as e:
            logger.error(f"Session discovery failed: {str(e)}")
            return []
    
    def initialize_shared_resources(self) -> None:
        """Initialize shared resources from configuration.
        
        This method initializes all shared resources (map, graph, calibration)
        that will be used across sessions.
        """
        logger.info("Initializing shared resources")
        
        try:
            # Initialize map provider
            if 'map_path' in self.config and 'map_settings' in self.config:
                map_provider_class = registry.get_shared_resource_plugin('map_provider')
                map_provider = map_provider_class()
                
                map_config = {
                    'map_path': self.config.map_path,
                    'map_settings': dict(self.config.map_settings)
                }
                
                map_provider.initialize_resource(map_config, logger)
                self.shared_resources['maze_map'] = map_provider
                logger.info("✓ Map provider initialized")
            
            # Initialize graph provider
            if 'graph' in self.config:
                graph_provider_class = registry.get_shared_resource_plugin('graph_provider')
                graph_provider = graph_provider_class()
                
                graph_config = dict(self.config.graph)
                graph_provider.initialize_resource(graph_config, logger)
                self.shared_resources['graph'] = graph_provider
                logger.info("✓ Graph provider initialized")
            
            # Initialize calibration provider
            calibration_config = self.config.get('calibrator_parameters', {})
            if calibration_config:
                calibration_provider_class = registry.get_shared_resource_plugin('calibration_provider')
                calibration_provider = calibration_provider_class()
                
                calib_resource_config = {
                    'pre_calculated_transform_matrix_path': calibration_config.get('pre_calculated_transform_matrix_path'),
                    'transform_matrix': calibration_config.get('transform_matrix')
                }
                
                calibration_provider.initialize_resource(calib_resource_config, logger)
                self.shared_resources['calibration'] = calibration_provider
                logger.info("✓ Calibration provider initialized")
            
            logger.info(f"Initialized {len(self.shared_resources)} shared resources")
            
        except Exception as e:
            logger.error(f"Shared resource initialization failed: {str(e)}")
            raise
    
    def create_sessions(self, discovered_sessions: List[Dict[str, str]]) -> None:
        """Create Session objects from discovered files.
        
        Args:
            discovered_sessions: List of session dictionaries from discovery
        """
        logger.info(f"Creating {len(discovered_sessions)} sessions")
        
        try:
            for i, session_files in enumerate(discovered_sessions):
                try:
                    # Create session configuration
                    session_config = {
                        'data_sources': self._get_data_source_config(session_files),
                        'shared_resources': self.shared_resources,
                        'session_settings': dict(self.config.get('location_settings', {})),
                        'analyze': dict(self.config.get('analyze', {})),
                        'reward_tile_id': self.config.get('reward_tile_id'),
                        'session_id': f"session_{i:03d}"
                    }
                    
                    # Create session
                    session = Session(session_config, logger)
                    self.sessions.append(session)
                    
                    logger.info(f"✓ Created session {i+1}/{len(discovered_sessions)}")
                    
                except Exception as e:
                    logger.error(f"Failed to create session {i+1}: {str(e)}")
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
        logger.info("Starting analysis phase")
        
        try:
            # Get available analyzer plugins
            available_analyzers = registry.list_all_plugins()['analyzers']
            logger.info(f"Available analyzers: {available_analyzers}")
            
            # Run single-session analysis
            session_results = {}
            
            for session in self.sessions:
                logger.info(f"Analyzing session: {session.session_id}")
                session_metrics = {}
                
                # Run each analyzer plugin
                for analyzer_name in available_analyzers:
                    try:
                        # Get analyzer class and instantiate
                        analyzer_class = registry.get_analyzer_plugin(analyzer_name)
                        analyzer = analyzer_class()
                        
                        # Run analysis
                        metrics = analyzer.analyze_session(session)
                        session_metrics.update(metrics)
                        
                        logger.debug(f"✓ {analyzer_name}: {len(metrics)} metrics computed")
                        
                    except Exception as e:
                        logger.error(f"Analyzer {analyzer_name} failed for {session.session_id}: {str(e)}")
                        continue
                
                session_results[session.session_id] = session_metrics
                logger.info(f"✓ Session {session.session_id}: {len(session_metrics)} total metrics")
            
            # Run cross-session analysis
            logger.info("Running cross-session analysis")
            cross_session_results = {}
            
            for analyzer_name in available_analyzers:
                try:
                    analyzer_class = registry.get_analyzer_plugin(analyzer_name)
                    analyzer = analyzer_class()
                    
                    cross_metrics = analyzer.analyze_cross_session(self.sessions, session_results)
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
            # Discover sessions
            discovered_sessions = self.discover_sessions()
            
            if not discovered_sessions:
                logger.warning("No sessions discovered - experiment terminated")
                return None
            
            # Handle calibration mode
            if CALIBRATE_MODE in self.system_mode:
                self.run_calibration(discovered_sessions)
            
            # Handle test mode
            if TEST_MODE in self.system_mode and CALIBRATE_MODE not in self.system_mode:
                self._test_saved_calibration(discovered_sessions)
            
            # Initialize shared resources for analysis/visualization
            if ANALYZE_MODE in self.system_mode or VISUALIZE_MODE in self.system_mode:
                self.initialize_shared_resources()
                self.create_sessions(discovered_sessions)
            
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
    
    def _get_data_source_config(self, session_files: Dict[str, str]) -> List[Dict[str, Any]]:
        """Create data source configuration for a session.
        
        Args:
            session_files: Dictionary with discovered file paths
            
        Returns:
            List of data source configurations
        """
        data_sources = []
        
        # DeepLabCut data source (if H5 file found)
        if 'h5_file' in session_files:
            data_sources.append({
                'plugin_name': 'deeplabcut',
                'discovered_file_path': session_files['h5_file'],
                'bodypart': self.config.get('location_settings', {}).get('bodypart', 'nose'),
                'likelihood_threshold': self.config.get('location_settings', {}).get('likelihood', 0.3)
            })
        
        # Map integration (if map provider available)
        if 'maze_map' in self.shared_resources:
            data_sources.append({
                'plugin_name': 'map_integration'
            })
        
        # Graph integration (if graph provider available)
        if 'graph' in self.shared_resources:
            data_sources.append({
                'plugin_name': 'graph_integration',
                'reward_tile_id': self.config.get('reward_tile_id')
            })
        
        return data_sources
    
    def _save_analysis_results(self, results_df: pd.DataFrame) -> None:
        """Save analysis results to configured outputs."""
        try:
            output_path = self.config.get(OUTPUT_PATH, '.')
            analyze_config = self.config.get('analyze', {})
            
            # Save as CSV
            if analyze_config.get('save_as_csv', False):
                csv_path = os.path.join(output_path, 'analysis_results.csv')
                results_df.to_csv(csv_path)
                logger.info(f"✓ Results saved to CSV: {csv_path}")
            
            # Save as PKL
            if analyze_config.get('save_as_pkl', False):
                pkl_path = os.path.join(output_path, 'analysis_results.pkl')
                results_df.to_pickle(pkl_path)
                logger.info(f"✓ Results saved to PKL: {pkl_path}")
            
            # Save raw data
            if analyze_config.get('save_raw_data_as_pkl', False):
                for session in self.sessions:
                    raw_df = session.get_integrated_dataframe()
                    raw_path = os.path.join(output_path, f'{session.session_id}_raw.pkl')
                    raw_df.to_pickle(raw_path)
                logger.info(f"✓ Raw data saved for {len(self.sessions)} sessions")
                
        except Exception as e:
            logger.warning(f"Failed to save some results: {str(e)}")
    
    def _test_saved_calibration(self, discovered_sessions: List[Dict[str, str]]) -> None:
        """Test saved calibration using calibration plugin."""
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
        """Run visualization mode (placeholder for future implementation)."""
        logger.info("Visualization mode requested")
        logger.warning("Visualization functionality not yet implemented in plugin architecture")
        # TODO: Implement visualization using plugin system