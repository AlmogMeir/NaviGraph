"""Visualization pipeline for NaviGraph.

This module provides a simple, clean pipeline for running visualization stages.
Each visualization is treated as a pipeline with one or more stages. The pipeline
manages data flow between stages and handles all file I/O through publishers.
"""

from typing import Dict, List, Any, Optional, Union, Iterator
import pandas as pd
from pathlib import Path
from loguru import logger
import numpy as np

from .interfaces import IVisualizer, Logger
from .session import Session
from .exceptions import NavigraphError
from .publishers import get_publisher


class VisualizationPipeline:
    """Pipeline for running visualization stages.
    
    The pipeline orchestrates visualization stages, managing data flow
    between them and handling output through publishers. All visualizations
    are treated uniformly as pipelines with N stages (where N >= 1).
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        plugin_registry = None,
        logger_instance: Optional[Logger] = None
    ):
        """Initialize visualization pipeline.
        
        Args:
            config: Full configuration dictionary
            plugin_registry: Optional plugin registry (defaults to global)
            logger_instance: Optional logger instance
        """
        self.config = config
        self.registry = plugin_registry
        self.logger = logger_instance or logger
    
    def create_session_visualizations(
        self,
        session: Session,
        output_path: Optional[str] = None,
        session_path: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Create all configured visualizations for a session.
        
        Args:
            session: Session object with integrated data
            output_path: Base output directory path
            session_path: Path to session directory for file discovery
            
        Returns:
            Dict mapping visualization names to list of output file paths
        """
        # Get visualizations config
        viz_configs = self.config.get('visualizations', {})
        if not viz_configs:
            self.logger.warning("No visualizations configured")
            return {}
        
        # Prepare common data and resources
        session_data = session.get_integrated_dataframe()
        shared_resources = session.shared_resources
        session_id = session.session_id
        reward_tile_id = session.config.get('reward_tile_id')
        
        # Process each visualization as a pipeline
        results = {}
        for viz_name, viz_config in viz_configs.items():
            self.logger.info(f"Processing visualization: {viz_name}")
            
            try:
                # Run the visualization pipeline
                output_files = self._run_visualization_pipeline(
                    viz_name=viz_name,
                    viz_config=viz_config,
                    session=session,
                    session_data=session_data,
                    session_id=session_id,
                    session_path=session_path,
                    shared_resources=shared_resources,
                    reward_tile_id=reward_tile_id,
                    output_path=output_path
                )
                
                results[viz_name] = output_files
                
                if output_files:
                    self.logger.info(f"✓ {viz_name}: {len(output_files)} output(s) created")
                else:
                    self.logger.warning(f"✗ {viz_name}: No outputs created")
                    
            except Exception as e:
                self.logger.error(f"Visualization {viz_name} failed: {str(e)}")
                results[viz_name] = []
        
        return results
    
    def _run_visualization_pipeline(
        self,
        viz_name: str,
        viz_config: Dict[str, Any],
        session,
        session_data: pd.DataFrame,
        session_id: str,
        session_path: Optional[str],
        shared_resources: Dict[str, Any],
        reward_tile_id: Any,
        output_path: Optional[str]
    ) -> List[str]:
        """Run a single visualization pipeline.
        
        Args:
            viz_name: Name of the visualization
            viz_config: Configuration for this visualization
            session: Session object with full data access
            session_data: Session DataFrame
            session_id: Session identifier
            session_path: Path to session files
            shared_resources: Shared resources (map, graph, etc.)
            reward_tile_id: Reward tile ID
            output_path: Base output directory
            
        Returns:
            List of created output file paths
        """
        # Get pipeline stages
        stages = viz_config.get('stages', [])
        if not stages:
            self.logger.error(f"{viz_name}: No stages configured")
            return []
        
        # Get publisher configuration
        publisher_type = viz_config.get('publisher', 'video')
        output_name = viz_config.get('output_name', viz_name)
        
        # Determine output directory
        if output_path:
            viz_output_path = Path(output_path) / 'visualizations'
        else:
            viz_output_path = Path('.') / 'visualizations'
        viz_output_path.mkdir(parents=True, exist_ok=True)
        
        # Common kwargs for all stages
        stage_kwargs = {
            'session_path': session_path,
            'session_id': session_id,
            'shared_resources': shared_resources,
            'reward_tile_id': reward_tile_id
        }
        
        # Discover video file for frame processing
        video_path = self._discover_video_file(stages, session_path)
        if not video_path:
            self.logger.error(f"No video file found for visualization {viz_name}")
            return []
        
        # Create visualizer instances for all stages
        visualizers = []
        self.logger.info(f"Running {len(stages)}-stage pipeline for {viz_name}")
        
        for i, stage_config in enumerate(stages):
            plugin_name = stage_config.get('plugin')
            stage_params = stage_config.get('config', {})
            
            if not plugin_name:
                self.logger.error(f"Stage {i+1} missing 'plugin' specification")
                return []
            
            # Get visualizer instance
            try:
                if not self.registry:
                    from .registry import registry
                    self.registry = registry
                visualizer_class = self.registry.get_visualizer_plugin(plugin_name)
                visualizer = visualizer_class.from_config(stage_params, self.logger)
                visualizers.append((plugin_name, visualizer))
            except Exception as e:
                self.logger.error(f"Failed to create visualizer {plugin_name}: {str(e)}")
                return []
            
            self.logger.info(f"Stage {i+1}/{len(stages)}: {plugin_name}")
        
        # Process video frame by frame
        try:
            processed_frames = self._process_video_frames(
                video_path=video_path,
                session=session,
                visualizers=visualizers,
                session_id=session_id
            )
            
            if not processed_frames:
                self.logger.error(f"No frames processed for {viz_name}")
                return []
                
        except Exception as e:
            self.logger.error(f"Frame processing failed for {viz_name}: {str(e)}")
            return []
        
        # Publish final output
        current_data = processed_frames  # Set processed frames as output
        try:
            publisher = get_publisher(
                publisher_type=publisher_type,
                output_path=str(viz_output_path),
                output_name=output_name
            )
            
            # Get FPS from first stage config if available (for video publisher)
            fps = stages[0].get('config', {}).get('output_fps', 30.0)
            
            output_file = publisher.publish(
                frames=current_data,
                fps=fps,
                session_id=session_id
            )
            
            if output_file:
                self.logger.info(f"✓ Published {viz_name}: {Path(output_file).name}")
                return [output_file]
            else:
                self.logger.error(f"Failed to publish {viz_name}")
                return []
                
        except Exception as e:
            self.logger.error(f"Publishing failed for {viz_name}: {str(e)}")
            return []
    
    def _discover_video_file(self, stages: List[Dict[str, Any]], session_path: str) -> Optional[str]:
        """Discover video file needed by the visualization stages.
        
        Args:
            stages: List of visualization stage configurations
            session_path: Path to session directory for file discovery
            
        Returns:
            Path to discovered video file, or None if not found
        """
        # Check if any stage requires video files
        video_patterns = set()
        
        for stage_config in stages:
            stage_params = stage_config.get('config', {})
            file_requirements = stage_params.get('file_requirements', {})
            
            # Look for video file requirements
            for req_name, pattern in file_requirements.items():
                if 'video' in req_name.lower():
                    video_patterns.add(pattern)
        
        if not video_patterns:
            self.logger.warning("No video file requirements found in stages")
            return None
        
        # Use first video pattern to discover file
        video_pattern = next(iter(video_patterns))
        
        # Use direct file discovery instead of BasePlugin 
        try:
            import re
            video_files = []
            search_path = Path(session_path)
            
            if search_path.exists():
                for file_path in search_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = str(file_path.relative_to(search_path))
                        if re.match(video_pattern, relative_path):
                            video_files.append(file_path)
            if video_files:
                video_path = str(video_files[0])
                self.logger.info(f"Discovered video file: {Path(video_path).name}")
                return video_path
            else:
                self.logger.error(f"No video file found matching pattern: {video_pattern}")
                return None
                
        except Exception as e:
            self.logger.error(f"Video file discovery failed: {str(e)}")
            return None
    
    def _process_video_frames(self, video_path: str, session, visualizers: List, session_id: str) -> List:
        """Process video frames through the visualizer pipeline.
        
        Args:
            video_path: Path to input video file
            session: Session object with full data access
            visualizers: List of (name, visualizer) tuples
            session_id: Session identifier for logging
            
        Returns:
            List of processed frames (numpy arrays)
        """
        import cv2
        import numpy as np
        
        processed_frames = []
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.logger.info(f"Processing {total_frames} frames from {Path(video_path).name}")
            
            frame_index = 0
            frames_processed = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_frame = frame.copy()
                
                # Pass frame through each visualizer stage
                for visualizer_name, visualizer in visualizers:
                    try:
                        current_frame = visualizer.process_frame(current_frame, frame_index, session)
                        if current_frame is None:
                            raise ValueError(f"Visualizer {visualizer_name} returned None frame")
                    except Exception as e:
                        self.logger.error(f"Frame {frame_index} failed in {visualizer_name}: {str(e)}")
                        # Use previous frame to continue processing
                        current_frame = frame.copy()
                
                processed_frames.append(current_frame)
                frames_processed += 1
                frame_index += 1
                
                # Log progress periodically
                if frame_index % 100 == 0:
                    self.logger.debug(f"Processed {frame_index}/{total_frames} frames")
            
            cap.release()
            
            self.logger.info(f"Successfully processed {frames_processed} frames for {session_id}")
            return processed_frames
            
        except Exception as e:
            self.logger.error(f"Frame processing failed: {str(e)}")
            if 'cap' in locals():
                cap.release()
            return []