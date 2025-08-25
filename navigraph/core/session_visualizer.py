"""Session Visualizer orchestrator for NaviGraph.

Manages visualization pipeline for a session.
Orchestrates the looping over video frames, passes each frame through
registered visualizer functions in sequence, and manages output creation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from loguru import logger

from .registry import registry
from .exceptions import NavigraphError


class SessionVisualizer:
    """Orchestrates visualization pipeline for a session."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with visualization configuration.
        
        Args:
            config: Dict with structure:
                {
                    'visualizations': {
                        'pipeline': ['trajectory', 'map_overlay', 'metrics'],
                        'output': {
                            'enabled': True,
                            'path': './output',
                            'format': 'mp4',
                            'fps': 30,
                            'codec': 'mp4v'
                        },
                        'visualizer_configs': {
                            'trajectory': {'color': [0, 255, 0]},
                            'map_overlay': {'opacity': 0.7}
                        }
                    }
                }
        """
        viz_config = config.get('visualizations', {})
        self.pipeline = viz_config.get('pipeline', [])
        self.output_config = viz_config.get('output', {})
        self.visualizer_configs = viz_config.get('visualizer_configs', {})
        self.logger = logger
    
    def process_video(
        self, 
        video_path: str,
        dataframe: pd.DataFrame, 
        shared_resources: Dict[str, Any],
        output_name: str = "output"
    ) -> Optional[str]:
        """Process video through visualization pipeline.
        
        Args:
            video_path: Path to input video
            dataframe: Session dataframe with frame-aligned data
            shared_resources: Shared resources (graph, mapping, etc.)
            output_name: Name for output file (without extension)
            
        Returns:
            Path to output video if created, None otherwise
        """
        # Check if output is enabled
        if not self.output_config.get('enabled', True):
            self.logger.info("Video output disabled in config")
            return None
        
        # Check if we have a pipeline
        if not self.pipeline:
            self.logger.warning("No visualizers in pipeline")
            return None
        
        # Validate and load visualizer functions
        visualizers = []
        for viz_name in self.pipeline:
            try:
                viz_func = registry.get_visualizer(viz_name)
                visualizers.append((viz_name, viz_func))
                self.logger.info(f"Loaded visualizer: {viz_name}")
            except NavigraphError as e:
                self.logger.error(f"Visualizer '{viz_name}' not found: {e}")
                if self.visualizer_configs.get(viz_name, {}).get('required', False):
                    raise
        
        if not visualizers:
            self.logger.error("No valid visualizers loaded")
            return None
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise NavigraphError(f"Failed to open video: {video_path}")
        
        try:
            # Get video properties
            fps = self.output_config.get('fps', cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup output
            output_path = Path(self.output_config.get('path', './output'))
            output_path.mkdir(parents=True, exist_ok=True)
            
            output_format = self.output_config.get('format', 'mp4')
            output_file = output_path / f"{output_name}.{output_format}"
            
            # Get codec
            codec_str = self.output_config.get('codec', 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            
            # Create video writer
            writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise NavigraphError(f"Failed to create video writer for: {output_file}")
            
            self.logger.info(f"Processing {total_frames} frames through {len(visualizers)} visualizers")
            self.logger.info(f"Output: {output_file} ({width}x{height} @ {fps}fps)")
            
            # Process frames
            frame_idx = 0
            last_progress = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                
                # Get frame data from dataframe
                if frame_idx < len(dataframe):
                    frame_data = dataframe.iloc[frame_idx]
                else:
                    # No data for this frame
                    frame_data = pd.Series()
                
                # Apply visualizer pipeline (each modifies the frame)
                for viz_name, viz_func in visualizers:
                    viz_config = self.visualizer_configs.get(viz_name, {})
                    try:
                        frame = viz_func(
                            frame=frame,
                            frame_data=frame_data,
                            shared_resources=shared_resources,
                            **viz_config
                        )
                        
                        # Validate frame is still valid
                        if frame is None or not isinstance(frame, np.ndarray):
                            raise TypeError(f"Visualizer '{viz_name}' returned invalid frame")
                            
                    except Exception as e:
                        self.logger.error(f"Visualizer '{viz_name}' failed on frame {frame_idx}: {e}")
                        if viz_config.get('required', False):
                            raise
                
                # Write processed frame
                writer.write(frame)
                
                # Progress logging (every 10%)
                progress = int((frame_idx / total_frames) * 100)
                if progress >= last_progress + 10:
                    self.logger.debug(f"Progress: {progress}%")
                    last_progress = progress
                
                frame_idx += 1
            
            self.logger.info(f"✓ Processed {frame_idx} frames")
            
        finally:
            # Cleanup
            cap.release()
            if 'writer' in locals():
                writer.release()
            cv2.destroyAllWindows()
        
        self.logger.info(f"✓ Video saved to: {output_file}")
        return str(output_file)
    
    def find_video(self, session_path: Path) -> Optional[Path]:
        """Find video file in session directory.
        
        Args:
            session_path: Path to session directory
            
        Returns:
            Path to video file or None if not found
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        for ext in video_extensions:
            videos = list(session_path.glob(f'*{ext}'))
            if videos:
                # Return first video found
                return videos[0]
        
        # Check subdirectories
        for ext in video_extensions:
            videos = list(session_path.rglob(f'*{ext}'))
            if videos:
                return videos[0]
        
        return None
    
    def validate_pipeline(self) -> List[str]:
        """Validate that all visualizers in pipeline are registered.
        
        Returns:
            List of missing visualizer names
        """
        missing = []
        for viz_name in self.pipeline:
            try:
                registry.get_visualizer(viz_name)
            except NavigraphError:
                missing.append(viz_name)
        
        return missing