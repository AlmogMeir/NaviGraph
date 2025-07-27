"""Trajectory visualizer plugin for NaviGraph.

This plugin visualizes animal movement trajectories by overlaying keypoint
positions on video frames, creating a visual trail of the animal's path.
"""

from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from ...core.interfaces import IVisualizer, Logger
from ...core.base_plugin import BasePlugin
from ...core.registry import register_visualizer_plugin


@register_visualizer_plugin("trajectory_visualizer")
class TrajectoryVisualizer(BasePlugin, IVisualizer):
    """Visualizes animal trajectories with customizable overlays.
    
    Features:
    - Draws current position markers
    - Shows trajectory trails
    - Configurable colors, sizes, and trail length
    - Optional confidence-based visualization
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance = None):
        """Factory method to create trajectory visualizer from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate trajectory visualizer configuration."""
        # All config keys are optional with sensible defaults
        pass
    
    def visualize(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> Optional[str]:
        """Create trajectory visualization on video frames.
        
        Args:
            data: DataFrame with keypoint coordinates and metadata
            config: Visualization-specific configuration
            shared_resources: Shared resources (not used for trajectory viz)
            output_path: Directory to save visualization outputs
            **kwargs: Additional parameters including:
                - video_path: Path to source video file
                - session_id: Session identifier for output naming
                
        Returns:
            Path to created visualization video file, or None if failed
        """
        try:
            # Extract parameters
            video_path = kwargs.get('video_path')
            if not video_path:
                self.logger.error("Trajectory visualization requires video_path")
                return None
                
            session_id = kwargs.get('session_id', 'unknown_session')
            
            # Get visualization settings with defaults
            viz_config = {
                'marker_radius': config.get('marker_radius', 5),
                'marker_color': config.get('marker_color', [0, 255, 0]),  # Green
                'marker_thickness': config.get('marker_thickness', -1),  # Filled
                'trail_length': config.get('trail_length', 30),  # Frames
                'trail_color': config.get('trail_color', [0, 255, 255]),  # Yellow
                'trail_thickness': config.get('trail_thickness', 2),
                'show_confidence': config.get('show_confidence', False),
                'confidence_threshold': config.get('confidence_threshold', 0.3),
                'output_fps': config.get('output_fps', None),  # Use source FPS if None
                'output_codec': config.get('output_codec', 'mp4v'),
                'bodypart': config.get('bodypart', 'keypoints')  # Column prefix
            }
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video: {video_path}")
                return None
                
            # Get video properties
            fps = viz_config['output_fps'] or cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Prepare output
            output_filename = f"trajectory_{session_id}_{Path(video_path).stem}.mp4"
            output_file = Path(output_path) / output_filename
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*viz_config['output_codec'])
            writer = cv2.VideoWriter(
                str(output_file),
                fourcc,
                fps,
                (frame_width, frame_height)
            )
            
            # Process frames
            frame_idx = 0
            trajectory_points = []
            
            self.logger.info(f"Creating trajectory visualization for {session_id}")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                    
                # Get coordinates for current frame
                if frame_idx in data.index:
                    x_col = f"{viz_config['bodypart']}_x"
                    y_col = f"{viz_config['bodypart']}_y"
                    
                    if x_col in data.columns and y_col in data.columns:
                        x = data.loc[frame_idx, x_col]
                        y = data.loc[frame_idx, y_col]
                        
                        # Check confidence if available
                        draw_point = True
                        if viz_config['show_confidence']:
                            conf_col = f"{viz_config['bodypart']}_likelihood"
                            if conf_col in data.columns:
                                confidence = data.loc[frame_idx, conf_col]
                                draw_point = confidence >= viz_config['confidence_threshold']
                        
                        if draw_point and not pd.isna(x) and not pd.isna(y):
                            point = (int(x), int(y))
                            trajectory_points.append(point)
                            
                            # Limit trail length
                            if len(trajectory_points) > viz_config['trail_length']:
                                trajectory_points.pop(0)
                            
                            # Draw trail
                            if len(trajectory_points) > 1:
                                for i in range(len(trajectory_points) - 1):
                                    # Fade trail based on age
                                    alpha = i / len(trajectory_points)
                                    color = [int(c * alpha) for c in viz_config['trail_color']]
                                    cv2.line(
                                        frame,
                                        trajectory_points[i],
                                        trajectory_points[i + 1],
                                        color,
                                        viz_config['trail_thickness']
                                    )
                            
                            # Draw current position marker
                            cv2.circle(
                                frame,
                                point,
                                viz_config['marker_radius'],
                                viz_config['marker_color'],
                                viz_config['marker_thickness']
                            )
                
                # Write frame
                writer.write(frame)
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 100 == 0:
                    progress = frame_idx / total_frames * 100
                    self.logger.debug(f"Trajectory visualization progress: {progress:.1f}%")
            
            # Cleanup
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            
            self.logger.info(f"Trajectory visualization saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Trajectory visualization failed: {str(e)}")
            return None
    
    @property
    def supported_formats(self) -> List[str]:
        """List of supported output formats."""
        return ['mp4', 'avi', 'mov']