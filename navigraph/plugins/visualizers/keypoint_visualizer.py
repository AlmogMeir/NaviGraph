"""Keypoint visualizer plugin for NaviGraph.

This plugin visualizes tracking keypoints (bodyparts) on video frames,
drawing circles at detected positions with configurable appearance.
"""

from typing import Dict, Any, Optional, List
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from ...core.interfaces import IVisualizer, Logger
from ...core.base_plugin import BasePlugin
from ...core.registry import register_visualizer_plugin


@register_visualizer_plugin("keypoint_visualizer")
class KeypointVisualizer(BasePlugin, IVisualizer):
    """Visualizes tracked keypoints on video frames.
    
    Features:
    - Draw circles at detected keypoint locations
    - Support multiple bodyparts with different colors
    - Configurable point size and appearance
    - Skip low-confidence detections
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance = None):
        """Factory method to create keypoint visualizer from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def generate_visualization(
        self,
        session_data: pd.DataFrame,
        config: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> Optional[str]:
        """Create keypoint visualization on video frames.
        
        Args:
            session_data: DataFrame with keypoint coordinates (x, y, likelihood columns)
            config: Visualization-specific configuration
            output_path: Directory to save visualization outputs
            **kwargs: Additional parameters including:
                - video_path: Path to source video file
                - session_id: Session identifier for output naming
                
        Returns:
            Path to created visualization video file, or None if failed
        """
        try:
            # Get file requirements
            session_path = kwargs.get('session_path')
            if not session_path:
                self.logger.error("Keypoint visualization requires session_path")
                return None
            
            files = self.get_file_requirements(session_path)
            video_path = files.get('video_file')
            if not video_path:
                self.logger.error("Keypoint visualization could not find required video file")
                return None
                
            session_id = kwargs.get('session_id', 'unknown_session')
            
            # Get visualization settings with defaults
            viz_config = {
                'bodyparts': config.get('bodyparts', ['all']),  # List of bodyparts or 'all'
                'point_size': config.get('point_size', 5),
                'point_thickness': config.get('point_thickness', -1),  # -1 for filled circle
                'point_colors': config.get('point_colors', {}),  # Dict mapping bodypart to color
                'default_color': config.get('default_color', [0, 255, 0]),  # Green default
                'likelihood_threshold': config.get('likelihood_threshold', 0.0),  # Min confidence
                'show_labels': config.get('show_labels', False),  # Draw bodypart names
                'label_font_scale': config.get('label_font_scale', 0.5),
                'label_color': config.get('label_color', [255, 255, 255]),  # White
                'output_fps': config.get('output_fps', None),
                'output_codec': config.get('output_codec', 'mp4v')
            }
            
            # Identify keypoint columns in data
            keypoint_info = self._extract_keypoint_columns(session_data)
            if not keypoint_info:
                self.logger.error("No keypoint data found in session DataFrame")
                return None
            
            # Filter bodyparts based on config
            if 'all' not in viz_config['bodyparts']:
                keypoint_info = {
                    bp: info for bp, info in keypoint_info.items() 
                    if bp in viz_config['bodyparts']
                }
            
            if not keypoint_info:
                self.logger.warning("No matching bodyparts found in data")
                return None
            
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
            output_filename = f"keypoints_{session_id}_{Path(video_path).stem}.mp4"
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
            self.logger.info(f"Creating keypoint visualization for {session_id}")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                
                # Draw keypoints for this frame
                if frame_idx in session_data.index:
                    for bodypart, cols in keypoint_info.items():
                        # Get coordinates and likelihood
                        x = session_data.loc[frame_idx, cols['x']]
                        y = session_data.loc[frame_idx, cols['y']]
                        likelihood = session_data.loc[frame_idx, cols['likelihood']] if cols['likelihood'] else 1.0
                        
                        # Skip if low confidence or NaN
                        if pd.isna(x) or pd.isna(y) or likelihood < viz_config['likelihood_threshold']:
                            continue
                        
                        # Get color for this bodypart
                        color = viz_config['point_colors'].get(bodypart, viz_config['default_color'])
                        
                        # Draw circle
                        cv2.circle(
                            frame,
                            (int(x), int(y)),
                            viz_config['point_size'],
                            color,
                            viz_config['point_thickness']
                        )
                        
                        # Draw label if enabled
                        if viz_config['show_labels']:
                            label_pos = (int(x) + 10, int(y) - 10)
                            cv2.putText(
                                frame,
                                bodypart,
                                label_pos,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                viz_config['label_font_scale'],
                                viz_config['label_color'],
                                1
                            )
                
                # Write frame
                writer.write(frame)
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 100 == 0:
                    progress = frame_idx / total_frames * 100
                    self.logger.debug(f"Keypoint visualization progress: {progress:.1f}%")
            
            # Cleanup
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            
            self.logger.info(f"Keypoint visualization saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Keypoint visualization failed: {str(e)}")
            return None
    
    def _extract_keypoint_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """Extract keypoint column information from DataFrame.
        
        Returns dict mapping bodypart names to column names for x, y, likelihood.
        """
        keypoint_info = {}
        
        # Check for multi-level columns (DeepLabCut format)
        if isinstance(df.columns, pd.MultiIndex):
            if df.columns.nlevels >= 3:
                # Format: (scorer, bodypart, coordinate)
                for scorer, bodypart, coord in df.columns:
                    if bodypart not in keypoint_info:
                        keypoint_info[bodypart] = {}
                    if coord in ['x', 'y', 'likelihood']:
                        keypoint_info[bodypart][coord] = (scorer, bodypart, coord)
            elif df.columns.nlevels == 2:
                # Format: (bodypart, coordinate)
                for bodypart, coord in df.columns:
                    if bodypart not in keypoint_info:
                        keypoint_info[bodypart] = {}
                    if coord in ['x', 'y', 'likelihood']:
                        keypoint_info[bodypart][coord] = (bodypart, coord)
        else:
            # Flat columns - look for pattern: bodypart_x, bodypart_y, bodypart_likelihood
            # or simple x, y, likelihood (single bodypart)
            if 'x' in df.columns and 'y' in df.columns:
                # Simple format - single bodypart
                keypoint_info['keypoint'] = {
                    'x': 'x',
                    'y': 'y',
                    'likelihood': 'likelihood' if 'likelihood' in df.columns else None
                }
            else:
                # Look for bodypart patterns
                for col in df.columns:
                    for suffix in ['_x', '_y', '_likelihood']:
                        if col.endswith(suffix):
                            bodypart = col.replace(suffix, '')
                            if bodypart not in keypoint_info:
                                keypoint_info[bodypart] = {}
                            coord = suffix[1:]  # Remove underscore
                            keypoint_info[bodypart][coord] = col
        
        # Validate that we have complete info for each bodypart
        valid_keypoints = {}
        for bodypart, cols in keypoint_info.items():
            if 'x' in cols and 'y' in cols:
                valid_keypoints[bodypart] = cols
        
        return valid_keypoints
    
    @property
    def supported_formats(self) -> List[str]:
        """List of supported output formats."""
        return ['mp4', 'avi', 'mov']