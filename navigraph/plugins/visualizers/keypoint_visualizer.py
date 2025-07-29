"""Keypoint visualizer plugin for NaviGraph.

This plugin visualizes tracking keypoints (bodyparts) on video frames,
drawing circles at detected positions with configurable appearance.
"""

from typing import Dict, Any, Optional, List, Iterator, Union
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
    
    This visualizer processes video frames and draws keypoint locations
    as circles with configurable colors and sizes. It supports multiple
    bodyparts and filters low-confidence detections.
    
    Features:
    - Draw circles at detected keypoint locations
    - Support multiple bodyparts with different colors
    - Configurable point size and appearance
    - Skip low-confidence detections
    - Optional bodypart labels
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance = None):
        """Factory method to create keypoint visualizer from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def process(
        self,
        session_data: pd.DataFrame,
        config: Dict[str, Any],
        input_data: Optional[Union[Iterator[np.ndarray], str]] = None,
        **kwargs
    ) -> Iterator[np.ndarray]:
        """Process video frames and add keypoint visualizations.
        
        Args:
            session_data: DataFrame with keypoint coordinates (x, y, likelihood columns)
            config: Visualization configuration including:
                - bodyparts: List of bodyparts to visualize or ['all']
                - point_size: Circle radius in pixels
                - point_thickness: Line thickness (-1 for filled)
                - point_colors: Dict mapping bodypart names to [B,G,R] colors
                - default_color: Default [B,G,R] color for unlisted bodyparts
                - likelihood_threshold: Minimum confidence to display keypoint
                - show_labels: Whether to draw bodypart names
                - label_font_scale: Text size for labels
                - label_color: [B,G,R] color for label text
            input_data: Video source - either frame iterator or video file path
            **kwargs: Additional parameters including:
                - session_path: Path to session directory for file discovery
                - session_id: Session identifier
                
        Yields:
            Processed frames with keypoint visualizations drawn
        """
        # Get visualization settings with defaults
        viz_config = {
            'bodyparts': config.get('bodyparts', ['all']),
            'point_size': config.get('point_size', 5),
            'point_thickness': config.get('point_thickness', -1),
            'point_colors': config.get('point_colors', {}),
            'default_color': config.get('default_color', [0, 255, 0]),
            'likelihood_threshold': config.get('likelihood_threshold', 0.0),
            'show_labels': config.get('show_labels', False),
            'label_font_scale': config.get('label_font_scale', 0.5),
            'label_color': config.get('label_color', [255, 255, 255])
        }
        
        # Extract keypoint column information
        keypoint_info = self._extract_keypoint_columns(session_data)
        if not keypoint_info:
            self.logger.error("No keypoint data found in session DataFrame")
            return
        
        # Filter bodyparts based on config
        if 'all' not in viz_config['bodyparts']:
            keypoint_info = {
                bp: info for bp, info in keypoint_info.items() 
                if bp in viz_config['bodyparts']
            }
        
        if not keypoint_info:
            self.logger.warning("No matching bodyparts found in data")
            return
        
        # Get frame source
        frame_source = self._get_frame_source(input_data, kwargs.get('session_path'))
        if frame_source is None:
            return
        
        # Process frames
        frame_idx = 0
        for frame in frame_source:
            # Draw keypoints for this frame
            if frame_idx in session_data.index:
                frame = self._draw_keypoints_on_frame(
                    frame, session_data, frame_idx, keypoint_info, viz_config
                )
            
            yield frame
            frame_idx += 1
    
    def _get_frame_source(
        self, 
        input_data: Optional[Union[Iterator[np.ndarray], str]], 
        session_path: Optional[str]
    ) -> Optional[Iterator[np.ndarray]]:
        """Get frame source from input data or file discovery.
        
        Args:
            input_data: Either frame iterator or video file path
            session_path: Path for file discovery if input_data is None
            
        Returns:
            Iterator yielding frames, or None if no source found
        """
        # If we have a frame iterator, use it directly
        if input_data is not None and not isinstance(input_data, str):
            return input_data
        
        # Otherwise, we need to open a video file
        video_path = None
        
        if isinstance(input_data, str):
            # Input is a video path
            video_path = input_data
        elif session_path:
            # Try file discovery
            files = self.get_file_requirements(session_path)
            video_path = files.get('video_file')
        
        if not video_path:
            self.logger.error("No video source available")
            return None
        
        # Open video and create frame generator
        return self._video_frame_generator(video_path)
    
    def _video_frame_generator(self, video_path: str) -> Iterator[np.ndarray]:
        """Generator that yields frames from a video file.
        
        Args:
            video_path: Path to video file
            
        Yields:
            Video frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                yield frame
        finally:
            cap.release()
    
    def _draw_keypoints_on_frame(
        self,
        frame: np.ndarray,
        session_data: pd.DataFrame,
        frame_idx: int,
        keypoint_info: Dict[str, Dict[str, str]],
        viz_config: Dict[str, Any]
    ) -> np.ndarray:
        """Draw keypoints on a single frame.
        
        Args:
            frame: Video frame to draw on
            session_data: DataFrame with keypoint data
            frame_idx: Current frame index
            keypoint_info: Mapping of bodyparts to column names
            viz_config: Visualization settings
            
        Returns:
            Frame with keypoints drawn
        """
        for bodypart, cols in keypoint_info.items():
            # Get coordinates and likelihood
            x = session_data.loc[frame_idx, cols['x']]
            y = session_data.loc[frame_idx, cols['y']]
            likelihood = session_data.loc[frame_idx, cols['likelihood']] if 'likelihood' in cols and cols['likelihood'] else 1.0
            
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
        
        return frame
    
    def _extract_keypoint_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """Extract keypoint column information from DataFrame.
        
        Handles various DataFrame formats including multi-level columns
        from DeepLabCut and flat column formats.
        
        Args:
            df: DataFrame with keypoint data
            
        Returns:
            Dict mapping bodypart names to column names for x, y, likelihood
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