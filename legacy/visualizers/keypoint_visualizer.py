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
    
    def process_frame(self, frame: np.ndarray, frame_index: int, session) -> np.ndarray:
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
            'bodyparts': self.config.get('bodyparts', ['all']),
            'point_size': self.config.get('point_size', 5),
            'point_thickness': self.config.get('point_thickness', -1),
            'point_colors': self.config.get('point_colors', {}),
            'default_color': self.config.get('default_color', [0, 255, 0]),
            'likelihood_threshold': self.config.get('likelihood_threshold', 0.0),
            'show_labels': self.config.get('show_labels', False),
            'label_font_scale': self.config.get('label_font_scale', 0.5),
            'label_color': self.config.get('label_color', [255, 255, 255])
        }
        
        # Get session data
        try:
            session_data = session.get_integrated_dataframe()
        except Exception as e:
            self.logger.error(f"Could not get session data: {str(e)}")
            return frame
        
        # Extract keypoint column information
        keypoint_info = self._extract_keypoint_columns(session_data)
        if not keypoint_info:
            self.logger.debug("No keypoint data found in session DataFrame")
            return frame
        
        # Check if frame_index is valid
        if frame_index >= len(session_data):
            self.logger.debug(f"Frame index {frame_index} beyond session data length {len(session_data)}")
            return frame
        
        # Filter bodyparts based on config
        available_bodyparts = list(keypoint_info.keys())
        self.logger.debug(f"Available bodyparts: {available_bodyparts}")
        
        target_bodyparts = self._filter_target_bodyparts(
            available_bodyparts, 
            viz_config['bodyparts']
        )
        self.logger.debug(f"Target bodyparts to visualize: {target_bodyparts}")
        
        # Copy frame to avoid modifying original
        output_frame = frame.copy()
        
        try:
            # Get keypoint data for this frame
            frame_data = session_data.iloc[frame_index]
            
            # Draw keypoints for each target bodypart
            for bodypart in target_bodyparts:
                # Get coordinates for this bodypart
                coords = self._get_bodypart_coordinates(
                    frame_data, 
                    bodypart, 
                    keypoint_info
                )
                
                if coords is None:
                    continue
                    
                x, y, likelihood = coords
                
                # Skip if likelihood below threshold
                if likelihood < viz_config['likelihood_threshold']:
                    continue
                
                # Get color for this bodypart
                color = viz_config['point_colors'].get(
                    bodypart, 
                    viz_config['default_color']
                )
                
                # Draw keypoint circle
                cv2.circle(
                    output_frame,
                    (int(x), int(y)),
                    viz_config['point_size'],
                    color,
                    viz_config['point_thickness']
                )
                
                # Draw label if enabled
                if viz_config['show_labels']:
                    label_pos = (int(x) + 10, int(y) - 10)
                    cv2.putText(
                        output_frame,
                        bodypart,
                        label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        viz_config['label_font_scale'],
                        viz_config['label_color'],
                        1
                    )
            
            return output_frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame {frame_index}: {str(e)}")
            return frame  # Return unprocessed frame to continue pipeline
    
    def _filter_target_bodyparts(self, available_bodyparts: List[str], requested_bodyparts) -> List[str]:
        """Filter bodyparts based on configuration.
        
        Args:
            available_bodyparts: List of bodyparts available in data
            requested_bodyparts: 'all', single bodypart name, or list of bodypart names
            
        Returns:
            List of bodyparts to visualize
        """
        # Handle string input
        if isinstance(requested_bodyparts, str):
            if requested_bodyparts.lower() == 'all':
                return available_bodyparts
            else:
                # Single bodypart name
                return [requested_bodyparts] if requested_bodyparts in available_bodyparts else []
        
        # Handle list input
        elif isinstance(requested_bodyparts, list):
            if 'all' in requested_bodyparts:
                return available_bodyparts
            else:
                return [bp for bp in requested_bodyparts if bp in available_bodyparts]
        
        # Default to all if input type is unexpected
        else:
            self.logger.warning(f"Unexpected bodyparts config type: {type(requested_bodyparts)}. Using all bodyparts.")
            return available_bodyparts
    
    def _get_bodypart_coordinates(self, frame_data: pd.Series, bodypart: str, keypoint_info: Dict[str, Dict[str, str]]) -> Optional[tuple]:
        """Get coordinates for a bodypart from frame data.
        
        Args:
            frame_data: Data for current frame
            bodypart: Name of bodypart
            keypoint_info: Mapping of bodyparts to column names
            
        Returns:
            Tuple of (x, y, likelihood) or None if not available
        """
        if bodypart not in keypoint_info:
            return None
            
        cols = keypoint_info[bodypart]
        
        try:
            x = frame_data[cols['x']]
            y = frame_data[cols['y']]
            likelihood = frame_data[cols['likelihood']] if 'likelihood' in cols and cols['likelihood'] else 1.0
            
            # Check for NaN values
            if pd.isna(x) or pd.isna(y):
                return None
                
            return (float(x), float(y), float(likelihood))
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.debug(f"Could not get coordinates for {bodypart}: {str(e)}")
            return None
    
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
        from DeepLabCut and flat column formats. Excludes non-keypoint columns.
        
        Args:
            df: DataFrame with keypoint data
            
        Returns:
            Dict mapping bodypart names to column names for x, y, likelihood
        """
        keypoint_info = {}
        
        # Define columns to exclude (these are not bodypart keypoints)
        excluded_columns = {
            'map', 'tile', 'graph', 'reward', 'calibration', 'timestamp', 
            'frame', 'session', 'index', 'level', 'metric', 'analysis'
        }
        
        # Check for multi-level columns (DeepLabCut format)
        if isinstance(df.columns, pd.MultiIndex):
            if df.columns.nlevels >= 3:
                # Format: (scorer, bodypart, coordinate)
                for scorer, bodypart, coord in df.columns:
                    # Skip if bodypart contains any excluded keywords
                    if any(excluded in bodypart.lower() for excluded in excluded_columns):
                        continue
                        
                    if bodypart not in keypoint_info:
                        keypoint_info[bodypart] = {}
                    if coord in ['x', 'y', 'likelihood']:
                        keypoint_info[bodypart][coord] = (scorer, bodypart, coord)
            elif df.columns.nlevels == 2:
                # Format: (bodypart, coordinate)
                for bodypart, coord in df.columns:
                    # Skip if bodypart contains any excluded keywords
                    if any(excluded in bodypart.lower() for excluded in excluded_columns):
                        continue
                        
                    if bodypart not in keypoint_info:
                        keypoint_info[bodypart] = {}
                    if coord in ['x', 'y', 'likelihood']:
                        keypoint_info[bodypart][coord] = (bodypart, coord)
        else:
            # Flat columns - look for pattern: bodypart_x, bodypart_y, bodypart_likelihood
            if 'x' in df.columns and 'y' in df.columns:
                # Simple format - single bodypart (only if no excluded keywords)
                if not any(excluded in 'keypoint' for excluded in excluded_columns):
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
                            
                            # Skip if bodypart contains any excluded keywords
                            if any(excluded in bodypart.lower() for excluded in excluded_columns):
                                continue
                            
                            if bodypart not in keypoint_info:
                                keypoint_info[bodypart] = {}
                            coord = suffix[1:]  # Remove underscore  
                            keypoint_info[bodypart][coord] = col
        
        # Validate that we have complete info for each bodypart
        valid_keypoints = {}
        for bodypart, cols in keypoint_info.items():
            if 'x' in cols and 'y' in cols:
                valid_keypoints[bodypart] = cols
        
        self.logger.debug(f"Extracted keypoint bodyparts: {list(valid_keypoints.keys())}")
        return valid_keypoints