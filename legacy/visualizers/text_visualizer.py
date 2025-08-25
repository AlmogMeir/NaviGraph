"""Text visualizer plugin for NaviGraph.

This plugin displays DataFrame column values as text overlays on video frames.
It's a flexible visualizer that can show any numeric or text data from the session DataFrame.
"""

from typing import Dict, Any, Optional, List, Iterator, Union, Tuple
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from ...core.interfaces import IVisualizer, Logger
from ...core.base_plugin import BasePlugin
from ...core.registry import register_visualizer_plugin


@register_visualizer_plugin("text_visualizer")
class TextVisualizer(BasePlugin, IVisualizer):
    """Visualizes DataFrame column values as text overlays on video frames.
    
    This visualizer can display any column values from the session DataFrame
    as formatted text on video frames. Useful for showing real-time values
    like yaw angles, neural activity, velocity, etc.
    
    Features:
    - Display multiple columns simultaneously
    - Configurable text positioning (corners)
    - Customizable colors and font scaling
    - Automatic value formatting with decimal precision
    - Optional background box for better readability
    - Handle missing/NaN values gracefully
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance = None):
        """Factory method to create text visualizer from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate text visualizer configuration."""
        # Set default values
        if 'columns' not in self.config:
            self.config['columns'] = []
            
        if 'position' not in self.config:
            self.config['position'] = 'top_right'
            
        if 'text_color' not in self.config:
            self.config['text_color'] = [255, 255, 255]  # White
            
        if 'font_scale' not in self.config:
            self.config['font_scale'] = 1.0
            
        if 'decimal_places' not in self.config:
            self.config['decimal_places'] = 2
            
        if 'background_opacity' not in self.config:
            self.config['background_opacity'] = 0.3
            
        if 'line_spacing' not in self.config:
            self.config['line_spacing'] = 30
            
        # Validate position
        valid_positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        if self.config['position'] not in valid_positions:
            self.logger.warning(f"Invalid position '{self.config['position']}', using 'top_right'")
            self.config['position'] = 'top_right'
    
    def process_frame(self, frame: np.ndarray, frame_index: int, session) -> np.ndarray:
        """Process a single frame and add text overlays.
        
        Args:
            frame: Input video frame
            frame_index: Current frame index
            session: Session object with full data access
            
        Returns:
            Frame with text overlays added
        """
        # Get columns to display
        columns_to_display = self.config.get('columns', [])
        if not columns_to_display:
            self.logger.debug("No columns specified for text visualization")
            return frame
        
        # Get session data
        try:
            session_data = session.get_integrated_dataframe()
        except Exception as e:
            self.logger.error(f"Could not get session data: {str(e)}")
            return frame
        
        # Validate columns exist
        available_columns = list(session_data.columns)
        missing_columns = [col for col in columns_to_display if col not in available_columns]
        if missing_columns:
            self.logger.warning(f"Missing columns for text visualization: {missing_columns}")
            columns_to_display = [col for col in columns_to_display if col in available_columns]
        
        if not columns_to_display:
            self.logger.debug("No valid columns found for text visualization")
            return frame
        
        # Check if frame_index is valid
        if frame_index >= len(session_data):
            self.logger.debug(f"Frame index {frame_index} beyond session data length {len(session_data)}")
            return frame
        
        try:
            # Add text overlays for this frame
            return self._add_text_overlays(frame, session_data, columns_to_display, frame_index)
        except Exception as e:
            self.logger.error(f"Failed to process frame {frame_index}: {str(e)}")
            return frame  # Return original frame on error
    
    def _add_text_overlays(
        self, 
        frame: np.ndarray, 
        session_data: pd.DataFrame, 
        columns: List[str], 
        frame_idx: int
    ) -> np.ndarray:
        """Add text overlays to a single frame.
        
        Args:
            frame: Input video frame
            session_data: DataFrame with column data
            columns: List of column names to display
            frame_idx: Current frame index
            
        Returns:
            Frame with text overlays added
        """
        # Create a copy to avoid modifying original
        output_frame = frame.copy()
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Get text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config['font_scale']
        color = tuple(self.config['text_color'])  # Convert to tuple for OpenCV
        thickness = 2
        line_spacing = self.config['line_spacing']
        decimal_places = self.config['decimal_places']
        
        # Calculate starting position based on configuration
        position = self.config['position']
        text_lines = []
        
        # Prepare text lines
        for col in columns:
            try:
                if frame_idx < len(session_data):
                    value = session_data.iloc[frame_idx][col]
                    formatted_value = self._format_value(value, decimal_places)
                    text_lines.append(f"{col}: {formatted_value}")
                else:
                    text_lines.append(f"{col}: N/A")
            except Exception as e:
                text_lines.append(f"{col}: Error")
                self.logger.debug(f"Error getting value for column {col}: {e}")
        
        if not text_lines:
            return output_frame
        
        # Calculate text dimensions for background box
        text_sizes = []
        max_width = 0
        total_height = 0
        
        for line in text_lines:
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            text_sizes.append((text_width, text_height))
            max_width = max(max_width, text_width)
            total_height += text_height + 5  # Add small padding between lines
        
        # Calculate position coordinates
        margin = 10
        if position == 'top_left':
            start_x = margin
            start_y = margin + text_sizes[0][1]
        elif position == 'top_right':
            start_x = frame_width - max_width - margin
            start_y = margin + text_sizes[0][1]
        elif position == 'bottom_left':
            start_x = margin
            start_y = frame_height - total_height - margin
        else:  # bottom_right
            start_x = frame_width - max_width - margin
            start_y = frame_height - total_height - margin
        
        # Draw background box if configured
        background_opacity = self.config.get('background_opacity', 0)
        if background_opacity > 0:
            # Create background rectangle
            bg_padding = 5
            bg_x1 = max(0, start_x - bg_padding)
            bg_y1 = max(0, start_y - text_sizes[0][1] - bg_padding)
            bg_x2 = min(frame_width, start_x + max_width + bg_padding)
            bg_y2 = min(frame_height, start_y + total_height + bg_padding)
            
            # Create overlay for transparency
            overlay = output_frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, background_opacity, output_frame, 1 - background_opacity, 0, output_frame)
        
        # Draw text lines
        current_y = start_y
        for i, line in enumerate(text_lines):
            cv2.putText(
                output_frame, 
                line, 
                (start_x, current_y), 
                font, 
                font_scale, 
                color, 
                thickness
            )
            current_y += text_sizes[i][1] + 5
        
        return output_frame
    
    def _format_value(self, value: Any, decimal_places: int) -> str:
        """Format a value for display.
        
        Args:
            value: Value to format
            decimal_places: Number of decimal places for numeric values
            
        Returns:
            Formatted string representation
        """
        if pd.isna(value):
            return "N/A"
        
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                return f"{value:.{decimal_places}f}"
            else:
                return str(value)
        
        return str(value)
    
    def get_required_columns(self) -> List[str]:
        """Get list of required DataFrame columns.
        
        Returns:
            List of column names that this visualizer needs
        """
        return self.config.get('columns', [])