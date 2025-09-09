"""Text display visualizer for NaviGraph.

Displays DataFrame column values as text overlays on video frames.
"""

import numpy as np
import cv2
import pandas as pd
from typing import Dict, Any, List, Tuple, Union

from ..core.registry import register_visualizer


@register_visualizer("text_display")
def visualize_text_display(frame: np.ndarray, frame_data: pd.Series, shared_resources: Dict[str, Any], **config) -> np.ndarray:
    """Display DataFrame column values as text on video frame.
    
    Args:
        frame: Input video frame (H, W, 3)
        frame_data: DataFrame row for current frame
        shared_resources: Session shared resources (unused)
        **config: Visualization configuration
        
    Config:
        columns: List of column names to display or single column name
        position: Text position - 'top_left', 'top_right', 'bottom_left', 'bottom_right', 
                 'center', or tuple (x, y) for custom position (default: 'top_left')
        font_face: OpenCV font face (default: cv2.FONT_HERSHEY_SIMPLEX)
        font_scale: Font size multiplier (default: 0.8)
        color: Text color [B,G,R] (default: [255, 255, 255] white)
        thickness: Text line thickness (default: 2)
        
        # Formatting
        decimal_places: Number of decimal places for float values (default: 2)
        prefix: Text prefix before values (default: "")
        suffix: Text suffix after values (default: "")
        separator: Separator between multiple columns (default: " | ")
        
        # Background
        background: Whether to draw background rectangle (default: True)
        background_color: Background color [B,G,R] (default: [0, 0, 0] black)
        background_opacity: Background transparency (0.0 to 1.0, default: 0.6)
        padding: Padding around text in pixels (default: 10)
        
        # Layout for multiple columns
        layout: 'horizontal' or 'vertical' for multiple columns (default: 'horizontal')
        line_spacing: Extra spacing between lines in vertical layout (default: 5)
        
    Returns:
        Frame with text overlay applied
    """
    if frame_data.empty:
        return frame
        
    # Configuration
    columns = config.get('columns', [])
    if isinstance(columns, str):
        columns = [columns]
    if not columns:
        return frame
        
    position = config.get('position', 'top_left')
    font_face = config.get('font_face', cv2.FONT_HERSHEY_SIMPLEX)
    font_scale = config.get('font_scale', 0.8)
    color = tuple(config.get('color', [255, 255, 255]))  # White
    thickness = config.get('thickness', 2)
    
    # Formatting
    decimal_places = config.get('decimal_places', 2)
    prefix = config.get('prefix', "")
    suffix = config.get('suffix', "")
    separator = config.get('separator', " | ")
    
    # Background
    background = config.get('background', True)
    background_color = tuple(config.get('background_color', [0, 0, 0]))  # Black
    background_opacity = config.get('background_opacity', 0.6)
    padding = config.get('padding', 10)
    
    # Layout
    layout = config.get('layout', 'horizontal')
    line_spacing = config.get('line_spacing', 5)
    
    # Collect text values
    text_values = []
    for col in columns:
        if col in frame_data.index:
            value = frame_data[col]
            if pd.isna(value):
                text_values.append("N/A")
            elif isinstance(value, (int, np.integer)):
                text_values.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                if decimal_places == 0:
                    text_values.append(str(int(value)))
                else:
                    text_values.append(f"{value:.{decimal_places}f}")
            else:
                text_values.append(str(value))
        else:
            text_values.append(f"{col}: N/A")
    
    if not text_values:
        return frame
        
    # Format text strings
    if layout == 'horizontal':
        display_text = prefix + separator.join(text_values) + suffix
        text_lines = [display_text]
    else:  # vertical
        text_lines = [prefix + value + suffix for value in text_values]
    
    # Calculate text dimensions
    line_heights = []
    line_widths = []
    
    for line in text_lines:
        (text_width, text_height), baseline = cv2.getTextSize(line, font_face, font_scale, thickness)
        line_widths.append(text_width)
        line_heights.append(text_height)
    
    max_width = max(line_widths) if line_widths else 0
    total_height = sum(line_heights) + line_spacing * (len(text_lines) - 1) if line_heights else 0
    
    # Calculate position
    frame_h, frame_w = frame.shape[:2]
    
    if isinstance(position, tuple) and len(position) == 2:
        x, y = position
    else:
        positions = {
            'top_left': (padding, line_heights[0] + padding if line_heights else padding),
            'top_right': (frame_w - max_width - padding, line_heights[0] + padding if line_heights else padding),
            'bottom_left': (padding, frame_h - total_height - padding + line_heights[0] if line_heights else frame_h - padding),
            'bottom_right': (frame_w - max_width - padding, frame_h - total_height - padding + line_heights[0] if line_heights else frame_h - padding),
            'center': (frame_w // 2 - max_width // 2, frame_h // 2 - total_height // 2 + line_heights[0] if line_heights else frame_h // 2)
        }
        x, y = positions.get(position, positions['top_left'])
    
    # Ensure text stays within frame bounds
    x = max(0, min(x, frame_w - max_width))
    y = max(line_heights[0] if line_heights else 0, min(y, frame_h))
    
    # Draw background if enabled
    if background and text_lines:
        bg_x1 = max(0, x - padding)
        bg_y1 = max(0, y - line_heights[0] - padding if line_heights else y - padding)
        bg_x2 = min(frame_w, x + max_width + padding)
        bg_y2 = min(frame_h, y + total_height - line_heights[0] + padding if line_heights else y + padding)
        
        # Create background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), background_color, -1)
        
        # Blend with original frame
        cv2.addWeighted(frame, 1 - background_opacity, overlay, background_opacity, 0, frame)
    
    # Draw text lines
    current_y = y
    for i, line in enumerate(text_lines):
        if i < len(line_heights):
            cv2.putText(frame, line, (x, current_y), font_face, font_scale, color, thickness, cv2.LINE_AA)
            current_y += line_heights[i] + line_spacing
    
    return frame