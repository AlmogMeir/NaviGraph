"""Bodypart visualizer for NaviGraph.

Draws bodypart positions from pose tracking data on video frames.
"""

import numpy as np
import cv2
import pandas as pd
from typing import Dict, Any, List, Tuple

from ..core.registry import register_visualizer


@register_visualizer("bodyparts")
def visualize_bodyparts(frame: np.ndarray, frame_data: pd.Series, shared_resources: Dict[str, Any], **config) -> np.ndarray:
    """Draw bodypart positions on video frame.
    
    Args:
        frame: Input video frame (H, W, 3)
        frame_data: DataFrame row for current frame with pose columns
        shared_resources: Session shared resources
        **config: Visualization configuration
        
    Config:
        bodyparts: List of bodypart names to visualize or 'all'
        colors: Dict mapping bodypart names to [B,G,R] colors
        default_color: Default color if bodypart not in colors dict
        radius: Circle radius in pixels (default: 5)
        thickness: Circle thickness, -1 for filled (default: -1)
        likelihood_threshold: Skip bodyparts below this likelihood (default: 0.3)
        show_labels: Whether to show bodypart names (default: False)
        font_scale: Text size for labels (default: 0.5)
        
    Returns:
        Frame with bodypart circles drawn
    """
    if frame_data.empty:
        return frame
    
    # Configuration
    bodyparts = config.get('bodyparts', 'all')
    colors = config.get('colors', {})
    default_color = config.get('default_color', [0, 255, 0])  # Green
    radius = config.get('radius', 5)
    thickness = config.get('thickness', -1)
    likelihood_threshold = config.get('likelihood_threshold', 0.3)
    show_labels = config.get('show_labels', False)
    font_scale = config.get('font_scale', 0.5)
    
    # Find all bodypart columns in frame data
    available_bodyparts = []
    for col in frame_data.index:
        if col.endswith('_x') and not col.startswith('centroid'):
            bodypart = col[:-2]  # Remove '_x' suffix
            if f"{bodypart}_y" in frame_data.index and f"{bodypart}_likelihood" in frame_data.index:
                available_bodyparts.append(bodypart)
    
    # Filter bodyparts based on config
    if bodyparts == 'all' or bodyparts is None:  # null or 'all' means all bodyparts
        target_bodyparts = available_bodyparts
    elif isinstance(bodyparts, list):
        target_bodyparts = [bp for bp in bodyparts if bp in available_bodyparts]
    else:
        target_bodyparts = [bodyparts] if bodyparts in available_bodyparts else []
    
    # Draw each bodypart
    for bodypart in target_bodyparts:
        x_col = f"{bodypart}_x"
        y_col = f"{bodypart}_y"
        likelihood_col = f"{bodypart}_likelihood"
        
        # Get coordinates and likelihood
        x = frame_data.get(x_col)
        y = frame_data.get(y_col) 
        likelihood = frame_data.get(likelihood_col)
        
        # Skip if missing data
        if pd.isna(x) or pd.isna(y):
            continue
        
        # Skip if likelihood check requested and below threshold
        if likelihood_threshold is not None and not pd.isna(likelihood):
            if likelihood < likelihood_threshold:
                continue
            
        # Convert to int coordinates
        center = (int(x), int(y))
        
        # Get color for this bodypart
        color = colors.get(bodypart, default_color)
        if isinstance(color, list) and len(color) == 3:
            color = tuple(color)  # Convert to tuple for OpenCV
        
        # Draw circle
        cv2.circle(frame, center, radius, color, thickness)
        
        # Draw label if requested
        if show_labels:
            text_pos = (center[0] + radius + 2, center[1] - radius - 2)
            cv2.putText(frame, bodypart, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, 1, cv2.LINE_AA)
    
    return frame