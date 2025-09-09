"""Trajectory visualizer for NaviGraph.

Draws movement trajectory trail on video frames.
"""

import numpy as np
import cv2
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

from ..core.registry import register_visualizer


@register_visualizer("trajectory")
def visualize_trajectory(frame: np.ndarray, frame_data: pd.Series, shared_resources: Dict[str, Any], **config) -> np.ndarray:
    """Draw trajectory trail on video frame.
    
    Args:
        frame: Input video frame (H, W, 3)
        frame_data: DataFrame row for current frame
        shared_resources: Session shared resources (unused)
        **config: Visualization configuration
        
    Config:
        # Position source
        bodypart: Bodypart name to track (default: 'Nose')
        coordinate_type: 'pixel' or 'map' coordinates (default: 'pixel')
        
        # Trail appearance  
        trail_length: Maximum number of points in trail (default: 50)
        line_thickness: Trail line thickness (default: 2)
        fade_trail: Whether to fade older points (default: True)
        
        # Colors
        current_color: Color for current position [B,G,R] (default: [0, 0, 255] red)
        trail_color: Color for trail [B,G,R] (default: [255, 255, 0] cyan)
        fade_to_color: Color to fade to [B,G,R] (default: [100, 100, 100] gray)
        
        # Current position marker
        show_current: Show current position marker (default: True)
        current_radius: Radius for current position circle (default: 5)
        current_thickness: Thickness for current position circle (default: -1 for filled)
        
        # Filter
        likelihood_threshold: Skip positions below this likelihood (default: 0.3)
        min_movement: Minimum movement distance to add point (default: 0.0)
        
    Returns:
        Frame with trajectory trail drawn
    """
    if frame_data.empty:
        return frame
        
    # Configuration
    bodypart = config.get('bodypart', 'Nose')
    coordinate_type = config.get('coordinate_type', 'pixel')
    trail_length = config.get('trail_length', 50)
    line_thickness = config.get('line_thickness', 2)
    fade_trail = config.get('fade_trail', True)
    
    # Colors
    current_color = tuple(config.get('current_color', [0, 0, 255]))  # Red
    trail_color = tuple(config.get('trail_color', [255, 255, 0]))    # Cyan
    fade_to_color = tuple(config.get('fade_to_color', [100, 100, 100]))  # Gray
    
    # Current position marker
    show_current = config.get('show_current', True)
    current_radius = config.get('current_radius', 5)
    current_thickness = config.get('current_thickness', -1)
    
    # Filters
    likelihood_threshold = config.get('likelihood_threshold', 0.3)
    min_movement = config.get('min_movement', 0.0)
    
    # Get position data based on coordinate type
    if coordinate_type == 'map':
        x_col = 'map_x'
        y_col = 'map_y'
        likelihood_col = f"{bodypart}_likelihood"  # Still use bodypart likelihood
    else:  # pixel coordinates
        x_col = f"{bodypart}_x"
        y_col = f"{bodypart}_y"
        likelihood_col = f"{bodypart}_likelihood"
    
    # Get current position
    x = frame_data.get(x_col)
    y = frame_data.get(y_col)
    likelihood = frame_data.get(likelihood_col)
    
    # Initialize trail storage in config if not exists
    if '_trail_points' not in config:
        config['_trail_points'] = deque(maxlen=trail_length)
    
    trail_points = config['_trail_points']
    
    # Check if we have valid position data
    valid_position = pd.notna(x) and pd.notna(y)
    
    # Check likelihood if threshold is specified
    valid_likelihood = True
    if likelihood_threshold is not None and pd.notna(likelihood):
        valid_likelihood = likelihood >= likelihood_threshold
    
    if valid_position and valid_likelihood:
        
        current_pos = (int(x), int(y))
        
        # Check minimum movement if trail has points
        add_point = True
        if trail_points and min_movement > 0:
            last_pos = trail_points[-1]
            distance = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
            if distance < min_movement:
                add_point = False
        
        # Add current position to trail
        if add_point:
            trail_points.append(current_pos)
    else:
        current_pos = None
    
    # Draw trail
    if len(trail_points) > 1:
        points_list = list(trail_points)
        
        if fade_trail:
            # Draw trail with fading effect
            for i in range(len(points_list) - 1):
                # Calculate fade factor (newer points are more opaque)
                fade_factor = (i + 1) / len(points_list)
                
                # Interpolate color between fade_to_color and trail_color
                color = _interpolate_color(fade_to_color, trail_color, fade_factor)
                
                # Draw line segment
                cv2.line(frame, points_list[i], points_list[i + 1], color, line_thickness, cv2.LINE_AA)
        else:
            # Draw trail with uniform color
            points_array = np.array(points_list, dtype=np.int32)
            cv2.polylines(frame, [points_array], False, trail_color, line_thickness, cv2.LINE_AA)
    
    # Draw current position marker
    if show_current and current_pos is not None:
        cv2.circle(frame, current_pos, current_radius, current_color, current_thickness)
    
    return frame


def _interpolate_color(color1: Tuple[int, int, int], color2: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    """Interpolate between two colors.
    
    Args:
        color1: Start color [B, G, R]
        color2: End color [B, G, R]
        factor: Interpolation factor (0.0 to 1.0)
        
    Returns:
        Interpolated color [B, G, R]
    """
    factor = max(0.0, min(1.0, factor))  # Clamp to [0, 1]
    
    b = int(color1[0] + (color2[0] - color1[0]) * factor)
    g = int(color1[1] + (color2[1] - color1[1]) * factor)
    r = int(color1[2] + (color2[2] - color1[2]) * factor)
    
    return (b, g, r)