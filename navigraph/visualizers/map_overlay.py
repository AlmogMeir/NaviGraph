"""Map overlay visualizer for NaviGraph.

General-purpose visualizer that displays bodypart positions and trajectories on a map image.
"""

import numpy as np
import cv2
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

from ..core.registry import register_visualizer


@register_visualizer("map_overlay")
def visualize_map_overlay(frame: np.ndarray, frame_data: pd.Series, shared_resources: Dict[str, Any], **config) -> np.ndarray:
    """Overlay map image on video frame with bodypart positions and trajectories.
    
    Args:
        frame: Input video frame (H, W, 3)
        frame_data: DataFrame row for current frame
        shared_resources: Session shared resources containing 'map_image'
        **config: Visualization configuration
        
    Config:
        # Map display
        position: 'bottom_right', 'bottom_left', 'top_right', 'top_left', 'side_by_side'
        size: Scale factor for overlay (0.1 to 1.0) when not side_by_side
        opacity: Map transparency (0.0 to 1.0, default: 0.8)
        
        # Bodyparts to show on map  
        bodyparts: List of bodypart names or 'all' (default: ['Nose'])
        colors: Dict mapping bodypart names to [B,G,R] colors
        default_color: Default color for bodyparts not in colors dict [B,G,R]
        radius: Circle radius for bodypart positions (default: 5)
        thickness: Circle thickness, -1 for filled (default: -1)
        likelihood_threshold: Skip bodyparts below this likelihood (default: None for no filtering)
        
        # Trajectory settings
        show_trajectory: Whether to show movement trails (default: True)
        trail_length: Maximum number of trail points per bodypart (default: 100)
        fade_trail: Whether to fade older trail points (default: True)  
        line_thickness: Thickness for trajectory lines (default: 2)
        trail_color: Color for trajectory lines [B,G,R] (default: same as bodypart)
        fade_to_color: Color to fade trail to [B,G,R] (default: [100, 100, 100])
        
    Returns:
        Frame with map overlay applied
    """
    if frame_data.empty:
        return frame
        
    # Get map image from shared resources
    map_image = shared_resources.get('map_image')
    if map_image is None:
        return frame
        
    # Configuration - Basic display
    position = config.get('position', 'bottom_right')
    size = config.get('size', 0.25)
    opacity = config.get('opacity', 0.8)
    
    # Configuration - Bodyparts
    bodyparts = config.get('bodyparts', ['Nose'])
    colors = config.get('colors', {})
    default_color = config.get('default_color', [255, 255, 255])  # White
    radius = config.get('radius', 5)
    thickness = config.get('thickness', -1)
    likelihood_threshold = config.get('likelihood_threshold', None)
    
    # Configuration - Trajectory
    show_trajectory = config.get('show_trajectory', True)
    trail_length = config.get('trail_length', 100)
    fade_trail = config.get('fade_trail', True)
    line_thickness = config.get('line_thickness', 2)
    trail_color = config.get('trail_color', None)  # Use bodypart color if None
    fade_to_color = tuple(config.get('fade_to_color', [100, 100, 100]))
    
    # Make a copy of the map image for drawing
    map_copy = map_image.copy()
    
    # Find available bodyparts with map coordinates
    available_bodyparts = []
    for col in frame_data.index:
        if col.endswith('_map_x'):
            bodypart = col[:-6]  # Remove '_map_x' suffix
            if f"{bodypart}_map_y" in frame_data.index:
                available_bodyparts.append(bodypart)
    
    # Filter bodyparts based on config
    if bodyparts == 'all' or bodyparts is None:
        target_bodyparts = available_bodyparts
    elif isinstance(bodyparts, list):
        target_bodyparts = [bp for bp in bodyparts if bp in available_bodyparts]
    else:
        target_bodyparts = [bodyparts] if bodyparts in available_bodyparts else []
    
    # Initialize trail storage if not exists
    if '_trail_data' not in config:
        config['_trail_data'] = {}
    
    trail_data = config['_trail_data']
    
    # Process each bodypart
    for bodypart in target_bodyparts:
        x_col = f"{bodypart}_map_x"
        y_col = f"{bodypart}_map_y"
        likelihood_col = f"{bodypart}_likelihood"
        
        # Get coordinates and likelihood
        x = frame_data.get(x_col)
        y = frame_data.get(y_col)
        likelihood = frame_data.get(likelihood_col)
        
        # Skip if missing data
        if pd.isna(x) or pd.isna(y):
            continue
            
        # Check likelihood threshold if specified
        if likelihood_threshold is not None and pd.notna(likelihood):
            if likelihood < likelihood_threshold:
                continue
        
        # Check if coordinates are within map bounds
        map_h, map_w = map_copy.shape[:2]
        is_in_bounds = (0 <= x < map_w and 0 <= y < map_h)
        
        if not is_in_bounds:
            # Skip out-of-bounds points silently
            continue
            
        # Convert to int coordinates
        current_pos = (int(x), int(y))
        
        # Get color for this bodypart
        bodypart_color = colors.get(bodypart, default_color)
        if isinstance(bodypart_color, list) and len(bodypart_color) == 3:
            bodypart_color = tuple(bodypart_color)
        
        # Handle trajectory - add point to trail first, then draw
        if show_trajectory:
            # Initialize trail for this bodypart if not exists
            if bodypart not in trail_data:
                trail_data[bodypart] = deque(maxlen=trail_length)
            
            bodypart_trail = trail_data[bodypart]
            
            # Add current position to trail
            bodypart_trail.append(current_pos)
            
            # Debug: Log trail info
            if len(bodypart_trail) > 1:
                print(f"DEBUG: {bodypart} trail has {len(bodypart_trail)} points, drawing with thickness {line_thickness}")
            
            # Draw trail if we have multiple points
            if len(bodypart_trail) > 1:
                trail_points = list(bodypart_trail)
                
                # Use bodypart color for trail if trail_color not specified
                current_trail_color = bodypart_color if trail_color is None else tuple(trail_color)
                
                if fade_trail:
                    # Draw trail with fading effect
                    for i in range(len(trail_points) - 1):
                        # Calculate fade factor (newer points are more opaque)
                        fade_factor = (i + 1) / len(trail_points)
                        
                        # Interpolate color
                        color = _interpolate_color(fade_to_color, current_trail_color, fade_factor)
                        
                        # Draw line segment
                        cv2.line(map_copy, trail_points[i], trail_points[i + 1], color, line_thickness)
                        print(f"DEBUG: Drawing faded line segment {i} with color {color} and thickness {line_thickness}")
                else:
                    # Draw trail with uniform color
                    if len(trail_points) >= 2:
                        # Draw as connected lines instead of polylines for debugging
                        for i in range(len(trail_points) - 1):
                            cv2.line(map_copy, trail_points[i], trail_points[i + 1], current_trail_color, line_thickness)
                        print(f"DEBUG: Drawing uniform trail with {len(trail_points)} points, color {current_trail_color}, thickness {line_thickness}")
        
        # Draw current position
        cv2.circle(map_copy, current_pos, radius, bodypart_color, thickness)
    
    # Apply overlay based on position
    if position == 'side_by_side':
        # Resize map to match frame height
        frame_h, frame_w = frame.shape[:2]
        map_h, map_w = map_copy.shape[:2]
        
        # Calculate new dimensions maintaining aspect ratio
        aspect_ratio = map_w / map_h
        new_height = frame_h
        new_width = int(frame_h * aspect_ratio)
        
        resized_map = cv2.resize(map_copy, (new_width, new_height))
        
        # Create combined frame
        combined_frame = np.zeros((frame_h, frame_w + new_width, 3), dtype=np.uint8)
        combined_frame[:, :frame_w] = frame
        combined_frame[:, frame_w:frame_w + new_width] = resized_map
        
        return combined_frame
        
    else:
        # Overlay on frame at specified position
        frame_h, frame_w = frame.shape[:2]
        map_h, map_w = map_copy.shape[:2]
        
        # Resize map based on size factor
        overlay_w = int(frame_w * size)
        overlay_h = int((overlay_w / map_w) * map_h)
        
        # Ensure overlay fits in frame
        if overlay_h > frame_h * 0.8:
            overlay_h = int(frame_h * 0.8)
            overlay_w = int((overlay_h / map_h) * map_w)
            
        resized_map = cv2.resize(map_copy, (overlay_w, overlay_h))
        
        # Calculate position
        positions = {
            'bottom_right': (frame_w - overlay_w - 10, frame_h - overlay_h - 10),
            'bottom_left': (10, frame_h - overlay_h - 10),
            'top_right': (frame_w - overlay_w - 10, 10),
            'top_left': (10, 10)
        }
        
        x, y = positions.get(position, positions['bottom_right'])
        
        # Ensure overlay doesn't go out of bounds
        x = max(0, min(x, frame_w - overlay_w))
        y = max(0, min(y, frame_h - overlay_h))
        
        # Apply overlay with opacity
        overlay_region = frame[y:y + overlay_h, x:x + overlay_w]
        blended = cv2.addWeighted(overlay_region, 1 - opacity, resized_map, opacity, 0)
        frame[y:y + overlay_h, x:x + overlay_w] = blended
        
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


