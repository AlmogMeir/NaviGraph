"""Map overlay visualizer for NaviGraph.

Overlays maze map on video frames with current position and visited tiles.
"""

import numpy as np
import cv2
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Set
from pathlib import Path

from ..core.registry import register_visualizer


@register_visualizer("map_overlay")
def visualize_map_overlay(frame: np.ndarray, frame_data: pd.Series, shared_resources: Dict[str, Any], **config) -> np.ndarray:
    """Overlay map image on video frame with position indicators.
    
    Args:
        frame: Input video frame (H, W, 3)
        frame_data: DataFrame row for current frame
        shared_resources: Session shared resources containing 'map_image'
        **config: Visualization configuration
        
    Config:
        position: 'bottom_right', 'bottom_left', 'top_right', 'top_left', 'side_by_side'
        size: Scale factor for overlay (0.1 to 1.0) when not side_by_side
        opacity: Map transparency (0.0 to 1.0, default: 0.7)
        highlight_current: Show current tile position (default: True)
        show_visited: Show trail of visited tiles (default: True)
        current_tile_color: Color for current tile [B,G,R] (default: [0, 0, 255] red)
        visited_tile_color: Color for visited tiles [B,G,R] (default: [0, 255, 0] green)  
        visited_opacity: Opacity for visited tiles (default: 0.3)
        show_tile_ids: Show tile ID numbers (default: False)
        tile_id_font_scale: Font size for tile IDs (default: 0.5)
        tile_id_color: Color for tile ID text [B,G,R] (default: [255, 255, 255] white)
        
    Returns:
        Frame with map overlay applied
    """
    if frame_data.empty:
        return frame
        
    # Get map image from shared resources
    map_image = shared_resources.get('map_image')
    if map_image is None:
        return frame
        
    # Configuration
    position = config.get('position', 'bottom_right')
    size = config.get('size', 0.3)
    opacity = config.get('opacity', 0.7)
    highlight_current = config.get('highlight_current', True)
    show_visited = config.get('show_visited', True)
    current_tile_color = tuple(config.get('current_tile_color', [0, 0, 255]))  # Red
    visited_tile_color = tuple(config.get('visited_tile_color', [0, 255, 0]))  # Green
    visited_opacity = config.get('visited_opacity', 0.3)
    show_tile_ids = config.get('show_tile_ids', False)
    tile_id_font_scale = config.get('tile_id_font_scale', 0.5)
    tile_id_color = tuple(config.get('tile_id_color', [255, 255, 255]))  # White
    
    # Get current position data
    current_tile_id = frame_data.get('tile_id')
    pixel_x = frame_data.get('pixel_x')
    pixel_y = frame_data.get('pixel_y')
    
    # Make a copy of the map image for drawing
    map_copy = map_image.copy()
    
    # Get map metadata for tile drawing
    map_metadata = shared_resources.get('map_metadata', {})
    segment_length = map_metadata.get('segment_length', 86)
    origin = map_metadata.get('origin', (47, 40))
    grid_size = map_metadata.get('grid_size', (17, 17))
    
    # Draw visited tiles if enabled (track visited tiles in config state)
    if show_visited:
        visited_tiles = config.get('_visited_tiles', set())
        if not isinstance(visited_tiles, set):
            visited_tiles = set()
            
        # Add current tile to visited
        if pd.notna(current_tile_id):
            visited_tiles.add(int(current_tile_id))
            config['_visited_tiles'] = visited_tiles  # Store back in config
            
        # Draw all visited tiles
        for tile_id in visited_tiles:
            if tile_id != current_tile_id:  # Don't draw current tile here
                tile_rect = _get_tile_rectangle(tile_id, segment_length, origin, grid_size)
                if tile_rect:
                    overlay = map_copy.copy()
                    cv2.rectangle(overlay, tile_rect[0], tile_rect[1], visited_tile_color, -1)
                    cv2.addWeighted(map_copy, 1 - visited_opacity, overlay, visited_opacity, 0, map_copy)
    
    # Draw current tile if available
    if highlight_current and pd.notna(current_tile_id):
        tile_rect = _get_tile_rectangle(int(current_tile_id), segment_length, origin, grid_size)
        if tile_rect:
            cv2.rectangle(map_copy, tile_rect[0], tile_rect[1], current_tile_color, 3)
            
            # Draw tile ID if requested
            if show_tile_ids:
                text_pos = (tile_rect[0][0] + 5, tile_rect[0][1] + 20)
                cv2.putText(map_copy, str(int(current_tile_id)), text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, tile_id_font_scale, tile_id_color, 1, cv2.LINE_AA)
    
    # Draw current pixel position if available
    if pd.notna(pixel_x) and pd.notna(pixel_y):
        pixel_pos = (int(pixel_x), int(pixel_y))
        cv2.circle(map_copy, pixel_pos, 3, current_tile_color, -1)
    
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


def _get_tile_rectangle(tile_id: int, segment_length: int, origin: Tuple[int, int], grid_size: Tuple[int, int]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Get rectangle coordinates for a tile ID.
    
    Args:
        tile_id: Tile identifier
        segment_length: Pixels per segment
        origin: Top-left corner of grid
        grid_size: Grid dimensions (width, height)
        
    Returns:
        ((x1, y1), (x2, y2)) rectangle coordinates or None if invalid
    """
    grid_w, grid_h = grid_size
    
    # Convert tile_id to grid coordinates
    # Assuming tile_id maps to row-major order
    row = tile_id // grid_w
    col = tile_id % grid_w
    
    if row >= grid_h or col >= grid_w or row < 0 or col < 0:
        return None
        
    # Calculate pixel coordinates
    x1 = origin[0] + col * segment_length
    y1 = origin[1] + row * segment_length
    x2 = x1 + segment_length
    y2 = y1 + segment_length
    
    return ((x1, y1), (x2, y2))