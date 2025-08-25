"""Map visualizer plugin for NaviGraph.

This plugin visualizes spatial navigation data by overlaying maze maps on video
frames, highlighting current tile positions and visited areas.
"""

from typing import Dict, Any, Optional, List, Tuple, Iterator, Union, Set
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from ...core.interfaces import IVisualizer, Logger
from ...core.base_plugin import BasePlugin
from ...core.registry import register_visualizer_plugin


@register_visualizer_plugin("map_visualizer")
class MapVisualizer(BasePlugin, IVisualizer):
    """Visualizes maze navigation with map overlays.
    
    This visualizer overlays a maze map on video frames, highlighting the
    current tile position and showing the history of visited tiles. It
    helps visualize spatial navigation patterns during experiments.
    
    Features:
    - Overlays maze map on video frames
    - Highlights current tile position
    - Shows visited tiles history
    - Configurable overlay position and opacity
    - Optional tile ID labels
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance = None):
        """Factory method to create map visualizer from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def process_frame(self, frame: np.ndarray, frame_index: int, session) -> np.ndarray:
        """Process a single frame and add map overlay.
        
        Args:
            frame: Input video frame
            frame_index: Current frame index  
            session: Session object with full data access
            
        Returns:
            Frame with map overlay applied
        """
        # Get visualization settings with defaults
        viz_config = {
            'overlay_position': self.config.get('overlay_position', 'bottom_right'),
            'overlay_size': self.config.get('overlay_size', 0.3), 
            'overlay_opacity': self.config.get('overlay_opacity', 0.7),
            'current_tile_color': self.config.get('current_tile_color', [255, 0, 0]),
            'visited_tile_color': self.config.get('visited_tile_color', [0, 255, 0]),
            'visited_tile_opacity': self.config.get('visited_tile_opacity', 0.3),
            'show_visited_tiles': self.config.get('show_visited_tiles', True),
            'show_tile_id': self.config.get('show_tile_id', True),
            'tile_id_font_scale': self.config.get('tile_id_font_scale', 1.0),
            'tile_id_color': self.config.get('tile_id_color', [0, 0, 255])
        }
        
        # Get map provider from shared resources
        try:
            shared_resources = session.shared_resources
            map_provider = shared_resources.get('map_integration') or shared_resources.get('maze_map')
            if not map_provider:
                self.logger.error("Map visualization requires map_integration or maze_map in shared_resources")
                return frame
                
            # Get session data
            session_data = session.get_integrated_dataframe()
            
            # Get map resources
            map_img = map_provider.get_map_image()
            map_config = map_provider.get_map_configuration()
            
        except Exception as e:
            self.logger.error(f"Could not get map resources: {str(e)}")
            return frame
        
        # Check if frame_index is valid
        if frame_index >= len(session_data):
            self.logger.debug(f"Frame index {frame_index} beyond session data length {len(session_data)}")
            return frame
        
        # Initialize visited tiles tracking (simple approach - get all visited up to current frame)
        visited_tiles: Set[int] = set()
        if 'tile_id' in session_data.columns:
            # Get all tiles visited up to current frame
            tiles_up_to_frame = session_data.iloc[:frame_index + 1]['tile_id'].dropna()
            visited_tiles = set(int(tile_id) for tile_id in tiles_up_to_frame if not pd.isna(tile_id))
        
        try:
            # Calculate overlay parameters
            overlay_params = self._calculate_overlay_params(
                frame.shape, map_img.shape, viz_config
            )
            
            # Create map overlay for this frame
            map_overlay = self._create_map_overlay(
                map_img,
                map_config,
                session_data,
                frame_index,
                visited_tiles,
                overlay_params,
                viz_config
            )
            
            # Apply overlay to frame
            return self._apply_overlay(
                frame,
                map_overlay,
                overlay_params,
                viz_config['overlay_opacity']
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process frame {frame_index}: {str(e)}")
            return frame  # Return original frame on error
    
    
    def _calculate_overlay_params(
        self,
        frame_shape: Tuple[int, int, int],
        map_shape: Tuple[int, int, int],
        viz_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overlay size and position parameters.
        
        Args:
            frame_shape: Shape of video frames
            map_shape: Shape of map image
            viz_config: Visualization configuration
            
        Returns:
            Dictionary with overlay parameters
        """
        frame_height, frame_width = frame_shape[:2]
        map_height, map_width = map_shape[:2]
        
        # Calculate overlay size
        overlay_width = int(frame_width * viz_config['overlay_size'])
        overlay_height = int(overlay_width * map_height / map_width)
        
        # Calculate position
        positions = {
            'top_left': (0, 0),
            'top_right': (frame_width - overlay_width, 0),
            'bottom_left': (0, frame_height - overlay_height),
            'bottom_right': (frame_width - overlay_width, frame_height - overlay_height)
        }
        overlay_x, overlay_y = positions.get(
            viz_config['overlay_position'], 
            positions['bottom_right']
        )
        
        return {
            'width': overlay_width,
            'height': overlay_height,
            'x': overlay_x,
            'y': overlay_y
        }
    
    def _create_map_overlay(
        self,
        map_img: np.ndarray,
        map_config: Dict[str, Any],
        session_data: pd.DataFrame,
        frame_idx: int,
        visited_tiles: Set[int],
        overlay_params: Dict[str, Any],
        viz_config: Dict[str, Any]
    ) -> np.ndarray:
        """Create map overlay for current frame.
        
        Args:
            map_img: Original map image
            map_config: Map configuration with grid info
            session_data: DataFrame with tile_id data
            frame_idx: Current frame index
            visited_tiles: Set of previously visited tiles
            overlay_params: Overlay size and position
            viz_config: Visualization settings
            
        Returns:
            Map overlay image
        """
        # Resize map to overlay size
        map_overlay = cv2.resize(
            map_img, 
            (overlay_params['width'], overlay_params['height'])
        )
        
        # Check if we have tile data for this frame
        if frame_idx in session_data.index and 'tile_id' in session_data.columns:
            current_tile_id = session_data.loc[frame_idx, 'tile_id']
            
            if not pd.isna(current_tile_id):
                # Update visited tiles
                visited_tiles.add(int(current_tile_id))
                
                # Draw visited tiles (if enabled)
                if viz_config['show_visited_tiles']:
                    for tile_id in visited_tiles:
                        self._highlight_tile_on_map(
                            map_overlay,
                            tile_id,
                            viz_config['visited_tile_color'],
                            viz_config['visited_tile_opacity'],
                            map_config,
                            overlay_params,
                            map_img.shape[:2]
                        )
                
                # Highlight current tile
                self._highlight_tile_on_map(
                    map_overlay,
                    int(current_tile_id),
                    viz_config['current_tile_color'],
                    0.7,
                    map_config,
                    overlay_params,
                    map_img.shape[:2]
                )
                
                # Draw tile ID if enabled
                if viz_config['show_tile_id']:
                    self._draw_tile_id(
                        map_overlay,
                        int(current_tile_id),
                        viz_config['tile_id_color'],
                        viz_config['tile_id_font_scale'],
                        map_config,
                        overlay_params,
                        map_img.shape[:2]
                    )
        
        return map_overlay
    
    def _highlight_tile_on_map(
        self,
        overlay: np.ndarray,
        tile_id: int,
        color: List[int],
        opacity: float,
        map_config: Dict[str, Any],
        overlay_params: Dict[str, Any],
        original_map_shape: Tuple[int, int]
    ) -> None:
        """Highlight a specific tile on the map overlay.
        
        Args:
            overlay: Map overlay image to draw on
            tile_id: ID of tile to highlight
            color: [B,G,R] color for highlight
            opacity: Opacity of highlight (0.0-1.0)
            map_config: Map configuration with grid info
            overlay_params: Overlay dimensions
            original_map_shape: Original map dimensions for scaling
        """
        # Get tile position in grid
        grid_cols = map_config['grid_size'][0]
        row = tile_id // grid_cols
        col = tile_id % grid_cols
        
        # Calculate tile bounds in original map coordinates
        segment_length = map_config['segment_length']
        origin_x, origin_y = map_config['origin']
        
        x1 = origin_x + col * segment_length
        y1 = origin_y + row * segment_length
        x2 = x1 + segment_length
        y2 = y1 + segment_length
        
        # Scale to overlay size
        scale_x = overlay_params['width'] / original_map_shape[1]
        scale_y = overlay_params['height'] / original_map_shape[0]
        
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)
        
        # Create highlight overlay
        highlight = overlay.copy()
        cv2.rectangle(highlight, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, -1)
        
        # Blend with original
        cv2.addWeighted(highlight, opacity, overlay, 1 - opacity, 0, overlay)
    
    def _draw_tile_id(
        self,
        overlay: np.ndarray,
        tile_id: int,
        color: List[int],
        font_scale: float,
        map_config: Dict[str, Any],
        overlay_params: Dict[str, Any],
        original_map_shape: Tuple[int, int]
    ) -> None:
        """Draw tile ID text on the map overlay.
        
        Args:
            overlay: Map overlay image to draw on
            tile_id: ID to draw
            color: [B,G,R] color for text
            font_scale: Text size scale
            map_config: Map configuration
            overlay_params: Overlay dimensions
            original_map_shape: Original map dimensions
        """
        # Get tile center position
        grid_cols = map_config['grid_size'][0]
        row = tile_id // grid_cols
        col = tile_id % grid_cols
        
        segment_length = map_config['segment_length']
        origin_x, origin_y = map_config['origin']
        
        center_x = origin_x + col * segment_length + segment_length // 2
        center_y = origin_y + row * segment_length + segment_length // 2
        
        # Scale to overlay size
        scale_x = overlay_params['width'] / original_map_shape[1]
        scale_y = overlay_params['height'] / original_map_shape[0]
        
        text_x = int(center_x * scale_x)
        text_y = int(center_y * scale_y)
        
        # Draw text
        cv2.putText(
            overlay,
            str(tile_id),
            (text_x - 10, text_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * scale_x,  # Scale font with overlay
            color,
            2
        )
    
    def _apply_overlay(
        self,
        frame: np.ndarray,
        overlay: np.ndarray,
        overlay_params: Dict[str, Any],
        opacity: float
    ) -> np.ndarray:
        """Apply map overlay to frame.
        
        Args:
            frame: Video frame
            overlay: Map overlay
            overlay_params: Position and size parameters
            opacity: Overlay opacity (0.0-1.0)
            
        Returns:
            Frame with overlay applied
        """
        x, y = overlay_params['x'], overlay_params['y']
        w, h = overlay_params['width'], overlay_params['height']
        
        # Extract region of interest
        roi = frame[y:y+h, x:x+w]
        
        # Apply overlay with opacity
        cv2.addWeighted(overlay, opacity, roi, 1 - opacity, 0, roi)
        
        # Put back into frame
        frame[y:y+h, x:x+w] = roi
        
        return frame