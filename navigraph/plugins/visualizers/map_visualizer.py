"""Map visualizer plugin for NaviGraph.

This plugin visualizes spatial navigation data by overlaying maze maps on video
frames, highlighting current tile positions and visited areas.
"""

from typing import Dict, Any, Optional, List, Tuple
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
    
    Features:
    - Overlays maze map on video frames
    - Highlights current tile position
    - Shows visited tiles history
    - Configurable overlay position and opacity
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance = None):
        """Factory method to create map visualizer from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate map visualizer configuration."""
        # All config keys are optional with sensible defaults
        pass
    
    def visualize(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> Optional[str]:
        """Create map visualization overlay on video frames.
        
        Args:
            data: DataFrame with tile_id and position data
            config: Visualization-specific configuration
            shared_resources: Must contain 'maze_map' resource
            output_path: Directory to save visualization outputs
            **kwargs: Additional parameters including:
                - video_path: Path to source video file
                - session_id: Session identifier for output naming
                
        Returns:
            Path to created visualization video file, or None if failed
        """
        try:
            # Extract parameters
            video_path = kwargs.get('video_path')
            if not video_path:
                self.logger.error("Map visualization requires video_path")
                return None
                
            session_id = kwargs.get('session_id', 'unknown_session')
            
            # Get map provider from shared resources
            map_provider = shared_resources.get('maze_map')
            if not map_provider:
                self.logger.error("Map visualization requires maze_map in shared_resources")
                return None
                
            # Get visualization settings with defaults
            viz_config = {
                'overlay_position': config.get('overlay_position', 'bottom_right'),
                'overlay_size': config.get('overlay_size', 0.3),  # Fraction of frame size
                'overlay_opacity': config.get('overlay_opacity', 0.7),
                'current_tile_color': config.get('current_tile_color', [255, 0, 0]),  # Red
                'visited_tile_color': config.get('visited_tile_color', [0, 255, 0]),  # Green
                'visited_tile_opacity': config.get('visited_tile_opacity', 0.3),
                'show_tile_id': config.get('show_tile_id', True),
                'tile_id_font_scale': config.get('tile_id_font_scale', 1.0),
                'tile_id_color': config.get('tile_id_color', [0, 0, 255]),  # Blue
                'output_fps': config.get('output_fps', None),
                'output_codec': config.get('output_codec', 'mp4v')
            }
            
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
            
            # Get map image
            map_img = map_provider.get_map_image()
            map_config = map_provider.get_map_configuration()
            
            # Calculate overlay size
            overlay_width = int(frame_width * viz_config['overlay_size'])
            overlay_height = int(overlay_width * map_img.shape[0] / map_img.shape[1])
            map_resized = cv2.resize(map_img, (overlay_width, overlay_height))
            
            # Calculate overlay position
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
            
            # Prepare output
            output_filename = f"map_{session_id}_{Path(video_path).stem}.mp4"
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
            visited_tiles = set()
            
            self.logger.info(f"Creating map visualization for {session_id}")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                    
                # Create map overlay for this frame
                map_overlay = map_resized.copy()
                
                # Get current tile
                current_tile_id = None
                if frame_idx in data.index and 'tile_id' in data.columns:
                    current_tile_id = data.loc[frame_idx, 'tile_id']
                    
                    if not pd.isna(current_tile_id) and current_tile_id >= 0:
                        visited_tiles.add(int(current_tile_id))
                        
                        # Highlight visited tiles
                        for tile_id in visited_tiles:
                            self._highlight_tile_on_map(
                                map_overlay,
                                tile_id,
                                viz_config['visited_tile_color'],
                                viz_config['visited_tile_opacity'],
                                map_config,
                                map_resized.shape[:2],
                                map_img.shape[:2]
                            )
                        
                        # Highlight current tile
                        self._highlight_tile_on_map(
                            map_overlay,
                            int(current_tile_id),
                            viz_config['current_tile_color'],
                            0.7,
                            map_config,
                            map_resized.shape[:2],
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
                                map_resized.shape[:2],
                                map_img.shape[:2]
                            )
                
                # Overlay map on frame
                self._overlay_image(
                    frame,
                    map_overlay,
                    overlay_x,
                    overlay_y,
                    viz_config['overlay_opacity']
                )
                
                # Write frame
                writer.write(frame)
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 100 == 0:
                    progress = frame_idx / total_frames * 100
                    self.logger.debug(f"Map visualization progress: {progress:.1f}%")
            
            # Cleanup
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            
            self.logger.info(f"Map visualization saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Map visualization failed: {str(e)}")
            return None
    
    def _highlight_tile_on_map(
        self,
        map_img: np.ndarray,
        tile_id: int,
        color: List[int],
        opacity: float,
        map_config: Dict[str, Any],
        resized_shape: Tuple[int, int],
        original_shape: Tuple[int, int]
    ) -> None:
        """Highlight a specific tile on the map image."""
        try:
            # Calculate tile position based on grid
            grid_size = map_config['grid_size']
            if tile_id >= grid_size[0] * grid_size[1]:
                return
                
            row = tile_id // grid_size[1]
            col = tile_id % grid_size[1]
            
            # Calculate pixel coordinates (scaled for resized map)
            scale_x = resized_shape[1] / original_shape[1]
            scale_y = resized_shape[0] / original_shape[0]
            
            origin = map_config['origin']
            segment_length = map_config['segment_length']
            
            x1 = int((origin[0] + col * segment_length) * scale_x)
            y1 = int((origin[1] + row * segment_length) * scale_y)
            x2 = int(x1 + segment_length * scale_x)
            y2 = int(y1 + segment_length * scale_y)
            
            # Create colored rectangle
            overlay = map_img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Blend with original
            cv2.addWeighted(overlay, opacity, map_img, 1 - opacity, 0, map_img)
            
        except Exception as e:
            self.logger.debug(f"Failed to highlight tile {tile_id}: {str(e)}")
    
    def _draw_tile_id(
        self,
        map_img: np.ndarray,
        tile_id: int,
        color: List[int],
        font_scale: float,
        map_config: Dict[str, Any],
        resized_shape: Tuple[int, int],
        original_shape: Tuple[int, int]
    ) -> None:
        """Draw tile ID number on the map."""
        try:
            # Calculate tile center position
            grid_size = map_config['grid_size']
            if tile_id >= grid_size[0] * grid_size[1]:
                return
                
            row = tile_id // grid_size[1]
            col = tile_id % grid_size[1]
            
            # Calculate pixel coordinates (scaled for resized map)
            scale_x = resized_shape[1] / original_shape[1]
            scale_y = resized_shape[0] / original_shape[0]
            
            origin = map_config['origin']
            segment_length = map_config['segment_length']
            
            center_x = int((origin[0] + (col + 0.5) * segment_length) * scale_x)
            center_y = int((origin[1] + (row + 0.5) * segment_length) * scale_y)
            
            # Draw text
            text = str(tile_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = max(1, int(2 * font_scale))
            
            # Get text size for centering
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
            
            cv2.putText(
                map_img, text, (text_x, text_y),
                font, font_scale, color, thickness
            )
            
        except Exception as e:
            self.logger.debug(f"Failed to draw tile ID {tile_id}: {str(e)}")
    
    def _overlay_image(
        self,
        background: np.ndarray,
        overlay: np.ndarray,
        x: int,
        y: int,
        opacity: float
    ) -> None:
        """Overlay an image on background at specified position."""
        h, w = overlay.shape[:2]
        
        # Ensure overlay fits within frame
        if x + w > background.shape[1]:
            w = background.shape[1] - x
        if y + h > background.shape[0]:
            h = background.shape[0] - y
            
        if w <= 0 or h <= 0:
            return
            
        # Extract ROI and blend
        roi = background[y:y+h, x:x+w]
        cv2.addWeighted(overlay[:h, :w], opacity, roi, 1 - opacity, 0, roi)
        background[y:y+h, x:x+w] = roi
    
    @property
    def supported_formats(self) -> List[str]:
        """List of supported output formats."""
        return ['mp4', 'avi', 'mov']