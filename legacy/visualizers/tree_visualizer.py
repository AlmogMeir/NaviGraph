"""Tree visualizer plugin for NaviGraph.

This plugin visualizes the graph/tree structure representing the spatial
navigation topology, highlighting current positions and paths.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from ...core.interfaces import IVisualizer, Logger
from ...core.base_plugin import BasePlugin
from ...core.registry import register_visualizer_plugin


@register_visualizer_plugin("tree_visualizer")
class TreeVisualizer(BasePlugin, IVisualizer):
    """Visualizes navigation graph/tree structure.
    
    Features:
    - Dynamic tree visualization showing current position
    - Path-to-goal highlighting
    - History tracking with fading colors
    - Configurable layout and styling
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance = None):
        """Factory method to create tree visualizer from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate tree visualizer configuration."""
        # All config keys are optional with sensible defaults
        pass
    
    def generate_visualization(
        self,
        session_data: pd.DataFrame,
        config: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> Optional[str]:
        """Create tree visualization overlay on video frames.
        
        Args:
            session_data: DataFrame with tile_id and tree_position data
            config: Visualization-specific configuration
            output_path: Directory to save visualization outputs
            **kwargs: Additional parameters including:
                - video_path: Path to source video file
                - session_id: Session identifier for output naming
                - reward_tile_id: Goal tile ID for path highlighting
                - shared_resources: Must contain 'graph' resource
                
        Returns:
            Path to created visualization video file, or None if failed
        """
        try:
            # Extract parameters
            video_path = kwargs.get('video_path')
            if not video_path:
                self.logger.error("Tree visualization requires video_path")
                return None
                
            session_id = kwargs.get('session_id', 'unknown_session')
            reward_tile_id = kwargs.get('reward_tile_id')
            
            # Get graph provider from shared resources
            shared_resources = kwargs.get('shared_resources', {})
            graph_provider = shared_resources.get('graph')
            if not graph_provider:
                self.logger.error("Tree visualization requires graph in shared_resources")
                return None
                
            # Get visualization settings with defaults
            viz_config = {
                'overlay_position': config.get('overlay_position', 'top_right'),
                'overlay_size': config.get('overlay_size', 0.3),
                'overlay_opacity': config.get('overlay_opacity', 0.8),
                'tree_bg_color': config.get('tree_bg_color', [255, 255, 255]),  # White
                'static_node_color': config.get('static_node_color', [200, 200, 200]),
                'static_edge_color': config.get('static_edge_color', [150, 150, 150]),
                'current_node_color': config.get('current_node_color', [255, 0, 0]),  # Red
                'current_edge_color': config.get('current_edge_color', [255, 0, 0]),
                'history_node_color': config.get('history_node_color', [139, 0, 0]),  # Dark red
                'history_edge_color': config.get('history_edge_color', [139, 0, 0]),
                'path_node_color': config.get('path_node_color', [0, 255, 0]),  # Green
                'path_edge_color': config.get('path_edge_color', [0, 255, 0]),
                'history_length': config.get('history_length', 30),  # Frames
                'output_fps': config.get('output_fps', None),
                'output_codec': config.get('output_codec', 'mp4v'),
                'display_mode': config.get('display_mode', 'overlay')  # 'overlay' or 'side_by_side'
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
            
            # Adjust output dimensions for side-by-side mode
            if viz_config['display_mode'] == 'side_by_side':
                output_width = frame_width * 2
                output_height = frame_height
            else:
                output_width = frame_width
                output_height = frame_height
            
            # Prepare output
            output_filename = f"tree_{session_id}_{Path(video_path).stem}.mp4"
            output_file = Path(output_path) / output_filename
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*viz_config['output_codec'])
            writer = cv2.VideoWriter(
                str(output_file),
                fourcc,
                fps,
                (output_width, output_height)
            )
            
            # Get graph instance and calculate path to reward if available
            graph = graph_provider.get_graph_instance()
            path_to_reward = []
            if reward_tile_id is not None:
                try:
                    # Get graph node for reward tile
                    reward_node = graph.get_tree_location(reward_tile_id)
                    if isinstance(reward_node, int):
                        path_to_reward = graph.get_shortest_path(source=0, target=reward_node)
                except Exception as e:
                    self.logger.debug(f"Could not calculate path to reward: {str(e)}")
            
            # Process frames
            frame_idx = 0
            position_history = []
            
            self.logger.info(f"Creating tree visualization for {session_id}")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                    
                # Get current tile and tree position
                current_tile_id = None
                current_tree_position = None
                
                if frame_idx in session_data.index:
                    if 'tile_id' in session_data.columns:
                        current_tile_id = session_data.loc[frame_idx, 'tile_id']
                    if 'tree_position' in session_data.columns:
                        current_tree_position = session_data.loc[frame_idx, 'tree_position']
                
                # Update position history
                if current_tree_position is not None and not pd.isna(current_tree_position):
                    position_history.append(current_tree_position)
                    if len(position_history) > viz_config['history_length']:
                        position_history.pop(0)
                
                # Generate tree visualization
                tree_img = self._generate_tree_image(
                    graph,
                    current_tree_position,
                    position_history,
                    path_to_reward,
                    viz_config,
                    (frame_width, frame_height)
                )
                
                # Combine with video frame
                if viz_config['display_mode'] == 'side_by_side':
                    # Side-by-side layout
                    output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                    output_frame[:, :frame_width] = frame
                    output_frame[:, frame_width:] = cv2.resize(tree_img, (frame_width, frame_height))
                else:
                    # Overlay mode
                    output_frame = self._overlay_tree_on_frame(
                        frame, tree_img, viz_config
                    )
                
                # Write frame
                writer.write(output_frame)
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 100 == 0:
                    progress = frame_idx / total_frames * 100
                    self.logger.debug(f"Tree visualization progress: {progress:.1f}%")
            
            # Cleanup
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            
            self.logger.info(f"Tree visualization saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Tree visualization failed: {str(e)}")
            return None
    
    def _generate_tree_image(
        self,
        graph: Any,
        current_position: Any,
        position_history: List[Any],
        path_to_reward: List[int],
        viz_config: Dict[str, Any],
        frame_size: Tuple[int, int]
    ) -> np.ndarray:
        """Generate tree visualization image."""
        try:
            # Create configuration for graph drawing
            draw_config = {
                'node_list': [],
                'edge_list': [],
                'color_mode': 'custom',
                'unique_path': []
            }
            
            # Add current position
            if current_position is not None:
                if isinstance(current_position, int):
                    draw_config['node_list'].append(current_position)
                elif isinstance(current_position, tuple):
                    draw_config['edge_list'].append(current_position)
                elif isinstance(current_position, frozenset):
                    for item in current_position:
                        if isinstance(item, int):
                            draw_config['node_list'].append(item)
                        elif isinstance(item, tuple):
                            draw_config['edge_list'].append(item)
            
            # Add path to reward
            if path_to_reward:
                draw_config['unique_path'] = path_to_reward
                # Also add edges for the path
                for i in range(len(path_to_reward) - 1):
                    draw_config['unique_path'].append(
                        (path_to_reward[i], path_to_reward[i + 1])
                    )
            
            # Configure colors based on viz_config
            graph.graph_config = type('GraphConfig', (), {
                'draw': type('DrawConfig', (), {
                    'with_labels': True,
                    'font_weight': 'bold',
                    'node_size': 500,
                    'font_size': 10
                })(),
                'options': type('OptionsConfig', (), {
                    'static_node_color': self._rgb_to_hex(viz_config['static_node_color']),
                    'static_edge_color': self._rgb_to_hex(viz_config['static_edge_color']),
                    'dynamic_node_color': self._rgb_to_hex(viz_config['current_node_color']),
                    'dynamic_edge_color': self._rgb_to_hex(viz_config['current_edge_color']),
                    'history_node_color': self._rgb_to_hex(viz_config['history_node_color']),
                    'history_edge_color': self._rgb_to_hex(viz_config['history_edge_color']),
                    'dynamic_reward_node_color': self._rgb_to_hex(viz_config['path_node_color']),
                    'dynamic_reward_edge_color': self._rgb_to_hex(viz_config['path_edge_color']),
                    'edge_width': 3
                })()
            })()
            
            # Draw tree
            graph.draw_tree(
                node_list=draw_config['node_list'],
                edge_list=draw_config['edge_list'],
                color_mode='current',
                unique_path=draw_config['unique_path']
            )
            
            # Convert to image
            tree_img = graph.tree_fig_to_img()
            
            # Ensure BGR format for OpenCV
            if tree_img.shape[2] == 4:  # RGBA
                tree_img = cv2.cvtColor(tree_img, cv2.COLOR_RGBA2BGR)
            elif tree_img.shape[2] == 3 and tree_img.dtype == np.uint8:
                # Assume RGB, convert to BGR
                tree_img = cv2.cvtColor(tree_img, cv2.COLOR_RGB2BGR)
            
            return tree_img
            
        except Exception as e:
            self.logger.debug(f"Failed to generate tree image: {str(e)}")
            # Return blank image
            blank = np.full((frame_size[1], frame_size[0], 3), 255, dtype=np.uint8)
            return blank
    
    def _overlay_tree_on_frame(
        self,
        frame: np.ndarray,
        tree_img: np.ndarray,
        viz_config: Dict[str, Any]
    ) -> np.ndarray:
        """Overlay tree visualization on video frame."""
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate overlay size
        overlay_width = int(frame_width * viz_config['overlay_size'])
        overlay_height = int(frame_height * viz_config['overlay_size'])
        
        # Resize tree image
        tree_resized = cv2.resize(tree_img, (overlay_width, overlay_height))
        
        # Calculate position
        positions = {
            'top_left': (0, 0),
            'top_right': (frame_width - overlay_width, 0),
            'bottom_left': (0, frame_height - overlay_height),
            'bottom_right': (frame_width - overlay_width, frame_height - overlay_height)
        }
        x, y = positions.get(viz_config['overlay_position'], positions['top_right'])
        
        # Create output frame
        output = frame.copy()
        
        # Overlay tree with opacity
        roi = output[y:y+overlay_height, x:x+overlay_width]
        cv2.addWeighted(
            tree_resized, 
            viz_config['overlay_opacity'],
            roi,
            1 - viz_config['overlay_opacity'],
            0,
            roi
        )
        output[y:y+overlay_height, x:x+overlay_width] = roi
        
        return output
    
    def _rgb_to_hex(self, rgb: List[int]) -> str:
        """Convert RGB color to hex string."""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
    
    @property
    def supported_formats(self) -> List[str]:
        """List of supported output formats."""
        return ['mp4', 'avi', 'mov']