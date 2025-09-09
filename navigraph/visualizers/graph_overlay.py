"""Graph overlay visualizer for NaviGraph.

Overlays graph structure on video frames with current node/edge highlighting.
Uses the graph builder's get_visualization() method for proper rendering.
"""

import numpy as np
import cv2
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Set, Union

from ..core.registry import register_visualizer


@register_visualizer("graph_overlay")
def visualize_graph_overlay(frame: np.ndarray, frame_data: pd.Series, shared_resources: Dict[str, Any], **config) -> np.ndarray:
    """Overlay graph structure on video frame with position highlighting.
    
    Args:
        frame: Input video frame (H, W, 3)
        frame_data: DataFrame row for current frame
        shared_resources: Session shared resources containing 'graph'
        **config: Visualization configuration
        
    Config:
        # Graph display settings
        mode: 'overlay' or 'side_by_side' (default: 'overlay')
        size: Scale factor for overlay (0.1 to 1.0) when mode='overlay' (default: 0.3)
        opacity: Graph transparency (0.0 to 1.0, default: 0.6)
        position: Position of graph overlay - 'top_right', 'top_left', 'bottom_right', 'bottom_left' (default: 'top_right')
        
        # Default visualization settings (if null, uses builder defaults)
        default_node_size: Default size for all nodes (default: null)
        default_node_color: Default color for all nodes (default: null)
        default_edge_width: Default width for all edges (default: null)  
        default_edge_color: Default color for all edges (default: null)
        with_labels: Show node labels (default: null)
        font_size: Font size for labels (default: null)
        font_weight: Font weight - 'normal', 'bold', etc. (default: null)
        font_color: Font color for labels (default: null)
        font_family: Font family - 'sans-serif', 'serif', 'monospace' (default: null)
        figsize: Figure size as [width, height] (default: null)
        
        # Current node/edge highlighting (passed to get_visualization)
        highlight_node_size: Size for current node (default: 500)
        highlight_node_color: Color for current node (default: 'red')
        highlight_edge_width: Width for current edge (default: 3.0)
        highlight_edge_color: Color for current edge (default: 'blue')
        
    Returns:
        Frame with graph overlay applied
    """
    if frame_data.empty:
        return frame
        
    # Get graph from shared resources
    graph = shared_resources.get('graph')
    if graph is None:
        return frame
    
    # Get current position data - try bodypart-specific columns first
    current_node_id = frame_data.get('graph_node_id')
    current_edge_id = frame_data.get('graph_edge_id')
    
    # If generic columns don't exist, try bodypart-specific ones (e.g., Nose_graph_node)
    if pd.isna(current_node_id):
        for col in frame_data.index:
            if col.endswith('_graph_node') and pd.notna(frame_data.get(col)):
                current_node_id = frame_data.get(col)
                break
    
    if pd.isna(current_edge_id):
        for col in frame_data.index:
            if col.endswith('_graph_edge') and pd.notna(frame_data.get(col)):
                current_edge_id = frame_data.get(col)
                break
    
    # Apply graph overlay
    return _apply_graph_overlay(frame, graph, current_node_id, current_edge_id, config)


def _apply_graph_overlay(frame: np.ndarray, graph, current_node_id, current_edge_id, config: Dict[str, Any]) -> np.ndarray:
    """Apply graph visualization overlay to frame."""
    
    # Graph overlay configuration
    mode = config.get('mode', 'overlay')
    size = config.get('size', 0.3)
    opacity = config.get('opacity', 0.6)
    position = config.get('position', 'top_right')
    
    # Build visualization parameters
    viz_params = _build_visualization_params(graph, current_node_id, current_edge_id, config)
    
    # Get graph visualization image
    try:
        graph_image = graph.get_visualization(**viz_params)
    except Exception as e:
        # Fall back to default visualization if highlighting fails
        graph_image = graph.get_visualization()
    
    if graph_image is None:
        return frame
    
    # Convert RGB to BGR for OpenCV (matplotlib outputs RGB, OpenCV expects BGR)
    graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
    
    # Apply overlay based on mode
    if mode == 'side_by_side':
        return _apply_side_by_side_overlay(frame, graph_image)
    else:  # overlay mode
        return _apply_overlay_mode(frame, graph_image, size, opacity, position)


def _build_visualization_params(graph, current_node_id, current_edge_id, config: Dict[str, Any]) -> Dict[str, Any]:
    """Build parameters for graph.get_visualization() with highlighting and custom defaults."""
    
    params = {}
    
    # Convert BGR colors to RGB for matplotlib (OpenCV uses BGR, matplotlib uses RGB)
    def bgr_to_rgb_color(color):
        if color is None:
            return color
        elif isinstance(color, str):
            return color  # Named colors work fine
        elif isinstance(color, (list, tuple)) and len(color) == 3:
            # Convert BGR to RGB
            return [color[2], color[1], color[0]]
        return color
    
    # Get configuration values
    highlight_node_size = config.get('highlight_node_size', 500)
    highlight_node_color = config.get('highlight_node_color', 'red')
    highlight_edge_width = config.get('highlight_edge_width', 3.0)
    highlight_edge_color = config.get('highlight_edge_color', 'blue')
    
    # Get default settings (use None if not specified to let builder use its defaults)
    default_node_size = config.get('default_node_size', None)
    default_node_color = config.get('default_node_color', None)
    default_edge_width = config.get('default_edge_width', None)
    default_edge_color = config.get('default_edge_color', None)
    
    # Always build node lists to ensure consistent sizing
    node_colors = []
    node_sizes = []
    
    # Convert highlight color from BGR to RGB
    rgb_node_color = bgr_to_rgb_color(highlight_node_color)
    rgb_default_node_color = bgr_to_rgb_color(default_node_color)
    
    for node in graph.graph.nodes():
        if pd.notna(current_node_id) and node == current_node_id:
            node_colors.append(rgb_node_color)
            node_sizes.append(highlight_node_size)
        else:
            # Use configured default or fallback
            node_colors.append(rgb_default_node_color if rgb_default_node_color is not None else 'lightblue')
            node_sizes.append(default_node_size if default_node_size is not None else 300)
    
    params['node_color'] = node_colors
    params['node_size'] = node_sizes
    
    # Always build edge lists for consistency
    edge_colors = []
    edge_widths = []
    
    # Convert highlight color from BGR to RGB
    rgb_edge_color = bgr_to_rgb_color(highlight_edge_color)
    rgb_default_edge_color = bgr_to_rgb_color(default_edge_color)
    
    # Convert edge ID to edge tuple if needed
    current_edge = _get_edge_from_id(graph, current_edge_id) if pd.notna(current_edge_id) else None
    
    for edge in graph.graph.edges():
        if current_edge and (edge == current_edge or edge[::-1] == current_edge):
            edge_colors.append(rgb_edge_color)
            edge_widths.append(highlight_edge_width)
        else:
            # Use configured default or fallback
            edge_colors.append(rgb_default_edge_color if rgb_default_edge_color is not None else 'gray')
            edge_widths.append(default_edge_width if default_edge_width is not None else 1.0)
    
    params['edge_color'] = edge_colors
    params['width'] = edge_widths
    
    # Add other configurable parameters (only if specified)
    optional_params = {
        'with_labels': config.get('with_labels'),
        'font_size': config.get('font_size'),
        'font_weight': config.get('font_weight'),
        'font_color': bgr_to_rgb_color(config.get('font_color')),
        'font_family': config.get('font_family'),
        'figsize': config.get('figsize')
    }
    
    # Only add non-None parameters
    for key, value in optional_params.items():
        if value is not None:
            params[key] = value
    
    return params


def _get_edge_from_id(graph, edge_id) -> Optional[Tuple]:
    """Convert edge ID to edge tuple."""
    if pd.isna(edge_id):
        return None
    
    # Try different methods to find the edge
    # Method 1: Direct edge lookup if graph has edge mapping
    if hasattr(graph, 'get_edge_by_id'):
        return graph.get_edge_by_id(edge_id)
    
    # Method 2: Check if edge_id is already a tuple
    if isinstance(edge_id, (tuple, list)) and len(edge_id) == 2:
        return tuple(edge_id)
    
    # Method 3: Search through edges (fallback)
    for edge in graph.graph.edges():
        # If edge has an 'id' attribute
        if graph.graph.edges[edge].get('id') == edge_id:
            return edge
    
    return None


def _apply_side_by_side_overlay(frame: np.ndarray, graph_image: np.ndarray) -> np.ndarray:
    """Apply side-by-side overlay of graph and frame."""
    
    frame_h, frame_w = frame.shape[:2]
    graph_h, graph_w = graph_image.shape[:2]
    
    # Calculate new dimensions maintaining aspect ratio
    aspect_ratio = graph_w / graph_h
    new_height = frame_h
    new_width = int(frame_h * aspect_ratio)
    
    resized_graph = cv2.resize(graph_image, (new_width, new_height))
    
    # Create combined frame
    combined_frame = np.zeros((frame_h, frame_w + new_width, 3), dtype=np.uint8)
    combined_frame[:, :frame_w] = frame
    combined_frame[:, frame_w:frame_w + new_width] = resized_graph
    
    return combined_frame


def _apply_overlay_mode(frame: np.ndarray, graph_image: np.ndarray, 
                       size: float, opacity: float, position: str) -> np.ndarray:
    """Apply overlay mode with graph positioned on frame."""
    
    frame_h, frame_w = frame.shape[:2]
    graph_h, graph_w = graph_image.shape[:2]
    
    # Resize graph based on size factor
    overlay_w = int(frame_w * size)
    overlay_h = int((overlay_w / graph_w) * graph_h)
    
    # Ensure overlay fits in frame
    if overlay_h > frame_h * 0.8:
        overlay_h = int(frame_h * 0.8)
        overlay_w = int((overlay_h / graph_h) * graph_w)
        
    resized_graph = cv2.resize(graph_image, (overlay_w, overlay_h))
    
    # Calculate position
    x, y = _calculate_overlay_position(frame_w, frame_h, overlay_w, overlay_h, position)
    
    # Apply overlay with opacity
    overlay_region = frame[y:y + overlay_h, x:x + overlay_w]
    blended = cv2.addWeighted(overlay_region, 1 - opacity, resized_graph, opacity, 0)
    frame[y:y + overlay_h, x:x + overlay_w] = blended
    
    return frame


def _calculate_overlay_position(frame_w: int, frame_h: int, overlay_w: int, overlay_h: int, position: str) -> Tuple[int, int]:
    """Calculate overlay position based on position string."""
    
    margin = 10
    
    if position == 'top_left':
        return margin, margin
    elif position == 'top_right':
        return frame_w - overlay_w - margin, margin
    elif position == 'bottom_left':
        return margin, frame_h - overlay_h - margin
    elif position == 'bottom_right':
        return frame_w - overlay_w - margin, frame_h - overlay_h - margin
    else:  # default to top_right
        return frame_w - overlay_w - margin, margin


