"""Graph overlay visualizer for NaviGraph.

Overlays graph structure on video frames with current node/edge highlighting.
"""

import numpy as np
import cv2
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Set
import networkx as nx

from ..core.registry import register_visualizer


@register_visualizer("graph_overlay")
def visualize_graph_overlay(frame: np.ndarray, frame_data: pd.Series, shared_resources: Dict[str, Any], **config) -> np.ndarray:
    """Overlay graph structure on video frame with position highlighting.
    
    Args:
        frame: Input video frame (H, W, 3)
        frame_data: DataFrame row for current frame
        shared_resources: Session shared resources containing 'graph' and 'graph_mapping'
        **config: Visualization configuration
        
    Config:
        mode: 'overlay' or 'side_by_side' (default: 'overlay')
        size: Scale factor for overlay (0.1 to 1.0) when mode='overlay'
        opacity: Graph transparency (0.0 to 1.0, default: 0.5)
        
        # Node styling
        node_color: Default node color [B,G,R] (default: [200, 200, 200] light gray)
        current_node_color: Color for current node [B,G,R] (default: [0, 0, 255] red)
        visited_node_color: Color for visited nodes [B,G,R] (default: [0, 255, 0] green)
        node_radius: Node circle radius (default: 8)
        
        # Edge styling  
        edge_color: Default edge color [B,G,R] (default: [100, 100, 100] gray)
        current_edge_color: Color for current edge [B,G,R] (default: [255, 0, 0] blue)
        visited_edge_color: Color for visited edges [B,G,R] (default: [0, 150, 0] dark green)
        edge_thickness: Edge line thickness (default: 2)
        
        # Path visualization
        show_visited: Show trail of visited nodes/edges (default: True)
        show_node_labels: Show node ID labels (default: False)
        font_scale: Font size for node labels (default: 0.4)
        label_color: Color for node labels [B,G,R] (default: [255, 255, 255] white)
        
        # Background
        background_color: Background color [B,G,R] (default: [50, 50, 50] dark gray)
        
    Returns:
        Frame with graph overlay applied
    """
    if frame_data.empty:
        return frame
        
    # Get graph components from shared resources
    graph = shared_resources.get('graph')
    graph_mapping = shared_resources.get('graph_mapping') 
    
    if graph is None or graph_mapping is None:
        return frame
    
    # Configuration
    mode = config.get('mode', 'overlay')
    size = config.get('size', 0.3)
    opacity = config.get('opacity', 0.5)
    
    # Node styling
    node_color = tuple(config.get('node_color', [200, 200, 200]))
    current_node_color = tuple(config.get('current_node_color', [0, 0, 255]))  # Red
    visited_node_color = tuple(config.get('visited_node_color', [0, 255, 0]))  # Green
    node_radius = config.get('node_radius', 8)
    
    # Edge styling
    edge_color = tuple(config.get('edge_color', [100, 100, 100]))
    current_edge_color = tuple(config.get('current_edge_color', [255, 0, 0]))  # Blue
    visited_edge_color = tuple(config.get('visited_edge_color', [0, 150, 0]))
    edge_thickness = config.get('edge_thickness', 2)
    
    # Path visualization
    show_visited = config.get('show_visited', True)
    show_node_labels = config.get('show_node_labels', False)
    font_scale = config.get('font_scale', 0.4)
    label_color = tuple(config.get('label_color', [255, 255, 255]))
    
    # Background
    background_color = tuple(config.get('background_color', [50, 50, 50]))
    
    # Get current position data
    current_node_id = frame_data.get('graph_node_id')
    current_edge_id = frame_data.get('graph_edge_id')
    
    # Create graph visualization
    graph_image = _create_graph_image(
        graph, graph_mapping, 
        current_node_id, current_edge_id,
        config, show_visited,
        node_color, current_node_color, visited_node_color, node_radius,
        edge_color, current_edge_color, visited_edge_color, edge_thickness,
        show_node_labels, font_scale, label_color, background_color
    )
    
    if graph_image is None:
        return frame
    
    # Apply overlay based on mode
    if mode == 'side_by_side':
        # Resize graph to match frame height
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
        
    else:  # overlay mode
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
        
        # Position at top-right corner
        x = frame_w - overlay_w - 10
        y = 10
        
        # Ensure overlay doesn't go out of bounds
        x = max(0, min(x, frame_w - overlay_w))
        y = max(0, min(y, frame_h - overlay_h))
        
        # Apply overlay with opacity
        overlay_region = frame[y:y + overlay_h, x:x + overlay_w]
        blended = cv2.addWeighted(overlay_region, 1 - opacity, resized_graph, opacity, 0)
        frame[y:y + overlay_h, x:x + overlay_w] = blended
        
        return frame


def _create_graph_image(
    graph, graph_mapping, current_node_id, current_edge_id, config, show_visited,
    node_color, current_node_color, visited_node_color, node_radius,
    edge_color, current_edge_color, visited_edge_color, edge_thickness,
    show_node_labels, font_scale, label_color, background_color
) -> Optional[np.ndarray]:
    """Create graph visualization image.
    
    Returns:
        Graph image as numpy array or None if failed
    """
    try:
        # Get graph layout positions
        pos = nx.spring_layout(graph.graph, k=1, iterations=50)
        
        # Scale positions to image coordinates
        image_size = 400
        margin = 50
        
        # Convert positions to pixel coordinates
        if not pos:
            return None
            
        # Get position bounds
        x_coords = [pos[node][0] for node in pos]
        y_coords = [pos[node][1] for node in pos]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Scale to image size with margin
        scale_x = (image_size - 2 * margin) / (x_max - x_min) if x_max != x_min else 1
        scale_y = (image_size - 2 * margin) / (y_max - y_min) if y_max != y_min else 1
        scale = min(scale_x, scale_y)
        
        pixel_pos = {}
        for node in pos:
            x = int((pos[node][0] - x_min) * scale + margin)
            y = int((pos[node][1] - y_min) * scale + margin)
            pixel_pos[node] = (x, y)
        
        # Create image
        image = np.full((image_size, image_size, 3), background_color, dtype=np.uint8)
        
        # Track visited nodes/edges if enabled
        if show_visited:
            visited_nodes = config.get('_visited_nodes', set())
            visited_edges = config.get('_visited_edges', set())
            
            if not isinstance(visited_nodes, set):
                visited_nodes = set()
            if not isinstance(visited_edges, set):
                visited_edges = set()
                
            # Add current to visited
            if pd.notna(current_node_id):
                visited_nodes.add(int(current_node_id))
            if pd.notna(current_edge_id):
                visited_edges.add(int(current_edge_id))
                
            # Store back in config
            config['_visited_nodes'] = visited_nodes
            config['_visited_edges'] = visited_edges
        else:
            visited_nodes = set()
            visited_edges = set()
        
        # Draw edges
        for edge in graph.graph.edges():
            node1, node2 = edge
            if node1 in pixel_pos and node2 in pixel_pos:
                pt1 = pixel_pos[node1]
                pt2 = pixel_pos[node2]
                
                # Determine edge color
                edge_id = graph.get_edge_id(node1, node2) if hasattr(graph, 'get_edge_id') else None
                if edge_id == current_edge_id:
                    color = current_edge_color
                elif edge_id in visited_edges:
                    color = visited_edge_color
                else:
                    color = edge_color
                    
                cv2.line(image, pt1, pt2, color, edge_thickness)
        
        # Draw nodes
        for node in graph.graph.nodes():
            if node in pixel_pos:
                center = pixel_pos[node]
                
                # Determine node color
                if node == current_node_id:
                    color = current_node_color
                elif node in visited_nodes:
                    color = visited_node_color
                else:
                    color = node_color
                    
                cv2.circle(image, center, node_radius, color, -1)
                
                # Draw node label if requested
                if show_node_labels:
                    label = str(node)
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                    text_pos = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)
                    cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                               font_scale, label_color, 1, cv2.LINE_AA)
        
        return image
        
    except Exception as e:
        # Return None if graph visualization fails
        return None