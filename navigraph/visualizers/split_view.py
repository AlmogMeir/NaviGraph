"""Split view visualizer for NaviGraph.

Displays video with mapping overlay on the left and graph with current position on the right.
Uses actual node positions from the mapping file for graph layout.
"""

import numpy as np
import cv2
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import pickle

from ..core.registry import register_visualizer
from ..core.coordinate_transform import transform_coordinates


@register_visualizer("split_view")
def visualize_split_view(
    frame: np.ndarray, 
    frame_data: pd.Series, 
    shared_resources: Dict[str, Any], 
    **config
) -> np.ndarray:
    """Create side-by-side view: video+mapping on left, graph with position on right.
    
    Args:
        frame: Input video frame (H, W, 3)
        frame_data: DataFrame row for current frame
        shared_resources: Session shared resources containing 'graph', 'map_image'
        **config: Visualization configuration
        
    Config:
        # Mapping overlay settings (for left side)
        mapping_alpha: Transparency of mapping overlay on video (default: 0.3)
        mapping_file: Path to mapping pickle file (required)
        show_node_boxes: Whether to show node bounding boxes (default: True)
        show_edge_boxes: Whether to show edge bounding boxes (default: False)
        node_box_color: Color for node boxes [B,G,R] (default: [0, 255, 0])
        edge_box_color: Color for edge boxes [B,G,R] (default: [255, 0, 0])
        
        # Graph visualization settings (for right side)
        graph_bg_color: Background color for graph [B,G,R] (default: [40, 40, 40])
        node_radius: Radius for graph nodes (default: 8)
        node_color: Color for inactive nodes [B,G,R] (default: [200, 200, 200])
        current_node_color: Color for current node [B,G,R] (default: [0, 255, 0])
        edge_color: Color for edges [B,G,R] (default: [100, 100, 100])
        edge_thickness: Thickness for edges (default: 1)
        show_node_labels: Whether to show node labels (default: True)
        label_font_scale: Font scale for labels (default: 0.4)
        label_color: Color for labels [B,G,R] (default: [255, 255, 255])
        
        # Bodypart tracking
        bodypart: Name of bodypart to track (default: 'headstage')
        
        # Layout settings
        graph_width_ratio: Width ratio for graph (0.0-1.0, default: 0.5)
        padding: Pixels of padding around graph (default: 20)
        
    Returns:
        Combined frame with video+mapping on left and graph on right
    """
    # Get configuration
    mapping_alpha = config.get('mapping_alpha', 0.3)
    mapping_file = config.get('mapping_file')
    show_node_boxes = config.get('show_node_boxes', True)
    show_edge_boxes = config.get('show_edge_boxes', False)
    node_box_color = tuple(config.get('node_box_color', [0, 255, 0]))
    edge_box_color = tuple(config.get('edge_box_color', [255, 0, 0]))
    
    graph_bg_color = tuple(config.get('graph_bg_color', [40, 40, 40]))
    node_radius = config.get('node_radius', 8)
    node_color = tuple(config.get('node_color', [200, 200, 200]))
    current_node_color = tuple(config.get('current_node_color', [0, 255, 0]))
    edge_color = tuple(config.get('edge_color', [100, 100, 100]))
    edge_thickness = config.get('edge_thickness', 1)
    show_node_labels = config.get('show_node_labels', True)
    label_font_scale = config.get('label_font_scale', 0.4)
    label_color = tuple(config.get('label_color', [0, 0, 255]))  # Red in BGR
    
    bodypart = config.get('bodypart', 'headstage')
    graph_width_ratio = config.get('graph_width_ratio', 0.5)
    padding = config.get('padding', 20)
    
    # Load mapping data if not cached
    if '_mapping_data' not in config:
        if mapping_file is None:
            # Try to get from shared resources
            mapping_file = shared_resources.get('graph_mapping_file')
        
        if mapping_file is None:
            return frame
        
        mapping_path = Path(mapping_file)
        if not mapping_path.exists():
            return frame
        
        with open(mapping_path, 'rb') as f:
            config['_mapping_data'] = pickle.load(f)
    
    mapping_data = config['_mapping_data']
    
    # Get calibration matrix from shared resources
    calibration_matrix = shared_resources.get('calibration_matrix')
    if calibration_matrix is None:
        # Try to load from mapping data
        calibration_matrix = mapping_data.get('transform_matrix')
    
    # Invert calibration matrix to go from map space to video space
    # (mapping polygons are in calibrated/map space, need to transform to video)
    inv_calibration_matrix = None
    if calibration_matrix is not None:
        try:
            inv_calibration_matrix = np.linalg.inv(calibration_matrix)
        except np.linalg.LinAlgError:
            # Matrix is singular, can't invert
            inv_calibration_matrix = None
    
    # Extract nodes and edges from mapping
    if 'mappings' not in mapping_data:
        return frame
    
    mappings = mapping_data['mappings']
    nodes_dict = mappings.get('nodes', {})
    edges_dict = mappings.get('edges', {})
    
    # LEFT SIDE: Video with mapping overlay
    left_frame = frame.copy()
    
    # Create overlay for mapping
    overlay = left_frame.copy()
    
    # Transform and draw node bounding boxes
    if show_node_boxes and inv_calibration_matrix is not None:
        for node_id, polygon_list in nodes_dict.items():
            if polygon_list and len(polygon_list) > 0:
                polygon = polygon_list[0]
                if len(polygon) >= 4:
                    # Extract x and y coordinates
                    x_coords = np.array([p[0] for p in polygon], dtype=float)
                    y_coords = np.array([p[1] for p in polygon], dtype=float)
                    
                    # Apply inverse calibration transformation (map -> video)
                    transformed_x, transformed_y = transform_coordinates(
                        x_coords, y_coords, inv_calibration_matrix
                    )
                    
                    # Filter out NaN values
                    valid_mask = ~(np.isnan(transformed_x) | np.isnan(transformed_y))
                    if np.any(valid_mask):
                        transformed_points = np.column_stack([
                            transformed_x[valid_mask],
                            transformed_y[valid_mask]
                        ]).astype(np.int32)
                        
                        cv2.polylines(overlay, [transformed_points], True, node_box_color, 2)
    elif show_node_boxes and inv_calibration_matrix is None:
        # No transformation available, draw original coordinates
        for node_id, polygon_list in nodes_dict.items():
            if polygon_list and len(polygon_list) > 0:
                polygon = polygon_list[0]
                if len(polygon) >= 4:
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.polylines(overlay, [pts], True, node_box_color, 2)
    
    # Transform and draw edge bounding boxes
    if show_edge_boxes and inv_calibration_matrix is not None:
        for edge_id, polygon_list in edges_dict.items():
            if polygon_list and len(polygon_list) > 0:
                polygon = polygon_list[0]
                if len(polygon) >= 4:
                    # Extract x and y coordinates
                    x_coords = np.array([p[0] for p in polygon], dtype=float)
                    y_coords = np.array([p[1] for p in polygon], dtype=float)
                    
                    # Apply inverse calibration transformation (map -> video)
                    transformed_x, transformed_y = transform_coordinates(
                        x_coords, y_coords, inv_calibration_matrix
                    )
                    
                    # Filter out NaN values
                    valid_mask = ~(np.isnan(transformed_x) | np.isnan(transformed_y))
                    if np.any(valid_mask):
                        transformed_points = np.column_stack([
                            transformed_x[valid_mask],
                            transformed_y[valid_mask]
                        ]).astype(np.int32)
                        
                        cv2.polylines(overlay, [transformed_points], True, edge_box_color, 1)
    elif show_edge_boxes and inv_calibration_matrix is None:
        # No transformation available, draw original coordinates
        for edge_id, polygon_list in edges_dict.items():
            if polygon_list and len(polygon_list) > 0:
                polygon = polygon_list[0]
                if len(polygon) >= 4:
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.polylines(overlay, [pts], True, edge_box_color, 1)
    
    # Blend overlay with original frame
    left_frame = cv2.addWeighted(left_frame, 1 - mapping_alpha, overlay, mapping_alpha, 0)
    
    # RIGHT SIDE: Graph with current position
    frame_h, frame_w = frame.shape[:2]
    graph_width = int(frame_w * graph_width_ratio)
    
    # Create graph canvas
    graph_frame = np.full((frame_h, graph_width, 3), graph_bg_color, dtype=np.uint8)
    
    # Calculate node positions (centers of bounding boxes)
    node_positions = {}
    for node_id, polygon_list in nodes_dict.items():
        if polygon_list and len(polygon_list) > 0:
            polygon = polygon_list[0]
            if len(polygon) >= 2:
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)
                node_positions[node_id] = (center_x, center_y)
    
    # Calculate scaling and translation to fit graph in canvas
    if node_positions:
        all_x = [pos[0] for pos in node_positions.values()]
        all_y = [pos[1] for pos in node_positions.values()]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Calculate scale to fit in canvas with padding
        available_width = graph_width - 2 * padding
        available_height = frame_h - 2 * padding
        
        data_width = max_x - min_x
        data_height = max_y - min_y
        
        if data_width > 0 and data_height > 0:
            scale_x = available_width / data_width
            scale_y = available_height / data_height
            scale = min(scale_x, scale_y)
            
            # Calculate offset to center the graph
            scaled_width = data_width * scale
            scaled_height = data_height * scale
            offset_x = padding + (available_width - scaled_width) / 2
            offset_y = padding + (available_height - scaled_height) / 2
            
            # Transform node positions to graph canvas coordinates
            graph_node_positions = {}
            for node_id, (x, y) in node_positions.items():
                new_x = int((x - min_x) * scale + offset_x)
                new_y = int((y - min_y) * scale + offset_y)
                graph_node_positions[node_id] = (new_x, new_y)
            
            # Get current node and edge for bodypart
            current_node = None
            current_edge = None
            current_edge_nodes = None
            
            node_col = f'{bodypart}_graph_node'
            edge_col = f'{bodypart}_graph_edge'
            
            if node_col in frame_data.index:
                current_node = frame_data[node_col]
                if pd.isna(current_node):
                    current_node = None
                else:
                    current_node = str(current_node)  # Ensure string
            
            if edge_col in frame_data.index:
                current_edge = frame_data[edge_col]
                if pd.isna(current_edge):
                    current_edge = None
                else:
                    # Handle different edge formats: tuple, string, etc.
                    if isinstance(current_edge, tuple) and len(current_edge) == 2:
                        current_edge_nodes = (str(current_edge[0]), str(current_edge[1]))
                    elif isinstance(current_edge, str):
                        current_edge_str = str(current_edge)
                        # Try to parse tuple string like "('node1', 'node2')"
                        if current_edge_str.startswith('(') and current_edge_str.endswith(')'):
                            # Remove parentheses and quotes, split by comma
                            cleaned = current_edge_str.strip('()').replace("'", "").replace('"', '')
                            parts = [p.strip() for p in cleaned.split(',')]
                            if len(parts) == 2:
                                current_edge_nodes = (parts[0], parts[1])
                        else:
                            # Try underscore-separated format
                            parts = current_edge_str.split('_')
                            if len(parts) >= 2:
                                current_edge_nodes = (parts[0], parts[1])
            
            # Draw edges first (so nodes appear on top)
            for edge_key in edges_dict.keys():
                nodes = edge_key.split('_')
                if len(nodes) == 2:
                    from_node, to_node = nodes
                    if from_node in graph_node_positions and to_node in graph_node_positions:
                        pt1 = graph_node_positions[from_node]
                        pt2 = graph_node_positions[to_node]
                        
                        # Highlight current edge - check if matches (either direction)
                        is_current = False
                        if current_edge_nodes is not None:
                            n1, n2 = current_edge_nodes
                            is_current = ((n1 == from_node and n2 == to_node) or 
                                         (n1 == to_node and n2 == from_node))
                        
                        if is_current:
                            cv2.line(graph_frame, pt1, pt2, current_node_color, edge_thickness + 3)
                        else:
                            cv2.line(graph_frame, pt1, pt2, edge_color, edge_thickness)
            
            # Draw nodes
            for node_id, (x, y) in graph_node_positions.items():
                # Determine color based on whether this is the current node
                color = current_node_color if node_id == current_node else node_color
                cv2.circle(graph_frame, (x, y), node_radius, color, -1)
                
                # Draw node label if enabled
                if show_node_labels:
                    # Get text size for centering
                    (text_w, text_h), baseline = cv2.getTextSize(
                        node_id, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, 1
                    )
                    text_x = x - text_w // 2
                    text_y = y + text_h // 2
                    cv2.putText(
                        graph_frame, node_id, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, 
                        label_color, 1, cv2.LINE_AA
                    )
            
            # Add status text to graph showing current location
            status_y = 30
            
            # Get frame index from frame_data
            if hasattr(frame_data, 'name') and frame_data.name is not None:
                frame_idx = frame_data.name
                cv2.putText(
                    graph_frame, f"Frame: {frame_idx}", 
                    (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1, cv2.LINE_AA
                )
                status_y += 20
            
            if current_node is not None:
                cv2.putText(
                    graph_frame, f"Node: {current_node}", 
                    (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1, cv2.LINE_AA
                )
                status_y += 20
            
            if current_edge_nodes is not None:
                edge_str = f"{current_edge_nodes[0]}_{current_edge_nodes[1]}"
                cv2.putText(
                    graph_frame, f"Edge: {edge_str}", 
                    (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1, cv2.LINE_AA
                )
    
    # Combine left and right frames
    combined_width = frame_w + graph_width
    combined_frame = np.zeros((frame_h, combined_width, 3), dtype=np.uint8)
    combined_frame[:, :frame_w] = left_frame
    combined_frame[:, frame_w:] = graph_frame
    
    return combined_frame
