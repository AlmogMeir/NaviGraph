"""Graph builder functions for NaviGraph.

This module provides various graph construction functions for common
topologies used in spatial navigation experiments.
"""

from typing import List, Tuple, Optional, Dict, Any, Callable
import networkx as nx
import numpy as np
from .structures import GraphStructure


def build_binary_tree(height: int, root_id: int = 0) -> GraphStructure:
    """Build a binary tree graph.
    
    Creates a complete binary tree of specified height. Nodes are labeled
    using a level-based scheme where the first digit indicates the level
    and remaining digits indicate position within that level.
    
    Args:
        height: Height of the binary tree (number of levels)
        root_id: ID for the root node
        
    Returns:
        GraphStructure containing the binary tree
    """
    graph = nx.Graph()
    
    # Build tree level by level
    for level in range(height):
        num_nodes_at_level = 2 ** level
        
        for node_idx in range(num_nodes_at_level):
            # Create node ID (level + position)
            if level == 0:
                node_id = root_id
            else:
                node_id = int(f"{level}{node_idx}")
            
            # Add node with level information
            graph.add_node(node_id, level=level, position_in_level=node_idx)
            
            # Add edges to children (except for leaf nodes)
            if level < height - 1:
                left_child_idx = node_idx * 2
                right_child_idx = node_idx * 2 + 1
                
                left_child_id = int(f"{level + 1}{left_child_idx}")
                right_child_id = int(f"{level + 1}{right_child_idx}")
                
                graph.add_edge(node_id, left_child_id, child_type='left')
                graph.add_edge(node_id, right_child_id, child_type='right')
    
    # Create GraphStructure with hierarchical positions
    structure = GraphStructure(graph, metadata={
        'type': 'binary_tree',
        'height': height,
        'root_id': root_id
    })
    
    # Set hierarchical positions
    positions = {}
    for node in graph.nodes():
        level = graph.nodes[node]['level']
        pos_in_level = graph.nodes[node]['position_in_level']
        
        # Calculate x position to spread nodes evenly at each level
        num_at_level = 2 ** level
        x = (pos_in_level + 0.5) / num_at_level
        y = 1.0 - (level / (height - 1)) if height > 1 else 0.5
        
        positions[node] = (x, y)
    
    structure.set_node_positions(positions)
    
    return structure


def build_dual_root_binary_tree(left_height: int, right_height: int, 
                                left_root_id: int = 0, right_root_id: int = 1) -> GraphStructure:
    """Build a dual-rooted binary tree graph.
    
    Creates two binary trees connected at their root nodes. Each tree can have
    a different height, providing flexibility for asymmetric structures.
    Nodes are labeled using a level-based scheme where the first character indicates
    the tree ('L' or 'R'), followed by level and position information.
    
    Args:
        left_height: Height of the left binary tree (number of levels)
        right_height: Height of the right binary tree (number of levels)
        left_root_id: ID for the left root node (default: 0)
        right_root_id: ID for the right root node (default: 1)
        
    Returns:
        GraphStructure containing the dual-rooted binary tree
        
    Example:
        # Create symmetric dual tree with height 3 on each side
        >>> structure = build_dual_root_binary_tree(3, 3)
        
        # Create asymmetric dual tree with different heights
        >>> structure = build_dual_root_binary_tree(left_height=4, right_height=2)
    """
    graph = nx.Graph()
    
    # Add root nodes
    graph.add_node(left_root_id, level=0, position_in_level=0, tree='left')
    graph.add_node(right_root_id, level=0, position_in_level=0, tree='right')
    
    # Connect the two roots
    graph.add_edge(left_root_id, right_root_id, edge_type='root_connection')
    
    # Helper function to build one side of the tree
    def build_tree_side(height: int, root_id: int, tree_side: str):
        """Build one side of the dual tree."""
        for level in range(1, height):  # Start from 1 since root is already added
            num_nodes_at_level = 2 ** level
            
            for node_idx in range(num_nodes_at_level):
                # Create unique node ID with tree prefix
                node_id = f"{tree_side[0].upper()}{level}{node_idx}"
                
                # Add node with metadata
                graph.add_node(node_id, level=level, position_in_level=node_idx, tree=tree_side)
                
                # Find parent and add edges
                if level == 1:
                    # Connect directly to root
                    parent_id = root_id
                else:
                    # Find parent in previous level
                    parent_idx = node_idx // 2
                    parent_id = f"{tree_side[0].upper()}{level - 1}{parent_idx}"
                
                # Determine if this is left or right child
                child_type = 'left' if node_idx % 2 == 0 else 'right'
                graph.add_edge(parent_id, node_id, child_type=child_type)
    
    # Build both sides of the tree
    build_tree_side(left_height, left_root_id, 'left')
    build_tree_side(right_height, right_root_id, 'right')
    
    # Create GraphStructure with metadata
    structure = GraphStructure(graph, metadata={
        'type': 'dual_root_binary_tree',
        'left_height': left_height,
        'right_height': right_height,
        'left_root_id': left_root_id,
        'right_root_id': right_root_id
    })
    
    # Set positions for visualization
    positions = {}
    max_height = max(left_height, right_height)
    
    # Position root nodes at the center top
    positions[left_root_id] = (0.4, 1.0)
    positions[right_root_id] = (0.6, 1.0)
    
    # Helper function to position nodes in a tree
    def position_tree_side(height: int, root_id: int, tree_side: str, x_offset: float, x_scale: float):
        """Calculate positions for one side of the tree."""
        for level in range(1, height):
            num_at_level = 2 ** level
            
            for node_idx in range(num_at_level):
                node_id = f"{tree_side[0].upper()}{level}{node_idx}"
                
                # Calculate x position with tree-specific offset and scaling
                level_width = x_scale / (num_at_level + 1)
                x = x_offset + (node_idx + 1) * level_width
                
                # Calculate y position (normalized by max height)
                y = 1.0 - (level / (max_height - 1)) if max_height > 1 else 0.5
                
                positions[node_id] = (x, y)
    
    # Position left tree (left half of space)
    if left_height > 1:
        position_tree_side(left_height, left_root_id, 'left', 0.0, 0.5)
    
    # Position right tree (right half of space)
    if right_height > 1:
        position_tree_side(right_height, right_root_id, 'right', 0.5, 0.5)
    
    structure.set_node_positions(positions)
    
    return structure


def build_grid_graph(rows: int, cols: int, connectivity: int = 4,
                     periodic: bool = False) -> GraphStructure:
    """Build a grid graph.
    
    Creates a 2D grid graph with specified connectivity pattern.
    Nodes are labeled as (row, col) tuples.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        connectivity: 4 for von Neumann neighborhood, 8 for Moore neighborhood
        periodic: If True, creates a torus (wraparound edges)
        
    Returns:
        GraphStructure containing the grid graph
    """
    if connectivity == 4:
        graph = nx.grid_2d_graph(rows, cols, periodic=periodic)
    elif connectivity == 8:
        # Start with 4-connected grid
        graph = nx.grid_2d_graph(rows, cols, periodic=periodic)
        
        # Add diagonal connections
        for r in range(rows):
            for c in range(cols):
                # Add diagonal neighbors
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    
                    if periodic:
                        nr = nr % rows
                        nc = nc % cols
                        graph.add_edge((r, c), (nr, nc))
                    elif 0 <= nr < rows and 0 <= nc < cols:
                        graph.add_edge((r, c), (nr, nc))
    else:
        raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")
    
    # Create GraphStructure
    structure = GraphStructure(graph, metadata={
        'type': 'grid',
        'rows': rows,
        'cols': cols,
        'connectivity': connectivity,
        'periodic': periodic
    })
    
    # Set grid positions
    positions = {}
    for node in graph.nodes():
        r, c = node
        x = c / (cols - 1) if cols > 1 else 0.5
        y = 1.0 - (r / (rows - 1)) if rows > 1 else 0.5
        positions[node] = (x, y)
    
    structure.set_node_positions(positions)
    
    return structure


def build_hexagonal_grid(rows: int, cols: int) -> GraphStructure:
    """Build a hexagonal grid graph.
    
    Creates a hexagonal lattice where each node has up to 6 neighbors.
    Nodes are labeled as (row, col) tuples.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        
    Returns:
        GraphStructure containing the hexagonal grid
    """
    graph = nx.hexagonal_lattice_graph(rows, cols)
    
    # Convert node labels to consistent format
    mapping = {}
    for i, node in enumerate(graph.nodes()):
        mapping[node] = i
    
    graph = nx.relabel_nodes(graph, mapping)
    
    # Create GraphStructure
    structure = GraphStructure(graph, metadata={
        'type': 'hexagonal_grid',
        'rows': rows,
        'cols': cols
    })
    
    # Positions are already set by NetworkX for hex lattice
    positions = nx.get_node_attributes(graph, 'pos')
    if positions:
        # Normalize positions to [0, 1] range
        x_vals = [pos[0] for pos in positions.values()]
        y_vals = [pos[1] for pos in positions.values()]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        
        normalized_positions = {}
        for node, (x, y) in positions.items():
            norm_x = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.5
            norm_y = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.5
            normalized_positions[node] = (norm_x, norm_y)
        
        structure.set_node_positions(normalized_positions)
    
    return structure


def build_custom_graph(edges: List[Tuple[Any, Any]], 
                      directed: bool = False,
                      node_attributes: Optional[Dict[Any, Dict[str, Any]]] = None,
                      edge_attributes: Optional[Dict[Tuple[Any, Any], Dict[str, Any]]] = None) -> GraphStructure:
    """Build a custom graph from edge list.
    
    Creates a graph from a list of edges with optional attributes.
    
    Args:
        edges: List of (source, target) tuples
        directed: If True, creates a directed graph
        node_attributes: Optional dictionary of node attributes
        edge_attributes: Optional dictionary of edge attributes
        
    Returns:
        GraphStructure containing the custom graph
    """
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
    
    # Add edges
    graph.add_edges_from(edges)
    
    # Add node attributes if provided
    if node_attributes:
        for node, attrs in node_attributes.items():
            if node in graph:
                graph.nodes[node].update(attrs)
    
    # Add edge attributes if provided
    if edge_attributes:
        for edge, attrs in edge_attributes.items():
            if graph.has_edge(*edge):
                graph.edges[edge].update(attrs)
    
    # Create GraphStructure
    structure = GraphStructure(graph, metadata={
        'type': 'custom',
        'directed': directed,
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges()
    })
    
    return structure


def build_from_adjacency(adj_matrix: np.ndarray, 
                        node_labels: Optional[List[Any]] = None,
                        directed: bool = False) -> GraphStructure:
    """Build graph from adjacency matrix.
    
    Args:
        adj_matrix: Square adjacency matrix
        node_labels: Optional list of node labels (default: 0 to n-1)
        directed: If True, creates a directed graph
        
    Returns:
        GraphStructure built from adjacency matrix
    """
    n = adj_matrix.shape[0]
    if adj_matrix.shape[1] != n:
        raise ValueError("Adjacency matrix must be square")
    
    # Default node labels
    if node_labels is None:
        node_labels = list(range(n))
    elif len(node_labels) != n:
        raise ValueError(f"Number of labels ({len(node_labels)}) must match matrix size ({n})")
    
    # Create appropriate graph type
    if directed:
        graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    else:
        graph = nx.from_numpy_array(adj_matrix)
    
    # Relabel nodes if custom labels provided
    if node_labels != list(range(n)):
        mapping = {i: label for i, label in enumerate(node_labels)}
        graph = nx.relabel_nodes(graph, mapping)
    
    # Create GraphStructure
    structure = GraphStructure(graph, metadata={
        'type': 'from_adjacency',
        'directed': directed
    })
    
    return structure


def build_hierarchical_graph(levels: List[int], 
                           branching_factor: Optional[int] = None) -> GraphStructure:
    """Build a hierarchical graph with variable branching.
    
    Creates a tree where each level can have a different number of nodes.
    
    Args:
        levels: List where each element is the number of nodes at that level
        branching_factor: If provided, connects each node to this many children
                         If None, distributes children evenly
        
    Returns:
        GraphStructure containing the hierarchical graph
    """
    graph = nx.Graph()
    
    # Track nodes at each level
    level_nodes = []
    node_counter = 0
    
    # Create nodes level by level
    for level, num_nodes in enumerate(levels):
        current_level_nodes = []
        for i in range(num_nodes):
            node_id = f"L{level}N{i}"
            graph.add_node(node_id, level=level, index_in_level=i)
            current_level_nodes.append(node_id)
        level_nodes.append(current_level_nodes)
    
    # Connect levels
    for level in range(len(levels) - 1):
        parent_nodes = level_nodes[level]
        child_nodes = level_nodes[level + 1]
        
        if branching_factor is not None:
            # Fixed branching factor
            child_idx = 0
            for parent in parent_nodes:
                for _ in range(min(branching_factor, len(child_nodes) - child_idx)):
                    if child_idx < len(child_nodes):
                        graph.add_edge(parent, child_nodes[child_idx])
                        child_idx += 1
        else:
            # Distribute children evenly
            children_per_parent = len(child_nodes) / len(parent_nodes)
            child_idx = 0
            
            for i, parent in enumerate(parent_nodes):
                # Calculate number of children for this parent
                start_child = int(i * children_per_parent)
                end_child = int((i + 1) * children_per_parent)
                
                for j in range(start_child, min(end_child, len(child_nodes))):
                    graph.add_edge(parent, child_nodes[j])
    
    # Create GraphStructure
    structure = GraphStructure(graph, metadata={
        'type': 'hierarchical',
        'levels': levels,
        'branching_factor': branching_factor
    })
    
    # Set hierarchical positions
    positions = {}
    for level, nodes in enumerate(level_nodes):
        y = 1.0 - (level / (len(levels) - 1)) if len(levels) > 1 else 0.5
        for i, node in enumerate(nodes):
            x = (i + 0.5) / len(nodes) if len(nodes) > 0 else 0.5
            positions[node] = (x, y)
    
    structure.set_node_positions(positions)
    
    return structure


def build_from_function(builder_func: Callable[..., nx.Graph], 
                       *args, **kwargs) -> GraphStructure:
    """Build graph using a custom function.
    
    This allows users to provide their own graph construction function
    that returns a NetworkX graph.
    
    Args:
        builder_func: Function that returns a NetworkX graph
        *args: Positional arguments for builder function
        **kwargs: Keyword arguments for builder function
        
    Returns:
        GraphStructure containing the graph
    """
    # Call user function
    graph = builder_func(*args, **kwargs)
    
    # Validate it's a NetworkX graph
    if not isinstance(graph, nx.Graph):
        raise ValueError(f"Builder function must return a NetworkX graph, got {type(graph)}")
    
    # Create GraphStructure
    structure = GraphStructure(graph, metadata={
        'type': 'custom_function',
        'builder': builder_func.__name__
    })
    
    return structure