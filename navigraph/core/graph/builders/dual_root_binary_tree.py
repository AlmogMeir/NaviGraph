"""Dual-rooted binary tree graph builder."""

import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple, Optional

from .base import GraphBuilder
from .registry import register_graph_builder


@register_graph_builder("dual_root_binary_tree")
class DualRootBinaryTreeBuilder(GraphBuilder):
    """Dual-rooted binary tree graph builder.
    
    Creates two binary trees connected at their root nodes. Each tree can have
    a different height, providing flexibility for asymmetric structures.
    This structure is useful for T-maze variants, choice paradigms, and
    bidirectional navigation experiments.
    
    Node naming convention:
    - Root nodes: 'L0' and 'R0' for left and right roots
    - Child nodes: '[L|R][level][position]' (e.g., 'L12' is left tree, level 1, position 2)
    """
    
    def __init__(self, left_height: int, right_height: int):
        """Initialize dual-rooted binary tree builder.
        
        Args:
            left_height: Height of the left binary tree (must be >= 1)
            right_height: Height of the right binary tree (must be >= 1)
            
        Raises:
            ValueError: If either height < 1
        """
        if left_height < 1 or right_height < 1:
            raise ValueError("Both tree heights must be >= 1")
        
        self.left_height = left_height
        self.right_height = right_height
        self._graph = None  # Cache the graph
    
    def build_graph(self) -> nx.Graph:
        """Build dual-rooted binary tree graph.
        
        Returns:
            NetworkX graph representing a dual-rooted binary tree
        """
        if self._graph is None:
            self._graph = self._build_dual_tree()
        return self._graph
    
    def _build_dual_tree(self, weight=lambda p_node, c_node: 1) -> nx.Graph:
        """Build a dual-rooted binary tree as a NetworkX graph.
        
        Args:
            weight: Function to calculate edge weights (default: constant 1)
            
        Returns:
            NetworkX graph representing the dual-rooted binary tree
        """
        g = nx.Graph()
        max_height = max(self.left_height, self.right_height)
        
        # Add root nodes at the top center
        left_root = 'L0'
        right_root = 'R0'
        
        # Add root nodes without positions (will use hierarchical layout)
        g.add_node(left_root, level=0, tree='left', subset=0)
        g.add_node(right_root, level=0, tree='right', subset=0)
        
        # Connect the two roots
        edge_weight = weight(left_root, right_root)
        g.add_edge(left_root, right_root, weight=edge_weight, edge_type='root_connection')
        
        # Build left tree
        if self.left_height > 1:
            self._build_tree_side(g, left_root, self.left_height, 'L', 
                                 weight_func=weight, tree_side='left')
        
        # Build right tree
        if self.right_height > 1:
            self._build_tree_side(g, right_root, self.right_height, 'R', 
                                 weight_func=weight, tree_side='right')
        
        # Use graphviz hierarchical layout for proper tree visualization
        # This prevents overlaps and creates a clean tree structure
        try:
            pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
        except:
            # Fallback to custom tree layout if graphviz not available
            pos = self._custom_tree_layout(g, max_height)
        
        # Set positions as node attributes
        nx.set_node_attributes(g, pos, 'pos')
        
        return g
    
    def _custom_tree_layout(self, graph: nx.Graph, max_height: int) -> Dict[str, Tuple[float, float]]:
        """Custom tree layout that prevents overlaps.
        
        Args:
            graph: The graph to layout
            max_height: Maximum tree height
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        pos = {}
        
        # Organize nodes by level
        levels = {}
        for node, data in graph.nodes(data=True):
            level = data.get('level', 0)
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        # Position nodes level by level
        for level, nodes in sorted(levels.items()):
            num_nodes = len(nodes)
            # Use exponential spacing to handle large numbers of leaf nodes
            width = max(num_nodes * 2.0, 10.0)
            
            # Sort nodes to keep tree structure organized
            sorted_nodes = sorted(nodes)
            
            for i, node in enumerate(sorted_nodes):
                x = (i - num_nodes / 2) * (width / num_nodes)
                y = (max_height - level) * 3.0  # Vertical spacing
                pos[node] = (x, y)
        
        return pos
    
    def _build_tree_side(self, graph: nx.Graph, root_id: str, height: int, 
                        side_prefix: str, weight_func, tree_side: str) -> None:
        """Build one side of the dual tree.
        
        Args:
            graph: Graph to add nodes and edges to
            root_id: ID of the root node for this side
            height: Height of this side of the tree
            side_prefix: Prefix for node IDs ('L' or 'R')
            weight_func: Function to calculate edge weights
            tree_side: 'left' or 'right' for positioning
        """
        # Build tree from root downward
        for level in range(1, height):
            num_nodes_at_level = 2 ** level
            
            for node_idx in range(num_nodes_at_level):
                current_node = f"{side_prefix}{level}{node_idx}"
                
                # Add node with subset attribute for multipartite layout
                # Subset determines vertical position (level)
                graph.add_node(current_node, level=level, tree=side_prefix.lower(), 
                             subset=level)
                
                # Connect to parent
                if level == 1:
                    parent_node = root_id
                else:
                    parent_idx = node_idx // 2
                    parent_node = f"{side_prefix}{level-1}{parent_idx}"
                
                edge_weight = weight_func(parent_node, current_node)
                child_type = 'left' if node_idx % 2 == 0 else 'right'
                graph.add_edge(parent_node, current_node, weight=edge_weight, 
                             child_type=child_type)
    
    def get_visualization(self, positions: Optional[Dict[Any, Tuple[float, float]]] = None,
                         **kwargs) -> np.ndarray:
        """Generate visualization of the dual-rooted binary tree.
        
        Args:
            positions: Optional node positions. If None, will use built-in positions.
            **kwargs: Additional visualization parameters
            
        Returns:
            RGB image array with shape (height, width, 3)
        """
        graph = self.build_graph()
        
        # Use built-in positions if none provided
        if positions is None:
            positions = nx.get_node_attributes(graph, 'pos')
        
        # Customize visualization for dual tree with much smaller nodes and text
        viz_kwargs = {
            'figsize': (30, 20),
            'node_size': 80,
            'node_color': self._get_node_colors(graph),
            'edge_color': 'gray',
            'width': 0.1,
            'with_labels': True,
            'font_size': 4,
            'font_weight': 'normal'
        }
        viz_kwargs.update(kwargs)
        
        return self._default_visualization(positions, **viz_kwargs)
    
    def _get_node_colors(self, graph: nx.Graph) -> list:
        """Generate color map for nodes based on tree side.
        
        Args:
            graph: The graph to color
            
        Returns:
            List of colors for each node
        """
        colors = []
        for node in graph.nodes():
            tree_side = graph.nodes[node].get('tree', '')
            if tree_side == 'left':
                colors.append('lightblue')
            elif tree_side == 'right':
                colors.append('lightcoral')
            else:
                colors.append('lightgreen')  # For any other nodes
        return colors
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the dual-rooted binary tree.
        
        Returns:
            Dictionary containing tree metadata
        """
        graph = self.build_graph()
        return {
            'builder_type': 'dual_root_binary_tree',
            'parameters': {
                'left_height': self.left_height,
                'right_height': self.right_height
            },
            'type': 'dual_root_binary_tree',
            'left_height': self.left_height,
            'right_height': self.right_height,
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'left_nodes': len([n for n, d in graph.nodes(data=True) if d.get('tree') == 'left']),
            'right_nodes': len([n for n, d in graph.nodes(data=True) if d.get('tree') == 'right']),
            'root_nodes': ['L0', 'R0']
        }
