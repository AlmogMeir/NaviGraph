"""Fully connected graph builder for complete graph layouts.

This builder creates a fully connected graph where every node is connected
to every other node, arranged in a circular pattern. Perfect for maze layouts
where all locations are directly accessible from any other location.
"""

import networkx as nx
import numpy as np
from typing import Dict, Any, Tuple, Optional

from .base import GraphBuilder
from .registry import register_graph_builder


@register_graph_builder("fully_connected")
class FullyConnectedGraphBuilder(GraphBuilder):
    """Builder that creates fully connected (complete) graphs.
    
    Creates a graph with:
    - n nodes (0, 1, 2, ..., n-1) arranged in a circle
    - Every node connected to every other node (complete graph)
    - No central node - all nodes are on the perimeter
    
    Perfect for maze layouts where animals can move directly
    between any two locations.
    """
    
    def __init__(self, n_nodes: int = 18):
        """Initialize fully connected graph builder.
        
        Args:
            n_nodes: Number of nodes to create (default 18)
            
        Raises:
            ValueError: If n_nodes is less than 2
        """
        if n_nodes < 2:
            raise ValueError(f"Number of nodes must be at least 2, got {n_nodes}")
        
        self.n_nodes = n_nodes
        self._graph: Optional[nx.Graph] = None
    
    def build_graph(self) -> nx.Graph:
        """Build and return the fully connected graph.
        
        Returns:
            NetworkX graph with complete topology (all nodes connected)
        """
        if self._graph is not None:
            return self._graph
        
        # Create complete graph with n nodes
        self._graph = nx.complete_graph(self.n_nodes)
        
        # Add node attributes for clarity
        for node in self._graph.nodes():
            self._graph.nodes[node]['node_type'] = 'peripheral'
        
        return self._graph
    
    def get_visualization(self, positions: Optional[Dict[Any, Tuple[float, float]]] = None,
                         **kwargs) -> np.ndarray:
        """Generate custom fully connected visualization.
        
        Args:
            positions: Ignored - uses custom circular layout
            **kwargs: Additional visualization parameters
            
        Returns:
            RGB image array with custom circular layout
        """
        # Override default parameters for fully connected visualization
        fc_params = {
            'figsize': kwargs.get('figsize', (12, 12)),
            'node_size': kwargs.get('node_size', 800),
            'node_color': kwargs.get('node_color', 'lightblue'),
            'edge_color': kwargs.get('edge_color', 'lightgray'),  # Lighter for dense edges
            'width': kwargs.get('width', 0.8),  # Thinner edges for clarity
            'with_labels': kwargs.get('with_labels', True),
            'font_size': kwargs.get('font_size', 10),
            'font_weight': 'normal',
            'font_color': kwargs.get('font_color', 'black'),
            'font_family': kwargs.get('font_family', 'sans-serif'),
        }
        
        # Create custom positions for circular layout
        circular_positions = self._create_circular_positions()
        
        return self._default_visualization(
            positions=circular_positions,
            **fc_params
        )
    
    def _create_circular_positions(self) -> Dict[int, Tuple[float, float]]:
        """Create positions for circular layout.
        
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        positions = {}
        
        # All nodes arranged in circle
        radius = 1.0
        for i in range(self.n_nodes):
            # Angle for this node (starting from top, going clockwise)
            angle = 2 * np.pi * i / self.n_nodes - np.pi/2
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions[i] = (x, y)
        
        return positions