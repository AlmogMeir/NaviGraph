"""Star graph builder for circular maze layouts.

This builder creates a star-shaped graph with one central node
connected to multiple peripheral nodes arranged in a circle.
Perfect for circular mazes with a center region and peripheral holes.
"""

import networkx as nx
import numpy as np
from typing import Dict, Any, Tuple, Optional

from .base import GraphBuilder
from .registry import register_graph_builder


@register_graph_builder("star")
class StarGraphBuilder(GraphBuilder):
    """Builder that creates star-shaped graphs.
    
    Creates a graph with:
    - 1 central node (node 0) at the center
    - n peripheral nodes (nodes 1, 2, ..., n) arranged in a circle
    - Edges connecting each peripheral node to the center
    
    Perfect for circular maze layouts where animals start at center
    and can move to any of the peripheral locations.
    """
    
    def __init__(self, n_peripheral_nodes: int = 18, connect_periphery: bool = False):
        """Initialize star graph builder.
        
        Args:
            n_peripheral_nodes: Number of nodes around the periphery (default 18)
            connect_periphery: Whether to add edges between adjacent peripheral nodes (default False)
            
        Raises:
            ValueError: If n_peripheral_nodes is less than 1
        """
        if n_peripheral_nodes < 1:
            raise ValueError(f"Number of peripheral nodes must be at least 1, got {n_peripheral_nodes}")
        
        self.n_peripheral_nodes = n_peripheral_nodes
        self.connect_periphery = connect_periphery
        self._graph: Optional[nx.Graph] = None
    
    def build_graph(self) -> nx.Graph:
        """Build and return the star-shaped graph.
        
        Returns:
            NetworkX graph with star topology
        """
        if self._graph is not None:
            return self._graph
        
        # Create graph
        self._graph = nx.Graph()
        
        # Add central node
        center_node = 0
        self._graph.add_node(center_node, node_type='center')
        
        # Add peripheral nodes and connect to center
        for i in range(1, self.n_peripheral_nodes + 1):
            self._graph.add_node(i, node_type='peripheral')
            self._graph.add_edge(center_node, i)
        
        # Optionally connect peripheral nodes to each other (ring)
        if self.connect_periphery:
            for i in range(1, self.n_peripheral_nodes + 1):
                next_node = i + 1 if i < self.n_peripheral_nodes else 1
                self._graph.add_edge(i, next_node)
        
        return self._graph
    
    def get_visualization(self, positions: Optional[Dict[Any, Tuple[float, float]]] = None,
                         **kwargs) -> np.ndarray:
        """Generate custom star visualization.
        
        Args:
            positions: Ignored - uses custom circular layout
            **kwargs: Additional visualization parameters
            
        Returns:
            RGB image array with custom star layout
        """
        # Override default parameters for star visualization
        star_params = {
            'figsize': kwargs.get('figsize', (10, 10)),
            'node_size': self._get_node_sizes(),
            'node_color': self._get_node_colors(),
            'edge_color': kwargs.get('edge_color', 'gray'),
            'width': kwargs.get('width', 2.0),
            'with_labels': kwargs.get('with_labels', True),
            'font_size': kwargs.get('font_size', 12),
            'font_weight': 'bold',
        }
        
        # Create custom positions for star layout
        star_positions = self._create_star_positions()
        
        return self._default_visualization(
            positions=star_positions,
            **star_params
        )
    
    def _create_star_positions(self) -> Dict[int, Tuple[float, float]]:
        """Create positions for star layout.
        
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        positions = {}
        
        # Center node at origin
        positions[0] = (0.0, 0.0)
        
        # Peripheral nodes arranged in circle
        radius = 1.0
        for i in range(1, self.n_peripheral_nodes + 1):
            # Angle for this node (starting from top, going clockwise)
            angle = 2 * np.pi * (i - 1) / self.n_peripheral_nodes - np.pi/2
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions[i] = (x, y)
        
        return positions
    
    def _get_node_sizes(self) -> list:
        """Get node sizes with larger center node.
        
        Returns:
            List of node sizes
        """
        graph = self.build_graph()
        sizes = []
        
        for node in graph.nodes():
            if node == 0:  # Center node - extra large
                sizes.append(1200)
            else:  # Peripheral nodes - large
                sizes.append(600)
        
        return sizes
    
    def _get_node_colors(self) -> list:
        """Get node colors - all nodes same color.
        
        Returns:
            List of node colors
        """
        graph = self.build_graph()
        colors = []
        
        # All nodes same color
        for node in graph.nodes():
            colors.append('lightblue')
        
        return colors