"""Graph structure classes for NaviGraph.

This module provides the base classes and implementations for navigation graphs,
supporting arbitrary graph topologies through NetworkX.
"""

from typing import Dict, Any, List, Tuple, Optional, Set, Union
from abc import ABC, abstractmethod
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class GraphStructure:
    """Base class for navigation graphs.
    
    This class wraps a NetworkX graph and provides common operations
    for spatial navigation analysis. It supports any graph topology
    that can be represented in NetworkX.
    
    Attributes:
        graph: NetworkX graph object
        metadata: Dictionary of graph-level metadata
        node_positions: Dictionary mapping node IDs to (x, y) positions for visualization
    """
    
    def __init__(self, graph: nx.Graph, metadata: Optional[Dict[str, Any]] = None):
        """Initialize graph structure.
        
        Args:
            graph: NetworkX graph object
            metadata: Optional metadata about the graph
        """
        self.graph = graph
        self.metadata = metadata or {}
        self.node_positions = {}
        
        # Generate default positions if not provided
        if not self.node_positions:
            self._generate_default_positions()
    
    def _generate_default_positions(self):
        """Generate default node positions using spring layout."""
        if len(self.graph.nodes()) > 0:
            self.node_positions = nx.spring_layout(self.graph)
    
    @property
    def num_nodes(self) -> int:
        """Get number of nodes in the graph."""
        return self.graph.number_of_nodes()
    
    @property
    def num_edges(self) -> int:
        """Get number of edges in the graph."""
        return self.graph.number_of_edges()
    
    @property
    def nodes(self) -> List[Any]:
        """Get list of all nodes."""
        return list(self.graph.nodes())
    
    @property
    def edges(self) -> List[Tuple[Any, Any]]:
        """Get list of all edges."""
        return list(self.graph.edges())
    
    def get_node_data(self, node: Any) -> Dict[str, Any]:
        """Get data associated with a node.
        
        Args:
            node: Node identifier
            
        Returns:
            Dictionary of node attributes
        """
        return dict(self.graph.nodes[node])
    
    def set_node_data(self, node: Any, **attributes):
        """Set attributes for a node.
        
        Args:
            node: Node identifier
            **attributes: Key-value pairs to set as node attributes
        """
        self.graph.nodes[node].update(attributes)
    
    def get_neighbors(self, node: Any) -> List[Any]:
        """Get neighboring nodes.
        
        Args:
            node: Node identifier
            
        Returns:
            List of neighboring node identifiers
        """
        return list(self.graph.neighbors(node))
    
    def has_node(self, node: Any) -> bool:
        """Check if node exists in graph.
        
        Args:
            node: Node identifier
            
        Returns:
            True if node exists
        """
        return node in self.graph
    
    def has_edge(self, node1: Any, node2: Any) -> bool:
        """Check if edge exists between two nodes.
        
        Args:
            node1: First node identifier
            node2: Second node identifier
            
        Returns:
            True if edge exists
        """
        return self.graph.has_edge(node1, node2)
    
    def get_shortest_path(self, source: Any, target: Any, 
                         weight: Optional[str] = None) -> List[Any]:
        """Calculate shortest path between two nodes.
        
        Args:
            source: Source node identifier
            target: Target node identifier
            weight: Optional edge attribute to use as weight
            
        Returns:
            List of nodes in shortest path, empty if no path exists
        """
        try:
            return nx.shortest_path(self.graph, source, target, weight=weight)
        except nx.NetworkXNoPath:
            return []
    
    def get_shortest_path_length(self, source: Any, target: Any,
                                weight: Optional[str] = None) -> float:
        """Calculate shortest path length between two nodes.
        
        Args:
            source: Source node identifier
            target: Target node identifier
            weight: Optional edge attribute to use as weight
            
        Returns:
            Path length, or infinity if no path exists
        """
        try:
            return nx.shortest_path_length(self.graph, source, target, weight=weight)
        except nx.NetworkXNoPath:
            return float('inf')
    
    def get_all_shortest_paths(self, source: Any, target: Any,
                              weight: Optional[str] = None) -> List[List[Any]]:
        """Get all shortest paths between two nodes.
        
        Args:
            source: Source node identifier
            target: Target node identifier
            weight: Optional edge attribute to use as weight
            
        Returns:
            List of paths (each path is a list of nodes)
        """
        try:
            return list(nx.all_shortest_paths(self.graph, source, target, weight=weight))
        except nx.NetworkXNoPath:
            return []
    
    def get_connected_components(self) -> List[Set[Any]]:
        """Get connected components of the graph.
        
        Returns:
            List of sets, each containing nodes in a connected component
        """
        if self.graph.is_directed():
            return list(nx.weakly_connected_components(self.graph))
        else:
            return list(nx.connected_components(self.graph))
    
    def is_connected(self) -> bool:
        """Check if graph is connected.
        
        Returns:
            True if graph is connected
        """
        if self.graph.is_directed():
            return nx.is_weakly_connected(self.graph)
        else:
            return nx.is_connected(self.graph)
    
    def get_degree(self, node: Any) -> int:
        """Get degree of a node.
        
        Args:
            node: Node identifier
            
        Returns:
            Node degree
        """
        return self.graph.degree(node)
    
    def get_subgraph(self, nodes: List[Any]) -> 'GraphStructure':
        """Create subgraph containing only specified nodes.
        
        Args:
            nodes: List of node identifiers to include
            
        Returns:
            New GraphStructure with subgraph
        """
        subgraph = self.graph.subgraph(nodes).copy()
        sub_positions = {n: self.node_positions.get(n) for n in nodes 
                        if n in self.node_positions}
        
        result = GraphStructure(subgraph, metadata=self.metadata.copy())
        result.node_positions = sub_positions
        return result
    
    def set_node_positions(self, positions: Dict[Any, Tuple[float, float]]):
        """Set node positions for visualization.
        
        Args:
            positions: Dictionary mapping node IDs to (x, y) coordinates
        """
        self.node_positions = positions
    
    def get_node_position(self, node: Any) -> Optional[Tuple[float, float]]:
        """Get position of a specific node.
        
        Args:
            node: Node identifier
            
        Returns:
            (x, y) tuple or None if position not set
        """
        return self.node_positions.get(node)
    
    def visualize(self, figsize: Tuple[int, int] = (10, 8),
                  node_color: str = 'lightblue',
                  edge_color: str = 'gray',
                  node_size: int = 500,
                  with_labels: bool = True,
                  font_size: int = 12,
                  layout: Optional[str] = None,
                  ax: Optional[plt.Axes] = None) -> plt.Figure:
        """Visualize the graph.
        
        Args:
            figsize: Figure size as (width, height)
            node_color: Color for nodes
            edge_color: Color for edges
            node_size: Size of nodes
            with_labels: Whether to show node labels
            font_size: Font size for labels
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', etc.)
            ax: Existing axes to draw on
            
        Returns:
            Matplotlib figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Determine positions
        if layout and len(self.graph.nodes()) > 0:
            if layout == 'spring':
                pos = nx.spring_layout(self.graph)
            elif layout == 'circular':
                pos = nx.circular_layout(self.graph)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(self.graph)
            elif layout == 'spectral':
                pos = nx.spectral_layout(self.graph)
            else:
                pos = self.node_positions
        else:
            pos = self.node_positions
        
        # Draw graph
        nx.draw(self.graph, pos=pos, ax=ax,
                node_color=node_color,
                edge_color=edge_color,
                node_size=node_size,
                with_labels=with_labels,
                font_size=font_size,
                font_weight='bold')
        
        ax.set_title(self.metadata.get('name', 'Graph Structure'))
        ax.axis('off')
        
        return fig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation.
        
        Returns:
            Dictionary with graph data
        """
        return {
            'nodes': list(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True)),
            'metadata': self.metadata,
            'node_positions': self.node_positions,
            'directed': self.graph.is_directed()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphStructure':
        """Create graph from dictionary representation.
        
        Args:
            data: Dictionary with graph data
            
        Returns:
            New GraphStructure instance
        """
        if data.get('directed', False):
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        
        # Add nodes
        graph.add_nodes_from(data['nodes'])
        
        # Add edges
        graph.add_edges_from(data['edges'])
        
        # Create structure
        structure = cls(graph, metadata=data.get('metadata', {}))
        structure.node_positions = data.get('node_positions', {})
        
        return structure
    
    def save(self, filepath: Union[str, Path], format: str = 'graphml'):
        """Save graph to file.
        
        Args:
            filepath: Path to save file
            format: File format ('graphml', 'gexf', 'json')
        """
        filepath = Path(filepath)
        
        if format == 'graphml':
            nx.write_graphml(self.graph, str(filepath))
        elif format == 'gexf':
            nx.write_gexf(self.graph, str(filepath))
        elif format == 'json':
            import json
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path], format: str = 'graphml') -> 'GraphStructure':
        """Load graph from file.
        
        Args:
            filepath: Path to load file
            format: File format ('graphml', 'gexf', 'json')
            
        Returns:
            New GraphStructure instance
        """
        filepath = Path(filepath)
        
        if format == 'graphml':
            graph = nx.read_graphml(str(filepath))
        elif format == 'gexf':
            graph = nx.read_gexf(str(filepath))
        elif format == 'json':
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return cls(graph)