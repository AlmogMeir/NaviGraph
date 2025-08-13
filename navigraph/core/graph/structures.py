"""Graph structure classes for NaviGraph.

This module provides a wrapper for NetworkX graphs with builder pattern support.
"""

from typing import Dict, Any, List, Tuple, Optional, Set, Union, TYPE_CHECKING
import networkx as nx
import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    from .builders.base import GraphBuilder


class GraphStructure:
    """Wrapper for NetworkX graphs with builder pattern support.
    
    This class provides a consistent interface for graphs created by different builders,
    caching the built graph and providing navigation analysis operations.
    
    Attributes:
        builder: The graph builder instance used to create this structure
        _graph: Cached NetworkX graph (built lazily)
        _visualization: Cached visualization array
    """
    
    def __init__(self, builder: 'GraphBuilder'):
        """Initialize graph structure with a builder.
        
        Args:
            builder: Graph builder instance
        """
        self.builder = builder
        self._graph: Optional[nx.Graph] = None
        self._visualization: Optional[np.ndarray] = None
    
    @property
    def graph(self) -> nx.Graph:
        """Get the NetworkX graph, building it if necessary."""
        if self._graph is None:
            self._graph = self.builder.build_graph()
        return self._graph
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata from the builder."""
        return self.builder.get_metadata()
    
    def get_visualization(self, **kwargs) -> np.ndarray:
        """Get visualization of the graph as an image array.
        
        Args:
            **kwargs: Additional visualization parameters
            
        Returns:
            RGB image array with shape (height, width, 3)
        """
        # Use cached visualization if no parameters changed
        if not kwargs and self._visualization is not None:
            return self._visualization
        
        # Generate new visualization
        visualization = self.builder.get_visualization(**kwargs)
        
        # Cache if using default parameters
        if not kwargs:
            self._visualization = visualization
        
        return visualization
    
    # Basic graph properties
    
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
    
    # Node operations
    
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
    
    def get_degree(self, node: Any) -> int:
        """Get degree of a node.
        
        Args:
            node: Node identifier
            
        Returns:
            Node degree
        """
        return self.graph.degree(node)
    
    # Path finding
    
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
    
    # Graph structure analysis
    
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
    
    def get_subgraph(self, nodes: List[Any]) -> nx.Graph:
        """Create subgraph containing only specified nodes.
        
        Args:
            nodes: List of node identifiers to include
            
        Returns:
            NetworkX subgraph (view)
        """
        return self.graph.subgraph(nodes)
    
    # Export method
    
    def save(self, filepath: Union[str, Path], format: str = 'graphml'):
        """Save graph to file in various formats.
        
        Args:
            filepath: Path to save file
            format: File format ('graphml', 'gexf', 'gml', 'adjlist', 'edgelist')
        """
        filepath = Path(filepath)
        
        if format == 'graphml':
            nx.write_graphml(self.graph, str(filepath))
        elif format == 'gexf':
            nx.write_gexf(self.graph, str(filepath))
        elif format == 'gml':
            nx.write_gml(self.graph, str(filepath))
        elif format == 'adjlist':
            nx.write_adjlist(self.graph, str(filepath))
        elif format == 'edgelist':
            nx.write_edgelist(self.graph, str(filepath))
        else:
            raise ValueError(f"Unsupported format: {format}. Supported: graphml, gexf, gml, adjlist, edgelist")