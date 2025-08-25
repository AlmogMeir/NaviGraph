"""Random graph builder for testing and examples."""

import networkx as nx
from typing import Optional

from .base import GraphBuilder
from .registry import register_graph_builder


@register_graph_builder("random")
class RandomGraphBuilder(GraphBuilder):
    """Builder that creates random graphs using NetworkX algorithms.
    
    Supports various random graph models for testing file loading,
    GUI visualization, and general graph analysis.
    """
    
    def __init__(self, 
                 graph_type: str = 'erdos_renyi',
                 n_nodes: int = 20,
                 edge_probability: float = 0.15,
                 m_edges: int = 2,
                 k_neighbors: int = 4,
                 p_rewire: float = 0.1,
                 seed: Optional[int] = 42):
        """Initialize random graph builder.
        
        Args:
            graph_type: Type of random graph ('erdos_renyi', 'barabasi_albert', 'watts_strogatz')
            n_nodes: Number of nodes in the graph
            edge_probability: Edge probability for Erdős–Rényi graphs
            m_edges: Number of edges for Barabási–Albert graphs
            k_neighbors: Each node connects to k nearest neighbors in Watts-Strogatz
            p_rewire: Probability of rewiring edges in Watts-Strogatz
            seed: Random seed for reproducibility (None for no seed)
            
        Raises:
            ValueError: If graph_type is not supported
        """
        supported_types = ['erdos_renyi', 'barabasi_albert', 'watts_strogatz']
        if graph_type not in supported_types:
            raise ValueError(f"Unsupported graph type: {graph_type}. Supported: {supported_types}")
        
        if n_nodes < 1:
            raise ValueError(f"Number of nodes must be positive, got {n_nodes}")
        
        self.graph_type = graph_type
        self.n_nodes = n_nodes
        self.edge_probability = edge_probability
        self.m_edges = min(m_edges, n_nodes - 1)  # Ensure valid for Barabási-Albert
        self.k_neighbors = min(k_neighbors, n_nodes - 1)  # Ensure valid for Watts-Strogatz
        self.p_rewire = p_rewire
        self.seed = seed
        self._graph: Optional[nx.Graph] = None
    
    def build_graph(self) -> nx.Graph:
        """Generate and return the random graph.
        
        Returns:
            NetworkX graph generated using specified random model
        """
        if self._graph is not None:
            return self._graph
        
        if self.graph_type == 'erdos_renyi':
            self._graph = nx.erdos_renyi_graph(
                n=self.n_nodes,
                p=self.edge_probability,
                seed=self.seed
            )
        elif self.graph_type == 'barabasi_albert':
            self._graph = nx.barabasi_albert_graph(
                n=self.n_nodes,
                m=self.m_edges,
                seed=self.seed
            )
        elif self.graph_type == 'watts_strogatz':
            self._graph = nx.watts_strogatz_graph(
                n=self.n_nodes,
                k=self.k_neighbors,
                p=self.p_rewire,
                seed=self.seed
            )
        
        return self._graph