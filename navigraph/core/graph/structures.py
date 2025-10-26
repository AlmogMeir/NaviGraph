"""Graph structure classes for NaviGraph.

This module provides a wrapper for NetworkX graphs with builder pattern support.
"""

from typing import Dict, Any, List, Tuple, Optional, Set, Union, TYPE_CHECKING
import networkx as nx
import numpy as np
from pathlib import Path
import multiprocessing
from functools import partial

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
    
    @classmethod
    def from_config(cls, builder_type: str, config: Dict[str, Any]) -> 'GraphStructure':
        """Create GraphStructure from builder type and configuration.
        
        Args:
            builder_type: Registry name of the builder (e.g., 'binary_tree')
            config: Builder configuration parameters
            
        Returns:
            New GraphStructure instance with configured builder
        """
        from .builders.registry import get_graph_builder
        
        # Get builder class from registry using the global function
        builder_class = get_graph_builder(builder_type)
        
        # Create builder instance with config
        builder = builder_class(**config)
        
        # Return new GraphStructure with the builder
        return cls(builder)
    
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

    # Random walks

    def random_walk(self,
                   start_node: Any,
                   max_steps: Optional[int] = None,
                   target_node: Optional[Any] = None,
                   terminate_on_target: bool = True,
                   backtrack_prob: float = 0.0,
                   use_edge_weights: bool = False,
                   seed: Optional[int] = None) -> List[Any]:
        """Generate a single random walk on the graph.

        Performs a random walk starting from the given node, with configurable
        backtracking probability and optional target-directed termination.

        Args:
            start_node: Starting node for the walk
            max_steps: Maximum number of steps. Required if target_node is None
            target_node: Optional target node. Walk terminates when reached (if terminate_on_target=True)
            terminate_on_target: If True, stop immediately upon reaching target.
                               If False, continue until max_steps even after reaching target
            backtrack_prob: Probability of moving back to previous node.
                          -1 = uniform distribution (treat previous node like any other neighbor)
                          0.0 = no backtracking (default - previous node excluded)
                          0.0-1.0 = explicit backtrack probability (remaining distributed to others)
                          1.0 = always backtrack if possible
            use_edge_weights: If True, use edge weights for transition probabilities.
                            If False, use uniform probability among neighbors
            seed: Random seed for reproducibility

        Returns:
            List of nodes representing the walk path (including start_node)

        Raises:
            ValueError: If start_node not in graph, or if neither max_steps nor target_node provided
            ValueError: If backtrack_prob not in [0, 1]

        Examples:
            >>> # Fixed-length walk from root (no backtracking)
            >>> graph = GraphStructure.from_config('binary_tree', {'height': 7})
            >>> path = graph.random_walk(start_node=0, max_steps=10)
            >>> len(path)
            11  # 10 steps = 11 nodes (including start)

            >>> # Uniform random walk (backtracking allowed, equal probability to all neighbors)
            >>> path = graph.random_walk(
            ...     start_node=0,
            ...     max_steps=20,
            ...     backtrack_prob=-1  # Treat previous node like any other neighbor
            ... )

            >>> # Walk to target without backtracking
            >>> path = graph.random_walk(
            ...     start_node=0,
            ...     target_node=127,
            ...     max_steps=50,
            ...     backtrack_prob=0.0
            ... )

            >>> # Walk with 30% backtracking probability
            >>> path = graph.random_walk(
            ...     start_node=0,
            ...     max_steps=20,
            ...     backtrack_prob=0.3,
            ...     seed=42  # Reproducible
            ... )

            >>> # Weighted random walk (if graph has edge weights)
            >>> path = graph.random_walk(
            ...     start_node=0,
            ...     max_steps=15,
            ...     use_edge_weights=True
            ... )
        """
        # Input validation
        if not self.has_node(start_node):
            raise ValueError(f"Start node {start_node} not in graph")

        if max_steps is None and target_node is None:
            raise ValueError("Must provide either max_steps or target_node")

        if backtrack_prob != -1 and not (0.0 <= backtrack_prob <= 1.0):
            raise ValueError(f"backtrack_prob must be -1 or in [0, 1], got {backtrack_prob}")

        if target_node is not None and not self.has_node(target_node):
            raise ValueError(f"Target node {target_node} not in graph")

        # Initialize random number generator
        rng = np.random.RandomState(seed)

        # Initialize walk
        path = [start_node]
        current_node = start_node
        previous_node = None
        steps = 0

        # Fast path 1: Uniform distribution with backtracking allowed
        # This is the simplest case - treat all neighbors equally
        if backtrack_prob == -1 and not use_edge_weights:
            import random
            if seed is not None:
                random.seed(seed)

            while True:
                # Check termination conditions
                if target_node is not None and current_node == target_node and terminate_on_target:
                    break
                if max_steps is not None and steps >= max_steps:
                    break

                # Get all neighbors (including where we came from)
                neighbors = list(self.graph.neighbors(current_node))

                if len(neighbors) == 0:
                    break

                # Simple uniform random choice among ALL neighbors (FAST PATH)
                next_node = random.choice(neighbors)

                # Update path
                previous_node = current_node
                current_node = next_node
                path.append(current_node)
                steps += 1

            return path

        # Fast path 2: Simple uniform random walk without backtracking or weights
        # This optimization provides 15-20x speedup for the most common case
        if backtrack_prob == 0.0 and not use_edge_weights:
            # Use Python's random.choice directly (much faster than numpy for small lists)
            import random
            if seed is not None:
                random.seed(seed)

            while True:
                # Check termination conditions
                if target_node is not None and current_node == target_node and terminate_on_target:
                    break
                if max_steps is not None and steps >= max_steps:
                    break

                # Get neighbors and handle backtracking by removal
                neighbors = list(self.graph.neighbors(current_node))

                if len(neighbors) == 0:
                    break

                # Remove previous node to prevent backtracking
                if previous_node is not None and previous_node in neighbors and len(neighbors) > 1:
                    neighbors.remove(previous_node)

                # Simple uniform random choice (FAST PATH)
                next_node = random.choice(neighbors)

                # Update path
                previous_node = current_node
                current_node = next_node
                path.append(current_node)
                steps += 1

            return path

        # Slow path: Full-featured walk with probabilistic backtracking and/or edge weights
        # Perform walk
        while True:
            # Check termination conditions
            if target_node is not None and current_node == target_node and terminate_on_target:
                break

            if max_steps is not None and steps >= max_steps:
                break

            # Get neighbors
            neighbors = list(self.graph.neighbors(current_node))

            # If no neighbors (dead end), terminate
            if len(neighbors) == 0:
                break

            # Calculate transition probabilities
            if use_edge_weights:
                # Get edge weights for all neighbors
                weights = []
                for neighbor in neighbors:
                    edge_data = self.graph.get_edge_data(current_node, neighbor)
                    weight = edge_data.get('weight', 1.0) if edge_data else 1.0
                    weights.append(weight)
                weights = np.array(weights)
            else:
                # Uniform weights
                weights = np.ones(len(neighbors))

            # Adjust for backtracking
            if previous_node is not None and previous_node in neighbors:
                prev_idx = neighbors.index(previous_node)

                # Set backtracking probability
                base_weights = weights.copy()
                base_weights[prev_idx] = 0  # Temporarily remove previous node

                # Normalize non-backtrack weights
                if base_weights.sum() > 0:
                    non_backtrack_probs = base_weights / base_weights.sum()
                    non_backtrack_probs *= (1.0 - backtrack_prob)

                    # Set probabilities
                    probs = non_backtrack_probs
                    probs[prev_idx] = backtrack_prob
                else:
                    # Only neighbor is previous node
                    probs = np.array([1.0])
            else:
                # No backtracking possible, normalize weights
                probs = weights / weights.sum()

            # Sample next node
            next_node = neighbors[rng.choice(len(neighbors), p=probs)]

            # Update path
            previous_node = current_node
            current_node = next_node
            path.append(current_node)
            steps += 1

        return path

    @staticmethod
    def _walk_worker(args: Tuple) -> List[Any]:
        """Worker function for parallel random walk execution.

        This static method enables multiprocessing by being picklable.

        Args:
            args: Tuple of (graph_structure, start_node, params_dict, walk_seed)

        Returns:
            Single walk path as list of nodes
        """
        graph_structure, start_node, params, walk_seed = args

        return graph_structure.random_walk(
            start_node=start_node,
            max_steps=params.get('max_steps'),
            target_node=params.get('target_node'),
            terminate_on_target=params.get('terminate_on_target', True),
            backtrack_prob=params.get('backtrack_prob', 0.0),
            use_edge_weights=params.get('use_edge_weights', False),
            seed=walk_seed
        )

    def random_walks(self,
                    start_node: Any,
                    n_walks: int = 1,
                    max_steps: Optional[int] = None,
                    target_node: Optional[Any] = None,
                    terminate_on_target: bool = True,
                    backtrack_prob: float = 0.0,
                    use_edge_weights: bool = False,
                    return_stats: bool = False,
                    seed: Optional[int] = None,
                    n_jobs: int = 1) -> Union[List[List[Any]], Tuple[List[List[Any]], Dict[str, Any]]]:
        """Generate multiple random walks on the graph.

        Performs multiple random walks with identical parameters, optionally using
        parallel processing for improved performance on multi-core systems.

        Args:
            start_node: Starting node for all walks
            n_walks: Number of random walks to generate
            max_steps: Maximum number of steps per walk. Required if target_node is None
            target_node: Optional target node. Walks terminate when reached (if terminate_on_target=True)
            terminate_on_target: If True, stop immediately upon reaching target.
                               If False, continue until max_steps even after reaching target
            backtrack_prob: Probability (0.0-1.0) of moving back to previous node.
                          0.0 = no backtracking (default), 1.0 = always backtrack if possible
            use_edge_weights: If True, use edge weights for transition probabilities.
                            If False, use uniform probability among neighbors
            return_stats: If True, return tuple of (paths, statistics_dict).
                        If False, return only paths
            seed: Random seed for reproducibility. Each walk gets a unique derived seed
            n_jobs: Number of parallel processes to use.
                   1 = serial execution (default)
                   -1 = use all available CPU cores
                   n > 1 = use n processes

        Returns:
            List[List[Any]]: List of walk paths (each path is a list of nodes)
            OR
            Tuple[List[List[Any]], Dict]: Paths and statistics if return_stats=True

        Statistics Dictionary (when return_stats=True):
            - 'mean_length': Mean path length across all walks
            - 'median_length': Median path length
            - 'std_length': Standard deviation of path lengths
            - 'min_length': Minimum path length
            - 'max_length': Maximum path length
            - 'success_rate': Fraction of walks reaching target (only if target_node provided)
            - 'successful_walks': Indices of walks that reached target (only if target_node provided)

        Raises:
            ValueError: If invalid parameters provided

        Performance Notes:
            - Multiprocessing (n_jobs > 1) has overhead from process spawning and serialization
            - Generally beneficial for n_walks > 100 or complex graphs
            - For small tasks (n_walks < 100), serial execution (n_jobs=1) is often faster
            - Memory usage scales with n_walks (stores all paths in memory)

        Examples:
            >>> # Basic example: 100 walks of fixed length
            >>> graph = GraphStructure.from_config('binary_tree', {'height': 7})
            >>> paths = graph.random_walks(
            ...     start_node=0,
            ...     n_walks=100,
            ...     max_steps=20
            ... )
            >>> len(paths)
            100

            >>> # Target-directed walks with statistics
            >>> paths, stats = graph.random_walks(
            ...     start_node=0,
            ...     target_node=127,
            ...     n_walks=1000,
            ...     max_steps=50,
            ...     backtrack_prob=0.0,
            ...     return_stats=True
            ... )
            >>> print(f"Success rate: {stats['success_rate']:.1%}")
            >>> print(f"Average path length: {stats['mean_length']:.2f}")

            >>> # Compare with vs without backtracking
            >>> paths_no_back, stats_no = graph.random_walks(
            ...     start_node=0, target_node=127, n_walks=1000,
            ...     max_steps=50, backtrack_prob=0.0, return_stats=True
            ... )
            >>> paths_with_back, stats_with = graph.random_walks(
            ...     start_node=0, target_node=127, n_walks=1000,
            ...     max_steps=50, backtrack_prob=0.5, return_stats=True
            ... )
            >>> print(f"No backtrack: {stats_no['mean_length']:.1f} steps")
            >>> print(f"50% backtrack: {stats_with['mean_length']:.1f} steps")

            >>> # Parallel processing for large-scale simulation
            >>> paths = graph.random_walks(
            ...     start_node=0,
            ...     n_walks=100000,
            ...     max_steps=30,
            ...     n_jobs=-1,  # Use all CPU cores
            ...     seed=42     # Reproducible
            ... )

            >>> # Weighted random walk
            >>> # (assuming graph has edge weights)
            >>> paths = graph.random_walks(
            ...     start_node=0,
            ...     n_walks=500,
            ...     max_steps=25,
            ...     use_edge_weights=True
            ... )

            >>> # Reproducible walks with specific seed
            >>> paths1 = graph.random_walks(start_node=0, n_walks=10, max_steps=5, seed=123)
            >>> paths2 = graph.random_walks(start_node=0, n_walks=10, max_steps=5, seed=123)
            >>> paths1 == paths2  # True - identical walks
        """
        # Input validation
        if n_walks < 1:
            raise ValueError(f"n_walks must be >= 1, got {n_walks}")

        if n_jobs < -1 or n_jobs == 0:
            raise ValueError(f"n_jobs must be -1 or >= 1, got {n_jobs}")

        # Determine number of processes
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        # Generate seeds for each walk (for reproducibility)
        if seed is not None:
            ss = np.random.SeedSequence(seed)
            child_seeds = [s.generate_state(1)[0] for s in ss.spawn(n_walks)]
        else:
            child_seeds = [None] * n_walks

        # Prepare parameters dictionary
        params = {
            'max_steps': max_steps,
            'target_node': target_node,
            'terminate_on_target': terminate_on_target,
            'backtrack_prob': backtrack_prob,
            'use_edge_weights': use_edge_weights
        }

        # Execute walks
        if n_jobs == 1:
            # Serial execution
            paths = []
            for walk_seed in child_seeds:
                path = self.random_walk(
                    start_node=start_node,
                    max_steps=max_steps,
                    target_node=target_node,
                    terminate_on_target=terminate_on_target,
                    backtrack_prob=backtrack_prob,
                    use_edge_weights=use_edge_weights,
                    seed=walk_seed
                )
                paths.append(path)
        else:
            # Parallel execution
            worker_args = [
                (self, start_node, params, walk_seed)
                for walk_seed in child_seeds
            ]

            with multiprocessing.Pool(processes=n_jobs) as pool:
                paths = pool.map(self._walk_worker, worker_args)

        # Compute statistics if requested
        if return_stats:
            path_lengths = np.array([len(path) - 1 for path in paths])  # -1 because length is edges, not nodes

            stats = {
                'mean_length': float(np.mean(path_lengths)),
                'median_length': float(np.median(path_lengths)),
                'std_length': float(np.std(path_lengths)),
                'min_length': int(np.min(path_lengths)),
                'max_length': int(np.max(path_lengths))
            }

            # Add target-specific statistics
            if target_node is not None:
                successful = [i for i, path in enumerate(paths) if path[-1] == target_node]
                stats['success_rate'] = len(successful) / len(paths)
                stats['successful_walks'] = successful

            return paths, stats

        return paths

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