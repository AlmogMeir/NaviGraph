"""Binary tree graph builder."""

import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple, Optional

from .base import GraphBuilder
from .registry import register_graph_builder


@register_graph_builder("binary_tree")
class BinaryTreeBuilder(GraphBuilder):
    """Binary tree graph builder.
    
    Creates a balanced binary tree with configurable height.
    Node naming follows the convention: level_digit + node_id_within_level.
    """
    
    def __init__(self, height: int):
        """Initialize binary tree builder.
        
        Args:
            height: Height of the binary tree (must be >= 1)
            
        Raises:
            ValueError: If height < 1
        """
        if height < 1:
            raise ValueError("Tree height must be >= 1")
        
        self.height = height
        self._graph = None  # Cache the graph
    
    def build_graph(self) -> nx.Graph:
        """Build binary tree graph.
        
        Returns:
            NetworkX graph representing a binary tree
        """
        if self._graph is None:
            self._graph = self._build_binary_tree(self.height)
        return self._graph
    
    def _build_binary_tree(self, height: int, weight=lambda p_node, c_node: 1) -> nx.Graph:
        """Build a binary tree as a NetworkX graph.
        
        Node naming convention: 2-digit number where first digit is level
        (starting from 0) and second digit is node ID within that level.
        
        Args:
            height: Height of the binary tree
            weight: Function to calculate edge weights (default: constant 1)
            
        Returns:
            NetworkX graph representing the binary tree
        """
        g = nx.Graph()
        x_pos = range(2**height)
        get_x_pos = lambda x: np.convolve(x, np.ones(2), 'valid')/2
        
        for level in range(height)[::-1]:
            num_nodes_at_level = 2**level
            
            # Set x position per level
            if level == height - 1:
                x_pos = np.arange(2 ** height)
            else:
                x_pos = get_x_pos(x_pos)[::2]  # rolling mean with step size
            
            # Add nodes and edges with child nodes
            for level_node_id in range(num_nodes_at_level):
                current_node = int(str(level) + str(level_node_id))
                g.add_node(current_node, pos=(x_pos[level_node_id], height - level))
                
                if level == height - 1:
                    continue
                
                left_most_child_id = level_node_id * 2
                right_most_child_id = (level_node_id * 2) + 1
                
                left_most_child = int(str(level + 1) + str(left_most_child_id))
                right_most_child = int(str(level + 1) + str(right_most_child_id))
                
                g.add_edge(current_node, left_most_child, weight=weight(current_node, left_most_child))
                g.add_edge(current_node, right_most_child, weight=weight(current_node, right_most_child))
        
        return g
    
    def get_visualization(self, positions: Optional[Dict[Any, Tuple[float, float]]] = None, 
                         **kwargs) -> np.ndarray:
        """Generate binary tree visualization using hierarchical layout.
        
        Args:
            positions: Node positions. If None, uses the tree's inherent positions
            **kwargs: Additional visualization parameters
            
        Returns:
            RGB image array with shape (height, width, 3)
        """
        # Build the graph to get its positions
        graph = self.build_graph()
        
        # Use the tree's built-in positions if none provided
        if positions is None:
            positions = nx.get_node_attributes(graph, 'pos')
        
        # Use enhanced styling for binary trees
        tree_kwargs = {
            'figsize': kwargs.get('figsize', (35, 20)),  # Match original size
            'node_size': kwargs.get('node_size', 1000),
            'node_color': kwargs.get('node_color', '#C9D6E8'),
            'edge_color': kwargs.get('edge_color', 'black'),
            'with_labels': kwargs.get('with_labels', True),
            'font_size': kwargs.get('font_size', 15),
            'font_weight': kwargs.get('font_weight', 'bold')
        }
        
        # Override with any provided kwargs
        tree_kwargs.update(kwargs)
        
        return self._binary_tree_visualization(positions, **tree_kwargs)
    
    def _binary_tree_visualization(self, positions: Dict[Any, Tuple[float, float]],
                                  figsize: Tuple[int, int] = (35, 20),
                                  node_size: int = 1000,
                                  node_color: str = '#C9D6E8',
                                  edge_color: str = 'black',
                                  width: float = 1.0,
                                  with_labels: bool = True,
                                  font_size: int = 15,
                                  font_weight: str = 'bold',
                                  **kwargs) -> np.ndarray:
        """Custom visualization for binary tree with proper aspect ratio.
        
        Returns:
            RGB image array with shape (height, width, 3)
        """
        # Import matplotlib only when needed
        import matplotlib
        import matplotlib.pyplot as plt
        
        # Save current backend and set to non-interactive
        original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        
        try:
            graph = self.build_graph()
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=figsize)
            ax.axis('off')
            ax.figure.tight_layout(pad=0)
            ax.margins(0)
            
            # Draw the graph
            nx.draw(graph, pos=positions, ax=ax,
                    node_size=node_size, node_color=node_color,
                    edge_color=edge_color, width=width, with_labels=with_labels,
                    font_size=font_size, font_weight=font_weight)
            
            # Convert to image array
            fig.canvas.draw()
            # Try new method first, fall back to old if not available
            try:
                # matplotlib >= 3.8
                buf = np.asarray(fig.canvas.buffer_rgba())
                buf = buf[:, :, :3]  # Drop alpha channel to get RGB
            except AttributeError:
                # matplotlib < 3.8 fallback
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return buf
            
        finally:
            # Restore original backend
            matplotlib.use(original_backend)