"""Base class for graph builders."""

from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, Tuple


class GraphBuilder(ABC):
    """Abstract base class for graph builders.
    
    This class provides a minimal interface for creating graph structures.
    Subclasses only need to implement the build_graph() method.
    
    The class provides a default visualization that outputs library-independent
    image arrays, which subclasses can override if needed.
    """
    
    # Registry name set by @register_graph_builder decorator
    _registry_name: Optional[str] = None
    
    @abstractmethod
    def build_graph(self) -> nx.Graph:
        """Build and return a NetworkX graph.
        
        Returns:
            NetworkX graph structure
        """
        pass
    
    def get_visualization(self, positions: Optional[Dict[Any, Tuple[float, float]]] = None, 
                         **kwargs) -> np.ndarray:
        """Generate visualization of the graph as an image array.
        
        Args:
            positions: Optional node positions. If None, will use spring layout.
            **kwargs: Additional visualization parameters
            
        Returns:
            RGB image array with shape (height, width, 3)
        """
        return self._default_visualization(positions, **kwargs)
    
    def _default_visualization(self, positions: Optional[Dict[Any, Tuple[float, float]]] = None,
                              figsize: Tuple[int, int] = (12, 8),
                              node_size: int = 300,
                              node_color: str = 'lightblue',
                              edge_color: str = 'gray',
                              width: float = 1.0,
                              with_labels: bool = True,
                              font_size: int = 10,
                              font_weight: str = 'normal',
                              font_color: str = 'black',
                              font_family: str = 'sans-serif',
                              dpi: int = 300,
                              **kwargs) -> np.ndarray:
        """Default graph visualization implementation.
        
        Args:
            positions: Node positions. If None, uses spring layout
            figsize: Figure size in inches
            node_size: Size of nodes (single value or list per node)
            node_color: Color of nodes (single color or list per node)
            edge_color: Color of edges (single color or list per edge)
            width: Width of edges (single value or list per edge)
            with_labels: Whether to show node labels
            font_size: Font size for labels
            font_weight: Font weight for labels ('normal', 'bold', etc.)
            font_color: Color of font for labels
            font_family: Font family for labels ('sans-serif', 'serif', 'monospace', etc.)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            RGB image array with shape (height, width, 3)
        """
        # Import matplotlib only when needed and set backend locally
        import matplotlib
        import matplotlib.pyplot as plt
        
        # Save current backend and set to non-interactive
        original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        
        try:
            graph = self.build_graph()
            
            # Create figure and axis with high DPI for better zoom quality
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.set_aspect('equal')
            
            # Determine positions
            if positions is None:
                positions = nx.spring_layout(graph, seed=42)
            
            # Draw the graph
            nx.draw(graph, pos=positions, ax=ax,
                    node_size=node_size, node_color=node_color,
                    edge_color=edge_color, width=width, with_labels=with_labels,
                    font_size=font_size, font_weight=font_weight, 
                    font_color=font_color, font_family=font_family)
            
            ax.set_title(f"Graph ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")
            
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
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GraphBuilder':
        """Create builder instance from configuration dictionary.
        
        Args:
            config: Configuration parameters for the builder
            
        Returns:
            Configured builder instance
        """
        # Extract constructor parameters using introspection
        import inspect
        sig = inspect.signature(cls.__init__)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            if param_name in config:
                params[param_name] = config[param_name]
        
        return cls(**params)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this builder instance.
        
        Returns:
            Dictionary with builder metadata extracted via introspection
        """
        import inspect
        
        metadata = {
            'builder_type': self._registry_name or self.__class__.__name__,
            'module': self.__class__.__module__,
            'parameters': {}
        }
        
        # Extract instance attributes that don't start with underscore
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                metadata['parameters'][attr_name] = getattr(self, attr_name)
        
        return metadata