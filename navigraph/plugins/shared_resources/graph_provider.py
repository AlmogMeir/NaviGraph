"""Graph provider shared resource for NaviGraph.

Provides graph instances and navigation utilities for spatial analysis.
"""

from typing import Dict, Any

# Import from local graph module
from .graph_module import Graph
from .graph_dictionary import graph_dict

from ...core.interfaces import ISharedResource, Logger
from ...core.exceptions import NavigraphError
from ...core.base_plugin import BasePlugin
from ...core.registry import register_shared_resource_plugin


@register_shared_resource_plugin("graph_provider")
class GraphProviderResource(BasePlugin, ISharedResource):
    """Provides graph instance and spatial navigation utilities.
    
    This shared resource initializes the graph instance needed for spatial
    navigation analysis. It wraps the existing Graph class while integrating
    with the new plugin architecture.
    """
    
    def __init__(self, config = None, logger_instance = None):
        """Initialize empty graph provider."""
        super().__init__(config, logger_instance)
        self._graph_instance = None
        self._graph_config = None
        self._resource_initialized = False
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance = None):
        """Factory method to create graph provider from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        instance.initialize_resource(config, instance.logger)
        return instance
    
    def _validate_config(self) -> None:
        """Validate graph provider configuration."""
        # graph settings are optional - using defaults if not provided
        pass
    
    @property
    def resource_type(self) -> str:
        """Type identifier for this resource."""
        return "graph"
    
    def initialize_resource(
        self, 
        resource_config: Dict[str, Any], 
        logger: Logger
    ) -> None:
        """Initialize graph provider with configuration.
        
        Args:
            resource_config: Configuration containing graph settings
            logger: Logger for initialization messages
            
        Raises:
            NavigraphError: If initialization fails
        """
        try:
            self.logger.info("Initializing graph provider resource")
            
            # Create a mock configuration for the Graph class
            # The original Graph class expects an OmegaConf DictConfig
            graph_config = self._create_graph_config(resource_config)
            
            # Initialize graph instance
            self._graph_instance = Graph(graph_config)
            self._graph_config = graph_config
            
            self._resource_initialized = True
            
            # Get graph statistics
            n_nodes = len(self._graph_instance.tree.nodes())
            n_edges = len(self._graph_instance.tree.edges())
            height = getattr(graph_config.graph, 'height', 'Unknown')
            
            self.logger.info(
                f"✓ Graph provider initialized: {n_nodes} nodes, {n_edges} edges, "
                f"height={height}"
            )
            
        except Exception as e:
            raise NavigraphError(
                f"Failed to initialize graph provider: {str(e)}"
            ) from e
    
    def cleanup_resource(self, logger: Logger) -> None:
        """Clean up graph provider resources."""
        self.logger.debug("Cleaning up graph provider resource")
        self._graph_instance = None
        self._graph_config = None
        self._resource_initialized = False
    
    def is_initialized(self) -> bool:
        """Check if graph provider is initialized."""
        return self._resource_initialized
    
    def get_required_config_keys(self) -> list:
        """Return required configuration keys."""
        return ['height']  # Minimum required for graph initialization
    
    def get_graph_instance(self) -> Graph:
        """Get the initialized graph instance.
        
        Returns:
            Graph instance for navigation analysis
            
        Raises:
            NavigraphError: If not initialized
        """
        if not self._resource_initialized:
            raise NavigraphError("Graph provider not initialized")
        return self._graph_instance
    
    def get_graph_configuration(self) -> Dict[str, Any]:
        """Get graph configuration.
        
        Returns:
            Dictionary with graph settings
            
        Raises:
            NavigraphError: If not initialized
        """
        if not self._resource_initialized:
            raise NavigraphError("Graph provider not initialized")
        return self._graph_config
    
    def get_tile_dictionary(self) -> Dict[int, Any]:
        """Get the tile-to-graph mapping dictionary.
        
        Returns:
            Dictionary mapping tile IDs to graph positions
        """
        if not self._resource_initialized:
            raise NavigraphError("Graph provider not initialized")
        return graph_dict
    
    def _create_graph_config(self, resource_config: Dict[str, Any]):
        """Create a mock configuration object for the Graph class.
        
        The original Graph class expects an OmegaConf DictConfig, but we'll create
        a simple object that provides the required attributes.
        
        Args:
            resource_config: Resource configuration from session config
            
        Returns:
            Mock configuration object
        """
        class MockGraphConfig:
            def __init__(self, config_dict):
                self.verbose = config_dict.get('verbose', False)
                self._config_dict = config_dict  # Store original dict for .get() method
                
                # Create graph sub-config
                self.graph = MockGraphSubConfig(config_dict)
            
            def get(self, key, default=None):
                """Dictionary-like get method for compatibility."""
                return self._config_dict.get(key, default)
        
        class MockGraphSubConfig:
            def __init__(self, config_dict):
                self.height = config_dict.get('height', 7)  # Default binary tree height
                
                # Create drawing options
                self.draw = MockDrawConfig(config_dict.get('draw', {}))
                self.options = MockOptionsConfig(config_dict.get('options', {}))
        
        class MockDrawConfig:
            def __init__(self, draw_config):
                # Default drawing parameters (preserving existing values)
                self.node_size = draw_config.get('node_size', 300)
                self.font_size = draw_config.get('font_size', 12)
                self.with_labels = draw_config.get('with_labels', True)
        
        class MockOptionsConfig:
            def __init__(self, options_config):
                # Default color options (preserving existing values)
                self.static_node_color = options_config.get('static_node_color', 'lightblue')
                self.static_edge_color = options_config.get('static_edge_color', 'gray')
                self.dynamic_node_color = options_config.get('dynamic_node_color', 'red')
                self.dynamic_edge_color = options_config.get('dynamic_edge_color', 'red')
                self.history_node_color = options_config.get('history_node_color', 'orange')
                self.history_edge_color = options_config.get('history_edge_color', 'orange')
                self.dynamic_reward_node_color = options_config.get('dynamic_reward_node_color', 'green')
                self.dynamic_reward_edge_color = options_config.get('dynamic_reward_edge_color', 'green')
                self.history_reward_node_color = options_config.get('history_reward_node_color', 'darkgreen')
                self.history_reward_edge_color = options_config.get('history_reward_edge_color', 'darkgreen')
                self.edge_width = options_config.get('edge_width', 2)
        
        return MockGraphConfig(resource_config)