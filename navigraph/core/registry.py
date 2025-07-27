"""Simple plugin registry for NaviGraph.

Central registry for all plugins in the NaviGraph system.
"""

from typing import Dict, Type, List, Any, Optional
from loguru import logger

# Type alias for logger  
Logger = type(logger)

from .interfaces import IDataSource, ISharedResource, IAnalyzer, IVisualizer
from .exceptions import NavigraphError


class PluginRegistry:
    """Central registry for all NaviGraph plugins."""
    
    def __init__(self, logger_instance: Optional[Logger] = None):
        """Initialize empty registry with optional logger."""
        self.logger = logger_instance or logger
        
        # Plugin storage by category
        self._data_sources: Dict[str, Type[IDataSource]] = {}
        self._shared_resources: Dict[str, Type[ISharedResource]] = {}
        self._analyzers: Dict[str, Type[IAnalyzer]] = {}
        self._visualizers: Dict[str, Type[IVisualizer]] = {}
        
        self.logger.debug("Initialized empty plugin registry")
    
    def register_data_source_plugin(self, plugin_name: str, plugin_class: Type[IDataSource]) -> None:
        """Register a data source plugin."""
        if not plugin_name or not plugin_class:
            raise NavigraphError(f"Invalid plugin name or class: {plugin_name}")
        
        if plugin_name in self._data_sources:
            self.logger.warning(f"Overriding existing data source plugin: {plugin_name}")
        
        self._data_sources[plugin_name] = plugin_class
        self.logger.info(f"Registered data source plugin: {plugin_name}")
    
    def get_data_source_plugin(self, plugin_name: str) -> Type[IDataSource]:
        """Get data source plugin by name."""
        if plugin_name not in self._data_sources:
            available_plugins = list(self._data_sources.keys())
            raise NavigraphError(
                f"Data source plugin '{plugin_name}' not found. "
                f"Available: {available_plugins}"
            )
        
        return self._data_sources[plugin_name]
    
    def list_all_plugins(self) -> Dict[str, List[str]]:
        """List all registered plugins by category."""
        return {
            'data_sources': list(self._data_sources.keys()),
            'shared_resources': list(self._shared_resources.keys()),
            'analyzers': list(self._analyzers.keys()),
            'visualizers': list(self._visualizers.keys())
        }
    
    
    def register_shared_resource_plugin(self, plugin_name: str, plugin_class: Type[ISharedResource]) -> None:
        """Register a shared resource plugin."""
        if not plugin_name or not plugin_class:
            raise NavigraphError(f"Invalid plugin name or class: {plugin_name}")
        
        if plugin_name in self._shared_resources:
            self.logger.warning(f"Overriding existing shared resource plugin: {plugin_name}")
        
        self._shared_resources[plugin_name] = plugin_class
        self.logger.info(f"Registered shared resource plugin: {plugin_name}")
    
    def get_shared_resource_plugin(self, plugin_name: str) -> Type[ISharedResource]:
        """Get a shared resource plugin by name."""
        if plugin_name not in self._shared_resources:
            available_plugins = list(self._shared_resources.keys())
            raise NavigraphError(
                f"Shared resource plugin '{plugin_name}' not found. "
                f"Available: {available_plugins}"
            )
        
        return self._shared_resources[plugin_name]
    
    def register_analyzer_plugin(self, plugin_name: str, plugin_class: Type[IAnalyzer]) -> None:
        """Register an analyzer plugin."""
        if not plugin_name or not plugin_class:
            raise NavigraphError(f"Invalid plugin name or class: {plugin_name}")
        
        if plugin_name in self._analyzers:
            self.logger.warning(f"Overriding existing analyzer plugin: {plugin_name}")
        
        self._analyzers[plugin_name] = plugin_class
        self.logger.info(f"Registered analyzer plugin: {plugin_name}")
    
    def get_analyzer_plugin(self, plugin_name: str) -> Type[IAnalyzer]:
        """Get an analyzer plugin by name."""
        if plugin_name not in self._analyzers:
            available_plugins = list(self._analyzers.keys())
            raise NavigraphError(
                f"Analyzer plugin '{plugin_name}' not found. "
                f"Available: {available_plugins}"
            )
        
        return self._analyzers[plugin_name]
    
    def register_visualizer_plugin(self, plugin_name: str, plugin_class: Type[IVisualizer]) -> None:
        """Register a visualizer plugin."""
        if not plugin_name or not plugin_class:
            raise NavigraphError(f"Invalid plugin name or class: {plugin_name}")
        
        if plugin_name in self._visualizers:
            self.logger.warning(f"Overriding existing visualizer plugin: {plugin_name}")
        
        self._visualizers[plugin_name] = plugin_class
        self.logger.info(f"Registered visualizer plugin: {plugin_name}")
    
    def get_visualizer_plugin(self, plugin_name: str) -> Type[IVisualizer]:
        """Get a visualizer plugin by name."""
        if plugin_name not in self._visualizers:
            available_plugins = list(self._visualizers.keys())
            raise NavigraphError(
                f"Visualizer plugin '{plugin_name}' not found. "
                f"Available: {available_plugins}"
            )
        
        return self._visualizers[plugin_name]
    
    def list_visualizer_plugins(self) -> List[str]:
        """List all registered visualizer plugins."""
        return list(self._visualizers.keys())


# Decorator functions for easy plugin registration
def register_data_source_plugin(plugin_name: str):
    """Decorator to register data source plugins."""
    def decorator(plugin_class):
        registry.register_data_source_plugin(plugin_name, plugin_class)
        return plugin_class
    return decorator

def register_shared_resource_plugin(plugin_name: str):
    """Decorator to register shared resource plugins."""
    def decorator(plugin_class):
        registry.register_shared_resource_plugin(plugin_name, plugin_class)
        return plugin_class
    return decorator

def register_analyzer_plugin(plugin_name: str):
    """Decorator to register analyzer plugins."""
    def decorator(plugin_class):
        registry.register_analyzer_plugin(plugin_name, plugin_class)
        return plugin_class
    return decorator

def register_visualizer_plugin(plugin_name: str):
    """Decorator to register visualizer plugins."""
    def decorator(plugin_class):
        registry.register_visualizer_plugin(plugin_name, plugin_class)
        return plugin_class
    return decorator

# Global registry instance
registry = PluginRegistry()