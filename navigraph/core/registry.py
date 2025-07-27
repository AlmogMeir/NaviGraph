"""Plugin registry system for NaviGraph.

This module provides the central registry for all plugins in the NaviGraph system.
It handles plugin registration, validation, discovery, and retrieval with
comprehensive error handling and user-friendly error messages.

Design principles:
- Clear error messages with specific guidance
- Comprehensive validation of plugin interfaces
- Automatic plugin discovery from plugin directories
- Type safety with proper type hints
"""

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, Type, List, Any, Optional, Set
from loguru import logger

# Type alias for logger  
Logger = type(logger)

from .interfaces import (
    IDataSource, ISharedResource, IAnalyzer, IVisualizer, IGraphProvider,
    NavigraphPluginError
)


class PluginRegistrationError(NavigraphPluginError):
    """Raised when plugin registration fails."""
    pass


class PluginNotFoundError(NavigraphPluginError):
    """Raised when requested plugin is not found."""
    pass


class PluginValidationError(NavigraphPluginError):
    """Raised when plugin validation fails."""
    pass


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
        self._visualizers: Dict[str, Type[IVisualizer]] = {}
        self._graph_providers: Dict[str, Type[IGraphProvider]] = {}
        
        self.logger.debug("Initialized empty plugin registry")
    
    def register_data_source_plugin(self, plugin_name: str, plugin_class: Type[IDataSource]) -> None:
        """Register a data source plugin with comprehensive validation."""
        self._validate_plugin_name(plugin_name)
        self._validate_data_source_interface(plugin_name, plugin_class)
        
        if plugin_name in self._data_sources:
            self.logger.warning(f"Overriding existing data source plugin: {plugin_name}")
        
        self._data_sources[plugin_name] = plugin_class
        self.logger.info(f"Registered data source plugin: {plugin_name}")
    
    def get_data_source_plugin(self, plugin_name: str) -> Type[IDataSource]:
        """Get data source plugin by name with helpful error messages."""
        if plugin_name not in self._data_sources:
            available_plugins = list(self._data_sources.keys())
            raise PluginNotFoundError(
                f"Data source plugin '{plugin_name}' not found. "
                f"Available data source plugins: {available_plugins}. "
                f"Make sure the plugin is registered and imported correctly."
            )
        
        return self._data_sources[plugin_name]
    
    def list_all_plugins(self) -> Dict[str, List[str]]:
        """List all registered plugins by category."""
        return {
            'data_sources': list(self._data_sources.keys()),
            'shared_resources': list(self._shared_resources.keys()),
            'analyzers': list(self._analyzers.keys()),
            'visualizers': list(self._visualizers.keys()),
            'graph_providers': list(self._graph_providers.keys())
        }
    
    def _validate_plugin_name(self, plugin_name: str) -> None:
        """Validate plugin name is non-empty and reasonable."""
        if not plugin_name or not isinstance(plugin_name, str):
            raise PluginRegistrationError("Plugin name must be a non-empty string")
        
        if not plugin_name.replace('_', '').replace('-', '').isalnum():
            raise PluginRegistrationError(
                f"Plugin name '{plugin_name}' contains invalid characters. "
                f"Use only alphanumeric characters, underscores, and hyphens."
            )
    
    def register_shared_resource_plugin(self, plugin_name: str, plugin_class: Type[ISharedResource]) -> None:
        """Register a shared resource plugin.
        
        Args:
            plugin_name: Unique name for the shared resource plugin
            plugin_class: Class implementing ISharedResource interface
            
        Raises:
            PluginRegistrationError: If registration fails
        """
        self._validate_plugin_name(plugin_name)
        self._validate_shared_resource_interface(plugin_name, plugin_class)
        
        if plugin_name in self._shared_resources:
            self.logger.warning(f"Overriding existing shared resource plugin: {plugin_name}")
        
        self._shared_resources[plugin_name] = plugin_class
        self.logger.info(f"Registered shared resource plugin: {plugin_name}")
    
    def get_shared_resource_plugin(self, plugin_name: str) -> Type[ISharedResource]:
        """Get a shared resource plugin by name.
        
        Args:
            plugin_name: Name of the shared resource plugin
            
        Returns:
            Shared resource plugin class
            
        Raises:
            PluginNotFoundError: If plugin not found
        """
        if plugin_name not in self._shared_resources:
            available_plugins = list(self._shared_resources.keys())
            raise PluginNotFoundError(
                f"Shared resource plugin '{plugin_name}' not found. "
                f"Available shared resource plugins: {available_plugins}"
            )
        
        return self._shared_resources[plugin_name]
    
    def register_analyzer_plugin(self, plugin_name: str, plugin_class: Type[IAnalyzer]) -> None:
        """Register an analyzer plugin.
        
        Args:
            plugin_name: Unique name for the analyzer plugin
            plugin_class: Class implementing IAnalyzer interface
            
        Raises:
            PluginRegistrationError: If registration fails
        """
        self._validate_plugin_name(plugin_name)
        self._validate_analyzer_interface(plugin_name, plugin_class)
        
        if plugin_name in self._analyzers:
            self.logger.warning(f"Overriding existing analyzer plugin: {plugin_name}")
        
        self._analyzers[plugin_name] = plugin_class
        self.logger.info(f"Registered analyzer plugin: {plugin_name}")
    
    def get_analyzer_plugin(self, plugin_name: str) -> Type[IAnalyzer]:
        """Get an analyzer plugin by name.
        
        Args:
            plugin_name: Name of the analyzer plugin
            
        Returns:
            Analyzer plugin class
            
        Raises:
            PluginNotFoundError: If plugin not found
        """
        if plugin_name not in self._analyzers:
            available_plugins = list(self._analyzers.keys())
            raise PluginNotFoundError(
                f"Analyzer plugin '{plugin_name}' not found. "
                f"Available analyzer plugins: {available_plugins}"
            )
        
        return self._analyzers[plugin_name]
    
    def register_visualizer_plugin(self, plugin_name: str, plugin_class: Type[IVisualizer]) -> None:
        """Register a visualizer plugin.
        
        Args:
            plugin_name: Unique name for the visualizer plugin
            plugin_class: Class implementing IVisualizer interface
            
        Raises:
            PluginRegistrationError: If registration fails
        """
        self._validate_plugin_name(plugin_name)
        self._validate_visualizer_interface(plugin_name, plugin_class)
        
        if plugin_name in self._visualizers:
            self.logger.warning(f"Overriding existing visualizer plugin: {plugin_name}")
        
        self._visualizers[plugin_name] = plugin_class
        self.logger.info(f"Registered visualizer plugin: {plugin_name}")
    
    def get_visualizer_plugin(self, plugin_name: str) -> Type[IVisualizer]:
        """Get a visualizer plugin by name.
        
        Args:
            plugin_name: Name of the visualizer plugin
            
        Returns:
            Visualizer plugin class
            
        Raises:
            PluginNotFoundError: If plugin not found
        """
        if plugin_name not in self._visualizers:
            available_plugins = list(self._visualizers.keys())
            raise PluginNotFoundError(
                f"Visualizer plugin '{plugin_name}' not found. "
                f"Available visualizer plugins: {available_plugins}"
            )
        
        return self._visualizers[plugin_name]
    
    def list_visualizer_plugins(self) -> List[str]:
        """List all registered visualizer plugins.
        
        Returns:
            List of visualizer plugin names
        """
        return list(self._visualizers.keys())

    def _validate_data_source_interface(self, plugin_name: str, plugin_class: Type) -> None:
        """Validate that plugin class properly implements IDataSource interface."""
        required_methods = {
            'integrate_data_into_session': 'Main integration method missing',
            'validate_session_prerequisites': 'Prerequisites validation method missing',
            'get_provided_column_names': 'Column names method missing'
        }
        
        self._validate_interface_methods(plugin_name, plugin_class, required_methods, "IDataSource")
    
    def _validate_shared_resource_interface(self, plugin_name: str, plugin_class: Type) -> None:
        """Validate that plugin class properly implements ISharedResource interface."""
        required_methods = {
            'initialize_resource': 'Resource initialization method missing',
            'cleanup_resource': 'Resource cleanup method missing',
            'is_initialized': 'Initialization check method missing',
            'get_required_config_keys': 'Required config keys method missing'
        }
        
        self._validate_interface_methods(plugin_name, plugin_class, required_methods, "ISharedResource")
        
        # Validate it's actually a subclass of ISharedResource
        if not issubclass(plugin_class, ISharedResource):
            raise PluginValidationError(
                f"Plugin '{plugin_name}' must inherit from ISharedResource. "
                f"Make sure your class definition includes: class {plugin_class.__name__}(ISharedResource)"
            )
    
    def _validate_analyzer_interface(self, plugin_name: str, plugin_class: Type) -> None:
        """Validate that plugin class properly implements IAnalyzer interface."""
        required_methods = {
            'analyze_session': 'Session analysis method missing',
            'analyze_cross_session': 'Cross-session analysis method missing'
        }
        
        self._validate_interface_methods(plugin_name, plugin_class, required_methods, "IAnalyzer")
        
        # Validate it's actually a subclass of IAnalyzer
        if not issubclass(plugin_class, IAnalyzer):
            raise PluginValidationError(
                f"Plugin '{plugin_name}' must inherit from IAnalyzer. "
                f"Make sure your class definition includes: class {plugin_class.__name__}(IAnalyzer)"
            )
    
    def _validate_visualizer_interface(self, plugin_name: str, plugin_class: Type) -> None:
        """Validate that plugin class properly implements IVisualizer interface."""
        required_methods = {
            'visualize': 'Visualization method missing',
            'supported_formats': 'Supported formats property missing'
        }
        
        self._validate_interface_methods(plugin_name, plugin_class, required_methods, "IVisualizer")
        
        # Validate it's actually a subclass of IVisualizer
        if not issubclass(plugin_class, IVisualizer):
            raise PluginValidationError(
                f"Plugin '{plugin_name}' must inherit from IVisualizer. "
                f"Make sure your class definition includes: class {plugin_class.__name__}(IVisualizer)"
            )
    
    def _validate_interface_methods(self, plugin_name: str, plugin_class: Type, 
                                   required_methods: Dict[str, str], interface_name: str) -> None:
        """Generic method to validate required methods exist on plugin class."""
        missing_methods = []
        
        for method_name, error_msg in required_methods.items():
            if not hasattr(plugin_class, method_name):
                missing_methods.append(f"  - {method_name}: {error_msg}")
        
        if missing_methods:
            missing_methods_str = "\n".join(missing_methods)
            raise PluginValidationError(
                f"Plugin '{plugin_name}' missing required {interface_name} methods:\n"
                f"{missing_methods_str}\n"
                f"Make sure your class implements all abstract methods from {interface_name}."
            )


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


# Global registry instance for the application  
registry = PluginRegistry()