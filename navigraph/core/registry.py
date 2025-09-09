"""Unified registry for NaviGraph plugins, analysis functions, and visualizers.

Single registry to manage all plugin types, analysis functions, and visualizers.
Clean and simple design with validation built-in.
"""

from typing import Dict, Type, Callable, Any, List
import inspect
import importlib
import pkgutil
from pathlib import Path
from loguru import logger

from .navigraph_plugin import NaviGraphPlugin
from .exceptions import NavigraphError


class UnifiedRegistry:
    """Single registry for plugins, analysis functions, and visualizers."""
    
    def __init__(self):
        self.plugins: Dict[str, Type[NaviGraphPlugin]] = {}
        self.visualizers: Dict[str, Callable] = {}
        self.session_metrics: Dict[str, Callable] = {}
        self.cross_session_metrics: Dict[str, Callable] = {}
        self.auto_discovered = False
    
    def _ensure_discovered(self):
        """Ensure auto-discovery has run (lazy loading)."""
        if not self.auto_discovered:
            try:
                self.discover_and_register()
            except Exception as e:
                logger.warning(f"Auto-discovery failed: {e}")
    
    def register_plugin(self, name: str, plugin_class: Type[NaviGraphPlugin]) -> None:
        """Register a plugin (data source type).
        
        Args:
            name: Plugin name for registry lookup
            plugin_class: Class inheriting from NaviGraphPlugin
        """
        if not name or not plugin_class:
            raise NavigraphError(f"Invalid plugin name or class: {name}")
        
        if not issubclass(plugin_class, NaviGraphPlugin):
            raise NavigraphError(f"Plugin must inherit from NaviGraphPlugin: {plugin_class}")
        
        if name in self.plugins:
            logger.warning(f"Overriding existing plugin: {name}")
        
        self.plugins[name] = plugin_class
        logger.debug(f"Registered plugin: {name}")
    
    
    def register_visualizer(self, name: str, func: Callable) -> None:
        """Register a visualizer function with validation.
        
        Args:
            name: Visualizer name for registry lookup  
            func: Function with signature (frame, frame_data, shared_resources, **config) -> frame
        
        Raises:
            ValueError: If function signature doesn't match requirements
        """
        # Validate function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        required = ['frame', 'frame_data', 'shared_resources']
        if not all(p in params for p in required):
            raise ValueError(
                f"Visualizer '{name}' must have parameters: "
                f"frame, frame_data, shared_resources, **config. Got: {params}"
            )
        
        self.visualizers[name] = func
        logger.debug(f"Registered visualizer: {name}")
    
    def register_session_metric(self, name: str, func: Callable) -> None:
        """Register a session-level metric function.
        
        Args:
            name: Metric function name for registry lookup
            func: Function with signature (session, **args) -> value
        
        Raises:
            ValueError: If function signature doesn't match requirements
        """
        # Validate function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        if not params or params[0] != 'session':
            raise ValueError(
                f"Session metric '{name}' must have 'session' as first parameter. Got: {params}"
            )
        
        self.session_metrics[name] = func
        logger.debug(f"Registered session metric: {name}")
    
    def register_cross_session_metric(self, name: str, func: Callable) -> None:
        """Register a cross-session metric function.
        
        Args:
            name: Metric function name for registry lookup  
            func: Function with signature (sessions, **args) -> value
        
        Raises:
            ValueError: If function signature doesn't match requirements
        """
        # Validate function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        if not params or params[0] != 'sessions':
            raise ValueError(
                f"Cross-session metric '{name}' must have 'sessions' as first parameter. Got: {params}"
            )
        
        self.cross_session_metrics[name] = func
        logger.debug(f"Registered cross-session metric: {name}")
    
    def get_plugin(self, name: str) -> Type[NaviGraphPlugin]:
        """Get plugin class by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin class
            
        Raises:
            NavigraphError: If plugin not found
        """
        self._ensure_discovered()
        if name not in self.plugins:
            raise NavigraphError(f"Plugin '{name}' not found. Available: {list(self.plugins.keys())}")
        return self.plugins[name]
    
    
    def get_visualizer(self, name: str) -> Callable:
        """Get visualizer function by name.
        
        Args:
            name: Visualizer name
            
        Returns:
            Visualizer function
            
        Raises:
            NavigraphError: If visualizer not found
        """
        self._ensure_discovered()
        if name not in self.visualizers:
            raise NavigraphError(f"Visualizer '{name}' not found. Available: {list(self.visualizers.keys())}")
        return self.visualizers[name]
    
    def get_session_metric(self, name: str) -> Callable:
        """Get session metric function by name.
        
        Args:
            name: Session metric name
            
        Returns:
            Session metric function
            
        Raises:
            NavigraphError: If metric not found
        """
        self._ensure_discovered()
        if name not in self.session_metrics:
            raise NavigraphError(f"Session metric '{name}' not found. Available: {list(self.session_metrics.keys())}")
        return self.session_metrics[name]
    
    def get_cross_session_metric(self, name: str) -> Callable:
        """Get cross-session metric function by name.
        
        Args:
            name: Cross-session metric name
            
        Returns:
            Cross-session metric function
            
        Raises:
            NavigraphError: If metric not found
        """
        self._ensure_discovered()
        if name not in self.cross_session_metrics:
            raise NavigraphError(f"Cross-session metric '{name}' not found. Available: {list(self.cross_session_metrics.keys())}")
        return self.cross_session_metrics[name]
    
    def list_all(self) -> Dict[str, list]:
        """List all registered components.
        
        Returns:
            Dict with keys 'plugins', 'session_metrics', 'cross_session_metrics', 'visualizers'
        """
        return {
            'plugins': list(self.plugins.keys()),
            'session_metrics': list(self.session_metrics.keys()),
            'cross_session_metrics': list(self.cross_session_metrics.keys()),
            'visualizers': list(self.visualizers.keys())
        }
    
    def discover_and_register(self) -> Dict[str, int]:
        """Auto-discover and register all components from standard locations.
        
        Searches in:
        - navigraph.plugins.* for plugins
        - navigraph.analysis.session.* for session analysis functions  
        - navigraph.analysis.cross_session.* for cross-session analysis functions
        - navigraph.visualizers.* for visualizer functions
        
        Returns:
            Dict with counts of discovered components
        """
        if self.auto_discovered:
            logger.debug("Auto-discovery already completed, skipping")
            return {'plugins': 0, 'session_metrics': 0, 'cross_session_metrics': 0, 'visualizers': 0}
        
        initial_counts = {
            'plugins': len(self.plugins),
            'session_metrics': len(self.session_metrics),
            'cross_session_metrics': len(self.cross_session_metrics), 
            'visualizers': len(self.visualizers)
        }
        
        # Define search paths
        search_paths = {
            'plugins': ['navigraph.plugins'],
            'session_metrics': ['navigraph.analysis.session'],
            'cross_session_metrics': ['navigraph.analysis.cross_session'],
            'visualizers': ['navigraph.visualizers']
        }
        
        logger.debug("Starting auto-discovery of components...")
        
        # Discover each type
        for component_type, packages in search_paths.items():
            for package_name in packages:
                try:
                    discovered = self._discover_in_package(package_name)
                    if discovered > 0:
                        logger.debug(f"Auto-discovered {discovered} modules in {package_name}")
                except ImportError:
                    logger.debug(f"Package {package_name} not found or empty")
                except Exception as e:
                    logger.warning(f"Error discovering in {package_name}: {e}")
        
        # Calculate what was newly discovered
        final_counts = {
            'plugins': len(self.plugins) - initial_counts['plugins'],
            'session_metrics': len(self.session_metrics) - initial_counts['session_metrics'],
            'cross_session_metrics': len(self.cross_session_metrics) - initial_counts['cross_session_metrics'],
            'visualizers': len(self.visualizers) - initial_counts['visualizers']
        }
        
        total_discovered = sum(final_counts.values())
        if total_discovered > 0:
            logger.info(f"Discovered {total_discovered} components: "
                       f"{final_counts['plugins']} plugins, "
                       f"{final_counts['session_metrics']} metrics, "
                       f"{final_counts['visualizers']} visualizers")
        
        self.auto_discovered = True
        return final_counts
    
    def _discover_in_package(self, package_name: str) -> int:
        """Discover and import all modules in a package.
        
        Args:
            package_name: Full package name (e.g., 'navigraph.plugins')
            
        Returns:
            Number of modules discovered and imported
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            return 0
        
        if not hasattr(package, '__path__'):
            # Not a package, try importing as module
            return 0
        
        discovered_count = 0
        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            # Skip __pycache__ and other non-python modules
            if module_name.startswith('_'):
                continue
                
            full_module_name = f"{package_name}.{module_name}"
            
            try:
                # Import the module - this triggers decorator registration
                importlib.import_module(full_module_name)
                discovered_count += 1
                logger.debug(f"  ✓ Imported: {full_module_name}")
                
            except Exception as e:
                logger.warning(f"  ✗ Failed to import {full_module_name}: {e}")
        
        return discovered_count
    
    # Temporary compatibility methods (will be removed after migration)
    def get_plugin_class(self, name: str) -> Type[NaviGraphPlugin]:
        """Compatibility alias for get_plugin."""
        return self.get_plugin(name)
    
    def get_data_source_plugin(self, name: str) -> Type[NaviGraphPlugin]:
        """Compatibility method for old data source plugin getter."""
        return self.get_plugin(name)
    
    def register_data_source_plugin(self, name: str, plugin_class: Type[NaviGraphPlugin]) -> None:
        """Compatibility method for old data source registration."""
        self.register_plugin(name, plugin_class)
    
    def get_analyzer_plugin(self, name: str) -> Type[NaviGraphPlugin]:
        """Compatibility method - analyzers are now analysis functions."""
        # This will fail since analyzers are not plugins anymore
        raise NavigraphError(f"Analyzers are now analysis functions. Use get_analysis('{name}') instead")
    
    def get_visualizer_plugin(self, name: str) -> Type[NaviGraphPlugin]:
        """Compatibility method - visualizers are now functions."""
        # This will fail since visualizers are not plugins anymore  
        raise NavigraphError(f"Visualizers are now functions. Use get_visualizer('{name}') instead")
    
    def list_all_plugins(self) -> Dict[str, list]:
        """List all registered components by category."""
        self._ensure_discovered()
        return {
            'plugins': list(self.plugins.keys()),
            'session_metrics': list(self.session_metrics.keys()),
            'cross_session_metrics': list(self.cross_session_metrics.keys()),
            'visualizers': list(self.visualizers.keys())
        }


# Global registry instance
registry = UnifiedRegistry()


# Decorator functions
def register_session_metric(name: str):
    """Decorator to register session-level metric functions.
    
    Args:
        name: Metric name for registry lookup
        
    Example:
        @register_session_metric('time_a_to_b')
        def time_a_to_b(session, a, b, **kwargs):
            return result
    """
    def decorator(func):
        registry.register_session_metric(name, func)
        return func
    return decorator


def register_cross_session_metric(name: str):
    """Decorator to register cross-session metric functions.
    
    Args:
        name: Metric name for registry lookup
        
    Example:
        @register_cross_session_metric('learning_progression')
        def learning_progression(sessions, metric_name, **kwargs):
            return result
    """
    def decorator(func):
        registry.register_cross_session_metric(name, func)
        return func
    return decorator

# Decorator functions
def register_plugin(name: str):
    """Decorator to register plugins.
    
    Usage:
        @register_plugin("pose_tracking")
        class PoseTrackingPlugin(NaviGraphPlugin):
            ...
    """
    def decorator(cls):
        registry.register_plugin(name, cls)
        return cls
    return decorator


def register_analysis(name: str):
    """Decorator to register analysis functions.
    
    Usage:
        @register_analysis("spatial_metrics")
        def analyze_spatial_metrics(dataframe, shared_resources, **config):
            ...
    """
    def decorator(func):
        registry.register_analysis(name, func)
        return func
    return decorator


def register_visualizer(name: str):
    """Decorator to register visualizer functions.
    
    Usage:
        @register_visualizer("trajectory")
        def visualize_trajectory(frame, frame_data, shared_resources, **config):
            ...
    """
    def decorator(func):
        registry.register_visualizer(name, func)
        return func
    return decorator


# Compatibility decorator for plugins
def register_data_source_plugin(name: str):
    """Decorator to register data source plugins."""
    def decorator(plugin_class):
        registry.register_plugin(name, plugin_class)
        return plugin_class
    return decorator


# Auto-discovery is now lazy - only happens when components are actually needed
# This prevents discovery logs from showing during --help or other non-functional commands