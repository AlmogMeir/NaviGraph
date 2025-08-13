"""Registry for graph builders with decorator-based registration."""

from typing import Dict, Type, List, Any
from .base import GraphBuilder


class GraphBuilderRegistry:
    """Registry for graph builder classes."""
    
    def __init__(self):
        self._builders: Dict[str, Type[GraphBuilder]] = {}
    
    def register(self, name: str, builder_class: Type[GraphBuilder]) -> None:
        """Register a graph builder class.
        
        Args:
            name: Unique name for the builder
            builder_class: Builder class to register
        """
        if not issubclass(builder_class, GraphBuilder):
            raise ValueError(f"Builder class must inherit from GraphBuilder")
        
        self._builders[name] = builder_class
    
    def get(self, name: str) -> Type[GraphBuilder]:
        """Get a registered builder class by name.
        
        Args:
            name: Name of the builder
            
        Returns:
            Builder class
            
        Raises:
            KeyError: If builder not found
        """
        if name not in self._builders:
            available = list(self._builders.keys())
            raise KeyError(f"Graph builder '{name}' not found. Available: {available}")
        
        return self._builders[name]
    
    def list_builders(self) -> List[str]:
        """List all registered builder names.
        
        Returns:
            List of builder names
        """
        return list(self._builders.keys())
    
    def get_builder_info(self, name: str) -> Dict[str, Any]:
        """Get information about a builder.
        
        Args:
            name: Name of the builder
            
        Returns:
            Dictionary with builder information
        """
        builder_class = self.get(name)
        
        # Extract constructor parameters
        import inspect
        sig = inspect.signature(builder_class.__init__)
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_info = {
                'type': param.annotation if param.annotation != param.empty else 'Any',
                'required': param.default == param.empty,
                'default': param.default if param.default != param.empty else None
            }
            parameters[param_name] = param_info
        
        return {
            'name': name,
            'class_name': builder_class.__name__,
            'module': builder_class.__module__,
            'docstring': builder_class.__doc__,
            'parameters': parameters
        }


# Global registry instance
_registry = GraphBuilderRegistry()


def register_graph_builder(name: str):
    """Decorator to register graph builder classes.
    
    Args:
        name: Unique name for the builder
        
    Example:
        @register_graph_builder("my_graph")
        class MyGraphBuilder(GraphBuilder):
            def __init__(self, size: int):
                self.size = size
            
            def build_graph(self) -> nx.Graph:
                # Implementation here
                pass
    """
    def decorator(builder_class: Type[GraphBuilder]) -> Type[GraphBuilder]:
        _registry.register(name, builder_class)
        return builder_class
    return decorator


def get_graph_builder(name: str) -> Type[GraphBuilder]:
    """Get a registered graph builder class.
    
    Args:
        name: Name of the builder
        
    Returns:
        Builder class
    """
    return _registry.get(name)


def list_graph_builders() -> List[str]:
    """List all registered graph builder names.
    
    Returns:
        List of builder names
    """
    return _registry.list_builders()


def get_graph_builder_info(name: str) -> Dict[str, Any]:
    """Get information about a graph builder.
    
    Args:
        name: Name of the builder
        
    Returns:
        Dictionary with builder information including parameters
    """
    return _registry.get_builder_info(name)