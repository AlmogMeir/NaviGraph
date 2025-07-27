"""Base plugin class for NaviGraph plugin system.

This module provides the base class for all plugins, offering common functionality
such as logging, initialization, and configuration management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar
from loguru import logger

# Type alias for logger
Logger = type(logger)

T = TypeVar('T', bound='BasePlugin')


class BasePlugin(ABC):
    """Abstract base class for all NaviGraph plugins.
    
    Provides common functionality:
    - Standardized initialization
    - Configuration validation
    - Logging setup
    - Factory pattern for instantiation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger_instance: Optional[Logger] = None):
        """Initialize base plugin with configuration and logger.
        
        Args:
            config: Plugin configuration dictionary
            logger_instance: Logger instance for plugin use
        """
        self.config = config or {}
        self.logger = logger_instance or logger
        self._initialized = False
        
        # Log initialization
        plugin_name = self.__class__.__name__
        self.logger.debug(f"Initializing {plugin_name} with config: {self.config}")
    
    def initialize(self) -> None:
        """Initialize the plugin after construction.
        
        This method is called after the plugin is constructed but before
        it's used. Override this method to perform any setup that requires
        the full configuration to be available.
        """
        if self._initialized:
            self.logger.warning(f"{self.__class__.__name__} already initialized")
            return
            
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self._validate_config()
        self._initialized = True
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate plugin configuration.
        
        Each plugin must implement this method to validate its specific
        configuration requirements.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls: Type[T], config: Dict[str, Any], logger_instance: Optional[Logger] = None) -> T:
        """Factory method to create plugin instance from configuration.
        
        Args:
            config: Configuration dictionary for the plugin
            logger_instance: Optional logger instance
            
        Returns:
            Configured plugin instance
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration for this plugin.
        
        Override this method to provide default configuration values.
        
        Returns:
            Dictionary of default configuration values
        """
        return {}
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this plugin.
        
        Returns:
            Dictionary containing plugin metadata
        """
        return {
            'name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'initialized': self._initialized,
            'config': self.config
        }
    
    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized