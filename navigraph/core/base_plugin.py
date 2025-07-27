"""Base plugin class for NaviGraph plugin system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar
from loguru import logger

# Type alias for logger
Logger = type(logger)
T = TypeVar('T', bound='BasePlugin')


class BasePlugin(ABC):
    """Base class for all NaviGraph plugins."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger_instance: Optional[Logger] = None):
        """Initialize plugin with configuration and logger."""
        self.config = config or {}
        self.logger = logger_instance or logger
        self._initialized = False
        
        # Log initialization
        self.logger.debug(f"Initializing {self.__class__.__name__} with config: {self.config}")
    
    def initialize(self) -> None:
        """Initialize the plugin after construction."""
        if self._initialized:
            self.logger.warning(f"{self.__class__.__name__} already initialized")
            return
            
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self._validate_config()
        self._initialized = True
    
    def _validate_config(self) -> None:
        """Validate plugin configuration. Override in subclasses if needed."""
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls: Type[T], config: Dict[str, Any], logger_instance: Optional[Logger] = None) -> T:
        """Factory method to create plugin instance from configuration."""
        pass