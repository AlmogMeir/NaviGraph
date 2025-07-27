"""Map provider shared resource for NaviGraph.

This plugin wraps the current MapLabeler functionality as a shared resource,
making maze map data and spatial configuration available to data sources
and analyzers that need spatial coordinate transformation.
"""

import cv2
import numpy as np
from typing import Dict, Any
from pathlib import Path
import os

from ...core.interfaces import ISharedResource, SharedResourceError, Logger
from ...core.base_plugin import BasePlugin
from ...core.registry import register_shared_resource_plugin


@register_shared_resource_plugin("map_provider")
class MapProviderResource(BasePlugin, ISharedResource):
    """Provides maze map and spatial transformation utilities.
    
    This shared resource initializes the maze map image and configuration
    needed for coordinate transformation and spatial analysis. It wraps
    the existing MapLabeler initialization logic.
    """
    
    def __init__(self, config = None, logger_instance = None):
        """Initialize empty map provider."""
        super().__init__(config, logger_instance)
        self._map_image = None
        self._map_config = None
        self._resource_initialized = False
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance = None):
        """Factory method to create map provider from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        instance.initialize_resource(config, instance.logger)
        return instance
    
    def _validate_config(self) -> None:
        """Validate map provider configuration."""
        required_keys = self.get_required_config_keys()
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
    
    def initialize_resource(
        self, 
        resource_config: Dict[str, Any], 
        logger: Logger
    ) -> None:
        """Initialize map provider with configuration.
        
        Args:
            resource_config: Configuration containing map_path and map_settings
            logger: Logger for initialization messages
            
        Raises:
            SharedResourceError: If initialization fails
        """
        try:
            self.logger.info("Initializing map provider resource")
            
            # Get map image path
            map_path = resource_config.get('map_path')
            if not map_path:
                raise SharedResourceError(
                    "Map provider requires 'map_path' in configuration"
                )
            
            # Resolve relative paths using experiment path
            if not os.path.isabs(map_path):
                experiment_path = resource_config.get('experiment_path', '.')
                map_path = os.path.join(experiment_path, map_path)
            
            # Load map image
            if not os.path.isfile(map_path):
                raise SharedResourceError(
                    f"Map image not found at path: {map_path}"
                )
            
            self._map_image = cv2.imread(map_path)
            if self._map_image is None:
                raise SharedResourceError(
                    f"Failed to load map image from: {map_path}"
                )
            
            # Get map settings
            map_settings = resource_config.get('map_settings', {})
            if not map_settings:
                raise SharedResourceError(
                    "Map provider requires 'map_settings' in configuration"
                )
            
            # Parse map configuration (preserving existing structure)
            self._map_config = self._parse_map_settings(map_settings)
            
            self._resource_initialized = True
            
            self.logger.info(
                f"✓ Map provider initialized: {Path(map_path).name} "
                f"({self._map_image.shape[1]}x{self._map_image.shape[0]}) "
                f"with {self._map_config['grid_size'][0]}x{self._map_config['grid_size'][1]} grid"
            )
            
        except Exception as e:
            raise SharedResourceError(
                f"Failed to initialize map provider: {str(e)}"
            ) from e
    
    def cleanup_resource(self, logger: Logger) -> None:
        """Clean up map provider resources."""
        self.logger.debug("Cleaning up map provider resource")
        self._map_image = None
        self._map_config = None
        self._resource_initialized = False
    
    def is_initialized(self) -> bool:
        """Check if map provider is initialized."""
        return self._resource_initialized
    
    def get_required_config_keys(self) -> list:
        """Return required configuration keys."""
        return ['map_path', 'map_settings']
    
    @property
    def resource_type(self) -> str:
        """Type identifier for this resource."""
        return "maze_map"
    
    def get_map_image(self) -> np.ndarray:
        """Get the loaded map image.
        
        Returns:
            Map image as numpy array
            
        Raises:
            SharedResourceError: If not initialized
        """
        if not self._resource_initialized:
            raise SharedResourceError("Map provider not initialized")
        return self._map_image
    
    def get_map_configuration(self) -> Dict[str, Any]:
        """Get map configuration for coordinate transformation.
        
        Returns:
            Dictionary with origin, grid_size, segment_length
            
        Raises:
            SharedResourceError: If not initialized
        """
        if not self._resource_initialized:
            raise SharedResourceError("Map provider not initialized")
        return self._map_config.copy()
    
    def get_map_bounds(self) -> Dict[str, int]:
        """Get map image bounds.
        
        Returns:
            Dictionary with width, height
        """
        if not self._resource_initialized:
            raise SharedResourceError("Map provider not initialized")
        
        height, width = self._map_image.shape[:2]
        return {'width': width, 'height': height}
    
    def _parse_map_settings(self, map_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Parse map settings from configuration (preserving existing logic).
        
        Args:
            map_settings: Raw map settings from configuration
            
        Returns:
            Parsed map configuration
            
        Raises:
            SharedResourceError: If required settings missing or invalid
        """
        try:
            # Required settings
            origin_str = map_settings.get('origin')
            grid_size_str = map_settings.get('grid_size')
            segment_length = map_settings.get('segment_length')
            
            if not all([origin_str, grid_size_str, segment_length]):
                raise SharedResourceError(
                    "Map settings must include 'origin', 'grid_size', and 'segment_length'"
                )
            
            # Parse origin (preserving existing eval logic for backward compatibility)
            if isinstance(origin_str, str):
                origin = eval(origin_str)  # Original code uses eval
            else:
                origin = origin_str
            
            # Parse grid size (preserving existing eval logic)
            if isinstance(grid_size_str, str):
                grid_size = eval(grid_size_str)  # Original code uses eval
            else:
                grid_size = grid_size_str
            
            # Validate parsed values
            if not (isinstance(origin, (list, tuple)) and len(origin) == 2):
                raise SharedResourceError(
                    f"Origin must be (x, y) tuple, got: {origin}"
                )
            
            if not (isinstance(grid_size, (list, tuple)) and len(grid_size) == 2):
                raise SharedResourceError(
                    f"Grid size must be (rows, cols) tuple, got: {grid_size}"
                )
            
            if not isinstance(segment_length, (int, float)) or segment_length <= 0:
                raise SharedResourceError(
                    f"Segment length must be positive number, got: {segment_length}"
                )
            
            return {
                'origin': tuple(origin),
                'grid_size': tuple(grid_size), 
                'segment_length': segment_length,
                'pixel_to_meter': map_settings.get('pixel_to_meter', None)
            }
            
        except Exception as e:
            raise SharedResourceError(
                f"Failed to parse map settings: {str(e)}"
            ) from e