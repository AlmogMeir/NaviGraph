"""Visualization pipeline orchestrator for NaviGraph.

This module provides a high-level interface for coordinating multiple
visualization plugins to create comprehensive visual outputs from session
data and analysis results.
"""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

from .interfaces import IVisualizer, Logger
from .registry import PluginRegistry, registry
from .session import Session
from .visualization_config import VisualizationConfig, OutputFormat
from .exceptions import VisualizationError


class VisualizationPipeline:
    """Orchestrates multiple visualization plugins.
    
    The pipeline can run visualizations sequentially or in parallel,
    coordinate shared resources, and manage output organization.
    """
    
    def __init__(
        self,
        config: Union[Dict[str, Any], VisualizationConfig],
        plugin_registry: Optional[PluginRegistry] = None,
        logger_instance: Optional[Logger] = None
    ):
        """Initialize visualization pipeline.
        
        Args:
            config: Pipeline configuration (dict or VisualizationConfig)
            plugin_registry: Plugin registry to use (defaults to global)
            logger_instance: Logger instance (defaults to global logger)
        """
        # Handle both dict and VisualizationConfig inputs
        if isinstance(config, VisualizationConfig):
            self.viz_config = config
            self.config = config.to_dict()
        else:
            self.config = config
            # Extract visualization settings if present
            viz_settings = config.get('visualization_settings', {})
            self.viz_config = VisualizationConfig.from_dict(viz_settings) if viz_settings else VisualizationConfig()
        
        self.registry = plugin_registry or registry
        self.logger = logger_instance or logger
        
        # Initialize visualizer instances
        self.visualizers: Dict[str, IVisualizer] = {}
        self._load_visualizers()
    
    def _load_visualizers(self) -> None:
        """Load and initialize visualizer plugins from configuration."""
        viz_configs = self.config.get('visualizations', {})
        
        for viz_name, viz_config in viz_configs.items():
            if not viz_config.get('enabled', True):
                self.logger.debug(f"Skipping disabled visualizer: {viz_name}")
                continue
                
            plugin_name = viz_config.get('plugin')
            if not plugin_name:
                self.logger.warning(f"No plugin specified for visualizer: {viz_name}")
                continue
            
            try:
                # Get plugin class and create instance
                plugin_class = self.registry.get_visualizer_plugin(plugin_name)
                
                # Merge visualizer config with theme/style settings
                visualizer_config = viz_config.get('config', {})
                visualizer_config.update(self.viz_config.get_visualizer_config(plugin_name))
                
                visualizer = plugin_class.from_config(
                    visualizer_config,
                    self.logger
                )
                
                self.visualizers[viz_name] = visualizer
                self.logger.info(f"Loaded visualizer: {viz_name} ({plugin_name})")
                
            except Exception as e:
                self.logger.error(f"Failed to load visualizer {viz_name}: {str(e)}")
    
    def create_session_visualizations(
        self,
        session: Session,
        output_path: Optional[str] = None,
        video_path: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Create all configured visualizations for a session.
        
        Args:
            session: Session object with integrated data
            output_path: Base directory for visualization outputs
            video_path: Path to source video file (if available)
            
        Returns:
            Dictionary mapping visualizer names to lists of created file paths
        """
        if not self.visualizers:
            self.logger.warning("No visualizers configured")
            return {}
        
        # Use configured output path or provided path
        base_output_path = output_path or self.viz_config.output_path
        if not base_output_path:
            raise VisualizationError("No output path specified")
        
        # Prepare data and resources
        data = session.get_integrated_dataframe()
        shared_resources = session.shared_resources
        session_id = session.session_id
        
        # Prepare common parameters
        common_kwargs = {
            'session_id': session_id,
            'video_path': video_path,
            'reward_tile_id': session.config.get('reward_tile_id')
        }
        
        results = {}
        
        for viz_name, visualizer in self.visualizers.items():
            self.logger.info(f"Creating {viz_name} visualization for {session_id}")
            
            try:
                # Create session-specific output directory
                viz_output_path = Path(base_output_path) / session_id / viz_name
                viz_output_path.mkdir(parents=True, exist_ok=True)
                
                # Get visualizer configuration
                viz_config = self.config['visualizations'][viz_name].get('config', {})
                
                # Run visualization
                output_file = visualizer.visualize(
                    data=data,
                    config=viz_config,
                    shared_resources=shared_resources,
                    output_path=str(viz_output_path),
                    **common_kwargs
                )
                
                if output_file:
                    results[viz_name] = [output_file]
                    self.logger.info(f"Created {viz_name}: {output_file}")
                else:
                    results[viz_name] = []
                    self.logger.warning(f"{viz_name} produced no output")
                    
            except Exception as e:
                self.logger.error(f"{viz_name} failed: {str(e)}")
                results[viz_name] = []
        
        return results
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get summary of configured visualizations.
        
        Returns:
            Dictionary with visualization pipeline information
        """
        return {
            'total_visualizers': len(self.visualizers),
            'enabled_visualizers': list(self.visualizers.keys()),
            'available_plugins': self.registry.list_visualizer_plugins(),
            'configuration': {
                name: {
                    'plugin': self.config['visualizations'][name].get('plugin'),
                    'enabled': self.config['visualizations'][name].get('enabled', True),
                    'config_keys': list(self.config['visualizations'][name].get('config', {}).keys())
                }
                for name in self.config.get('visualizations', {}).keys()
            }
        }