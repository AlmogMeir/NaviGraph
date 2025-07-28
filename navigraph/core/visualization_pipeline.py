"""Simple visualization pipeline for NaviGraph."""

from typing import Dict, List, Any, Optional, Union
import pandas as pd
from pathlib import Path
from loguru import logger

from .interfaces import IVisualizer, Logger
from .registry import PluginRegistry, registry
from .session import Session
from .visualization_config import VisualizationConfig, OutputFormat
from .exceptions import NavigraphError


class VisualizationPipeline:
    """Simple visualization pipeline that runs configured visualizers."""
    
    def __init__(
        self,
        config: Union[Dict[str, Any], VisualizationConfig],
        plugin_registry: Optional[PluginRegistry] = None,
        logger_instance: Optional[Logger] = None
    ):
        """Initialize visualization pipeline."""
        # Handle both dict and VisualizationConfig inputs
        if isinstance(config, VisualizationConfig):
            self.viz_config = config
            self.config = config.to_dict()
        else:
            self.config = config
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
                
                # Merge visualizer config with basic settings
                visualizer_config = viz_config.get('config', {})
                visualizer_config.update(self.viz_config.to_dict())
                
                # Add file_requirements from the visualizer config
                if 'file_requirements' in viz_config:
                    visualizer_config['file_requirements'] = viz_config['file_requirements']
                
                visualizer = plugin_class.from_config(visualizer_config, self.logger)
                self.visualizers[viz_name] = visualizer
                self.logger.info(f"Loaded visualizer: {viz_name} ({plugin_name})")
                
            except Exception as e:
                self.logger.error(f"Failed to load visualizer {viz_name}: {str(e)}")
    
    def create_session_visualizations(
        self,
        session: Session,
        output_path: Optional[str] = None,
        session_path: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Create all configured visualizations for a session.
        
        Args:
            session: Session object with integrated data
            output_path: Override output directory path
            session_path: Path to session directory for file discovery
            
        Returns:
            Dict mapping visualizer names to list of output file paths
        """
        if not self.visualizers:
            self.logger.warning("No visualizers configured")
            return {}
        
        # Use configured output path or provided path
        base_output_path = output_path or self.viz_config.output_path
        if not base_output_path:
            raise NavigraphError("No output path specified for visualization")
        
        # Prepare data and resources
        data = session.get_integrated_dataframe()
        shared_resources = session.shared_resources
        session_id = session.session_id
        
        # Get session metadata (like reward_tile_id)
        session_metadata = session.get_session_metadata()
        reward_tile_id = session.config.get('reward_tile_id')
        
        results = {}
        
        for viz_name, visualizer in self.visualizers.items():
            self.logger.info(f"Creating {viz_name} visualization for {session_id}")
            
            try:
                # Create session-specific output directory
                viz_output_path = Path(base_output_path) / session_id / viz_name
                viz_output_path.mkdir(parents=True, exist_ok=True)
                
                # Get visualizer configuration
                viz_config = self.config['visualizations'][viz_name].get('config', {})
                
                # Prepare kwargs for visualizations
                viz_kwargs = {
                    'session_path': session_path,
                    'session_id': session_id,
                    'shared_resources': shared_resources,
                    'reward_tile_id': reward_tile_id
                }
                
                # Run visualization
                output_file = visualizer.generate_visualization(
                    session_data=data,
                    config=viz_config,
                    output_path=str(viz_output_path),
                    **viz_kwargs
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