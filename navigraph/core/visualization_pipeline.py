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
            # Skip pipeline configuration - it's handled separately
            if viz_name == 'pipeline':
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
        # Check if we have a pipeline configuration first
        if 'pipeline' in self.config.get('visualizations', {}):
            # Pipeline mode: chain visualizers together
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
            
            return self._run_pipeline_mode(data, session_id, session_path, shared_resources, reward_tile_id, base_output_path)
        
        # Independent mode: check if we have individual visualizers
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
        
        # Independent mode: run each visualizer separately
        return self._run_independent_mode(data, session_id, session_path, shared_resources, reward_tile_id, base_output_path)
    
    def _run_independent_mode(
        self, 
        data: pd.DataFrame, 
        session_id: str, 
        session_path: str, 
        shared_resources: Dict[str, Any], 
        reward_tile_id: Any, 
        base_output_path: str
    ) -> Dict[str, List[str]]:
        """Run visualizers independently (current behavior)."""
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
                
                # Run visualization (no input_video_path, uses file discovery)
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
    
    def _run_pipeline_mode(
        self, 
        data: pd.DataFrame, 
        session_id: str, 
        session_path: str, 
        shared_resources: Dict[str, Any], 
        reward_tile_id: Any, 
        base_output_path: str
    ) -> Dict[str, List[str]]:
        """Run visualizers in pipeline mode (chained together)."""
        pipeline_config = self.config['visualizations']['pipeline']
        stages = pipeline_config.get('stages', [])
        output_name = pipeline_config.get('output_name', 'pipeline_output')
        
        if not stages:
            self.logger.error("Pipeline mode requires 'stages' configuration")
            return {'pipeline': []}
        
        self.logger.info(f"Running {len(stages)}-stage visualization pipeline for {session_id}")
        
        # Create pipeline output directory
        pipeline_output_path = Path(base_output_path) / session_id / 'pipeline'
        pipeline_output_path.mkdir(parents=True, exist_ok=True)
        
        # Common kwargs for all stages (same as independent mode)
        viz_kwargs = {
            'session_path': session_path,
            'session_id': session_id,
            'shared_resources': shared_resources,
            'reward_tile_id': reward_tile_id
        }
        
        # Start with original video for first stage
        current_video_path = None  # First stage will use file discovery
        
        try:
            for i, stage_config in enumerate(stages):
                plugin_name = stage_config.get('plugin')
                stage_config_dict = stage_config.get('config', {})
                
                if not plugin_name:
                    self.logger.error(f"Pipeline stage {i+1} missing 'plugin' specification")
                    return {'pipeline': []}
                
                # Get visualizer instance (same as independent mode would)
                if plugin_name not in self.registry._visualizers:
                    available_plugins = list(self.registry._visualizers.keys())
                    self.logger.error(f"Unknown visualizer plugin: {plugin_name}. Available: {available_plugins}")
                    return {'pipeline': []}
                
                visualizer_class = self.registry.get_visualizer_plugin(plugin_name)
                visualizer = visualizer_class.from_config(stage_config_dict, self.logger)
                
                self.logger.info(f"Pipeline stage {i+1}/{len(stages)}: {plugin_name}")
                
                # Create stage-specific output directory (simple naming)
                stage_output_path = pipeline_output_path / f"stage_{i+1}_{plugin_name}"
                stage_output_path.mkdir(parents=True, exist_ok=True)
                
                # Run visualization - EXACTLY like independent mode, just with input_video_path
                output_file = visualizer.generate_visualization(
                    session_data=data,
                    config=stage_config_dict,
                    output_path=str(stage_output_path),
                    input_video_path=current_video_path,  # Only difference from independent mode!
                    **viz_kwargs
                )
                
                if not output_file:
                    self.logger.error(f"Pipeline stage {i+1} ({plugin_name}) produced no output")
                    return {'pipeline': []}
                
                # Output from this stage becomes input for next stage
                current_video_path = output_file
                self.logger.info(f"✓ Stage {i+1} complete: {Path(output_file).name}")
            
            # Move final output to pipeline root and clean up intermediate files
            if current_video_path:
                # Create final output with desired name
                final_output_path = pipeline_output_path / f"{output_name}_{session_id}.mp4"
                Path(current_video_path).rename(final_output_path)
                
                # Clean up intermediate stage directories (keep only final output)
                for stage_dir in pipeline_output_path.glob("stage_*"):
                    if stage_dir.is_dir():
                        import shutil
                        shutil.rmtree(stage_dir)
                
                self.logger.info(f"✓ Pipeline complete: {final_output_path.name}")
                return {'pipeline': [str(final_output_path)]}
            else:
                return {'pipeline': []}
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {'pipeline': []}