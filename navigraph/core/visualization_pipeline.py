"""Visualization pipeline for NaviGraph.

This module provides a simple, clean pipeline for running visualization stages.
Each visualization is treated as a pipeline with one or more stages. The pipeline
manages data flow between stages and handles all file I/O through publishers.
"""

from typing import Dict, List, Any, Optional, Union, Iterator
import pandas as pd
from pathlib import Path
from loguru import logger
import numpy as np

from .interfaces import IVisualizer, Logger
from .session import Session
from .exceptions import NavigraphError
from .publishers import get_publisher


class VisualizationPipeline:
    """Pipeline for running visualization stages.
    
    The pipeline orchestrates visualization stages, managing data flow
    between them and handling output through publishers. All visualizations
    are treated uniformly as pipelines with N stages (where N >= 1).
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        plugin_registry = None,
        logger_instance: Optional[Logger] = None
    ):
        """Initialize visualization pipeline.
        
        Args:
            config: Full configuration dictionary
            plugin_registry: Optional plugin registry (defaults to global)
            logger_instance: Optional logger instance
        """
        self.config = config
        self.registry = plugin_registry
        self.logger = logger_instance or logger
    
    def create_session_visualizations(
        self,
        session: Session,
        output_path: Optional[str] = None,
        session_path: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Create all configured visualizations for a session.
        
        Args:
            session: Session object with integrated data
            output_path: Base output directory path
            session_path: Path to session directory for file discovery
            
        Returns:
            Dict mapping visualization names to list of output file paths
        """
        # Get visualizations config
        viz_configs = self.config.get('visualizations', {})
        if not viz_configs:
            self.logger.warning("No visualizations configured")
            return {}
        
        # Prepare common data and resources
        session_data = session.get_integrated_dataframe()
        shared_resources = session.shared_resources
        session_id = session.session_id
        reward_tile_id = session.config.get('reward_tile_id')
        
        # Process each visualization as a pipeline
        results = {}
        for viz_name, viz_config in viz_configs.items():
            self.logger.info(f"Processing visualization: {viz_name}")
            
            try:
                # Run the visualization pipeline
                output_files = self._run_visualization_pipeline(
                    viz_name=viz_name,
                    viz_config=viz_config,
                    session_data=session_data,
                    session_id=session_id,
                    session_path=session_path,
                    shared_resources=shared_resources,
                    reward_tile_id=reward_tile_id,
                    output_path=output_path
                )
                
                results[viz_name] = output_files
                
                if output_files:
                    self.logger.info(f"✓ {viz_name}: {len(output_files)} output(s) created")
                else:
                    self.logger.warning(f"✗ {viz_name}: No outputs created")
                    
            except Exception as e:
                self.logger.error(f"Visualization {viz_name} failed: {str(e)}")
                results[viz_name] = []
        
        return results
    
    def _run_visualization_pipeline(
        self,
        viz_name: str,
        viz_config: Dict[str, Any],
        session_data: pd.DataFrame,
        session_id: str,
        session_path: Optional[str],
        shared_resources: Dict[str, Any],
        reward_tile_id: Any,
        output_path: Optional[str]
    ) -> List[str]:
        """Run a single visualization pipeline.
        
        Args:
            viz_name: Name of the visualization
            viz_config: Configuration for this visualization
            session_data: Session DataFrame
            session_id: Session identifier
            session_path: Path to session files
            shared_resources: Shared resources (map, graph, etc.)
            reward_tile_id: Reward tile ID
            output_path: Base output directory
            
        Returns:
            List of created output file paths
        """
        # Get pipeline stages
        stages = viz_config.get('stages', [])
        if not stages:
            self.logger.error(f"{viz_name}: No stages configured")
            return []
        
        # Get publisher configuration
        publisher_type = viz_config.get('publisher', 'video')
        output_name = viz_config.get('output_name', viz_name)
        
        # Determine output directory
        if output_path:
            viz_output_path = Path(output_path) / 'visualizations'
        else:
            viz_output_path = Path('.') / 'visualizations'
        viz_output_path.mkdir(parents=True, exist_ok=True)
        
        # Common kwargs for all stages
        stage_kwargs = {
            'session_path': session_path,
            'session_id': session_id,
            'shared_resources': shared_resources,
            'reward_tile_id': reward_tile_id
        }
        
        # Process stages in sequence
        self.logger.info(f"Running {len(stages)}-stage pipeline for {viz_name}")
        
        current_data = None  # First stage will use file discovery
        
        for i, stage_config in enumerate(stages):
            plugin_name = stage_config.get('plugin')
            stage_params = stage_config.get('config', {})
            
            if not plugin_name:
                self.logger.error(f"Stage {i+1} missing 'plugin' specification")
                return []
            
            # Get visualizer instance
            try:
                if not self.registry:
                    from .registry import registry
                    self.registry = registry
                visualizer_class = self.registry.get_visualizer_plugin(plugin_name)
                visualizer = visualizer_class.from_config(stage_params, self.logger)
            except Exception as e:
                self.logger.error(f"Failed to create visualizer {plugin_name}: {str(e)}")
                return []
            
            self.logger.info(f"Stage {i+1}/{len(stages)}: {plugin_name}")
            
            # Process data through this stage
            try:
                current_data = visualizer.process(
                    session_data=session_data,
                    config=stage_params,
                    input_data=current_data,
                    **stage_kwargs
                )
                
                # Ensure we have data
                if current_data is None:
                    self.logger.error(f"Stage {i+1} ({plugin_name}) produced no data")
                    return []
                    
            except Exception as e:
                self.logger.error(f"Stage {i+1} ({plugin_name}) failed: {str(e)}")
                return []
        
        # Publish final output
        try:
            publisher = get_publisher(
                publisher_type=publisher_type,
                output_path=str(viz_output_path),
                output_name=output_name
            )
            
            # Get FPS from first stage config if available (for video publisher)
            fps = stages[0].get('config', {}).get('output_fps', 30.0)
            
            output_file = publisher.publish(
                frames=current_data,
                fps=fps,
                session_id=session_id
            )
            
            if output_file:
                self.logger.info(f"✓ Published {viz_name}: {Path(output_file).name}")
                return [output_file]
            else:
                self.logger.error(f"Failed to publish {viz_name}")
                return []
                
        except Exception as e:
            self.logger.error(f"Publishing failed for {viz_name}: {str(e)}")
            return []