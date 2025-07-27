#!/usr/bin/env python3
"""Plugin Development Guide for NaviGraph.

This example demonstrates how to create custom plugins for each plugin type
in the NaviGraph framework.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import time
from abc import ABC

from navigraph.core import (
    IDataSource, ISharedResource, IAnalyzer, IVisualizer,
    BasePlugin, register_data_source_plugin, register_shared_resource_plugin,
    register_analyzer_plugin, register_visualizer_plugin,
    AnalysisResult, AnalysisMetadata, VisualizationResult
)
from navigraph.core.utils import compute_configuration_hash
from navigraph.core.exceptions import DataSourceError, AnalysisError


# =============================================================================
# CUSTOM DATA SOURCE PLUGIN
# =============================================================================

@register_data_source_plugin("custom_camera")
class CustomCameraDataSource(BasePlugin, IDataSource):
    """Example custom data source for camera-based tracking.
    
    This example shows how to create a data source that processes
    custom camera tracking data and integrates it into the session.
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance=None):
        """Factory method to create instance from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate configuration for custom camera data source."""
        required_keys = ['video_path', 'tracking_algorithm']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Custom camera data source requires '{key}' in config")
    
    def get_provided_column_names(self) -> List[str]:
        """Return column names this data source provides."""
        return [
            'camera_x', 'camera_y', 'camera_confidence',
            'velocity', 'direction', 'movement_quality'
        ]
    
    def validate_session_prerequisites(
        self, 
        current_dataframe: pd.DataFrame, 
        shared_resources: Dict[str, Any]
    ) -> bool:
        """Validate that prerequisites are met for integration."""
        # Check if required file exists
        file_path = self.config.get('discovered_file_path')
        if not file_path:
            self.logger.error("No file path provided for custom camera data")
            return False
        
        # Check if previous data sources provided required data
        if len(current_dataframe) > 0:
            # If we have existing data, validate compatibility
            self.logger.info("Existing data found, will merge with camera data")
        
        return True
    
    def integrate_data_into_session(
        self,
        current_dataframe: pd.DataFrame,
        data_source_config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        logger
    ) -> pd.DataFrame:
        """Integrate custom camera data into session DataFrame."""
        logger.info("Processing custom camera data...")
        
        # Simulate loading and processing camera tracking data
        file_path = data_source_config.get('discovered_file_path')
        algorithm = self.config.get('tracking_algorithm', 'optical_flow')
        
        # In a real implementation, you would load your custom data format
        # For this example, we'll simulate the data
        n_frames = len(current_dataframe) if len(current_dataframe) > 0 else 1000
        
        camera_data = {
            'camera_x': np.random.uniform(0, 640, n_frames),
            'camera_y': np.random.uniform(0, 480, n_frames),
            'camera_confidence': np.random.uniform(0.8, 1.0, n_frames),
            'velocity': np.random.uniform(0, 50, n_frames),
            'direction': np.random.uniform(0, 360, n_frames),
            'movement_quality': np.random.uniform(0.5, 1.0, n_frames)
        }
        
        camera_df = pd.DataFrame(camera_data)
        
        # Integrate with existing data
        if len(current_dataframe) == 0:
            # First data source
            result_df = camera_df
        else:
            # Merge with existing data
            result_df = pd.concat([current_dataframe, camera_df], axis=1)
        
        logger.info(f"Custom camera integration complete: {len(camera_df.columns)} columns added")
        return result_df


# =============================================================================
# CUSTOM SHARED RESOURCE PLUGIN
# =============================================================================

@register_shared_resource_plugin("arena_geometry")
class ArenaGeometryProvider(BasePlugin, ISharedResource):
    """Example shared resource for arena geometry information.
    
    This provides geometric information about experimental arenas
    that can be shared across sessions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger_instance=None):
        super().__init__(config, logger_instance)
        self.arena_geometry = None
        self.calibration_matrix = None
        self._is_initialized = False
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance=None):
        """Factory method to create instance from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize_resource()
        return instance
    
    def get_required_config_keys(self) -> List[str]:
        """Return required configuration keys."""
        return ['arena_type', 'dimensions']
    
    def initialize_resource(self) -> None:
        """Initialize the arena geometry resource."""
        if self._is_initialized:
            return
        
        arena_type = self.config.get('arena_type', 'rectangular')
        dimensions = self.config.get('dimensions', [100, 100])
        
        # Create arena geometry
        if arena_type == 'rectangular':
            self.arena_geometry = {
                'type': 'rectangular',
                'width': dimensions[0],
                'height': dimensions[1],
                'area': dimensions[0] * dimensions[1],
                'perimeter': 2 * (dimensions[0] + dimensions[1])
            }
        elif arena_type == 'circular':
            radius = dimensions[0]
            self.arena_geometry = {
                'type': 'circular', 
                'radius': radius,
                'area': np.pi * radius**2,
                'perimeter': 2 * np.pi * radius
            }
        
        # Create calibration matrix (example)
        self.calibration_matrix = np.eye(3)  # Identity for simplicity
        
        self._is_initialized = True
        self.logger.info(f"Arena geometry initialized: {arena_type} arena")
    
    def cleanup_resource(self) -> None:
        """Clean up the resource."""
        self.arena_geometry = None
        self.calibration_matrix = None
        self._is_initialized = False
        self.logger.info("Arena geometry resource cleaned up")
    
    def is_initialized(self) -> bool:
        """Check if resource is initialized."""
        return self._is_initialized
    
    def get_arena_info(self) -> Dict[str, Any]:
        """Get arena geometry information."""
        if not self._is_initialized:
            raise RuntimeError("Arena geometry not initialized")
        return self.arena_geometry.copy()
    
    def get_calibration_matrix(self) -> np.ndarray:
        """Get calibration matrix for coordinate transformations."""
        if not self._is_initialized:
            raise RuntimeError("Arena geometry not initialized")
        return self.calibration_matrix.copy()


# =============================================================================
# CUSTOM ANALYZER PLUGIN
# =============================================================================

@register_analyzer_plugin("custom_behavior_metrics")
class CustomBehaviorAnalyzer(BasePlugin, IAnalyzer):
    """Example custom analyzer for behavioral metrics.
    
    This analyzer computes custom behavioral metrics from session data.
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance=None):
        """Factory method to create instance from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate analyzer configuration."""
        # Custom analyzers can have specific configuration requirements
        metrics_config = self.config.get('metrics', {})
        if not metrics_config:
            self.logger.warning("No specific metrics configured, will compute defaults")
    
    def analyze_session(self, session) -> AnalysisResult:
        """Analyze a single session for custom behavioral metrics."""
        start_time = time.time()
        
        try:
            # Get session data
            dataframe = session.get_integrated_dataframe()
            session_config = session.get_session_config()
            
            self.logger.info(f"Computing custom behavior metrics for {session.session_id}")
            
            # Compute custom metrics
            metrics = {}
            
            # Example 1: Activity level
            if 'velocity' in dataframe.columns:
                velocity = dataframe['velocity'].dropna()
                metrics['mean_velocity'] = float(velocity.mean())
                metrics['max_velocity'] = float(velocity.max())
                metrics['activity_ratio'] = float((velocity > 5).mean())  # Fraction of time moving
            
            # Example 2: Spatial distribution
            if 'camera_x' in dataframe.columns and 'camera_y' in dataframe.columns:
                x_coords = dataframe['camera_x'].dropna()
                y_coords = dataframe['camera_y'].dropna()
                
                # Calculate spatial spread
                x_range = float(x_coords.max() - x_coords.min())
                y_range = float(y_coords.max() - y_coords.min())
                metrics['spatial_spread_x'] = x_range
                metrics['spatial_spread_y'] = y_range
                metrics['exploration_area'] = x_range * y_range
            
            # Example 3: Movement patterns
            if 'direction' in dataframe.columns:
                directions = dataframe['direction'].dropna()
                direction_changes = np.abs(np.diff(directions))
                direction_changes = np.minimum(direction_changes, 360 - direction_changes)  # Handle wrap-around
                metrics['mean_direction_change'] = float(direction_changes.mean())
                metrics['directional_stability'] = float((direction_changes < 30).mean())
            
            # Example 4: Quality metrics
            if 'movement_quality' in dataframe.columns:
                quality = dataframe['movement_quality'].dropna()
                metrics['mean_tracking_quality'] = float(quality.mean())
                metrics['low_quality_ratio'] = float((quality < 0.7).mean())
            
            # Create metadata
            metadata = AnalysisMetadata(
                analyzer_name="custom_behavior_metrics",
                version="1.0.0",
                timestamp=datetime.now(),
                computation_time=time.time() - start_time,
                config_hash=compute_configuration_hash(self.config)
            )
            
            self.logger.info(f"Custom behavior metrics computed: {list(metrics.keys())}")
            
            return AnalysisResult(
                session_id=session.session_id,
                analyzer_name="custom_behavior_metrics",
                metrics=metrics,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Custom behavior analysis failed: {str(e)}")
            
            # Return empty result with error metadata
            metadata = AnalysisMetadata(
                analyzer_name="custom_behavior_metrics",
                version="1.0.0",
                timestamp=datetime.now(),
                computation_time=time.time() - start_time,
                config_hash=""
            )
            return AnalysisResult(
                session_id=session.session_id,
                analyzer_name="custom_behavior_metrics",
                metrics={},
                metadata=metadata
            )
    
    def analyze_cross_session(
        self,
        sessions: List,
        session_results: Dict[str, AnalysisResult]
    ) -> Dict[str, Any]:
        """Perform cross-session analysis for custom behavior metrics."""
        cross_session_results = {}
        
        if not session_results:
            return cross_session_results
        
        # Collect metrics across sessions
        all_metrics = {}
        for session_id, result in session_results.items():
            for metric_name, metric_value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                if metric_value is not None:
                    all_metrics[metric_name].append(metric_value)
        
        # Compute cross-session statistics
        for metric_name, values in all_metrics.items():
            if values:
                cross_session_results[f"{metric_name}_mean"] = np.mean(values)
                cross_session_results[f"{metric_name}_std"] = np.std(values)
                cross_session_results[f"{metric_name}_min"] = np.min(values)
                cross_session_results[f"{metric_name}_max"] = np.max(values)
                cross_session_results[f"{metric_name}_count"] = len(values)
        
        return cross_session_results


# =============================================================================
# CUSTOM VISUALIZER PLUGIN  
# =============================================================================

@register_visualizer_plugin("custom_heatmap")
class CustomHeatmapVisualizer(BasePlugin, IVisualizer):
    """Example custom visualizer for creating spatial heatmaps.
    
    This visualizer creates custom heatmap visualizations of animal positions.
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance=None):
        """Factory method to create instance from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate visualizer configuration."""
        # Set default values if not provided
        if 'bin_size' not in self.config:
            self.config['bin_size'] = 20
        if 'colormap' not in self.config:
            self.config['colormap'] = 'viridis'
    
    @property
    def supported_formats(self) -> List[str]:
        """Return supported output formats."""
        return ['png', 'jpg', 'svg', 'pdf']
    
    def visualize(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> VisualizationResult:
        """Create custom heatmap visualization."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            from matplotlib.patches import Rectangle
            
            self.logger.info("Creating custom heatmap visualization...")
            
            # Extract position data
            if 'camera_x' in data.columns and 'camera_y' in data.columns:
                x_data = data['camera_x'].dropna()
                y_data = data['camera_y'].dropna()
            elif 'keypoints_x' in data.columns and 'keypoints_y' in data.columns:
                x_data = data['keypoints_x'].dropna()
                y_data = data['keypoints_y'].dropna()
            else:
                raise ValueError("No position data found in DataFrame")
            
            if len(x_data) == 0 or len(y_data) == 0:
                raise ValueError("No valid position data available")
            
            # Get configuration
            bin_size = config.get('bin_size', 20)
            colormap = config.get('colormap', 'viridis')
            session_id = kwargs.get('session_id', 'unknown')
            
            # Create figure
            fig_size = config.get('figure_size', (10, 8))
            fig, ax = plt.subplots(figsize=fig_size)
            
            # Create 2D histogram (heatmap)
            x_range = [x_data.min(), x_data.max()]
            y_range = [y_data.min(), y_data.max()]
            
            # Calculate number of bins
            x_bins = int((x_range[1] - x_range[0]) / bin_size)
            y_bins = int((y_range[1] - y_range[0]) / bin_size)
            
            # Create heatmap
            heatmap, xedges, yedges = np.histogram2d(
                x_data, y_data, 
                bins=[x_bins, y_bins],
                range=[x_range, y_range]
            )
            
            # Plot heatmap
            im = ax.imshow(
                heatmap.T, 
                origin='lower',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap=colormap,
                aspect='auto'
            )
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Time Spent (frames)', rotation=270, labelpad=20)
            
            # Customize plot
            ax.set_xlabel('X Position (pixels)')
            ax.set_ylabel('Y Position (pixels)')
            ax.set_title(f'Spatial Heatmap - {session_id}')
            
            # Add arena boundary if available
            if 'arena_geometry' in shared_resources:
                arena = shared_resources['arena_geometry'].get_arena_info()
                if arena['type'] == 'rectangular':
                    rect = Rectangle(
                        (0, 0), arena['width'], arena['height'],
                        linewidth=2, edgecolor='white', facecolor='none'
                    )
                    ax.add_patch(rect)
            
            # Save figure
            output_file = f"{output_path}/heatmap_{session_id}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Custom heatmap saved: {output_file}")
            
            return VisualizationResult(
                visualizer_name="custom_heatmap",
                output_files=[output_file],
                metadata={
                    'bin_size': bin_size,
                    'colormap': colormap,
                    'data_points': len(x_data)
                },
                success=True
            )
            
        except Exception as e:
            error_msg = f"Custom heatmap visualization failed: {str(e)}"
            self.logger.error(error_msg)
            
            return VisualizationResult(
                visualizer_name="custom_heatmap",
                output_files=[],
                success=False,
                error_message=error_msg
            )


# =============================================================================
# PLUGIN DEVELOPMENT EXAMPLES AND GUIDELINES
# =============================================================================

def demonstrate_plugin_usage():
    """Demonstrate how to use the custom plugins."""
    print("=== Custom Plugin Development Examples ===\n")
    
    # Show plugin registration
    from navigraph.core import registry
    plugins = registry.list_all_plugins()
    
    print("Custom plugins registered:")
    custom_plugins = {
        'data_sources': [p for p in plugins['data_sources'] if 'custom' in p],
        'shared_resources': [p for p in plugins['shared_resources'] if 'arena' in p],
        'analyzers': [p for p in plugins['analyzers'] if 'custom' in p],
        'visualizers': [p for p in plugins['visualizers'] if 'custom' in p]
    }
    
    for category, plugin_list in custom_plugins.items():
        print(f"  {category}: {plugin_list}")
    
    # Example configuration for custom plugins
    print("\nExample configuration:")
    example_config = {
        'data_sources': [
            {
                'name': 'custom_camera',
                'type': 'custom_camera',
                'config': {
                    'video_path': '/path/to/video.mp4',
                    'tracking_algorithm': 'optical_flow'
                }
            }
        ],
        'shared_resources': [
            {
                'name': 'arena_geometry',
                'type': 'arena_geometry',
                'config': {
                    'arena_type': 'rectangular',
                    'dimensions': [120, 80]
                }
            }
        ],
        'analyzers': [
            {
                'name': 'custom_behavior_metrics',
                'type': 'custom_behavior_metrics',
                'config': {
                    'metrics': ['activity', 'exploration', 'movement_patterns']
                }
            }
        ],
        'visualizers': [
            {
                'name': 'custom_heatmap',
                'type': 'custom_heatmap',
                'config': {
                    'bin_size': 15,
                    'colormap': 'hot'
                }
            }
        ]
    }
    
    import json
    print(json.dumps(example_config, indent=2))


def plugin_development_guidelines():
    """Print plugin development guidelines."""
    guidelines = """
=== Plugin Development Guidelines ===

1. DATA SOURCE PLUGINS:
   - Inherit from BasePlugin and IDataSource
   - Implement get_provided_column_names(), validate_session_prerequisites(), integrate_data_into_session()
   - Use @register_data_source_plugin("name") decorator
   - Handle file discovery and data loading
   - Integrate with existing DataFrame structure

2. SHARED RESOURCE PLUGINS:
   - Inherit from BasePlugin and ISharedResource  
   - Implement initialize_resource(), cleanup_resource(), is_initialized()
   - Use @register_shared_resource_plugin("name") decorator
   - Provide resources used across multiple sessions
   - Handle resource lifecycle management

3. ANALYZER PLUGINS:
   - Inherit from BasePlugin and IAnalyzer
   - Implement analyze_session() returning AnalysisResult
   - Implement analyze_cross_session() for group statistics
   - Use @register_analyzer_plugin("name") decorator
   - Include metadata and timing information
   - Handle errors gracefully

4. VISUALIZER PLUGINS:
   - Inherit from BasePlugin and IVisualizer
   - Implement visualize() returning VisualizationResult
   - Define supported_formats property
   - Use @register_visualizer_plugin("name") decorator
   - Support multiple output formats
   - Use configuration for customization

5. GENERAL BEST PRACTICES:
   - Use factory pattern with from_config() class method
   - Validate configuration in _validate_config()
   - Use structured logging throughout
   - Handle errors with appropriate exceptions
   - Document configuration requirements
   - Write comprehensive tests
   - Follow type hints and documentation standards
   
6. TESTING YOUR PLUGINS:
   - Test factory methods and configuration validation
   - Test integration with mock data
   - Test error handling and edge cases
   - Test cross-plugin compatibility
   - Include performance tests for large datasets
"""
    print(guidelines)


def main():
    """Run plugin development examples."""
    print("NaviGraph Plugin Development Guide")
    print("=" * 50)
    
    demonstrate_plugin_usage()
    print()
    plugin_development_guidelines()
    
    print("\nPlugin development examples complete!")
    print("Use these examples as templates for creating your own plugins.")


if __name__ == "__main__":
    main()