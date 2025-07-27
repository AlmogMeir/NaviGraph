"""Spatial metrics analyzer plugin for NaviGraph.

This plugin wraps the spatial analysis functionality from the original SessionAnalyzer,
preserving all existing behavior while adapting to the new plugin architecture.
It provides time and velocity measurements between spatial locations.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import operator
from functools import partial
from datetime import datetime
import time

# Import from local utils module
from .utils import a_to_b, count_unique_type_specific_objects, Condition

from ...core.interfaces import IAnalyzer, Logger
from ...core.base_plugin import BasePlugin
from ...core.registry import register_analyzer_plugin
from ...core.types import AnalysisResult, AnalysisMetadata
from ...core.utils import compute_configuration_hash

# Constants from original SessionAnalyzer
FPS = 'fps'
TILE_ID = 'tile_id'
TREE_POSITION = 'tree_position'
NODE_POSSIBLE_DTYPES = (int, frozenset)
DEFAULT_MIN_NODES_ON_PATH = 0

# Helper condition - preserving existing logic
count_unique_nodes = partial(count_unique_type_specific_objects, dtype=NODE_POSSIBLE_DTYPES)
condition = Condition(column_name=TREE_POSITION,
                      func=count_unique_nodes,
                      threshold=DEFAULT_MIN_NODES_ON_PATH,
                      operator=operator.gt)


@register_analyzer_plugin("spatial_metrics")
class SpatialMetricsAnalyzer(BasePlugin, IAnalyzer):
    """Provides spatial analysis metrics: time and velocity between locations.
    
    This analyzer wraps the original time_a_to_b and velocity_a_to_b methods,
    preserving all existing functionality while integrating with the new
    plugin architecture.
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance = None):
        """Factory method to create spatial metrics analyzer from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate spatial metrics analyzer configuration."""
        # No required config keys for this analyzer
        pass
    
    def analyze_session(self, session) -> AnalysisResult:
        """Analyze a single session for spatial metrics.
        
        Args:
            session: Session object with integrated DataFrame and configuration
            
        Returns:
            AnalysisResult with computed spatial metrics
        """
        start_time = time.time()
        
        try:
            # Get session data
            dataframe = session.get_integrated_dataframe()
            session_config = session.get_session_config()
            logger = session.logger
            
            logger.info("Computing spatial metrics for session")
            
            # Extract configuration for spatial analysis
            analyzer_config = session_config.get('analyze', {}).get('metrics', {})
            results = {}
            
            # Process each configured metric
            for metric_name, metric_config in analyzer_config.items():
                func_name = metric_config.get('func_name')
                args = metric_config.get('args', {})
                
                if func_name == 'time_a_to_b':
                    results[metric_name] = self._time_a_to_b(
                        dataframe, session, args, logger
                    )
                elif func_name == 'velocity_a_to_b':
                    results[metric_name] = self._velocity_a_to_b(
                        dataframe, session, args, logger
                    )
            
            logger.info(f"Spatial metrics computed: {list(results.keys())}")
            
            # Create metadata
            metadata = AnalysisMetadata(
                analyzer_name="spatial_metrics",
                version="1.0.0",
                timestamp=datetime.now(),
                computation_time=time.time() - start_time,
                config_hash=compute_configuration_hash(analyzer_config)
            )
            
            # Return structured result
            return AnalysisResult(
                session_id=session.session_id,
                analyzer_name="spatial_metrics",
                metrics=results,
                metadata=metadata
            )
            
        except Exception as e:
            session.logger.error(f"Spatial metrics analysis failed: {str(e)}")
            # Return empty result with error metadata
            metadata = AnalysisMetadata(
                analyzer_name="spatial_metrics",
                version="1.0.0",
                timestamp=datetime.now(),
                computation_time=time.time() - start_time,
                config_hash=""
            )
            return AnalysisResult(
                session_id=session.session_id,
                analyzer_name="spatial_metrics",
                metrics={},
                metadata=metadata
            )
    
    def analyze_cross_session(
        self,
        sessions: List,
        session_results: Dict[str, AnalysisResult]
    ) -> Dict[str, Any]:
        """Perform cross-session analysis for spatial metrics.
        
        Args:
            sessions: List of Session objects
            session_results: Results from analyze_session for each session
            
        Returns:
            Dictionary of cross-session statistics
        """
        cross_session_results = {}
        
        # Get all metric names from first session
        if session_results:
            first_result = next(iter(session_results.values()))
            metric_names = list(first_result.metrics.keys())
            
            for metric_name in metric_names:
                # Collect values across sessions
                values = []
                for session_id, result in session_results.items():
                    if result.has_metric(metric_name):
                        metric_value = result.get_metric(metric_name)
                        if metric_value:
                            if isinstance(metric_value, list):
                                values.extend(metric_value)
                            else:
                                values.append(metric_value)
                
                if values:
                    cross_session_results[f"{metric_name}_mean"] = np.mean(values)
                    cross_session_results[f"{metric_name}_std"] = np.std(values)
                    cross_session_results[f"{metric_name}_median"] = np.median(values)
                    cross_session_results[f"{metric_name}_count"] = len(values)
        
        return cross_session_results
    
    def _time_a_to_b(self, dataframe: pd.DataFrame, session, args: Dict[str, Any], logger) -> List[float]:
        """Calculate time intervals between two spatial locations (preserving original logic)."""
        try:
            # Extract parameters (preserving original defaults)
            a = args.get('a')
            b = args.get('b')
            min_nodes_on_path = args.get('min_nodes_on_path', DEFAULT_MIN_NODES_ON_PATH)
            
            if a is None or b is None:
                logger.warning("time_a_to_b requires 'a' and 'b' parameters")
                return []
            
            # Set condition threshold (preserving original logic)
            condition.threshold = min_nodes_on_path
            
            # Find a-to-b indices using original utility
            a_to_b_indices = a_to_b(dataframe, TILE_ID, a, b, condition)
            
            # Calculate times (preserving original logic)
            session_stream_info = session.get_session_stream_info()
            fps_value = session_stream_info.get(FPS, 30)  # Default FPS if missing
            
            times_between_a_to_b = [
                (ind_b + 1 - ind_a) * (1 / fps_value) 
                for (ind_a, ind_b) in a_to_b_indices
            ]
            
            logger.debug(f"time_a_to_b: Found {len(times_between_a_to_b)} intervals between {a} and {b}")
            return times_between_a_to_b
            
        except Exception as e:
            logger.error(f"time_a_to_b calculation failed: {str(e)}")
            return []
    
    def _velocity_a_to_b(self, dataframe: pd.DataFrame, session, args: Dict[str, Any], logger) -> List[float]:
        """Calculate velocities between two spatial locations (preserving original logic)."""
        try:
            # Extract parameters
            a = args.get('a')
            b = args.get('b')
            min_nodes_on_path = args.get('min_nodes_on_path', DEFAULT_MIN_NODES_ON_PATH)
            
            if a is None or b is None:
                logger.warning("velocity_a_to_b requires 'a' and 'b' parameters")
                return []
            
            # Set condition threshold
            condition.threshold = min_nodes_on_path
            
            # Find a-to-b indices
            a_to_b_indices = a_to_b(dataframe, TILE_ID, a, b, condition)
            
            # Calculate times and distances (preserving original logic)
            session_stream_info = session.get_session_stream_info()
            fps_value = session_stream_info.get(FPS, 30)
            
            times_between_a_to_b = [
                (ind_b + 1 - ind_a) * (1 / fps_value) 
                for (ind_a, ind_b) in a_to_b_indices
            ]
            
            # Calculate path distances (preserving original logic)
            path_traveled_between_a_to_b = []
            
            # Get pixel-to-meter conversion from shared resources
            map_provider = session.shared_resources.get('maze_map')
            if map_provider:
                map_config = map_provider.get_map_configuration()
                pixel_to_meter = map_config.get('pixel_to_meter', 1.0)
            else:
                pixel_to_meter = 1.0
                logger.warning("No map provider found, using pixel_to_meter=1.0")
            
            for ind_a, ind_b in a_to_b_indices:
                try:
                    # Calculate path segments using keypoint coordinates
                    segment_data = dataframe.iloc[ind_a:ind_b + 1]
                    
                    # Use keypoint coordinates from data source plugins
                    x_coords = segment_data.get('keypoints_x', pd.Series())
                    y_coords = segment_data.get('keypoints_y', pd.Series())
                    
                    if not x_coords.empty and not y_coords.empty:
                        path_segments_length_pixels = (
                            (x_coords.diff() ** 2) + (y_coords.diff() ** 2)
                        ) ** 0.5
                        total_path_length_meters = path_segments_length_pixels.sum() / pixel_to_meter
                        path_traveled_between_a_to_b.append(total_path_length_meters)
                    else:
                        logger.warning(f"No keypoint data found for segment {ind_a}:{ind_b}")
                        path_traveled_between_a_to_b.append(0.0)
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate path distance for segment {ind_a}:{ind_b}: {e}")
                    path_traveled_between_a_to_b.append(0.0)
            
            # Calculate velocities
            velocities = []
            for distance, time_interval in zip(path_traveled_between_a_to_b, times_between_a_to_b):
                if time_interval > 0:
                    velocities.append(distance / time_interval)
                else:
                    velocities.append(0.0)
            
            logger.debug(f"velocity_a_to_b: Calculated {len(velocities)} velocities between {a} and {b}")
            return velocities
            
        except Exception as e:
            logger.error(f"velocity_a_to_b calculation failed: {str(e)}")
            return []