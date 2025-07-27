"""Exploration metrics analyzer plugin for NaviGraph.

Computes exploration patterns and node visit statistics.
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
AVG_NODE_TIME_BY_PATH = 'by_path'
AVG_NODE_TIME_OVERALL = 'overall'

# Helper condition - preserving existing logic
count_unique_nodes = partial(count_unique_type_specific_objects, dtype=NODE_POSSIBLE_DTYPES)
condition = Condition(column_name=TREE_POSITION,
                      func=count_unique_nodes,
                      threshold=DEFAULT_MIN_NODES_ON_PATH,
                      operator=operator.gt)


@register_analyzer_plugin("exploration_metrics")
class ExplorationMetricsAnalyzer(BasePlugin, IAnalyzer):
    """Provides exploration analysis metrics: exploration patterns and node visits.
    
    This analyzer wraps the original exploration_percentage and avg_node_time methods,
    preserving all existing functionality while integrating with the new plugin architecture.
    """
    
    @property
    def required_columns(self) -> List[str]:
        """Required DataFrame columns for this analyzer."""
        return ['tile_id', 'tree_position']  # Required for exploration analysis
    
    @property
    def analyzer_type(self) -> str:
        """Type of analyzer: supports both session and cross-session analysis."""
        return 'both'
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance: Logger = None):
        """Factory method to create exploration metrics analyzer from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate exploration metrics analyzer configuration."""
        # No required config keys for this analyzer
        pass
    
    def analyze_session(self, session) -> AnalysisResult:
        """Analyze a single session for exploration metrics.
        
        Args:
            session: Session object with integrated DataFrame and configuration
            
        Returns:
            AnalysisResult with computed exploration metrics
        """
        start_time = time.time()
        
        try:
            # Get session data
            dataframe = session.get_integrated_dataframe()
            session_config = session.get_session_config()
            logger = session.logger
            
            logger.info("Computing exploration metrics for session")
            
            # Extract configuration for exploration analysis
            analyzer_config = session_config.get('analyze', {}).get('metrics', {})
            results = {}
            
            # Process each configured metric
            for metric_name, metric_config in analyzer_config.items():
                func_name = metric_config.get('func_name')
                args = metric_config.get('args', {})
                
                if func_name == 'exploration_percentage':
                    results[metric_name] = self._exploration_percentage(
                        dataframe, session, args, logger
                    )
                elif func_name == 'avg_node_time':
                    results[metric_name] = self._avg_node_time(
                        dataframe, session, args, logger
                    )
            
            logger.info(f"Exploration metrics computed: {list(results.keys())}")
            
            # Create metadata
            metadata = AnalysisMetadata(
                analyzer_name="exploration_metrics",
                version="1.0.0",
                timestamp=datetime.now(),
                computation_time=time.time() - start_time,
                config_hash=compute_configuration_hash(analyzer_config)
            )
            
            # Return structured result
            return AnalysisResult(
                session_id=session.session_id,
                analyzer_name="exploration_metrics",
                metrics=results,
                metadata=metadata
            )
            
        except Exception as e:
            session.logger.error(f"Exploration metrics analysis failed: {str(e)}")
            # Return empty result with error metadata
            metadata = AnalysisMetadata(
                analyzer_name="exploration_metrics",
                version="1.0.0",
                timestamp=datetime.now(),
                computation_time=time.time() - start_time,
                config_hash=""
            )
            return AnalysisResult(
                session_id=session.session_id,
                analyzer_name="exploration_metrics",
                metrics={},
                metadata=metadata
            )
    
    def analyze_cross_session(
        self,
        sessions: List,
        session_results: Dict[str, AnalysisResult]
    ) -> Dict[str, Any]:
        """Perform cross-session analysis for exploration metrics.
        
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
                # Handle different metric types
                if 'exploration_percentage' in metric_name:
                    # For exploration percentage, compute final values
                    final_values = []
                    for session_id, result in session_results.items():
                        if result.has_metric(metric_name):
                            # Get final exploration percentage
                            values = result.get_metric(metric_name)
                            if isinstance(values, list) and values:
                                final_values.append(values[-1])
                            elif isinstance(values, (int, float)):
                                final_values.append(values)
                    
                    if final_values:
                        cross_session_results[f"{metric_name}_mean"] = np.mean(final_values)
                        cross_session_results[f"{metric_name}_std"] = np.std(final_values)
                        cross_session_results[f"{metric_name}_median"] = np.median(final_values)
                        
                elif 'avg_node_time' in metric_name:
                    # For node time metrics, handle dictionary format
                    all_node_avgs = []
                    all_node_medians = []
                    
                    for session_id, result in session_results.items():
                        if result.has_metric(metric_name):
                            node_data = result.get_metric(metric_name)
                            if isinstance(node_data, dict):
                                if 'node_avg' in node_data:
                                    all_node_avgs.append(node_data['node_avg'])
                                if 'node_median' in node_data:
                                    all_node_medians.append(node_data['node_median'])
                            elif isinstance(node_data, list):
                                # Handle list format for by_path mode
                                for item in node_data:
                                    if isinstance(item, dict):
                                        if 'node_avg' in item:
                                            all_node_avgs.append(item['node_avg'])
                                        if 'node_median' in item:
                                            all_node_medians.append(item['node_median'])
                    
                    if all_node_avgs:
                        cross_session_results[f"{metric_name}_avg_mean"] = np.mean(all_node_avgs)
                        cross_session_results[f"{metric_name}_avg_std"] = np.std(all_node_avgs)
                    
                    if all_node_medians:
                        cross_session_results[f"{metric_name}_median_mean"] = np.mean(all_node_medians)
                        cross_session_results[f"{metric_name}_median_std"] = np.std(all_node_medians)
        
        return cross_session_results
    
    def _exploration_percentage(self, dataframe: pd.DataFrame, session, args: Dict[str, Any], logger) -> List[float]:
        """Calculate exploration percentage over time (preserving original logic)."""
        try:
            # Get graph provider for total node count
            graph_provider = session.shared_resources.get('graph')
            if not graph_provider:
                logger.error("Graph provider required for exploration percentage calculation")
                return []
            
            graph_instance = graph_provider.get_graph_instance()
            total_nodes = graph_instance.tree.number_of_nodes()
            
            if total_nodes == 0:
                logger.warning("No nodes found in graph")
                return []
            
            # Extract node positions (preserving original logic)
            def keep_node_positions(x):
                if isinstance(x, int):
                    return x
                elif isinstance(x, frozenset):
                    node_items = [i for i in x if isinstance(i, int)]
                    return node_items[0] if node_items else None
                else:
                    return None
            
            # Apply node extraction to tree positions
            tree_positions = dataframe[TREE_POSITION]
            nodes_visited = tree_positions.apply(keep_node_positions)
            
            # Calculate cumulative unique node visits (preserving original logic)
            na_indicator = nodes_visited.isna()
            flag_unique = ~nodes_visited.duplicated() & ~na_indicator
            cumulative_exploration = flag_unique.cumsum() / total_nodes
            
            # Convert to list for consistency
            exploration_percentages = cumulative_exploration.tolist()
            
            logger.debug(f"exploration_percentage: Final exploration = {exploration_percentages[-1]:.3f}")
            return exploration_percentages
            
        except Exception as e:
            logger.error(f"exploration_percentage calculation failed: {str(e)}")
            return []
    
    def _avg_node_time(self, dataframe: pd.DataFrame, session, args: Dict[str, Any], logger) -> Dict[str, Any]:
        """Calculate average time spent in nodes (preserving original logic)."""
        try:
            # Extract parameters
            mode = args.get('mode', AVG_NODE_TIME_OVERALL)
            a = args.get('a', None)
            b = args.get('b', None)
            min_nodes_on_path = args.get('min_nodes_on_path', DEFAULT_MIN_NODES_ON_PATH)
            
            # Get session FPS for time conversion
            session_stream_info = session.get_session_stream_info()
            fps_value = session_stream_info.get(FPS, 30)
            
            # Helper function to calculate time in nodes (preserving original logic)
            def time_in_node_sec(nodes: pd.Series, time_in_sec=False):
                nodes = nodes.dropna()
                visits = {}
                
                for node in nodes:
                    node_id = node if isinstance(node, int) else next(
                        (item for item in node if isinstance(item, int)), None
                    )
                    
                    if node_id is not None:
                        if node_id not in visits:
                            visits[node_id] = 0
                        
                        if time_in_sec:
                            visits[node_id] += 1 / fps_value  # Convert frame to seconds
                        else:
                            visits[node_id] += 1
                
                if visits:
                    node_avg = np.mean(list(visits.values()))
                    node_median = np.median(list(visits.values()))
                    visits['node_avg'] = node_avg
                    visits['node_median'] = node_median
                
                return visits
            
            # Calculate based on mode
            if mode == AVG_NODE_TIME_OVERALL:
                # Overall mode - analyze entire session
                result = time_in_node_sec(dataframe[TREE_POSITION], time_in_sec=True)
                logger.debug(f"avg_node_time (overall): {len(result)-2} unique nodes analyzed")
                return result
                
            elif mode == AVG_NODE_TIME_BY_PATH:
                # By path mode - analyze individual paths
                if a is None or b is None:
                    logger.warning("avg_node_time by_path mode requires 'a' and 'b' parameters")
                    return {}
                
                condition.threshold = min_nodes_on_path
                a_to_b_indices = a_to_b(dataframe, TILE_ID, a, b, condition)
                
                path_results = []
                for ind_a, ind_b in a_to_b_indices:
                    tree_frame_position = dataframe[ind_a:ind_b + 1][TREE_POSITION]
                    path_result = time_in_node_sec(tree_frame_position, time_in_sec=True)
                    path_results.append(path_result)
                
                logger.debug(f"avg_node_time (by_path): Analyzed {len(path_results)} paths")
                return path_results
                
            else:
                logger.error(f"Unsupported avg_node_time mode: {mode}")
                return {}
            
        except Exception as e:
            logger.error(f"avg_node_time calculation failed: {str(e)}")
            return {}