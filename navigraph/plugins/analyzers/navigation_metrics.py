"""Navigation metrics analyzer plugin for NaviGraph.

Computes path analysis and graph-based navigation measurements.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import operator
from functools import partial
from datetime import datetime
import time

# Import from local utils module
from .utils import a_to_b, count_unique_type_specific_objects, count_node_visits_eliminating_sequences, Condition

from ...core.interfaces import IAnalyzer, Logger
from ...core.base_plugin import BasePlugin
from ...core.registry import register_analyzer_plugin
from ...core.types import AnalysisResult, AnalysisMetadata
from ...core.utils import compute_configuration_hash

# Constants from original SessionAnalyzer
TILE_ID = 'tile_id'
TREE_POSITION = 'tree_position'
NODE_POSSIBLE_DTYPES = (int, frozenset)
DEFAULT_MIN_NODES_ON_PATH = 0
NODE_PATH_TOPOLOGICAL_MODE = 'topological_distance'
NODE_PATH_EXPLORATION_MODE = 'exploration'

# Helper condition - preserving existing logic
count_unique_nodes = partial(count_unique_type_specific_objects, dtype=NODE_POSSIBLE_DTYPES)
condition = Condition(column_name=TREE_POSITION,
                      func=count_unique_nodes,
                      threshold=DEFAULT_MIN_NODES_ON_PATH,
                      operator=operator.gt)


@register_analyzer_plugin("navigation_metrics")
class NavigationMetricsAnalyzer(BasePlugin, IAnalyzer):
    """Provides navigation analysis metrics: path lengths and shortest path analysis.
    
    This analyzer wraps the original num_nodes_in_path and shortest_path_from_a_to_b methods,
    preserving all existing functionality while integrating with the new plugin architecture.
    """
    
    @property
    def required_columns(self) -> List[str]:
        """Required DataFrame columns for this analyzer."""
        return ['tile_id', 'tree_position']  # Required for navigation analysis
    
    @property
    def analyzer_type(self) -> str:
        """Type of analyzer: supports both session and cross-session analysis."""
        return 'both'
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance: Logger = None):
        """Factory method to create navigation metrics analyzer from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate navigation metrics analyzer configuration."""
        # No required config keys for this analyzer
        pass
    
    def analyze_session(self, session) -> AnalysisResult:
        """Analyze a single session for navigation metrics.
        
        Args:
            session: Session object with integrated DataFrame and configuration
            
        Returns:
            AnalysisResult with computed navigation metrics
        """
        start_time = time.time()
        
        try:
            # Get session data
            dataframe = session.get_integrated_dataframe()
            session_config = session.get_session_config()
            logger = session.logger
            
            logger.info("Computing navigation metrics for session")
            
            # Extract configuration for navigation analysis
            analyzer_config = session_config.get('analyze', {}).get('metrics', {})
            results = {}
            
            # Process each configured metric
            for metric_name, metric_config in analyzer_config.items():
                func_name = metric_config.get('func_name')
                args = metric_config.get('args', {})
                
                if func_name == 'num_nodes_in_path':
                    results[metric_name] = self._num_nodes_in_path(
                        dataframe, session, args, logger
                    )
                elif func_name == 'shortest_path_from_a_to_b':
                    results[metric_name] = self._shortest_path_from_a_to_b(
                        dataframe, session, args, logger
                    )
            
            logger.info(f"Navigation metrics computed: {list(results.keys())}")
            
            # Create metadata
            metadata = AnalysisMetadata(
                analyzer_name="navigation_metrics",
                version="1.0.0",
                timestamp=datetime.now(),
                computation_time=time.time() - start_time,
                config_hash=compute_configuration_hash(analyzer_config)
            )
            
            # Return structured result
            return AnalysisResult(
                session_id=session.session_id,
                analyzer_name="navigation_metrics",
                metrics=results,
                metadata=metadata
            )
            
        except Exception as e:
            session.logger.error(f"Navigation metrics analysis failed: {str(e)}")
            # Return empty result with error metadata
            metadata = AnalysisMetadata(
                analyzer_name="navigation_metrics",
                version="1.0.0",
                timestamp=datetime.now(),
                computation_time=time.time() - start_time,
                config_hash=""
            )
            return AnalysisResult(
                session_id=session.session_id,
                analyzer_name="navigation_metrics",
                metrics={},
                metadata=metadata
            )
    
    def analyze_cross_session(
        self,
        sessions: List,
        session_results: Dict[str, AnalysisResult]
    ) -> Dict[str, Any]:
        """Perform cross-session analysis for navigation metrics.
        
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
    
    def _num_nodes_in_path(self, dataframe: pd.DataFrame, session, args: Dict[str, Any], logger) -> List[int]:
        """Calculate number of nodes in paths between locations (preserving original logic)."""
        try:
            # Extract parameters (preserving original defaults)
            a = args.get('a')
            b = args.get('b')
            min_nodes_on_path = args.get('min_nodes_on_path', DEFAULT_MIN_NODES_ON_PATH)
            mode = args.get('mode', NODE_PATH_TOPOLOGICAL_MODE)
            min_frame_rep = args.get('min_frame_rep', 1)
            
            if a is None or b is None:
                logger.warning("num_nodes_in_path requires 'a' and 'b' parameters")
                return []
            
            # Set condition threshold (preserving original logic)
            condition.threshold = min_nodes_on_path
            
            # Find a-to-b indices using original utility
            a_to_b_indices = a_to_b(dataframe, TILE_ID, a, b, condition)
            
            # Calculate node counts for each path
            num_nodes_in_path = []
            
            # Get graph provider for exploration mode
            graph_provider = session.shared_resources.get('graph')
            total_nodes = None
            if graph_provider and mode == NODE_PATH_EXPLORATION_MODE:
                graph_instance = graph_provider.get_graph_instance()
                total_nodes = graph_instance.tree.number_of_nodes()
            
            for ind_a, ind_b in a_to_b_indices:
                try:
                    if mode == NODE_PATH_TOPOLOGICAL_MODE:
                        # Topological distance mode (preserving original logic)
                        tree_frame_position = dataframe[ind_a:ind_b + 1][TREE_POSITION]
                        node_count = count_node_visits_eliminating_sequences(tree_frame_position)
                        num_nodes_in_path.append(node_count)
                        
                    elif mode == NODE_PATH_EXPLORATION_MODE:
                        # Exploration mode (preserving original logic)
                        tree_frame_position = dataframe[ind_a:ind_b + 1][TREE_POSITION]
                        unique_nodes = count_unique_type_specific_objects(
                            tree_frame_position, NODE_POSSIBLE_DTYPES
                        )
                        
                        if total_nodes and total_nodes > 0:
                            exploration_ratio = unique_nodes / total_nodes
                            num_nodes_in_path.append(exploration_ratio)
                        else:
                            logger.warning("Total nodes not available for exploration mode")
                            num_nodes_in_path.append(unique_nodes)
                    else:
                        logger.warning(f"Unknown mode: {mode}")
                        num_nodes_in_path.append(0)
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate nodes for path {ind_a}:{ind_b}: {e}")
                    num_nodes_in_path.append(0)
            
            logger.debug(f"num_nodes_in_path: Calculated {len(num_nodes_in_path)} path lengths between {a} and {b}")
            return num_nodes_in_path
            
        except Exception as e:
            logger.error(f"num_nodes_in_path calculation failed: {str(e)}")
            return []
    
    def _shortest_path_from_a_to_b(self, dataframe: pd.DataFrame, session, args: Dict[str, Any], logger) -> List[int]:
        """Calculate shortest path lengths with strike allowance (preserving original logic)."""
        try:
            # Extract parameters
            a = args.get('a')
            b = args.get('b')
            min_nodes_on_path = args.get('min_nodes_on_path', DEFAULT_MIN_NODES_ON_PATH)
            levels = args.get('levels', None)
            strikes = args.get('strikes', None)
            
            if a is None or b is None:
                logger.warning("shortest_path_from_a_to_b requires 'a' and 'b' parameters")
                return []
            
            # Set condition threshold
            condition.threshold = min_nodes_on_path
            
            # Find a-to-b indices
            a_to_b_indices = a_to_b(dataframe, TILE_ID, a, b, condition)
            
            # Get graph provider to find reward node
            graph_provider = session.shared_resources.get('graph')
            if not graph_provider:
                logger.error("Graph provider required for shortest path calculation")
                return []
            
            graph_instance = graph_provider.get_graph_instance()
            
            # Get reward node location (preserving original logic)
            b_node = graph_instance.get_tree_location(b)
            if isinstance(b_node, int):
                reward_node = b_node
            elif isinstance(b_node, frozenset):
                reward_node = next((item for item in b_node if isinstance(item, int)), None)
                if reward_node is None:
                    logger.error(f"No node found in reward location: {b_node}")
                    return []
            else:
                logger.error(f"Invalid reward node format: {b_node}")
                return []
            
            # Calculate shortest paths with strike logic (preserving original algorithm)
            num_nodes_on_shortest_path = []
            
            for ind_a, ind_b in a_to_b_indices:
                try:
                    tree_frame_position = dataframe[ind_a:ind_b + 1][TREE_POSITION]
                    ind_on_shortest_path = -1
                    shortest_path_from_current_node_to_reward = None
                    strike = 0
                    previous_node = None
                    
                    for frame_num, cur_node in tree_frame_position.items():
                        # Skip any type of nan value
                        if cur_node is None or (isinstance(cur_node, float) and np.isnan(cur_node)):
                            continue
                        
                        # Skip edges
                        if isinstance(cur_node, tuple):
                            continue
                        
                        # Handle cases where the current node is both a node and an edge
                        if isinstance(cur_node, frozenset):
                            cur_node = next((item for item in cur_node if isinstance(item, int)), None)
                            if cur_node is None:
                                continue
                        
                        # On the first non-nan value update the shortest path
                        if cur_node is not None and ind_on_shortest_path == -1:
                            shortest_path_from_current_node_to_reward = graph_instance.get_shortest_path(
                                cur_node, reward_node
                            )
                            ind_on_shortest_path = 0
                            continue
                        
                        # Skip the current node if it is the same as the previous node
                        if cur_node == previous_node:
                            continue
                        
                        # Update the previous node
                        previous_node = cur_node
                        
                        # Allow for up to strikes mistakes on specific tree levels
                        if levels is not None and strikes is not None:
                            if cur_node != shortest_path_from_current_node_to_reward[ind_on_shortest_path + 1]:
                                if int(str(cur_node)[0]) in levels:
                                    if strike < strikes:
                                        strike += 1
                                        continue
                        
                        if cur_node == shortest_path_from_current_node_to_reward[ind_on_shortest_path + 1]:
                            ind_on_shortest_path += 1
                            strike = 0
                            # Check if we reached the reward - if so, break
                            if ind_on_shortest_path == len(shortest_path_from_current_node_to_reward) - 1:
                                break
                            else:
                                continue
                        else:
                            # Reset the shortest path
                            shortest_path_from_current_node_to_reward = graph_instance.get_shortest_path(
                                cur_node, reward_node
                            )
                            ind_on_shortest_path = 0
                    
                    # Add the shortest path length to list
                    if shortest_path_from_current_node_to_reward:
                        num_nodes_on_shortest_path.append(len(shortest_path_from_current_node_to_reward))
                    else:
                        num_nodes_on_shortest_path.append(0)
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate shortest path for segment {ind_a}:{ind_b}: {e}")
                    num_nodes_on_shortest_path.append(0)
            
            logger.debug(f"shortest_path_from_a_to_b: Calculated {len(num_nodes_on_shortest_path)} shortest paths")
            return num_nodes_on_shortest_path
            
        except Exception as e:
            logger.error(f"shortest_path_from_a_to_b calculation failed: {str(e)}")
            return []