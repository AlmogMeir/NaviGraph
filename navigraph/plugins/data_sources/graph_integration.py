"""Graph integration plugin for NaviGraph.

This plugin wraps the current Graph functionality as a data source plugin,
preserving all existing behavior while adapting to the new plugin architecture.
It maps tile IDs from spatial data to graph nodes and edges.

The plugin requires tile_id data from previous data sources (typically map_integration)
and uses the graph configuration and tile-to-graph dictionary to determine
graph positions for spatial navigation analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Tuple
from pathlib import Path

from ...core.interfaces import IDataSource, DataSourceIntegrationError
from ...core.registry import register_data_source_plugin


@register_data_source_plugin("graph_integration")
class GraphIntegrationDataSource(IDataSource):
    """Integrates graph-based spatial navigation data - requires tile_id from spatial sources.
    
    This plugin converts tile IDs to graph positions (nodes, edges) using the existing
    Graph class logic. It preserves the tile-to-graph mapping dictionary while
    integrating with the new plugin architecture.
    
    The plugin adds these columns to the session DataFrame:
    - tree_position: Graph node/edge identifier (int for nodes, tuple for edges, frozenset for mixed)
    - graph_node: Specific graph node ID if position is a node (None otherwise)
    - graph_edge: Specific graph edge tuple if position is an edge (None otherwise)
    - path_to_reward: Shortest path from current position to reward (if reward configured)
    """
    
    def integrate_data_into_session(
        self,
        current_dataframe: pd.DataFrame,
        session_config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        logger
    ) -> pd.DataFrame:
        """Add graph-based columns using existing Graph logic.
        
        This method preserves the existing tile-to-graph mapping and pathfinding
        logic while adapting it to work with the new plugin system.
        
        Args:
            current_dataframe: DataFrame with tile_id from previous sources
            session_config: Configuration for this data source
            shared_resources: Shared resources including graph provider
            logger: Logger for progress reporting
            
        Returns:
            DataFrame with added graph columns (tree_position, graph_node, etc.)
            
        Raises:
            DataSourceIntegrationError: If prerequisites not met or processing fails
        """
        logger.info("Starting graph integration - mapping tiles to graph positions")
        
        try:
            # Get required resources
            graph_provider = shared_resources.get('graph')
            if not graph_provider:
                raise DataSourceIntegrationError(
                    "Graph integration requires 'graph' shared resource. "
                    "Make sure graph_provider is configured in shared_resources."
                )
            
            # Check for tile_id column from previous data source
            if 'tile_id' not in current_dataframe.columns:
                raise DataSourceIntegrationError(
                    "Graph integration requires 'tile_id' column from previous data sources. "
                    "Make sure map_integration or equivalent is configured before graph_integration."
                )
            
            # Process graph mapping for all frames
            graph_data = self._process_tile_to_graph_mapping(
                current_dataframe, graph_provider, session_config, logger
            )
            
            # Add new columns to existing DataFrame
            for col_name, col_data in graph_data.items():
                current_dataframe[col_name] = col_data
            
            # Log success statistics
            valid_positions = current_dataframe['tree_position'].notna()
            valid_count = valid_positions.sum()
            total_count = len(current_dataframe)
            
            logger.info(
                f"✓ Graph integration complete: {valid_count}/{total_count} frames "
                f"mapped to graph positions ({valid_count/total_count*100:.1f}%)"
            )
            
            return current_dataframe
            
        except Exception as e:
            raise DataSourceIntegrationError(
                f"Graph integration failed: {str(e)}"
            ) from e
    
    def validate_session_prerequisites(
        self, 
        current_dataframe: pd.DataFrame, 
        shared_resources: Dict[str, Any]
    ) -> bool:
        """Check for tile_id column and graph resources.
        
        This method validates that:
        1. Tile ID data is available from previous data sources
        2. Required shared resources (graph provider) are available
        
        Args:
            current_dataframe: Current DataFrame state
            shared_resources: Available shared resources
            
        Returns:
            True if all prerequisites are met
        """
        # Check for required columns
        has_tile_id = 'tile_id' in current_dataframe.columns
        
        # Check for required resources  
        required_resources = self.get_required_shared_resources()
        has_resources = all(res in shared_resources for res in required_resources)
        
        return has_tile_id and has_resources
    
    def get_provided_column_names(self) -> List[str]:
        """Return column names this data source provides."""
        return ['tree_position', 'graph_node', 'graph_edge', 'path_to_reward']
    
    def get_required_columns(self) -> List[str]:
        """Return column names required by this data source."""
        return ['tile_id']
    
    def get_required_shared_resources(self) -> List[str]:
        """Return shared resource names required by this data source."""
        return ['graph']
    
    def _process_tile_to_graph_mapping(
        self, 
        dataframe: pd.DataFrame,
        graph_provider,
        session_config: Dict[str, Any],
        logger
    ) -> Dict[str, pd.Series]:
        """Process tile-to-graph mapping using existing Graph logic.
        
        This method preserves the existing tile-to-graph mapping dictionary
        and pathfinding logic while adapting it to the plugin system.
        
        Args:
            dataframe: DataFrame with tile_id column
            graph_provider: Graph provider resource
            session_config: Session configuration
            logger: Logger instance
            
        Returns:
            Dictionary with new column data
        """
        n_frames = len(dataframe)
        
        # Initialize output arrays
        tree_positions = [None] * n_frames
        graph_nodes = [None] * n_frames
        graph_edges = [None] * n_frames
        paths_to_reward = [None] * n_frames
        
        # Get graph instance and configuration
        graph_instance = graph_provider.get_graph_instance()
        reward_tile_id = session_config.get('reward_tile_id', None)
        
        # Calculate reward path if configured
        reward_path = None
        if reward_tile_id is not None:
            reward_path = self._calculate_reward_path(
                graph_instance, reward_tile_id, logger
            )
        
        # Process each frame (preserving existing logic)
        for frame_idx in range(n_frames):
            try:
                tile_id = dataframe.iloc[frame_idx]['tile_id']
                
                # Skip invalid tile IDs
                if pd.isna(tile_id) or tile_id == -1:
                    continue
                
                # Get tree position using existing logic
                tree_position = self._get_tree_position_from_tile(
                    int(tile_id), graph_instance
                )
                
                if tree_position is not None:
                    tree_positions[frame_idx] = tree_position
                    
                    # Extract specific node/edge information
                    node_info, edge_info = self._extract_node_edge_info(tree_position)
                    graph_nodes[frame_idx] = node_info
                    graph_edges[frame_idx] = edge_info
                    
                    # Calculate path to reward if available
                    if reward_path is not None and node_info is not None:
                        paths_to_reward[frame_idx] = reward_path
                
            except Exception as e:
                logger.warning(f"Failed to process graph mapping for frame {frame_idx}: {e}")
                continue
        
        # Log mapping statistics
        valid_positions = sum(1 for pos in tree_positions if pos is not None)
        valid_nodes = sum(1 for node in graph_nodes if node is not None)
        valid_edges = sum(1 for edge in graph_edges if edge is not None)
        
        logger.debug(
            f"Graph mapping: {valid_positions}/{n_frames} valid positions "
            f"({valid_nodes} nodes, {valid_edges} edges)"
        )
        
        return {
            'tree_position': pd.Series(tree_positions, index=dataframe.index),
            'graph_node': pd.Series(graph_nodes, index=dataframe.index),
            'graph_edge': pd.Series(graph_edges, index=dataframe.index),
            'path_to_reward': pd.Series(paths_to_reward, index=dataframe.index)
        }
    
    def _get_tree_position_from_tile(self, tile_id: int, graph_instance) -> Union[int, Tuple, frozenset, None]:
        """Get tree position from tile ID using existing Graph logic."""
        try:
            # Use the existing get_tree_location method from Graph class
            return graph_instance.get_tree_location(tile_id)
        except Exception:
            return None
    
    def _extract_node_edge_info(self, tree_position) -> Tuple[Union[int, None], Union[Tuple, None]]:
        """Extract specific node and edge information from tree position.
        
        The original logic handles:
        - int: Pure node
        - tuple: Pure edge  
        - frozenset: Mixed node/edge combination
        
        Args:
            tree_position: Tree position from graph mapping
            
        Returns:
            Tuple of (node_id, edge_tuple) where either can be None
        """
        if tree_position is None:
            return None, None
        
        # Handle different tree position types (preserving existing logic)  
        if isinstance(tree_position, int):
            # Pure node
            return tree_position, None
        elif isinstance(tree_position, tuple):
            # Pure edge
            return None, tree_position
        elif isinstance(tree_position, frozenset):
            # Mixed node/edge - extract both (preserving existing logic)
            node_id = None
            edge_tuple = None
            
            for item in tree_position:
                if isinstance(item, int):
                    node_id = item
                elif isinstance(item, tuple):
                    edge_tuple = item
            
            return node_id, edge_tuple
        else:
            # Unknown format
            return None, None
    
    def _calculate_reward_path(self, graph_instance, reward_tile_id: int, logger) -> List:
        """Calculate shortest path to reward using existing Graph logic."""
        try:
            # Get reward position in graph
            reward_tree_location = graph_instance.get_tree_location(reward_tile_id)
            
            if reward_tree_location is None:
                logger.warning(f"Reward tile {reward_tile_id} not found in graph")
                return None
            
            # Determine reward node ID (preserving existing logic)
            if isinstance(reward_tree_location, int):
                reward_node_id = reward_tree_location
            elif isinstance(reward_tree_location, tuple):
                logger.warning("Reward tile maps to edge, not node - cannot calculate shortest path")
                return None
            elif isinstance(reward_tree_location, frozenset):
                # Extract node from frozenset (preserving existing logic)
                reward_node_id = None
                for item in reward_tree_location:
                    if isinstance(item, int):
                        reward_node_id = item
                        break
                
                if reward_node_id is None:
                    logger.warning("No node found in reward tile mapping")
                    return None
            else:
                logger.warning(f"Unknown reward tree location format: {reward_tree_location}")
                return None
            
            # Calculate shortest path from source (node 0) to reward
            path_to_reward = graph_instance.get_shortest_path(source=0, target=reward_node_id)
            
            # Add edges to path (preserving existing logic from Session class)
            path_with_edges = path_to_reward + [
                (node_1, node_2) for node_1, node_2 in 
                zip(path_to_reward, path_to_reward[1:])
            ]
            
            logger.info(f"Calculated reward path: {len(path_to_reward)} nodes, {len(path_with_edges) - len(path_to_reward)} edges")
            return path_with_edges
            
        except Exception as e:
            logger.warning(f"Failed to calculate reward path: {e}")
            return None