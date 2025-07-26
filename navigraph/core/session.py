"""Enhanced Session class for NaviGraph.

This module provides the core Session class that orchestrates multi-source data
integration in configuration-specified order.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import queue
from loguru import logger

# Type alias for logger
Logger = type(logger)

from .interfaces import IDataSource, DataSourceIntegrationError, PluginPrerequisiteError
from .registry import PluginRegistry, registry
from .file_discovery import FileDiscoveryEngine


class SessionInitializationError(Exception):
    """Raised when session initialization fails."""
    pass


class Session:
    """Core session that orchestrates multi-source data integration."""
    
    def __init__(
        self,
        session_configuration: Dict[str, Any],
        logger_instance: Logger,
        session_identifier: Optional[str] = None,
        file_discovery_engine: Optional[FileDiscoveryEngine] = None,
        plugin_registry: Optional[PluginRegistry] = None
    ):
        """Initialize session with data source orchestration."""
        self.session_id = session_identifier or session_configuration.get('session_id', 'unknown_session')
        self.config = session_configuration
        self.shared_resources = session_configuration.get('shared_resources', {})
        self.file_discovery = file_discovery_engine or FileDiscoveryEngine()
        self.registry = plugin_registry or registry  # Use global registry if not provided
        self.logger = logger_instance
        
        # Lazy-loaded data - populated on first access
        self._integrated_dataframe: Optional[pd.DataFrame] = None
        self._node_level_features: Optional[pd.DataFrame] = None
        
        # Data source instances in configuration order
        self.data_source_instances: List[Tuple[str, IDataSource, Dict[str, Any]]] = []
        
        # File discovery results
        self._discovered_file_paths: Dict[str, Optional[str]] = {}
        
        # Backward compatibility properties
        self._session_name: Optional[str] = None
        self._video_file_path: Optional[str] = None
        self._likelihood_threshold: Optional[float] = None
        self._bodyparts: Optional[List[str]] = None
        self._coords: Optional[List[str]] = None
        
        # Tree visualization state (for backward compatibility)
        self.session_history = queue.Queue()
        self._path_to_reward: Optional[List[int]] = None
        
        try:
            self._initialize_session()
            self.logger.info(f"Initialized session: {self.session_id}")
        except Exception as e:
            raise SessionInitializationError(
                f"Failed to initialize session {self.session_id}: {str(e)}"
            ) from e
    
    def get_integrated_dataframe(self) -> pd.DataFrame:
        """Get the fully integrated DataFrame from all data sources."""
        if self._integrated_dataframe is not None:
            return self._integrated_dataframe
        
        return self._integrate_all_data_sources()
    
    def get_graph_structure(self) -> Any:
        """Get the graph structure from shared resources."""
        return self.shared_resources.get('graph')
    
    def get_session_config(self) -> Dict[str, Any]:
        """Get session configuration for analyzer plugins."""
        return self.config.copy()
    
    def get_session_stream_info(self) -> Dict[str, Any]:
        """Get session stream information extracted from video file."""
        # Try to get video file path from session configuration
        video_path = None
        
        # Look for video file in discovered files or configuration
        for ds_config in self.config.get('data_sources', []):
            if ds_config.get('plugin_name') == 'deeplabcut':
                video_path = ds_config.get('discovered_file_path')
                # Convert H5 path to video path by replacing extension
                if video_path and video_path.endswith('.h5'):
                    # Try common video extensions
                    for ext in ['.mp4', '.avi', '.mov']:
                        potential_video = video_path.replace('.h5', ext)
                        if Path(potential_video).exists():
                            video_path = potential_video
                            break
                break
        
        # If no video path found, return defaults
        if not video_path or not Path(video_path).exists():
            df_len = len(self.get_integrated_dataframe()) if self._integrated_dataframe is not None else 0
            return {
                'fps': 30.0,  # Default FPS
                'frame_count': df_len,
                'duration': df_len / 30.0
            }
        
        # Extract real video information
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count, 
                'duration': duration
            }
        except Exception as e:
            self.logger.warning(f"Could not extract video info from {video_path}: {str(e)}")
            df_len = len(self.get_integrated_dataframe()) if self._integrated_dataframe is not None else 0
            return {
                'fps': 30.0,  # Default FPS
                'frame_count': df_len,
                'duration': df_len / 30.0
            }
    
    def get_session_metadata(self) -> Dict[str, Any]:
        """Get comprehensive metadata about this session."""
        metadata = {
            'session_id': self.session_id,
            'data_sources_count': len(self.data_source_instances),
            'data_source_names': [name for name, _, _ in self.data_source_instances],
            'discovered_files': self._discovered_file_paths.copy(),
            'shared_resources_available': list(self.shared_resources.keys()),
            'integration_status': 'not_started'
        }
        
        # Add data statistics if integration has been performed
        if self._integrated_dataframe is not None:
            df = self._integrated_dataframe
            metadata.update({
                'total_frames': len(df),
                'total_columns': len(df.columns),
                'frame_range': [int(df.index.min()), int(df.index.max())],
                'column_names': list(df.columns),
                'integration_status': 'completed'
            })
        
        return metadata
    
    # =============================================================================
    # BACKWARD COMPATIBILITY METHODS (from old session/session.py)
    # =============================================================================
    
    @property
    def session_name(self) -> str:
        """Get session name for backward compatibility."""
        if self._session_name is None:
            # Extract from session_id or configuration
            self._session_name = self.session_id.replace('session_', '')
        return self._session_name
    
    @property 
    def bodyparts(self) -> List[str]:
        """Get available body parts from integrated DataFrame."""
        if self._bodyparts is None:
            df = self.get_integrated_dataframe()
            # Extract bodyparts from multi-level columns if they exist
            if isinstance(df.columns, pd.MultiIndex):
                if df.columns.nlevels >= 2:
                    self._bodyparts = list(df.columns.get_level_values(1).unique())
                else:
                    self._bodyparts = []
            else:
                # Look for coordinate columns pattern (bodypart_x, bodypart_y, bodypart_likelihood)
                bodypart_cols = [col for col in df.columns if col.endswith(('_x', '_y', '_likelihood'))]
                bodyparts_set = set()
                for col in bodypart_cols:
                    for suffix in ['_x', '_y', '_likelihood']:
                        if col.endswith(suffix):
                            bodyparts_set.add(col.replace(suffix, ''))
                            break
                self._bodyparts = list(bodyparts_set)
        return self._bodyparts or []
    
    @property
    def coords(self) -> List[str]:
        """Get coordinate types for backward compatibility."""
        if self._coords is None:
            self._coords = ['x', 'y', 'likelihood']  # Standard coordinate types
        return self._coords
    
    @property
    def likelihood(self) -> Optional[float]:
        """Get likelihood threshold for backward compatibility."""
        if self._likelihood_threshold is None:
            session_settings = self.config.get('session_settings', {})
            self._likelihood_threshold = session_settings.get('likelihood', 0.3)
        return self._likelihood_threshold
    
    @property
    def map_labeler(self) -> Any:
        """Get map labeler from shared resources for backward compatibility."""
        map_provider = self.shared_resources.get('maze_map')
        if map_provider:
            return map_provider
        else:
            raise AttributeError("No map provider available in shared resources")
    
    @property 
    def tree(self) -> Any:
        """Get graph/tree from shared resources for backward compatibility."""
        graph_provider = self.shared_resources.get('graph')
        if graph_provider:
            return graph_provider.get_graph_instance()
        else:
            raise AttributeError("No graph provider available in shared resources")
    
    @property
    def path_to_reward(self) -> Optional[List[int]]:
        """Get shortest path to reward tile for backward compatibility."""
        if self._path_to_reward is None:
            try:
                reward_tile_id = self.config.get('reward_tile_id')
                if reward_tile_id is not None and 'graph' in self.shared_resources:
                    graph = self.tree
                    tree_loc = graph.get_tree_location(reward_tile_id)
                    
                    if isinstance(tree_loc, int):
                        reward_node_id = tree_loc
                    elif isinstance(tree_loc, tuple):
                        raise ValueError('tile_id must be related to a node, not an edge, for shortest path calculation')
                    else:
                        # Handle frozenset or other complex types
                        reward_node_id = [item for item in tree_loc if isinstance(item, int)][0]
                    
                    self._path_to_reward = graph.get_shortest_path(source=0, target=reward_node_id)
            except Exception as e:
                self.logger.warning(f"Could not calculate path to reward: {str(e)}")
                self._path_to_reward = []
        
        return self._path_to_reward
    
    def get_coords(self, frame_number: int, body_part: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get coordinates for specific frame and body part.
        
        Args:
            frame_number: Frame index to get coordinates for
            body_part: Body part name (e.g., 'nose', 'tailbase')
            
        Returns:
            Tuple of (x, y, likelihood) or (None, None, None) if not found/low confidence
        """
        if body_part not in self.bodyparts:
            raise KeyError(f'Required bodypart not documented in session. Available bodyparts: {self.bodyparts}')
        
        df = self.get_integrated_dataframe()
        
        if frame_number not in df.index:
            return None, None, None
        
        try:
            # Try multi-level column access first (DeepLabCut format)
            if isinstance(df.columns, pd.MultiIndex):
                if df.columns.nlevels >= 2:
                    # Look for scorer/bodypart/coord structure
                    scorer_level = df.columns.get_level_values(0)[0]  # Get first scorer
                    x = df.loc[frame_number, (scorer_level, body_part, 'x')]
                    y = df.loc[frame_number, (scorer_level, body_part, 'y')]
                    likelihood = df.loc[frame_number, (scorer_level, body_part, 'likelihood')]
                else:
                    # Two-level: bodypart/coord
                    x = df.loc[frame_number, (body_part, 'x')]
                    y = df.loc[frame_number, (body_part, 'y')]
                    likelihood = df.loc[frame_number, (body_part, 'likelihood')]
            else:
                # Flat column structure
                x = df.loc[frame_number, f'{body_part}_x']
                y = df.loc[frame_number, f'{body_part}_y']
                likelihood = df.loc[frame_number, f'{body_part}_likelihood']
            
            # Check likelihood threshold
            if self.likelihood is not None and likelihood < self.likelihood:
                return None, None, None
            else:
                return float(x), float(y), float(likelihood)
                
        except (KeyError, IndexError) as e:
            self.logger.debug(f"Could not get coordinates for frame {frame_number}, bodypart {body_part}: {str(e)}")
            return None, None, None
    
    def get_map_coords(self, frame_number: int, body_part: str) -> Tuple[Optional[int], Optional[int]]:
        """Get map coordinates for specific frame and body part.
        
        Args:
            frame_number: Frame index 
            body_part: Body part name
            
        Returns:
            Tuple of (map_x, map_y) or (None, None) if coordinates unavailable
        """
        x, y, _ = self.get_coords(frame_number, body_part)
        if x is None or y is None:
            return None, None
        
        try:
            return self.map_labeler.get_map_coords(row=int(y), col=int(x))
        except Exception as e:
            self.logger.debug(f"Could not get map coordinates: {str(e)}")
            return None, None
    
    def get_map_tile(self, frame_number: int, body_part: str) -> Tuple[Optional[Any], Optional[int]]:
        """Get map tile information for specific frame and body part.
        
        Args:
            frame_number: Frame index
            body_part: Body part name
            
        Returns:
            Tuple of (tile_bbox, tile_id) or (None, None) if unavailable
        """
        x, y, likelihood = self.get_coords(frame_number, body_part)
        if x is None or y is None:
            return None, None
        
        try:
            return self.map_labeler.get_tile_by_img_coords(row=int(y), col=int(x))
        except Exception as e:
            self.logger.debug(f"Could not get map tile: {str(e)}")
            return None, None
    
    def get_df(self, body_part: str) -> pd.DataFrame:
        """Get DataFrame for specific body part with tile and tree information.
        
        Args:
            body_part: Body part to get DataFrame for
            
        Returns:
            DataFrame with coordinates, tile info, and tree positions
        """
        df = self.get_integrated_dataframe()
        
        # Extract bodypart data
        if isinstance(df.columns, pd.MultiIndex):
            if df.columns.nlevels >= 2:
                # Multi-level columns: get first scorer's data for this bodypart
                scorer_level = df.columns.get_level_values(0)[0]
                try:
                    bodypart_df = df[(scorer_level, body_part)].copy()
                except KeyError:
                    raise KeyError(f"Body part '{body_part}' not found in data")
            else:
                # Two-level: bodypart/coord
                try:
                    bodypart_df = df[body_part].copy()
                except KeyError:
                    raise KeyError(f"Body part '{body_part}' not found in data")
        else:
            # Flat columns - extract columns matching bodypart pattern
            bodypart_cols = [col for col in df.columns if col.startswith(f'{body_part}_')]
            if not bodypart_cols:
                raise KeyError(f"No columns found for body part '{body_part}'")
            bodypart_df = df[bodypart_cols].copy()
            # Rename columns to remove bodypart prefix
            bodypart_df.columns = [col.replace(f'{body_part}_', '') for col in bodypart_cols]
        
        # Add tile data if map labeler available
        if 'maze_map' in self.shared_resources:
            try:
                tile_data = bodypart_df.apply(
                    lambda row: self.map_labeler.get_tile_by_img_coords(row=row.y, col=row.x) 
                    if pd.notna(row.x) and pd.notna(row.y) else (None, -1), 
                    axis=1
                )
                bodypart_df[['tile_box', 'tile_id']] = pd.DataFrame(tile_data.tolist(), index=bodypart_df.index)
            except Exception as e:
                self.logger.debug(f"Could not add tile data: {str(e)}")
                bodypart_df['tile_box'] = None
                bodypart_df['tile_id'] = -1
        
        # Add tree position data if graph available
        if 'graph' in self.shared_resources:
            try:
                def get_tree_position(tile_id):
                    if pd.isna(tile_id) or tile_id == -1:
                        return None
                    try:
                        return self.tree.get_tree_location(int(tile_id))
                    except:
                        return None
                
                bodypart_df['tree_position'] = bodypart_df['tile_id'].apply(get_tree_position)
            except Exception as e:
                self.logger.debug(f"Could not add tree position data: {str(e)}")
                bodypart_df['tree_position'] = None
        
        # Apply likelihood threshold
        if self.likelihood is not None and 'likelihood' in bodypart_df.columns:
            low_conf_mask = bodypart_df['likelihood'] < self.likelihood
            bodypart_df.loc[low_conf_mask] = np.nan
        
        return bodypart_df
    
    def draw_tree(self, tile_id: Union[int, List[int]], mode: str = 'current') -> np.ndarray:
        """Draw tree visualization with current tile highlighted.
        
        Args:
            tile_id: Tile ID or list of tile IDs to highlight
            mode: Drawing mode ('current' or 'history')
            
        Returns:
            Tree image as numpy array
        """
        if 'graph' not in self.shared_resources:
            raise AttributeError("No graph provider available for tree drawing")
        
        # Recursively draw history
        if self.session_history.qsize() > 0:
            self.draw_tree(self.session_history.get(), mode='history')
        
        if mode != 'history':
            self.session_history.put(tile_id)
        
        if tile_id == -1:
            node, edge = None, None
        else:
            tree_loc = self.tree.get_tree_location(tile_id)
            if isinstance(tree_loc, tuple):
                edge = [tree_loc]
                node = None
            elif isinstance(tree_loc, int):
                edge = None  
                node = [tree_loc]
            elif isinstance(tree_loc, frozenset):
                node = []
                edge = []
                for item in tree_loc:
                    if isinstance(item, tuple):
                        edge.append(item)
                    elif isinstance(item, int):
                        node.append(item)
            else:
                node, edge = None, None
        
        # Get unique path to reward
        unique_path = []
        if self.path_to_reward:
            unique_path = self.path_to_reward + [
                (node_1, node_2) for node_1, node_2 in 
                zip(self.path_to_reward, self.path_to_reward[1:])
            ]
        
        # Draw tree using graph provider
        self.tree.draw_tree(node_list=node, edge_list=edge, color_mode=mode, unique_path=unique_path)
        return self.tree.tree_fig_to_img()
    
    def insert_data(self, body_part: str, col_name: str, col_values: Any) -> None:
        """Insert additional data column for a specific body part.
        
        Args:
            body_part: Body part name
            col_name: Column name to add
            col_values: Column values to insert
        """
        try:
            df = self.get_integrated_dataframe()
            
            # Handle multi-level vs flat column structure
            if isinstance(df.columns, pd.MultiIndex):
                if df.columns.nlevels >= 2:
                    # Add to multi-level structure
                    scorer_level = df.columns.get_level_values(0)[0]  # Use first scorer
                    df[(scorer_level, body_part, col_name)] = col_values
                else:
                    # Two-level structure
                    df[(body_part, col_name)] = col_values
            else:
                # Flat structure
                df[f'{body_part}_{col_name}'] = col_values
            
            # Update the cached dataframe
            self._integrated_dataframe = df
            
        except Exception as e:
            self.logger.error(f"Failed to insert data for {body_part}.{col_name}: {str(e)}")
            raise
    
    def _initialize_session(self) -> None:
        """Initialize session by discovering files and loading data sources."""
        self._discover_session_files()
        self._load_and_validate_data_sources()
        
        self.logger.debug(
            f"Session initialization complete: {len(self.data_source_instances)} data sources loaded"
        )
    
    def _discover_session_files(self) -> None:
        """Discover files for this session using configured patterns."""
        if 'data_sources' not in self.config:
            raise SessionInitializationError(
                f"No data sources configured for session {self.session_id}. "
                f"Make sure your configuration includes a 'data_sources' section."
            )
        
        # Extract file patterns from data source configurations
        file_patterns = {}
        for ds_config in self.config['data_sources']:
            if 'file_pattern' in ds_config:
                file_patterns[ds_config['name']] = ds_config['file_pattern']
        
        if file_patterns:
            self._discovered_file_paths = self.file_discovery.match_files_in_session(
                self.session_id, file_patterns
            )
            
            # Log discovery results
            found_files = sum(1 for path in self._discovered_file_paths.values() if path is not None)
            self.logger.info(
                f"File discovery for {self.session_id}: {found_files}/{len(file_patterns)} files found"
            )
        else:
            self.logger.debug(f"No file patterns specified for session {self.session_id}")
    
    def _load_and_validate_data_sources(self) -> None:
        """Load and validate data sources in configuration order (NO SORTING)."""
        for ds_config in self.config['data_sources']:
            ds_name = ds_config['name']
            ds_type = ds_config['type']
            
            try:
                # Get plugin class from registry
                ds_class = self.registry.get_data_source_plugin(ds_type)
                ds_instance = ds_class()
                
                # Store in configuration order (no sorting!)
                self.data_source_instances.append((ds_name, ds_instance, ds_config))
                
                self.logger.debug(f"Loaded data source plugin: {ds_name} ({ds_type})")
                
            except Exception as e:
                error_msg = (
                    f"Failed to load data source '{ds_name}' of type '{ds_type}' "
                    f"for session {self.session_id}: {str(e)}"
                )
                self.logger.error(error_msg)
                
                # Check if this is a required data source
                if ds_config.get('required', True):
                    raise SessionInitializationError(error_msg) from e
                else:
                    self.logger.warning(f"Skipping optional data source: {ds_name}")
    
    def _integrate_all_data_sources(self) -> pd.DataFrame:
        """Sequentially integrate all data sources in configuration order."""
        if not self.data_source_instances:
            raise DataSourceIntegrationError(
                f"No data sources available for integration in session {self.session_id}"
            )
        
        # Start with empty DataFrame - first data source will establish structure
        current_dataframe = pd.DataFrame()
        
        self.logger.info(
            f"Starting data integration for {self.session_id}: "
            f"{len(self.data_source_instances)} data sources"
        )
        
        for position, (ds_name, ds_instance, ds_config) in enumerate(self.data_source_instances):
            self.logger.info(f"Integrating data source {position+1}/{len(self.data_source_instances)}: {ds_name}")
            
            # Validate prerequisites
            if not ds_instance.validate_session_prerequisites(current_dataframe, self.shared_resources):
                error_message = f"Prerequisites not met for data source '{ds_name}'"
                self.logger.error(error_message)
                raise PluginPrerequisiteError(error_message)
            
            # Add discovered file path to configuration
            if ds_name in self._discovered_file_paths:
                ds_config = ds_config.copy()  # Don't modify original
                ds_config['discovered_file_path'] = self._discovered_file_paths[ds_name]
            
            # Perform integration
            try:
                current_dataframe = ds_instance.integrate_data_into_session(
                    current_dataframe, ds_config, self.shared_resources, self.logger
                )
                
                provided_columns = ds_instance.get_provided_column_names()
                self.logger.info(
                    f"✓ Successfully integrated {ds_name}: "
                    f"+{len(provided_columns)} columns, {len(current_dataframe)} frames total"
                )
                
            except Exception as e:
                error_message = (
                    f"Data source '{ds_name}' integration failed for session {self.session_id}: {str(e)}"
                )
                self.logger.error(error_message)
                raise DataSourceIntegrationError(error_message) from e
        
        self._integrated_dataframe = current_dataframe
        
        self.logger.info(
            f"✓ Data integration complete for {self.session_id}: "
            f"{len(current_dataframe.columns)} columns, {len(current_dataframe)} frames"
        )
        
        return current_dataframe