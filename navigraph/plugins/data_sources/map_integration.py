"""Map integration plugin for NaviGraph.

This plugin wraps the current MapLabeler functionality as a data source plugin,
preserving all existing behavior while adapting to the new plugin architecture.
It converts keypoint coordinates to spatial map coordinates and tile locations.

The plugin requires keypoint data (x, y coordinates) from previous data sources
and uses calibration data from shared resources to perform coordinate transformation.
"""

import pandas as pd
import numpy as np
import cv2
from typing import Dict, Any, List
from pathlib import Path

from ...core.interfaces import IDataSource, DataSourceIntegrationError
from ...core.registry import register_data_source_plugin


@register_data_source_plugin("map_integration")
class MapIntegrationDataSource(IDataSource):
    """Integrates spatial map data - requires keypoints and calibration.
    
    This plugin converts image coordinates from keypoint data to map coordinates
    and determines which tile/region of the maze the animal is in. It preserves
    the existing MapLabeler logic while integrating with the new architecture.
    
    The plugin adds these columns to the session DataFrame:
    - map_x, map_y: Coordinates in the map reference frame
    - tile_id: Integer identifier of the spatial tile/region  
    - tile_bbox: Bounding box coordinates of the tile
    """
    
    def integrate_data_into_session(
        self,
        current_dataframe: pd.DataFrame,
        session_config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        logger
    ) -> pd.DataFrame:
        """Add map-based columns using existing MapLabeler logic.
        
        This method preserves the existing coordinate transformation and tile
        detection logic while adapting it to work with the new plugin system.
        
        Args:
            current_dataframe: DataFrame with keypoint data from previous sources
            session_config: Configuration for this data source
            shared_resources: Shared resources including map and calibration
            logger: Logger for progress reporting
            
        Returns:
            DataFrame with added map columns (map_x, map_y, tile_id, tile_bbox)
            
        Raises:
            DataSourceIntegrationError: If prerequisites not met or processing fails
        """
        logger.info("Starting map integration - converting keypoints to spatial coordinates")
        
        try:
            # Get required resources
            map_provider = shared_resources.get('maze_map')
            calibration = shared_resources.get('calibration')
            
            if not map_provider:
                raise DataSourceIntegrationError(
                    "Map integration requires 'maze_map' shared resource. "
                    "Make sure map_provider is configured in shared_resources."
                )
            
            if not calibration:
                raise DataSourceIntegrationError(
                    "Map integration requires 'calibration' shared resource. "
                    "Make sure camera_calibrator is configured in shared_resources."
                )
            
            # Get keypoint columns from previous data source
            keypoint_x_col = self._find_keypoint_column(current_dataframe, 'x', logger)
            keypoint_y_col = self._find_keypoint_column(current_dataframe, 'y', logger)
            
            # Transform coordinates and get tile information
            map_data = self._process_coordinate_transformation(
                current_dataframe, keypoint_x_col, keypoint_y_col, 
                map_provider, calibration, logger
            )
            
            # Add new columns to existing DataFrame
            for col_name, col_data in map_data.items():
                current_dataframe[col_name] = col_data
            
            # Log success statistics
            valid_coordinates = map_data['tile_id'] != -1
            valid_count = valid_coordinates.sum()
            total_count = len(current_dataframe)
            
            logger.info(
                f"✓ Map integration complete: {valid_count}/{total_count} frames "
                f"mapped to valid tiles ({valid_count/total_count*100:.1f}%)"
            )
            
            return current_dataframe
            
        except Exception as e:
            raise DataSourceIntegrationError(
                f"Map integration failed: {str(e)}"
            ) from e
    
    def validate_session_prerequisites(
        self, 
        current_dataframe: pd.DataFrame, 
        shared_resources: Dict[str, Any]
    ) -> bool:
        """Check for keypoints and map resources.
        
        This method validates that:
        1. Keypoint coordinates are available from previous data sources
        2. Required shared resources (map, calibration) are available
        
        Args:
            current_dataframe: Current DataFrame state
            shared_resources: Available shared resources
            
        Returns:
            True if all prerequisites are met
        """
        # Check for required columns
        required_columns = self.get_required_columns()
        has_columns = all(
            any(col.endswith(suffix) for col in current_dataframe.columns)
            for suffix in ['_x', '_y']
        )
        
        # Check for required resources  
        required_resources = self.get_required_shared_resources()
        has_resources = all(res in shared_resources for res in required_resources)
        
        return has_columns and has_resources
    
    def get_provided_column_names(self) -> List[str]:
        """Return column names this data source provides."""
        return ['map_x', 'map_y', 'tile_id', 'tile_bbox']
    
    def get_required_columns(self) -> List[str]:
        """Return column names required by this data source."""
        return ['keypoints_x', 'keypoints_y']
    
    def get_required_shared_resources(self) -> List[str]:
        """Return shared resource names required by this data source."""
        return ['maze_map', 'calibration']
    
    def _find_keypoint_column(self, dataframe: pd.DataFrame, coord_type: str, logger) -> str:
        """Find keypoint coordinate column (handling different naming conventions).
        
        This method looks for columns that contain keypoint coordinates,
        supporting various naming patterns from different data sources.
        
        Args:
            dataframe: DataFrame to search
            coord_type: 'x' or 'y' coordinate type
            logger: Logger for warnings
            
        Returns:
            Column name containing the requested coordinate type
            
        Raises:
            DataSourceIntegrationError: If coordinate column not found
        """
        # Look for columns ending with the coordinate type
        candidate_columns = [
            col for col in dataframe.columns 
            if col.endswith(f'_{coord_type}')
        ]
        
        if not candidate_columns:
            raise DataSourceIntegrationError(
                f"No {coord_type} coordinate column found. "
                f"Available columns: {list(dataframe.columns)}. "
                f"Make sure keypoint data source is configured before map integration."
            )
        
        if len(candidate_columns) > 1:
            # Prefer 'keypoints_x' over other options
            preferred_col = f'keypoints_{coord_type}'
            if preferred_col in candidate_columns:
                selected_col = preferred_col
            else:
                selected_col = candidate_columns[0]
                logger.warning(
                    f"Multiple {coord_type} columns found: {candidate_columns}. "
                    f"Using: {selected_col}"
                )
        else:
            selected_col = candidate_columns[0]
        
        logger.debug(f"Using {coord_type} coordinate column: {selected_col}")
        return selected_col
    
    def _process_coordinate_transformation(
        self, 
        dataframe: pd.DataFrame,
        x_col: str, 
        y_col: str,
        map_provider,
        calibration,
        logger
    ) -> Dict[str, pd.Series]:
        """Process coordinate transformation using existing MapLabeler logic.
        
        This method preserves the existing coordinate transformation pipeline:
        1. Image coordinates → Map coordinates (using calibration)
        2. Map coordinates → Tile identification (using map grid)
        
        Args:
            dataframe: DataFrame with keypoint coordinates
            x_col: Name of x coordinate column
            y_col: Name of y coordinate column  
            map_provider: Map provider resource
            calibration: Calibration resource
            logger: Logger instance
            
        Returns:
            Dictionary with new column data
        """
        n_frames = len(dataframe)
        
        # Initialize output arrays
        map_x_coords = np.full(n_frames, np.nan)
        map_y_coords = np.full(n_frames, np.nan)
        tile_ids = np.full(n_frames, -1, dtype=int)
        tile_bboxes = [None] * n_frames
        
        # Get transformation matrix from calibration
        transform_matrix = calibration.get_transformation_matrix()
        
        # Get map configuration from map provider
        map_config = map_provider.get_map_configuration()
        
        # Process each frame (preserving existing logic)
        for frame_idx in range(n_frames):
            try:
                # Get image coordinates (note: MapLabeler expects row=y, col=x)
                img_x = dataframe.iloc[frame_idx][x_col]
                img_y = dataframe.iloc[frame_idx][y_col]
                
                # Skip invalid coordinates
                if pd.isna(img_x) or pd.isna(img_y):
                    continue
                
                # Transform to map coordinates (preserving existing logic)
                # Note: MapLabeler uses col=x, row=y convention
                map_coords = self._transform_image_to_map_coords(
                    img_x, img_y, transform_matrix
                )
                
                if map_coords is not None:
                    map_x, map_y = map_coords
                    map_x_coords[frame_idx] = map_x
                    map_y_coords[frame_idx] = map_y
                    
                    # Get tile information (preserving existing logic)
                    tile_info = self._get_tile_by_map_coords(
                        map_x, map_y, map_config
                    )
                    
                    if tile_info is not None:
                        tile_bbox, tile_id = tile_info
                        tile_ids[frame_idx] = tile_id
                        tile_bboxes[frame_idx] = tile_bbox
                
            except Exception as e:
                logger.warning(f"Failed to process frame {frame_idx}: {e}")
                continue
        
        # Log transformation statistics
        valid_transforms = ~np.isnan(map_x_coords)
        valid_tiles = tile_ids != -1
        
        logger.debug(
            f"Coordinate transformation: {valid_transforms.sum()}/{n_frames} valid map coordinates"
        )
        logger.debug(
            f"Tile identification: {valid_tiles.sum()}/{n_frames} valid tiles"
        )
        
        return {
            'map_x': pd.Series(map_x_coords, index=dataframe.index),
            'map_y': pd.Series(map_y_coords, index=dataframe.index),
            'tile_id': pd.Series(tile_ids, index=dataframe.index),
            'tile_bbox': pd.Series(tile_bboxes, index=dataframe.index)
        }
    
    def _transform_image_to_map_coords(self, img_x: float, img_y: float, transform_matrix) -> tuple:
        """Transform image coordinates to map coordinates (preserving existing logic)."""
        try:
            # Use OpenCV perspective transform (preserving existing logic)
            map_coords = cv2.perspectiveTransform(
                np.array([[(img_x, img_y)]], dtype='float32'), 
                transform_matrix
            ).ravel().astype(int)
            
            map_x, map_y = map_coords
            return map_x, map_y
            
        except Exception:
            return None
    
    def _get_tile_by_map_coords(self, map_x: int, map_y: int, map_config: Dict) -> tuple:
        """Get tile information from map coordinates (preserving existing MapLabeler logic).
        
        NOTE: Original MapLabeler uses map_col=x, map_row=y convention but calculations
        expect row/col grid indexing. This preserves the exact original logic.
        """
        try:
            # Extract map configuration (preserving existing structure)
            origin_y, origin_x = map_config['origin']  # Note: original uses (row, col) = (y, x)
            grid_rows, grid_cols = map_config['grid_size']
            segment_length = map_config['segment_length']
            
            # Check bounds (preserving existing logic)
            # Original: map_row < origin_row or map_col < origin_col or map_row > max_grid_row or map_col > max_grid_col
            max_grid_y = (grid_rows + 1) * segment_length  
            max_grid_x = (grid_cols + 1) * segment_length
            
            if (map_y < origin_y or map_x < origin_x or 
                map_y > max_grid_y or map_x > max_grid_x):
                return None
            
            # Calculate tile position (preserving existing logic)
            # Original: row_origin_offset = map_row - origin_row, col_origin_offset = map_col - origin_col
            row_origin_offset = map_y - origin_y
            col_origin_offset = map_x - origin_x
            
            row_segment_multiplier = row_origin_offset // segment_length
            col_segment_multiplier = col_origin_offset // segment_length
            
            # Calculate tile ID (preserving existing logic)
            # Original: tile_id = (grid_size[0] * row_segment_multiplier) + col_segment_multiplier
            tile_id = (grid_rows * row_segment_multiplier) + col_segment_multiplier
            
            # Calculate bounding box (preserving existing logic) 
            # Original returns [x, y, w, h] format
            y_min = (segment_length * row_segment_multiplier) + origin_y
            x_min = (segment_length * col_segment_multiplier) + origin_x
            x_max = x_min + segment_length - 1
            y_max = y_min + segment_length - 1
            
            # Original returns [x_min, y_min, x_max - x_min, y_max - y_min] = [x, y, w, h]
            tile_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            
            return tile_bbox, int(tile_id)
            
        except Exception:
            return None