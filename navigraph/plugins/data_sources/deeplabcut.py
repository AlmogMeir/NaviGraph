"""DeepLabCut data source plugin for NaviGraph.

Loads keypoint detection data from DeepLabCut .h5 files and establishes
the primary temporal index for session data integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
from pathlib import Path

from ...core.interfaces import IDataSource, Logger
from ...core.exceptions import DataSourceError
from ...core.base_plugin import BasePlugin
from ...core.registry import register_data_source_plugin


@register_data_source_plugin("deeplabcut")
class DeepLabCutDataSource(BasePlugin, IDataSource):
    """DeepLabCut keypoint data source - establishes primary temporal index."""
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance = None):
        """Factory method to create DeepLabCut data source from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    
    def integrate_data_into_session(
        self,
        current_dataframe: pd.DataFrame,
        session_config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        logger: Logger
    ) -> pd.DataFrame:
        """Load DeepLabCut data and establish frame index structure."""
        h5_file_path = session_config.get('discovered_file_path')
        if not h5_file_path:
            raise DataSourceError(
                "DeepLabCut H5 file path not provided. Make sure file discovery "
                "found a matching .h5 file for this session."
            )
        
        # Extract config from session_config structure
        config_section = session_config.get('config', {})
        
        # Support both single bodypart (backward compatibility) and multiple bodyparts
        target_bodyparts = config_section.get('bodyparts', None)
        if target_bodyparts is None:
            # Backward compatibility: use single bodypart from config
            target_bodyparts = config_section.get('bodypart', 'nose')
        
        self.logger.debug(f"DeepLabCut config: {session_config}")
        self.logger.info(f"Target bodyparts configuration: {target_bodyparts}")
        
        likelihood_threshold = config_section.get('likelihood_threshold', 0.3)
        
        self.logger.info(f"Loading DeepLabCut data from: {Path(h5_file_path).name}")
        
        try:
            # Load DeepLabCut HDF5 file
            dlc_dataframe = pd.read_hdf(h5_file_path)
            
            # Extract session information (preserving existing logic)
            session_metadata = self._extract_session_metadata(dlc_dataframe, self.logger)
            
            # Extract bodypart data based on configuration
            if isinstance(target_bodyparts, str) and target_bodyparts.lower() == 'all':
                # Extract all available bodyparts
                processed_data = self._extract_all_bodyparts_data(
                    dlc_dataframe, session_metadata, likelihood_threshold, self.logger
                )
            elif isinstance(target_bodyparts, list):
                # Extract multiple specific bodyparts
                processed_data = self._extract_multiple_bodyparts_data(
                    dlc_dataframe, session_metadata, target_bodyparts, likelihood_threshold, self.logger
                )
            else:
                # Single bodypart (backward compatibility)
                processed_data = self._extract_single_bodypart_data(
                    dlc_dataframe, session_metadata, target_bodyparts, likelihood_threshold, self.logger
                )
            
            # Create or merge with existing dataframe
            if current_dataframe.empty:
                # First data source - establish frame index (existing behavior)
                integrated_dataframe = processed_data.copy()
                integrated_dataframe.index.name = 'frame_number'
                self.logger.info("Established primary frame index from DeepLabCut data")
            else:
                # This shouldn't happen if DeepLabCut is first, but handle gracefully
                self.logger.warning("DeepLabCut data source is not first - this may cause issues")
                for column_name, column_data in processed_data.items():
                    current_dataframe[column_name] = column_data
                integrated_dataframe = current_dataframe
            
            # Log success
            self.logger.info(f"DeepLabCut integration complete: {len(processed_data.columns)} columns, {len(integrated_dataframe)} frames")
            
            return integrated_dataframe
            
        except Exception as e:
            raise DataSourceError(
                f"Failed to load DeepLabCut data from {h5_file_path}: {str(e)}"
            ) from e
    
    def validate_session_prerequisites(
        self, 
        current_dataframe: pd.DataFrame, 
        shared_resources: Dict[str, Any]
    ) -> bool:
        """DeepLabCut typically runs first - minimal prerequisites."""
        # DeepLabCut is typically the primary data source with no dependencies
        return True
    
    def get_provided_column_names(self) -> List[str]:
        """Return column names this data source provides."""
        # Note: This is dynamic based on bodyparts configuration
        # For single bodypart: keypoints_x, keypoints_y, keypoints_likelihood
        # For multiple: BodypartName_x, BodypartName_y, BodypartName_likelihood
        return ['keypoints_x', 'keypoints_y', 'keypoints_likelihood']  # Default single bodypart
    
    def get_required_columns(self) -> List[str]:
        """Return column names required by this data source."""
        return []
    
    def get_required_shared_resources(self) -> List[str]:
        """Return shared resource names required by this data source."""
        return []
    
    def _extract_session_metadata(self, dlc_dataframe: pd.DataFrame, logger) -> Dict[str, Any]:
        """Extract session metadata from DeepLabCut DataFrame (preserving existing logic)."""
        try:
            # Handle multi-level column structure
            if hasattr(dlc_dataframe.columns, 'levels') and len(dlc_dataframe.columns.levels) >= 2:
                session_ids = list(dlc_dataframe.columns.levels[0])
                session_id = session_ids if len(session_ids) > 1 else session_ids[0]
                bodyparts = list(dlc_dataframe.columns.levels[1])
                coordinates = list(dlc_dataframe.columns.levels[2])
            else:
                # Fallback for simpler column structure
                session_id = 'unknown_session'
                bodyparts = ['unknown_bodypart']
                coordinates = ['x', 'y', 'likelihood']
                self.logger.warning("DeepLabCut file has unexpected column structure")
            
            metadata = {
                'session_id': session_id,
                'bodyparts': bodyparts,
                'coordinates': coordinates,
                'n_frames': len(dlc_dataframe),
                'n_bodyparts': len(bodyparts)
            }
            
            self.logger.debug(f"DeepLabCut metadata: {len(bodyparts)} bodyparts, {len(dlc_dataframe)} frames")
            
            return metadata
            
        except Exception as e:
            raise DataSourceError(
                f"Failed to extract session metadata from DeepLabCut file: {str(e)}"
            ) from e
    
    def _extract_single_bodypart_data(
        self, 
        dlc_dataframe: pd.DataFrame, 
        session_metadata: Dict[str, Any], 
        target_bodypart: str,
        likelihood_threshold: float,
        logger
    ) -> pd.DataFrame:
        """Extract data for single bodypart (preserving existing behavior)."""
        try:
            session_id = session_metadata['session_id']
            available_bodyparts = session_metadata['bodyparts']
            
            # Validate bodypart exists
            if target_bodypart not in available_bodyparts:
                # Try case-insensitive match
                bodypart_matches = [
                    bp for bp in available_bodyparts 
                    if bp.lower() == target_bodypart.lower()
                ]
                
                if bodypart_matches:
                    actual_bodypart = bodypart_matches[0]
                    self.logger.warning(f"Bodypart '{target_bodypart}' not found, using '{actual_bodypart}' instead")
                else:
                    raise DataSourceError(
                        f"Bodypart '{target_bodypart}' not found in DeepLabCut data. "
                        f"Available bodyparts: {available_bodyparts}"
                    )
            else:
                actual_bodypart = target_bodypart
            
            # Extract bodypart data (preserving existing logic)
            if isinstance(session_id, list):
                # Multiple sessions - this is complex, use first one
                session_id = session_id[0]
                self.logger.warning(f"Multiple sessions in file, using: {session_id}")
            
            bodypart_data = dlc_dataframe[session_id][actual_bodypart]
            
            # Create clean column structure with descriptive names
            keypoint_columns = {
                'keypoints_x': bodypart_data['x'],
                'keypoints_y': bodypart_data['y'],
                'keypoints_likelihood': bodypart_data['likelihood']
            }
            
            # Apply likelihood filtering using vectorized operations (preserving existing logic)
            if likelihood_threshold is not None and likelihood_threshold > 0:
                likelihood_mask = keypoint_columns['keypoints_likelihood'] >= likelihood_threshold
                
                # Apply mask to all columns
                filtered_columns = {
                    col_name: col_data.where(likelihood_mask, np.nan)
                    for col_name, col_data in keypoint_columns.items()
                }
                
                valid_frames = likelihood_mask.sum()
                total_frames = len(likelihood_mask)
                
                self.logger.info(
                    f"Applied likelihood threshold {likelihood_threshold}: "
                    f"{valid_frames}/{total_frames} frames retained "
                    f"({valid_frames/total_frames*100:.1f}%)"
                )
            else:
                filtered_columns = keypoint_columns
                self.logger.info("No likelihood filtering applied")
            
            return pd.DataFrame(filtered_columns)
            
        except Exception as e:
            raise DataSourceError(
                f"Failed to extract bodypart '{target_bodypart}' from DeepLabCut data: {str(e)}"
            ) from e
    
    def _extract_multiple_bodyparts_data(
        self,
        dlc_dataframe: pd.DataFrame,
        session_metadata: Dict[str, Any],
        target_bodyparts: List[str],
        likelihood_threshold: float,
        logger
    ) -> pd.DataFrame:
        """Extract data for multiple specific bodyparts."""
        all_columns = {}
        
        for bodypart in target_bodyparts:
            try:
                # Extract single bodypart data
                single_data = self._extract_single_bodypart_data(
                    dlc_dataframe, session_metadata, bodypart, likelihood_threshold, logger
                )
                
                # Rename columns to include bodypart name
                for col in single_data.columns:
                    # Replace generic 'keypoints' with actual bodypart name
                    new_col_name = col.replace('keypoints', bodypart)
                    all_columns[new_col_name] = single_data[col]
                    
            except DataSourceError as e:
                logger.warning(f"Skipping bodypart '{bodypart}': {str(e)}")
                continue
        
        if not all_columns:
            raise DataSourceError(
                f"None of the requested bodyparts {target_bodyparts} could be extracted"
            )
        
        logger.info(f"Extracted {len(target_bodyparts)} bodyparts with {len(all_columns)} total columns")
        return pd.DataFrame(all_columns)
    
    def _extract_all_bodyparts_data(
        self,
        dlc_dataframe: pd.DataFrame,
        session_metadata: Dict[str, Any],
        likelihood_threshold: float,
        logger
    ) -> pd.DataFrame:
        """Extract data for all available bodyparts."""
        available_bodyparts = session_metadata['bodyparts']
        logger.info(f"Extracting all {len(available_bodyparts)} available bodyparts: {available_bodyparts}")
        
        return self._extract_multiple_bodyparts_data(
            dlc_dataframe, session_metadata, available_bodyparts, likelihood_threshold, logger
        )