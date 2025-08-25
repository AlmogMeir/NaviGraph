"""Pose tracking plugin for NaviGraph unified architecture.

Loads pose data from DeepLabCut H5 files and adds bodypart tracking columns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from ...core.navigraph_plugin import NaviGraphPlugin
from ...core.exceptions import NavigraphError
from ...core.registry import register_data_source_plugin


@register_data_source_plugin("pose_tracking")
class PoseTrackingPlugin(NaviGraphPlugin):
    """Loads pose tracking data and adds bodypart columns to DataFrame."""
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.config.get('name', 'pose_tracking'),
            'type': 'pose_tracking',
            'description': 'Loads pose tracking data from DeepLabCut H5 files',
            'provides': [],
            'augments': 'bodypart_x, bodypart_y, bodypart_likelihood columns'
        }
    
    def augment_data(self, dataframe: pd.DataFrame, shared_resources: Dict[str, Any]) -> pd.DataFrame:
        """Load pose data and add bodypart columns to DataFrame.
        
        Args:
            dataframe: Current DataFrame (may be empty)
            shared_resources: Available shared resources
            
        Returns:
            DataFrame with bodypart tracking columns added
        """
        # Validate we have discovered files
        if not self.discovered_files:
            raise NavigraphError(
                f"PoseTrackingPlugin requires H5 file but none found. "
                f"Check file_pattern in config: {self.config.get('file_pattern')}"
            )
        
        h5_file = self.discovered_files[0]  # Use first discovered file
        self.logger.info(f"Loading pose data from: {h5_file.name}")
        
        try:
            # Load DeepLabCut HDF5 file
            dlc_dataframe = pd.read_hdf(h5_file)
            
            # Extract available bodyparts from file
            available_bodyparts = self._extract_available_bodyparts(dlc_dataframe)
            
            # Get bodyparts configuration
            config_bodyparts = self.config.get('bodyparts', [])
            
            # Determine which bodyparts to process
            if not config_bodyparts:  # Empty list = all bodyparts
                bodyparts = available_bodyparts
                self.logger.info(f"Processing all {len(bodyparts)} available bodyparts")
            else:
                # Validate requested bodyparts exist
                bodyparts = []
                for bp in config_bodyparts:
                    if bp in available_bodyparts:
                        bodyparts.append(bp)
                    else:
                        self.logger.warning(f"Bodypart '{bp}' not found in data. Available: {available_bodyparts}")
                
                if not bodyparts:
                    raise NavigraphError(
                        f"None of requested bodyparts {config_bodyparts} found. "
                        f"Available: {available_bodyparts}"
                    )
            
            # Get likelihood thresholds
            global_threshold = self.config.get('likelihood_threshold', 0.3)
            bodypart_thresholds = self.config.get('bodypart_thresholds', {})
            
            # Create result dataframe
            if dataframe.empty:
                # First plugin - establish frame index
                result_df = pd.DataFrame(index=dlc_dataframe.index)
                result_df.index.name = 'frame_number'
            else:
                result_df = dataframe.copy()
            
            # Process each bodypart
            for bodypart in bodyparts:
                # Get threshold for this bodypart
                threshold = bodypart_thresholds.get(bodypart, global_threshold)
                
                # Extract data for this bodypart
                x_data, y_data, likelihood_data = self._extract_bodypart_data(
                    dlc_dataframe, bodypart
                )
                
                # Apply likelihood threshold (set to NaN if below)
                if threshold > 0:
                    low_conf_mask = likelihood_data < threshold
                    x_data = np.where(low_conf_mask, np.nan, x_data)
                    y_data = np.where(low_conf_mask, np.nan, y_data)
                    
                    valid_frames = (~low_conf_mask).sum()
                    total_frames = len(low_conf_mask)
                    self.logger.debug(
                        f"{bodypart}: {valid_frames}/{total_frames} frames above threshold {threshold:.2f} "
                        f"({valid_frames/total_frames*100:.1f}%)"
                    )
                
                # Add columns to dataframe
                result_df[f'{bodypart}_x'] = x_data
                result_df[f'{bodypart}_y'] = y_data  
                result_df[f'{bodypart}_likelihood'] = likelihood_data
            
            # Process derived bodyparts if configured
            derived_bodyparts = self.config.get('derived_bodyparts', {})
            for derived_name, derived_config in derived_bodyparts.items():
                self._add_derived_bodypart(result_df, derived_name, derived_config)
            
            self.logger.info(
                f"Pose tracking complete: {len(bodyparts)} bodyparts, "
                f"{len(derived_bodyparts)} derived, {len(result_df)} frames"
            )
            
            return result_df
            
        except Exception as e:
            raise NavigraphError(f"Failed to load pose data from {h5_file}: {str(e)}") from e
    
    def _extract_available_bodyparts(self, dlc_dataframe: pd.DataFrame) -> List[str]:
        """Extract list of available bodyparts from DeepLabCut data.
        
        Args:
            dlc_dataframe: DeepLabCut DataFrame
            
        Returns:
            List of bodypart names
        """
        if isinstance(dlc_dataframe.columns, pd.MultiIndex):
            # Multi-level columns: extract bodypart level
            if dlc_dataframe.columns.nlevels >= 3:
                # Format: (scorer, bodypart, coordinate)
                return list(dlc_dataframe.columns.get_level_values(1).unique())
            elif dlc_dataframe.columns.nlevels >= 2:
                # Format: (bodypart, coordinate)
                return list(dlc_dataframe.columns.get_level_values(0).unique())
        
        # Fallback: scan for pattern in flat columns
        bodyparts = set()
        for col in dlc_dataframe.columns:
            if isinstance(col, str) and col.endswith(('_x', '_y', '_likelihood')):
                for suffix in ['_x', '_y', '_likelihood']:
                    if col.endswith(suffix):
                        bodyparts.add(col.replace(suffix, ''))
                        break
        
        if not bodyparts:
            raise NavigraphError("Could not extract bodyparts from DeepLabCut data structure")
        
        return sorted(list(bodyparts))
    
    def _extract_bodypart_data(
        self, 
        dlc_dataframe: pd.DataFrame, 
        bodypart: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract x, y, likelihood data for a specific bodypart.
        
        Args:
            dlc_dataframe: DeepLabCut DataFrame
            bodypart: Name of bodypart to extract
            
        Returns:
            Tuple of (x_data, y_data, likelihood_data) as numpy arrays
        """
        try:
            if isinstance(dlc_dataframe.columns, pd.MultiIndex):
                # Multi-level columns
                if dlc_dataframe.columns.nlevels >= 3:
                    # Format: (scorer, bodypart, coordinate)
                    scorer = dlc_dataframe.columns.get_level_values(0)[0]
                    x_data = dlc_dataframe[(scorer, bodypart, 'x')].values
                    y_data = dlc_dataframe[(scorer, bodypart, 'y')].values
                    likelihood_data = dlc_dataframe[(scorer, bodypart, 'likelihood')].values
                elif dlc_dataframe.columns.nlevels >= 2:
                    # Format: (bodypart, coordinate)
                    x_data = dlc_dataframe[(bodypart, 'x')].values
                    y_data = dlc_dataframe[(bodypart, 'y')].values
                    likelihood_data = dlc_dataframe[(bodypart, 'likelihood')].values
                else:
                    raise NavigraphError(f"Unexpected multi-level column structure")
            else:
                # Flat columns
                x_data = dlc_dataframe[f'{bodypart}_x'].values
                y_data = dlc_dataframe[f'{bodypart}_y'].values
                likelihood_data = dlc_dataframe[f'{bodypart}_likelihood'].values
            
            return x_data, y_data, likelihood_data
            
        except (KeyError, IndexError) as e:
            raise NavigraphError(f"Failed to extract data for bodypart '{bodypart}': {str(e)}") from e
    
    def _add_derived_bodypart(
        self, 
        dataframe: pd.DataFrame, 
        derived_name: str, 
        config: Dict[str, Any]
    ) -> None:
        """Add a derived bodypart based on centroid of existing bodyparts.
        
        Args:
            dataframe: DataFrame with existing bodypart data
            derived_name: Name for the derived bodypart
            config: Configuration for derivation (function, source_bodyparts)
        """
        function = config.get('function', 'centroid')
        source_bodyparts = config.get('source_bodyparts', [])
        
        if not source_bodyparts:
            self.logger.warning(f"No source bodyparts specified for derived '{derived_name}'")
            return
        
        # Validate source bodyparts exist
        missing = []
        for bp in source_bodyparts:
            if f'{bp}_x' not in dataframe.columns:
                missing.append(bp)
        
        if missing:
            self.logger.warning(
                f"Cannot create derived bodypart '{derived_name}': "
                f"missing source bodyparts {missing}"
            )
            return
        
        # Calculate centroid (works for 2 or more points)
        if function == 'centroid':
            # Collect coordinates from all source bodyparts
            x_values = []
            y_values = []
            likelihood_values = []
            
            for bp in source_bodyparts:
                x_values.append(dataframe[f'{bp}_x'].values)
                y_values.append(dataframe[f'{bp}_y'].values)
                likelihood_values.append(dataframe[f'{bp}_likelihood'].values)
            
            # Stack arrays for vectorized computation
            x_stack = np.vstack(x_values)
            y_stack = np.vstack(y_values)
            likelihood_stack = np.vstack(likelihood_values)
            
            # Calculate centroid (mean position) handling NaN values
            # nanmean will ignore NaN values in the calculation
            dataframe[f'{derived_name}_x'] = np.nanmean(x_stack, axis=0)
            dataframe[f'{derived_name}_y'] = np.nanmean(y_stack, axis=0)
            
            # For likelihood, use minimum (most conservative) or mean
            # Using minimum ensures centroid is only high confidence if all points are
            dataframe[f'{derived_name}_likelihood'] = np.nanmin(likelihood_stack, axis=0)
            
            self.logger.info(
                f"Added derived bodypart '{derived_name}' "
                f"(centroid of {len(source_bodyparts)} points: {source_bodyparts})"
            )
            
        else:
            self.logger.warning(f"Unknown derivation function: {function}. Use 'centroid'.")
    
    def get_expected_columns(self) -> List[str]:
        """Generate expected column names based on configuration.
        
        Returns:
            List of column names this plugin will create
        """
        columns = []
        
        # Get configured bodyparts
        bodyparts = self.config.get('bodyparts', [])
        
        # If empty, we don't know until runtime
        if not bodyparts:
            return []
        
        # Add columns for each bodypart
        for bp in bodyparts:
            columns.extend([f"{bp}_x", f"{bp}_y", f"{bp}_likelihood"])
        
        # Add derived bodyparts
        derived = self.config.get('derived_bodyparts', {})
        for derived_name in derived.keys():
            columns.extend([f"{derived_name}_x", f"{derived_name}_y", f"{derived_name}_likelihood"])
        
        return columns