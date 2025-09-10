"""Neural activity plugin for NaviGraph unified architecture.

Loads neural activity data from Minian zarr format.
"""

import zarr
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as darr
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

from ..core.navigraph_plugin import NaviGraphPlugin
from ..core.exceptions import NavigraphError
from ..core.registry import register_data_source_plugin


@register_data_source_plugin("neural_activity")
class NeuralActivityPlugin(NaviGraphPlugin):
    """Loads neural activity data from Minian zarr format."""
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.config.get('name', 'neural_activity'),
            'type': 'neural_activity',
            'description': 'Loads neural activity data from Minian zarr format',
            'provides': [],
            'augments': 'neuron_* columns for each detected neuron'
        }
    
    def augment_data(self, dataframe: pd.DataFrame, shared_resources: Dict[str, Any]) -> pd.DataFrame:
        """Add neural activity columns to the dataframe.
        
        Args:
            dataframe: DataFrame with temporal index (usually from pose tracking)
            shared_resources: Available shared resources
            
        Returns:
            DataFrame with neural activity columns added
        """
        # Validate we have discovered files
        if not self.discovered_files:
            self.logger.warning("No Minian data discovered, skipping neural activity data")
            return dataframe

        minian_path = self.discovered_files[0]  # Use first discovered folder
        self.logger.info(f"Loading neural activity from: {minian_path.name}")
        
        try:
            # Load Minian zarr data
            neural_data = self._load_minian_data(minian_path)
            
            if neural_data is None:
                self.logger.warning("No neural data loaded, returning unchanged dataframe")
                return dataframe
            
            # Align neural data with dataframe temporal index
            result_df = self._align_and_add_neural_data(neural_data, dataframe)
            
            # Add derived neural metrics if requested
            result_df = self._add_derived_neural_metrics(result_df)
            
            neuron_count = len(neural_data)
            derived_info = ""
            plugin_config = self.config.get('config', {})
            if plugin_config.get('derived_metrics'):
                derived_info = f" (with {len(plugin_config['derived_metrics'])} derived metrics)"
            
            self.logger.info(
                f"✓ Neural activity loaded: {neuron_count} neurons, "
                f"{len(result_df)} timepoints{derived_info}"
            )
            
            return result_df
            
        except Exception as e:
            raise NavigraphError(
                f"Failed to load neural activity from {minian_path}: {str(e)}"
            ) from e
    
    def _load_minian_data(self, minian_path: Path) -> Optional[Dict[int, np.ndarray]]:
        """Load neural activity data from Minian zarr format.
        
        Args:
            minian_path: Path to Minian folder containing zarr data
            
        Returns:
            Dictionary mapping neuron ID to activity trace, or None if no data
        """
        # Look for different Minian zarr files
        zarr_candidates = [
            minian_path / "C.zarr",  # Calcium traces
            minian_path / "S.zarr",  # Spike inference
            minian_path / "A.zarr",  # Spatial footprints (for neuron count)
        ]
        
        # Try to load calcium traces first (most common)
        for zarr_file in zarr_candidates:
            if zarr_file.exists():
                try:
                    self.logger.debug(f"Loading zarr file: {zarr_file}")
                    
                    # Try xarray loading first (preferred method)
                    try:
                        zarr_data = list(xr.open_zarr(str(zarr_file)).values())[0]
                        zarr_data.data = darr.from_zarr(os.path.join(str(zarr_file), zarr_data.name), inline_array=True)
                        activity_matrix = zarr_data.values
                    except Exception:
                        # Fallback to direct zarr loading
                        zarr_data = zarr.open(str(zarr_file), mode='r')
                        activity_matrix = np.array(zarr_data)
                    
                    # Validate we have a 2D matrix
                    if len(activity_matrix.shape) < 2:
                        self.logger.debug(f"Skipping {zarr_file}: not a 2D activity matrix")
                        continue
                    
                    # Determine orientation - neurons should be fewer than timepoints
                    if activity_matrix.shape[0] > activity_matrix.shape[1]:
                        # Transpose if needed: (timepoints, neurons) -> (neurons, timepoints)
                        activity_matrix = activity_matrix.T
                    
                    # Create neuron dictionary
                    neural_data = {}
                    n_neurons, n_timepoints = activity_matrix.shape
                    
                    for neuron_id in range(n_neurons):
                        neural_data[neuron_id] = activity_matrix[neuron_id, :]
                    
                    self.logger.info(
                        f"Loaded neural data from {zarr_file.name}: {n_neurons} neurons, {n_timepoints} timepoints"
                    )
                    
                    return neural_data
                    
                except Exception as e:
                    self.logger.debug(f"Failed to load {zarr_file}: {e}")
                    continue
        
        # No valid zarr data found
        self.logger.warning(f"No valid Minian zarr data found in {minian_path}")
        return None
    
    def _align_and_add_neural_data(
        self, 
        neural_data: Dict[int, np.ndarray], 
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """Align neural data with dataframe temporal index and add columns.
        
        Args:
            neural_data: Dictionary mapping neuron ID to activity trace
            dataframe: DataFrame with temporal index to align to
            
        Returns:
            DataFrame with neural activity columns added
        """
        target_length = len(dataframe)
        result_df = dataframe.copy()
        
        plugin_config = self.config.get('config', {})
        unit_prefix = plugin_config.get('unit_prefix', 'neuron_')
        
        for neuron_id, activity in neural_data.items():
            column_name = f"{unit_prefix}{neuron_id}"
            
            if len(activity) == target_length:
                # Perfect match
                result_df[column_name] = activity
            elif len(activity) > target_length:
                # Truncate neural data
                self.logger.debug(
                    f"Truncating {column_name} from {len(activity)} to {target_length}"
                )
                result_df[column_name] = activity[:target_length]
            else:
                # Pad with NaN
                self.logger.debug(
                    f"Padding {column_name} from {len(activity)} to {target_length}"
                )
                padded = np.full(target_length, np.nan)
                padded[:len(activity)] = activity
                result_df[column_name] = padded
        
        # Log coverage statistics
        neuron_columns = [col for col in result_df.columns if col.startswith(unit_prefix)]
        if neuron_columns:
            # Check coverage for first neuron as representative
            first_neuron_data = result_df[neuron_columns[0]]
            valid_count = np.sum(~np.isnan(first_neuron_data))
            coverage_pct = (valid_count / target_length) * 100 if target_length > 0 else 0
            
            self.logger.info(
                f"Neural activity coverage: {valid_count}/{target_length} frames ({coverage_pct:.1f}%)"
            )
        
        return result_df
    
    def _add_derived_neural_metrics(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add derived neural metrics based on configuration.
        
        Args:
            dataframe: DataFrame with neural activity columns
            
        Returns:
            DataFrame with derived neural metrics added
        """
        plugin_config = self.config.get('config', {})
        derived_metrics = plugin_config.get('derived_metrics', {})
        if not derived_metrics:
            return dataframe
        
        unit_prefix = plugin_config.get('unit_prefix', 'neuron_')
        neuron_columns = [col for col in dataframe.columns if col.startswith(unit_prefix)]
        
        if not neuron_columns:
            self.logger.warning("No neuron columns found for derived metrics calculation")
            return dataframe
        
        result_df = dataframe.copy()
        
        for metric_name, metric_config in derived_metrics.items():
            try:
                if metric_config == 'mean' or metric_config.get('function') == 'mean':
                    # Calculate mean across specified neurons or all neurons
                    neurons_to_use = metric_config.get('neurons', 'all') if isinstance(metric_config, dict) else 'all'
                    
                    if neurons_to_use == 'all':
                        selected_columns = neuron_columns
                    else:
                        # Specific neurons specified
                        selected_columns = []
                        for neuron_spec in neurons_to_use:
                            if isinstance(neuron_spec, int):
                                col_name = f"{unit_prefix}{neuron_spec}"
                                if col_name in neuron_columns:
                                    selected_columns.append(col_name)
                            elif isinstance(neuron_spec, str) and neuron_spec in neuron_columns:
                                selected_columns.append(neuron_spec)
                    
                    if selected_columns:
                        result_df[metric_name] = result_df[selected_columns].mean(axis=1)
                        self.logger.debug(
                            f"Added derived metric '{metric_name}': mean of {len(selected_columns)} neurons"
                        )
                    else:
                        self.logger.warning(f"No valid neurons found for derived metric '{metric_name}'")
                
                else:
                    self.logger.warning(f"Unknown derived metric function: {metric_config}")
                    
            except Exception as e:
                self.logger.error(f"Failed to calculate derived metric '{metric_name}': {str(e)}")
        
        return result_df
    
    def get_expected_columns(self) -> List[str]:
        """Generate expected column names based on available data.
        
        Returns:
            List of column names this plugin will create
        """
        columns = []
        
        # Add derived metrics columns if specified
        plugin_config = self.config.get('config', {})
        derived_metrics = plugin_config.get('derived_metrics', {})
        if derived_metrics:
            columns.extend(list(derived_metrics.keys()))
        
        # We don't know individual neuron count until runtime
        return columns