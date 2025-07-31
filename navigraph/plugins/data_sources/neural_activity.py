"""Neural activity data source plugin for NaviGraph.

Loads neural activity data (df/f values) from zarr/xarray format files (e.g., Minian output)
and integrates them into session DataFrames for behavioral-neural analysis.
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import dask.array as darr
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from ...core.interfaces import IDataSource, Logger
from ...core.exceptions import DataSourceError
from ...core.base_plugin import BasePlugin
from ...core.registry import register_data_source_plugin


@register_data_source_plugin("neural_activity")
class NeuralActivityDataSource(BasePlugin, IDataSource):
    """Neural activity data source for calcium imaging or similar neural data.
    
    Loads neural data from zarr/xarray format (e.g., Minian package output) and
    integrates df/f values into session DataFrames. Each neuron becomes a column
    in the final DataFrame, indexed by frame number.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger_instance: Optional[Logger] = None):
        """Initialize neural activity data source.
        
        Args:
            config: Configuration dictionary containing:
                - minian_path: Path to neural data directory or file
                - merge_mode: How to merge with session data ('left', 'inner', 'outer')
                - unit_prefix: Prefix for neuron unit columns (default: 'neuron_')
            logger_instance: Logger instance
        """
        super().__init__(config, logger_instance)
        self._neural_data: Optional[pd.DataFrame] = None
        self._spatial_footprints: Optional[Dict[str, np.ndarray]] = None
        self._metadata: Dict[str, Any] = {}
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance: Optional[Logger] = None):
        """Factory method to create neural activity data source from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate neural activity data source configuration."""
        # No specific validation needed - path will be discovered
        pass
    
    def initialize(self) -> None:
        """Initialize neural data loading."""
        super().initialize()
        # Initialization deferred until integrate_data_into_session when we have the discovered path
    
    def _load_minian_data(self, minian_path: str) -> Union[xr.Dataset, Dict[str, xr.DataArray]]:
        """Load neural data from zarr/xarray format.
        
        Args:
            minian_path: Path to neural data directory or file
            
        Returns:
            xarray Dataset or dictionary of DataArrays
            
        Raises:
            DataSourceError: If data cannot be loaded
        """
        try:
            if os.path.isfile(minian_path):
                # Single file - load as dataset
                self.logger.debug(f"Loading neural data from file: {minian_path}")
                dataset = xr.open_dataset(minian_path).chunk()
                return dataset
                
            elif os.path.isdir(minian_path):
                # Directory - load individual zarr arrays
                self.logger.debug(f"Loading neural data from directory: {minian_path}")
                data_arrays = []
                
                for d in os.listdir(minian_path):
                    arr_path = os.path.join(minian_path, d)
                    if os.path.isdir(arr_path):
                        try:
                            # Use proven loading method from user's function
                            arr = list(xr.open_zarr(arr_path).values())[0]
                            arr.data = darr.from_zarr(os.path.join(arr_path, arr.name), inline_array=True)
                            data_arrays.append(arr)
                            self.logger.debug(f"Loaded array: {arr.name} with shape {arr.shape}")
                            
                        except Exception as e:
                            self.logger.warning(f"Could not load {arr_path}: {str(e)}")
                            continue
                
                if not data_arrays:
                    raise DataSourceError("No valid zarr arrays found in directory")
                
                # Merge arrays into dataset using no_conflicts compatibility
                return xr.merge(data_arrays, compat="no_conflicts")
                
            else:
                raise DataSourceError(f"Path does not exist: {minian_path}")
                
        except Exception as e:
            raise DataSourceError(f"Failed to load Minian data: {str(e)}")
    
    def _process_neural_data(self, raw_data: Union[xr.Dataset, Dict[str, xr.DataArray]]) -> None:
        """Process raw neural data into DataFrame format using efficient extraction.
        
        Args:
            raw_data: Raw neural data from zarr/xarray
        """
        try:
            # Convert to dataset if needed
            if isinstance(raw_data, dict):
                raw_data = xr.Dataset(raw_data)
            
            # Look for calcium traces data (C array)
            neural_vars = ['C', 'df_f', 'fluorescence', 'activity']
            activity_var = None
            
            for var in neural_vars:
                if var in raw_data.data_vars:
                    activity_var = var
                    break
            
            if activity_var is None:
                # Fallback: use first data variable with frame and unit_id dims
                for var_name, var_data in raw_data.data_vars.items():
                    if 'frame' in var_data.dims and 'unit_id' in var_data.dims:
                        activity_var = var_name
                        self.logger.warning(
                            f"No standard neural activity variable found, using: {activity_var}"
                        )
                        break
                
                if activity_var is None:
                    raise DataSourceError("No neural activity data with frame and unit_id dimensions found")
            
            self.logger.info(f"Using neural activity variable: {activity_var}")
            
            # Extract calcium data efficiently
            calcium_data = raw_data[activity_var]  # Shape: (unit_id, frame)
            self.logger.debug(f"Calcium data shape: {calcium_data.shape}, dims: {calcium_data.dims}")
            
            # Convert to DataFrame and pivot efficiently
            calcium_df = calcium_data.to_dataframe(name=activity_var)
            neural_pivot = calcium_df.reset_index().pivot(
                index='frame', 
                columns='unit_id', 
                values=activity_var
            )
            
            # Rename columns with prefix
            unit_prefix = self.config.get('unit_prefix', 'neuron_')
            neural_pivot.columns = [f"{unit_prefix}{col}" for col in neural_pivot.columns]
            neural_pivot.columns.name = None  # Remove column name
            
            # Sort by frame index
            neural_pivot = neural_pivot.sort_index()
            
            self._neural_data = neural_pivot
            
            # Store metadata
            self._metadata.update({
                'activity_variable': activity_var,
                'frame_column': 'frame',
                'unit_column': 'unit_id',
                'num_neurons': len(neural_pivot.columns),
                'num_frames': len(neural_pivot),
                'frame_range': (int(neural_pivot.index.min()), int(neural_pivot.index.max()))
            })
            
            self.logger.info(
                f"Processed neural data: {self._metadata['num_neurons']} neurons, "
                f"{self._metadata['num_frames']} frames"
            )
            
            # Extract spatial footprints if available
            self._extract_spatial_footprints(raw_data, 'unit_id')
                
        except Exception as e:
            raise DataSourceError(f"Failed to process neural data: {str(e)}")
    
    def _extract_spatial_footprints(self, raw_data: xr.Dataset, unit_col: str) -> None:
        """Extract spatial footprints for neurons if available.
        
        Args:
            raw_data: Raw neural dataset
            unit_col: Column name for neuron units
        """
        try:
            # Look for spatial footprint variables
            footprint_vars = ['A', 'spatial', 'footprint', 'roi']
            
            for var in footprint_vars:
                if var in raw_data.data_vars:
                    self.logger.info(f"Extracting spatial footprints from variable: {var}")
                    
                    footprint_data = raw_data[var]
                    
                    # Convert to dictionary by unit
                    self._spatial_footprints = {}
                    
                    if unit_col in footprint_data.dims:
                        for unit_id in footprint_data.coords[unit_col].values:
                            footprint = footprint_data.sel({unit_col: unit_id}).values
                            self._spatial_footprints[f"neuron_{unit_id}"] = footprint
                    
                    self.logger.info(f"Extracted {len(self._spatial_footprints)} spatial footprints")
                    break
                    
        except Exception as e:
            self.logger.warning(f"Could not extract spatial footprints: {str(e)}")
    
    def integrate_data_into_session(
        self,
        current_dataframe: pd.DataFrame,
        session_config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        logger: Logger
    ) -> pd.DataFrame:
        """Integrate neural activity data into session DataFrame.
        
        Args:
            current_dataframe: Current session DataFrame
            session_config: Session configuration
            shared_resources: Shared resources dictionary
            logger: Logger instance
            
        Returns:
            DataFrame with neural activity data added
            
        Raises:
            DataSourceError: If integration fails
        """
        # Load neural data if not already loaded
        if self._neural_data is None:
            # Get discovered file path
            discovered_path = session_config.get('discovered_file_path')
            if not discovered_path:
                logger.warning("Neural activity data source: no file discovered")
                return current_dataframe
            
            logger.info(f"Loading neural data from discovered path: {discovered_path}")
            
            try:
                # Load neural data
                raw_data = self._load_minian_data(discovered_path)
                self._process_neural_data(raw_data)
                
                logger.info(
                    f"Neural data loaded: {len(self._neural_data)} frames, "
                    f"{len(self._neural_data.columns)} neurons"
                )
                
            except Exception as e:
                raise DataSourceError(f"Failed to load neural data: {str(e)}")
        
        logger.info("Integrating neural activity data into session")
        
        try:
            # Get merge configuration
            merge_mode = self.config.get('merge_mode', 'left')
            
            # Align neural data with session DataFrame index
            # Current DataFrame should be indexed by frame number
            logger.debug(
                f"Session DataFrame shape: {current_dataframe.shape}, "
                f"index range: {current_dataframe.index.min()}-{current_dataframe.index.max()}"
            )
            logger.debug(
                f"Neural data shape: {self._neural_data.shape}, "
                f"index range: {self._neural_data.index.min()}-{self._neural_data.index.max()}"
            )
            
            # Merge DataFrames
            enhanced_df = current_dataframe.merge(
                self._neural_data,
                left_index=True,
                right_index=True,
                how=merge_mode,
                suffixes=('', '_neural')
            )
            
            # Add summary statistics columns if requested
            if self.config.get('add_summary_stats', True):
                enhanced_df = self._add_summary_statistics(enhanced_df, logger)
            
            logger.info(
                f"Neural data integration complete. Added {len(self._neural_data.columns)} "
                f"neural activity columns. Final shape: {enhanced_df.shape}"
            )
            
            # Log overlap statistics
            overlap_frames = len(enhanced_df.dropna(subset=self._neural_data.columns))
            total_frames = len(enhanced_df)
            overlap_pct = (overlap_frames / total_frames) * 100 if total_frames > 0 else 0
            
            logger.info(
                f"Neural-behavioral data overlap: {overlap_frames}/{total_frames} "
                f"frames ({overlap_pct:.1f}%)"
            )
            
            return enhanced_df
            
        except Exception as e:
            raise DataSourceError(f"Failed to integrate neural data: {str(e)}")
    
    def get_neural_metadata(self) -> Dict[str, Any]:
        """Get metadata about loaded neural data.
        
        Returns:
            Dictionary containing neural data metadata
        """
        return self._metadata.copy()
    
    def get_spatial_footprints(self) -> Optional[Dict[str, np.ndarray]]:
        """Get spatial footprints for neurons.
        
        Returns:
            Dictionary mapping neuron names to spatial footprint arrays,
            or None if not available
        """
        return self._spatial_footprints.copy() if self._spatial_footprints else None
    
    def get_neuron_activity(self, neuron_name: str) -> Optional[pd.Series]:
        """Get activity time series for a specific neuron.
        
        Args:
            neuron_name: Name of neuron (e.g., 'neuron_0')
            
        Returns:
            Time series of neural activity, or None if neuron not found
        """
        if self._neural_data is None or neuron_name not in self._neural_data.columns:
            return None
        
        return self._neural_data[neuron_name].copy()
    
    def get_active_neurons(self, frame_idx: int, threshold: float = 0.1) -> List[str]:
        """Get list of active neurons at a specific frame.
        
        Args:
            frame_idx: Frame index
            threshold: Activity threshold for considering a neuron "active"
            
        Returns:
            List of neuron names that are active at the given frame
        """
        if self._neural_data is None or frame_idx not in self._neural_data.index:
            return []
        
        frame_data = self._neural_data.loc[frame_idx]
        active = frame_data[frame_data > threshold]
        
        return active.index.tolist()
    
    def _add_summary_statistics(self, dataframe: pd.DataFrame, logger) -> pd.DataFrame:
        """Add neural activity summary statistics columns.
        
        Args:
            dataframe: DataFrame with neural activity columns
            logger: Logger instance
            
        Returns:
            DataFrame with added summary statistics columns
        """
        # Get all neural activity columns (those starting with unit_prefix)
        unit_prefix = self.config.get('unit_prefix', 'neuron_')
        neural_columns = [col for col in dataframe.columns if col.startswith(unit_prefix)]
        
        if not neural_columns:
            logger.warning("No neural activity columns found for summary statistics")
            return dataframe
        
        logger.debug(f"Computing summary statistics for {len(neural_columns)} neural columns")
        
        # Extract just the neural data for calculations
        neural_data = dataframe[neural_columns]
        
        # Calculate summary statistics
        try:
            # Mean activity across all neurons per frame
            dataframe['neuron_mean_activity'] = neural_data.mean(axis=1)
            
            # Maximum activity across all neurons per frame
            dataframe['neuron_max_activity'] = neural_data.max(axis=1)
            
            # Count of active neurons per frame (above threshold)
            activity_threshold = self.config.get('activity_threshold', 0.1)
            dataframe['neuron_active_count'] = (neural_data > activity_threshold).sum(axis=1)
            
            # Standard deviation of activity across neurons per frame
            dataframe['neuron_activity_std'] = neural_data.std(axis=1)
            
            logger.info(
                f"Added neural summary statistics: mean, max, active_count, std "
                f"(threshold={activity_threshold})"
            )
            
        except Exception as e:
            logger.error(f"Failed to compute neural summary statistics: {str(e)}")
        
        return dataframe
    
    def get_provided_column_names(self) -> List[str]:
        """Get names of columns provided by this data source.
        
        Returns:
            List of column names that will be added to the session DataFrame
        """
        columns = []
        
        # Add neural activity columns
        if self._neural_data is not None:
            columns.extend(list(self._neural_data.columns))
        
        # Add summary statistics columns if enabled
        if self.config.get('add_summary_stats', True):
            summary_columns = [
                'neuron_mean_activity',
                'neuron_max_activity', 
                'neuron_active_count',
                'neuron_activity_std'
            ]
            columns.extend(summary_columns)
        
        return columns