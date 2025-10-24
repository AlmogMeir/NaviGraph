"""Neural activity plugin for NaviGraph unified architecture.

Loads neural activity data from Minian zarr format, extracting calcium traces,
spike activity, and spatial footprints. Also discovers and provides associated
video files for visualization.

Minian Component Descriptions:
- C: Calcium traces (df/f - change in fluorescence over baseline)
- S: Deconvolved spike activity (inferred neural firing events)
- A: Spatial footprints (2D masks showing each neuron's location, extracted as contours)
- b: Background fluorescence spatial pattern
- b0: Background fluorescence baseline values
- c0: Baseline calcium level for each neuron
- f: Global fluorescence signal across the field of view
- max_proj: Maximum projection of the entire video
- motion: Motion correction shifts applied during preprocessing

Configuration Options:
- components: List of components to extract ['C', 'S', 'A'], or null for all (default: all)
- neurons: List of specific neuron IDs to extract, or null for all (default: all)
- derived_metrics: Dictionary of derived metrics to compute (e.g., mean across neurons)

Example config:
    config:
        components: null  # Extract all components (C, S, A)
        neurons: null     # Extract all neurons
        derived_metrics:
            neuron_mean_activity: 'mean'  # Calculate mean of all C traces
"""

import os
import cv2
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as darr
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..core.navigraph_plugin import NaviGraphPlugin
from ..core.exceptions import NavigraphError
from ..core.registry import register_data_source_plugin


# Minian dimension constants
UNIT_ID = 'unit_id'  # Neuron identifier dimension
FRAME = 'frame'      # Temporal dimension
HEIGHT = 'height'    # Spatial dimension (y-axis)
WIDTH = 'width'      # Spatial dimension (x-axis)


@register_data_source_plugin("neural_activity")
class NeuralActivityPlugin(NaviGraphPlugin):
    """Loads neural activity data from Minian zarr format with comprehensive extraction."""

    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.config.get('name', 'neural_activity'),
            'type': 'neural_activity',
            'description': 'Loads neural activity data from Minian zarr format with C, S, A extraction',
            'provides': ['neural_video'] if self._find_video_file() else [],
            'augments': 'unit_id_*_C, unit_id_*_S, unit_id_*_A_contour columns for each neuron'
        }

    def provide(self, shared_resources: Dict[str, Any]) -> None:
        """Provide associated video file to shared resources if found.

        Args:
            shared_resources: Dictionary to store shared objects
        """
        video_file = self._find_video_file()
        if video_file:
            self.logger.info(f"Found neural activity video: {video_file.name}")

            # Extract basic video info
            try:
                cap = cv2.VideoCapture(str(video_file))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

                    shared_resources['neural_video'] = {
                        'path': str(video_file),
                        'fps': fps,
                        'frame_count': frame_count,
                        'width': width,
                        'height': height
                    }
                    self.logger.info(f"✓ Neural video provided: {width}x{height} @ {fps:.1f} fps")
            except Exception as e:
                self.logger.warning(f"Could not extract video info: {e}")

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
            # Load full Minian dataset
            minian_data = self._load_minian_data(str(minian_path))

            if minian_data is None:
                self.logger.warning("No neural data loaded, returning unchanged dataframe")
                return dataframe

            # Extract and add neural activity data
            result_df = self._extract_and_add_neural_data(minian_data, dataframe)

            # Add derived neural metrics if requested
            result_df = self._add_derived_neural_metrics(result_df)

            # Log summary statistics
            self._log_loading_summary(minian_data, result_df)

            return result_df

        except Exception as e:
            raise NavigraphError(
                f"Failed to load neural activity from {minian_path}: {str(e)}"
            ) from e

    def _load_minian_data(self, dpath: str) -> Optional[xr.Dataset]:
        """Load Minian data from directory or file.

        Args:
            dpath: Path to Minian directory or dataset file

        Returns:
            xarray Dataset containing all Minian components, or None if loading fails
        """
        if os.path.isfile(dpath):
            # Load from single file
            self.logger.debug(f"Loading Minian dataset from file: {dpath}")
            try:
                ds = xr.open_dataset(dpath).chunk()
                return ds
            except Exception as e:
                self.logger.error(f"Failed to load dataset file: {e}")
                return None

        elif os.path.isdir(dpath):
            # Load all zarr arrays from directory
            self.logger.debug(f"Loading Minian zarr arrays from directory: {dpath}")
            dslist = []

            for d in sorted(os.listdir(dpath)):
                arr_path = os.path.join(dpath, d)
                if os.path.isdir(arr_path) and d.endswith('.zarr'):
                    try:
                        # Open with explicit consolidated=False to avoid warning
                        arr = list(xr.open_zarr(arr_path, consolidated=False).values())[0]

                        # Load the actual data using dask
                        zarr_path = os.path.join(arr_path, arr.name)
                        if os.path.exists(zarr_path):
                            arr.data = darr.from_zarr(zarr_path, inline_array=True)
                        else:
                            # Try without the nested name directory
                            arr.data = darr.from_zarr(arr_path, inline_array=True)

                        dslist.append(arr)
                        self.logger.debug(f"Loaded {d}: shape={arr.shape}, dims={arr.dims}")

                    except Exception as e:
                        self.logger.debug(f"Could not load {d}: {e}")
                        continue

            if not dslist:
                self.logger.warning(f"No valid zarr arrays found in {dpath}")
                return None

            # Merge all arrays into single dataset
            try:
                ds = xr.merge(dslist, compat="no_conflicts")
                self.logger.info(f"Loaded Minian dataset with variables: {list(ds.data_vars)}")
                return ds
            except Exception as e:
                self.logger.error(f"Failed to merge zarr arrays: {e}")
                return None

        return None

    def _extract_and_add_neural_data(
        self,
        minian_data: xr.Dataset,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract C, S, and A components and add to dataframe.

        Args:
            minian_data: Minian xarray Dataset
            dataframe: DataFrame to augment

        Returns:
            DataFrame with neural activity columns added
        """
        result_df = dataframe.copy()
        target_length = len(dataframe)

        # Get available neurons
        if UNIT_ID not in minian_data.dims:
            self.logger.warning(f"No '{UNIT_ID}' dimension found in Minian data")
            return result_df

        unit_ids = minian_data.coords[UNIT_ID].values
        self.logger.info(f"Found {len(unit_ids)} neurons in Minian data")

        # Check config for component selection
        plugin_config = self.config.get('config', {})
        components_to_extract = plugin_config.get('components', None)
        selected_neurons = plugin_config.get('neurons', None)

        # Default: extract all components if not specified or if null
        if components_to_extract is None:
            components_to_extract = ['C', 'S', 'A']
        elif not components_to_extract:  # Empty list
            self.logger.warning("No components specified for extraction")
            return result_df

        # Filter neurons if specified
        if selected_neurons is not None:
            # Convert to list of unit_ids
            selected_ids = []
            for neuron_spec in selected_neurons:
                if neuron_spec in unit_ids:
                    selected_ids.append(neuron_spec)
                elif isinstance(neuron_spec, int) and neuron_spec in unit_ids:
                    selected_ids.append(neuron_spec)
            unit_ids = selected_ids
            if not unit_ids:
                self.logger.warning("No valid neurons found in selection")
                return result_df
            self.logger.info(f"Extracting data for {len(unit_ids)} selected neurons")

        # Collect all new columns to add at once (avoids DataFrame fragmentation)
        new_columns = {}

        # Extract C (calcium traces) if requested
        if 'C' in components_to_extract and 'C' in minian_data.data_vars:
            self.logger.debug("Extracting C (calcium traces)")
            for unit_id in unit_ids:
                try:
                    c_values = minian_data['C'].sel({UNIT_ID: unit_id}).values
                    column_name = f"unit_id_{unit_id}_C"
                    new_columns[column_name] = self._align_to_dataframe(c_values, target_length, column_name)
                except Exception as e:
                    self.logger.debug(f"Could not extract C for unit {unit_id}: {e}")

        # Extract S (spike activity) if requested
        if 'S' in components_to_extract and 'S' in minian_data.data_vars:
            self.logger.debug("Extracting S (spike activity)")
            for unit_id in unit_ids:
                try:
                    s_values = minian_data['S'].sel({UNIT_ID: unit_id}).values
                    column_name = f"unit_id_{unit_id}_S"
                    new_columns[column_name] = self._align_to_dataframe(s_values, target_length, column_name)
                except Exception as e:
                    self.logger.debug(f"Could not extract S for unit {unit_id}: {e}")

        # Extract A (spatial footprints) as contours with activity if requested
        if 'A' in components_to_extract and 'A' in minian_data.data_vars:
            self.logger.debug("Extracting A (spatial footprints) as activity-modulated contours")
            for unit_id in unit_ids:
                try:
                    # Extract spatial footprint contour (static)
                    a_matrix = minian_data['A'].sel({UNIT_ID: unit_id}).values
                    contour = self._extract_contour_from_footprint(a_matrix)

                    # Get corresponding calcium activity (dynamic)
                    c_values = None
                    if 'C' in minian_data.data_vars:
                        try:
                            c_values = minian_data['C'].sel({UNIT_ID: unit_id}).values
                        except Exception:
                            pass

                    column_name = f"unit_id_{unit_id}_A_contour"

                    # Store activity-modulated contour data for each frame
                    contour_data = []
                    for frame_idx in range(target_length):
                        if c_values is not None and frame_idx < len(c_values):
                            activity = float(c_values[frame_idx])
                        else:
                            activity = 0.0

                        # Store tuple of (contour_points, activity_value)
                        contour_data.append((contour, activity))

                    new_columns[column_name] = contour_data

                except Exception as e:
                    self.logger.debug(f"Could not extract A contour for unit {unit_id}: {e}")

        # Add all new columns at once to avoid DataFrame fragmentation
        if new_columns:
            new_df = pd.DataFrame(new_columns, index=result_df.index)
            result_df = pd.concat([result_df, new_df], axis=1)

        return result_df

    def _align_to_dataframe(
        self,
        values: np.ndarray,
        target_length: int,
        column_name: str
    ) -> np.ndarray:
        """Align neural data array to target dataframe length.

        Args:
            values: Neural activity values
            target_length: Target length to align to
            column_name: Name of column for logging

        Returns:
            Aligned array of target length
        """
        if len(values) == target_length:
            return values
        elif len(values) > target_length:
            self.logger.debug(f"Truncating {column_name} from {len(values)} to {target_length}")
            return values[:target_length]
        else:
            self.logger.debug(f"Padding {column_name} from {len(values)} to {target_length}")
            padded = np.full(target_length, np.nan)
            padded[:len(values)] = values
            return padded

    def _extract_contour_from_footprint(
        self,
        footprint: np.ndarray,
        threshold_percentile: float = 50
    ) -> List[Tuple[float, float]]:
        """Extract contour points from spatial footprint matrix.

        Args:
            footprint: 2D spatial footprint matrix
            threshold_percentile: Percentile for threshold (default 50)

        Returns:
            List of (x, y) contour points
        """
        # Normalize footprint to 0-255 range
        footprint_norm = footprint - footprint.min()
        if footprint_norm.max() > 0:
            footprint_norm = (footprint_norm / footprint_norm.max() * 255).astype(np.uint8)
        else:
            return []

        # Apply threshold to create binary mask
        threshold = np.percentile(footprint_norm[footprint_norm > 0], threshold_percentile) if footprint_norm.max() > 0 else 1
        _, binary_mask = cv2.threshold(footprint_norm, threshold, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Convert to list of (x, y) tuples
        contour_points = [(float(point[0][0]), float(point[0][1])) for point in largest_contour]

        return contour_points

    def _find_video_file(self) -> Optional[Path]:
        """Find associated video file for neural activity.

        Searches in:
        1. The Minian directory itself
        2. A 'video' subdirectory within Minian
        3. The parent directory of Minian
        4. A 'video' subdirectory in the parent

        Returns:
            Path to video file if found, None otherwise
        """
        if not self.discovered_files:
            return None

        minian_path = self.discovered_files[0]

        # Define search locations
        search_dirs = [
            minian_path,                          # In Minian directory
            minian_path / 'video',                # In video subdirectory
            minian_path / 'videos',               # In videos subdirectory
            minian_path.parent,                   # In parent directory
            minian_path.parent / 'video',         # In sibling video directory
            minian_path.parent / 'videos',        # In sibling videos directory
        ]

        video_extensions = ['.avi', '.mp4', '.mov', '.mkv', '.wmv']

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Look for video files
            for ext in video_extensions:
                video_files = list(search_dir.glob(f'*{ext}'))
                if video_files:
                    # Return first video found
                    self.logger.debug(f"Found video file: {video_files[0]}")
                    return video_files[0]

        return None

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

        # Find all C columns for derived metrics
        c_columns = [col for col in dataframe.columns if col.endswith('_C')]

        if not c_columns:
            self.logger.warning("No calcium trace columns found for derived metrics")
            return dataframe

        result_df = dataframe.copy()

        # Collect derived metrics to add at once (avoids DataFrame fragmentation)
        derived_columns = {}

        for metric_name, metric_config in derived_metrics.items():
            try:
                if metric_config == 'mean' or (isinstance(metric_config, dict) and metric_config.get('function') == 'mean'):
                    # Calculate mean across all C traces
                    derived_columns[metric_name] = result_df[c_columns].mean(axis=1)
                    self.logger.debug(
                        f"Added derived metric '{metric_name}': mean of {len(c_columns)} calcium traces"
                    )
                else:
                    self.logger.warning(f"Unknown derived metric function: {metric_config}")

            except Exception as e:
                self.logger.error(f"Failed to calculate derived metric '{metric_name}': {str(e)}")

        # Add all derived metrics at once
        if derived_columns:
            derived_df = pd.DataFrame(derived_columns, index=result_df.index)
            result_df = pd.concat([result_df, derived_df], axis=1)

        return result_df

    def _log_loading_summary(self, minian_data: xr.Dataset, result_df: pd.DataFrame) -> None:
        """Log summary of loaded neural activity data.

        Args:
            minian_data: Loaded Minian dataset
            result_df: DataFrame with neural columns added
        """
        # Count neurons
        n_neurons = len(minian_data.coords[UNIT_ID]) if UNIT_ID in minian_data.dims else 0

        # Count different column types
        c_columns = [col for col in result_df.columns if col.endswith('_C')]
        s_columns = [col for col in result_df.columns if col.endswith('_S')]
        contour_columns = [col for col in result_df.columns if col.endswith('_A_contour')]

        # Check coverage
        coverage_info = ""
        if c_columns:
            first_c = result_df[c_columns[0]]
            valid_count = np.sum(~np.isnan(first_c))
            coverage_pct = (valid_count / len(result_df)) * 100 if len(result_df) > 0 else 0
            coverage_info = f", coverage={coverage_pct:.1f}%"

        components = []
        if c_columns:
            components.append(f"{len(c_columns)} C")
        if s_columns:
            components.append(f"{len(s_columns)} S")
        if contour_columns:
            components.append(f"{len(contour_columns)} activity-modulated contours")

        components_str = f" ({', '.join(components)})" if components else ""

        self.logger.info(
            f"✓ Neural activity loaded: {n_neurons} neurons{components_str}, "
            f"{len(result_df)} frames{coverage_info}"
        )

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

        # We don't know individual neuron columns until runtime
        return columns