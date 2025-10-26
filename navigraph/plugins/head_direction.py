"""Head direction plugin for NaviGraph unified architecture.

Loads head direction data from CSV files containing quaternion data and converts
to Euler angles (yaw, pitch, roll) for integration into session DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from ..core.navigraph_plugin import NaviGraphPlugin
from ..core.exceptions import NavigraphError
from ..core.registry import register_data_source_plugin
from ..core.conversion_utils import quaternions_to_euler


@register_data_source_plugin("head_direction")
class HeadDirectionPlugin(NaviGraphPlugin):
    """Head direction plugin for orientation tracking.
    
    Loads quaternion data from CSV files and converts to Euler angles,
    with support for sampling rate adjustment via skip_index.
    """
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.config.get('name', 'head_direction'),
            'type': 'head_direction',
            'description': 'Loads head orientation data from CSV files with quaternion conversion',
            'provides': [],
            'augments': 'yaw, pitch, roll columns'
        }
    
    def augment_data(self, dataframe: pd.DataFrame, shared_resources: Dict[str, Any]) -> pd.DataFrame:
        """Add head direction columns to the dataframe.
        
        Args:
            dataframe: DataFrame with temporal index (usually from pose tracking)
            shared_resources: Available shared resources
            
        Returns:
            DataFrame with yaw, pitch, roll columns added
        """
        # Validate we have discovered files
        if not self.discovered_files:
            self.logger.warning("No head direction file discovered, skipping head direction data")
            return dataframe
        
        head_direction_file = self.discovered_files[0]  # Use first discovered file
        self.logger.info(f"Loading head direction from: {head_direction_file.name}")
        
        try:
            # Load raw CSV data
            raw_data = pd.read_csv(head_direction_file)
            self.logger.debug(f"Loaded {len(raw_data)} quaternion records")
            
            # Validate quaternion columns
            required_cols = ['qw', 'qx', 'qy', 'qz']
            missing_cols = [col for col in required_cols if col not in raw_data.columns]
            if missing_cols:
                raise NavigraphError(f"Missing required quaternion columns: {missing_cols}")
            
            # Convert quaternions to Euler angles
            config_params = self.config.get('config', {})
            yaw_offset = config_params.get('yaw_offset', -167)
            positive_direction = config_params.get('positive_direction', -1)

            euler_angles = quaternions_to_euler(
                raw_data,
                yaw_offset=yaw_offset,
                positive_direction=positive_direction
            )
            
            # Add head direction columns to dataframe
            result_df = dataframe.copy()
            yaw_list, pitch_list, roll_list = [], [], []

            # Get skip_index for frame-to-IMU mapping (like frame_idx += 2)
            skip_index = config_params.get('skip_index', 1)

            # Use relative frame position instead of absolute DataFrame index
            frame_indices = list(result_df.index)

            for i, _ in enumerate(frame_indices):
                # Map relative frame position to IMU sample (like your frame_idx += 2)
                # i=0 → IMU[0], i=1 → IMU[2], i=2 → IMU[4], etc.
                imu_idx = i * skip_index

                if imu_idx < len(euler_angles):
                    angles = euler_angles[imu_idx]
                    yaw_list.append(float(angles[0]))
                    pitch_list.append(float(angles[1]))
                    roll_list.append(float(angles[2]))
                else:
                    # No corresponding IMU data - use NaN
                    yaw_list.append(np.nan)
                    pitch_list.append(np.nan)
                    roll_list.append(np.nan)
            
            result_df["yaw"] = yaw_list
            result_df["pitch"] = pitch_list
            result_df["roll"] = roll_list
            
            # Log coverage statistics
            valid_yaw = np.sum(~np.isnan(yaw_list))
            total_frames = len(yaw_list)
            coverage_pct = (valid_yaw / total_frames) * 100 if total_frames > 0 else 0

            self.logger.info(f"✓ Head direction loaded: {valid_yaw}/{total_frames} frames ({coverage_pct:.1f}% coverage)")

            if coverage_pct < 50:
                self.logger.warning(f"Low head direction coverage ({coverage_pct:.1f}%). Check data alignment.")
            
            return result_df
            
        except Exception as e:
            raise NavigraphError(
                f"Failed to load head direction from {head_direction_file}: {str(e)}"
            ) from e
    
    def get_expected_columns(self) -> List[str]:
        """Generate expected column names.
        
        Returns:
            List of column names this plugin will create
        """
        return ['yaw', 'pitch', 'roll']