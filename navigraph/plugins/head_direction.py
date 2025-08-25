"""Head direction plugin for NaviGraph unified architecture.

Loads head direction data from CSV files containing quaternion data and converts
to Euler angles (yaw, pitch, roll) for integration into session DataFrames.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from ...core.navigraph_plugin import NaviGraphPlugin
from ...core.exceptions import NavigraphError
from ...core.registry import register_data_source_plugin
from ...core.conversion_utils import quaternions_to_euler


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
            yaw_offset = self.config.get('yaw_offset', -167)
            positive_direction = self.config.get('positive_direction', -1)
            
            euler_angles = quaternions_to_euler(
                raw_data, 
                yaw_offset=yaw_offset,
                positive_direction=positive_direction
            )
            
            self.logger.info(
                f"Converted {len(euler_angles)} quaternions to Euler angles "
                f"(yaw_offset={yaw_offset}, positive_direction={positive_direction})"
            )
            
            # Apply sampling adjustment (manual sync option) if specified
            skip_index = self.config.get('skip_index', None)
            if skip_index is not None and skip_index > 1:
                adjusted_angles = euler_angles[::skip_index]
                self.logger.info(
                    f"Applied sampling adjustment (skip_index={skip_index}): "
                    f"{len(adjusted_angles)} samples"
                )
            else:
                adjusted_angles = euler_angles
            
            # Add head direction columns to dataframe
            result_df = dataframe.copy()
            yaw_list, pitch_list, roll_list = [], [], []
            
            for frame_idx in result_df.index:
                if frame_idx < len(adjusted_angles):
                    angles = adjusted_angles[frame_idx]
                    yaw_list.append(float(angles[0]))
                    pitch_list.append(float(angles[1]))
                    roll_list.append(float(angles[2]))
                else:
                    # No corresponding data - use NaN
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
            
            self.logger.info(
                f"✓ Head direction loaded: {valid_yaw}/{total_frames} frames ({coverage_pct:.1f}% coverage)"
            )
            
            if coverage_pct < 50:
                self.logger.warning(
                    f"Low head direction coverage ({coverage_pct:.1f}%). "
                    f"Check skip_index parameter or data alignment."
                )
            
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