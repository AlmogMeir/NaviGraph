"""Head direction data source plugin for NaviGraph.

Loads head direction data from CSV files containing quaternion data and converts
to Euler angles (yaw, pitch, roll) for integration into session DataFrames.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from ...core.interfaces import IDataSource, Logger
from ...core.exceptions import DataSourceError
from ...core.base_plugin import BasePlugin
from ...core.registry import register_data_source_plugin
from ...core.conversion_utils import quaternions_to_euler


@register_data_source_plugin("head_direction")
class HeadDirectionDataSource(BasePlugin, IDataSource):
    """Head direction data source for orientation tracking.
    
    Loads quaternion data from CSV files and converts to Euler angles,
    with support for sampling rate adjustment and calibration corrections.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger_instance: Optional[Logger] = None):
        """Initialize head direction data source.
        
        Args:
            config: Configuration dictionary containing:
                - head_direction_path: Path to CSV file with quaternion data
                - yaw_offset: Yaw offset for calibration (default: -167)
                - positive_direction: Direction multiplier for yaw (default: -1)
                - skip_index: Sampling adjustment factor (default: 2)
                - merge_mode: How to merge with session data (default: 'left')
            logger_instance: Logger instance
        """
        super().__init__(config, logger_instance)
        self._raw_data: Optional[pd.DataFrame] = None
        self._euler_angles: Optional[np.ndarray] = None
        self._adjusted_angles: Optional[np.ndarray] = None
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance: Optional[Logger] = None):
        """Factory method to create head direction data source from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate head direction data source configuration."""
        # No specific validation needed - path will be discovered
        pass
    
    def initialize(self) -> None:
        """Initialize head direction data loading."""
        super().initialize()
        # Initialization deferred until integrate_data_into_session when we have the discovered path
    
    def integrate_data_into_session(
        self,
        current_dataframe: pd.DataFrame,
        session_config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        logger: Logger
    ) -> pd.DataFrame:
        """Integrate head direction data into session DataFrame.
        
        Args:
            current_dataframe: Current session DataFrame
            session_config: Session configuration
            shared_resources: Shared resources dictionary
            logger: Logger instance
            
        Returns:
            DataFrame with yaw, pitch, roll columns added
            
        Raises:
            DataSourceError: If integration fails
        """
        # Load head direction data if not already loaded
        if self._adjusted_angles is None:
            # Get discovered file path
            discovered_path = session_config.get('discovered_file_path')
            if not discovered_path:
                logger.warning("Head direction data source: no file discovered")
                return current_dataframe
            
            logger.info(f"Loading head direction data from discovered path: {discovered_path}")
            
            try:
                # Load raw CSV data
                self._raw_data = pd.read_csv(discovered_path)
                logger.debug(f"Loaded {len(self._raw_data)} quaternion records")
                
                # Validate quaternion columns
                required_cols = ['qw', 'qx', 'qy', 'qz']
                missing_cols = [col for col in required_cols if col not in self._raw_data.columns]
                if missing_cols:
                    raise DataSourceError(f"Missing required quaternion columns: {missing_cols}")
                
                # Convert quaternions to Euler angles
                yaw_offset = self.config.get('yaw_offset', -167)
                positive_direction = self.config.get('positive_direction', -1)
                
                self._euler_angles = quaternions_to_euler(
                    self._raw_data, 
                    yaw_offset=yaw_offset,
                    positive_direction=positive_direction
                )
                
                logger.info(
                    f"Converted {len(self._euler_angles)} quaternions to Euler angles "
                    f"(yaw_offset={yaw_offset}, positive_direction={positive_direction})"
                )
                
                # Apply sampling adjustment
                skip_index = self.config.get('skip_index', 2)
                if skip_index > 1:
                    self._adjusted_angles = self._euler_angles[::skip_index]
                    logger.info(
                        f"Applied sampling adjustment (skip_index={skip_index}): "
                        f"{len(self._adjusted_angles)} samples"
                    )
                else:
                    self._adjusted_angles = self._euler_angles.copy()
                    
            except Exception as e:
                raise DataSourceError(f"Failed to load head direction data: {str(e)}")
        
        logger.info("Integrating head direction data into session")
        
        try:
            # Create head direction DataFrame aligned with session index
            enhanced_df = current_dataframe.copy()
            
            # Initialize angle columns
            yaw_values = []
            pitch_values = []
            roll_values = []
            
            # Get session frame indices
            session_indices = enhanced_df.index.tolist()
            
            logger.debug(
                f"Session frames: {len(session_indices)} "
                f"(range: {min(session_indices)}-{max(session_indices)})"
            )
            logger.debug(f"Available head direction samples: {len(self._adjusted_angles)}")
            
            # Map session frames to head direction data
            for frame_idx in session_indices:
                yaw, pitch, roll = self._get_angles_for_frame(frame_idx)
                yaw_values.append(yaw)
                pitch_values.append(pitch)
                roll_values.append(roll)
            
            # Add angle columns to DataFrame
            enhanced_df['yaw'] = yaw_values
            enhanced_df['pitch'] = pitch_values
            enhanced_df['roll'] = roll_values
            
            # Log integration statistics
            valid_yaw = np.sum(~np.isnan(yaw_values))
            total_frames = len(yaw_values)
            coverage_pct = (valid_yaw / total_frames) * 100 if total_frames > 0 else 0
            
            logger.info(
                f"Head direction integration complete. "
                f"Coverage: {valid_yaw}/{total_frames} frames ({coverage_pct:.1f}%)"
            )
            
            if coverage_pct < 50:
                logger.warning(
                    f"Low head direction coverage ({coverage_pct:.1f}%). "
                    f"Check skip_index parameter or data alignment."
                )
            
            return enhanced_df
            
        except Exception as e:
            raise DataSourceError(f"Failed to integrate head direction data: {str(e)}")
    
    def _get_angles_for_frame(self, frame_idx: int) -> Tuple[float, float, float]:
        """Get yaw, pitch, roll angles for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Tuple of (yaw, pitch, roll) in degrees, with NaN for unavailable data
        """
        if (self._adjusted_angles is None or 
            frame_idx < 0 or 
            frame_idx >= len(self._adjusted_angles)):
            return (np.nan, np.nan, np.nan)
        
        angles = self._adjusted_angles[frame_idx]
        return float(angles[0]), float(angles[1]), float(angles[2])
    
    def get_raw_quaternion_data(self) -> Optional[pd.DataFrame]:
        """Get raw quaternion data.
        
        Returns:
            DataFrame with original quaternion data, or None if not loaded
        """
        return self._raw_data.copy() if self._raw_data is not None else None
    
    def get_euler_angles(self, adjusted: bool = True) -> Optional[np.ndarray]:
        """Get Euler angles array.
        
        Args:
            adjusted: If True, return sampling-adjusted angles; if False, return raw conversion
            
        Returns:
            Array of shape (N, 3) with [yaw, pitch, roll] angles, or None if not loaded
        """
        if adjusted and self._adjusted_angles is not None:
            return self._adjusted_angles.copy()
        elif not adjusted and self._euler_angles is not None:
            return self._euler_angles.copy()
        else:
            return None
    
    def get_head_direction_at_frame(self, frame_idx: int) -> Dict[str, float]:
        """Get head direction angles for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Dictionary with 'yaw', 'pitch', 'roll' keys and angle values
        """
        yaw, pitch, roll = self._get_angles_for_frame(frame_idx)
        
        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        }
    
    def get_yaw_statistics(self) -> Dict[str, float]:
        """Get statistical summary of yaw angles.
        
        Returns:
            Dictionary with yaw angle statistics
        """
        if self._adjusted_angles is None:
            return {}
        
        yaw_angles = self._adjusted_angles[:, 0]
        valid_yaw = yaw_angles[~np.isnan(yaw_angles)]
        
        if len(valid_yaw) == 0:
            return {}
        
        return {
            'mean_yaw': float(np.mean(valid_yaw)),
            'std_yaw': float(np.std(valid_yaw)),
            'min_yaw': float(np.min(valid_yaw)),
            'max_yaw': float(np.max(valid_yaw)),
            'range_yaw': float(np.max(valid_yaw) - np.min(valid_yaw)),
            'valid_samples': len(valid_yaw),
            'total_samples': len(yaw_angles)
        }
    
    def compute_angular_velocity(self, dt: float = 1.0) -> Optional[np.ndarray]:
        """Compute angular velocity from yaw angles.
        
        Args:
            dt: Time step between samples
            
        Returns:
            Array of angular velocities in degrees per time unit, or None if no data
        """
        if self._adjusted_angles is None:
            return None
        
        try:
            from ...core.conversion_utils import compute_angular_velocity
            
            yaw_angles = self._adjusted_angles[:, 0]
            return compute_angular_velocity(yaw_angles, dt)
            
        except Exception as e:
            self.logger.error(f"Failed to compute angular velocity: {str(e)}")
            return None
    
    def detect_head_turns(
        self, 
        velocity_threshold: float = 30.0, 
        min_duration: int = 3
    ) -> List[Dict[str, Any]]:
        """Detect significant head turning events.
        
        Args:
            velocity_threshold: Minimum angular velocity for turn detection (degrees/frame)
            min_duration: Minimum duration of turn in frames
            
        Returns:
            List of dictionaries describing detected turns
        """
        angular_velocity = self.compute_angular_velocity()
        if angular_velocity is None:
            return []
        
        # Find periods above threshold
        above_threshold = np.abs(angular_velocity) > velocity_threshold
        
        # Find continuous segments
        diff = np.diff(np.concatenate(([False], above_threshold, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Filter by minimum duration
        turns = []
        for start, end in zip(starts, ends):
            duration = end - start
            if duration >= min_duration:
                segment_velocity = angular_velocity[start:end]
                turns.append({
                    'start_frame': int(start),
                    'end_frame': int(end),
                    'duration': int(duration),
                    'max_velocity': float(np.max(np.abs(segment_velocity))),
                    'mean_velocity': float(np.mean(segment_velocity)),
                    'direction': 'clockwise' if np.mean(segment_velocity) > 0 else 'counterclockwise'
                })
        
        self.logger.info(f"Detected {len(turns)} head turning events")
        return turns
    
    def get_provided_column_names(self) -> List[str]:
        """Get names of columns provided by this data source.
        
        Returns:
            List of column names that will be added to the session DataFrame
        """
        # Head direction provides yaw, pitch, and roll columns
        return ['yaw', 'pitch', 'roll']