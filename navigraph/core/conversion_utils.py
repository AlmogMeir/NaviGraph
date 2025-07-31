"""Conversion utilities for angle and quaternion operations.

This module provides utilities for converting between different angle representations
and coordinate transformations commonly used in behavioral analysis.
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from typing import Union, Tuple


def wrap_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap an angle in degrees to the range [-180, 180].
    
    Args:
        angle: Angle(s) in degrees to wrap
        
    Returns:
        Angle(s) wrapped to [-180, 180] range
        
    Examples:
        >>> wrap_angle(270)
        -90.0
        >>> wrap_angle([270, 360, -200])
        array([-90.,   0., 160.])
    """
    return (angle + 180) % 360 - 180


def quaternions_to_euler(
    data: pd.DataFrame, 
    yaw_offset: float = -167, 
    positive_direction: float = -1
) -> np.ndarray:
    """Convert quaternion values in a DataFrame to Euler angles.
    
    Converts quaternion data to Euler angles using ZYX convention (yaw, pitch, roll).
    Applies calibration offset and direction correction to yaw values.
    
    Args:
        data: DataFrame containing quaternion columns ['qw', 'qx', 'qy', 'qz']
        yaw_offset: Offset to subtract from yaw in degrees (default: -167)
        positive_direction: Multiplier for yaw direction correction (default: -1)
        
    Returns:
        Array of shape (N, 3) with Euler angles [yaw, pitch, roll] in degrees
        
    Raises:
        KeyError: If required quaternion columns are missing
        ValueError: If no valid quaternion data is found
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'qw': [1.0, 0.707],
        ...     'qx': [0.0, 0.0], 
        ...     'qy': [0.0, 0.0],
        ...     'qz': [0.0, 0.707]
        ... })
        >>> euler = quaternions_to_euler(df)
        >>> euler.shape
        (2, 3)
    """
    # Validate required columns
    required_cols = ['qw', 'qx', 'qy', 'qz']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Missing required quaternion columns: {missing_cols}")
    
    # Extract and validate quaternion data
    quats = data[required_cols].dropna().values
    if len(quats) == 0:
        raise ValueError("No valid quaternion data found after dropping NaN values")
    
    # Reorder quaternions from [qw, qx, qy, qz] to [qx, qy, qz, qw] for scipy
    quats_reordered = quats[:, [1, 2, 3, 0]]
    
    # Convert to Euler angles using ZYX convention (yaw, pitch, roll)
    euler_angles = Rotation.from_quat(quats_reordered).as_euler('zyx', degrees=True)
    
    # Apply yaw calibration: offset, wrap, and direction correction
    euler_angles[:, 0] = wrap_angle(euler_angles[:, 0] - yaw_offset)
    euler_angles[:, 0] *= positive_direction
    
    return euler_angles


def degrees_to_radians(degrees: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert degrees to radians.
    
    Args:
        degrees: Angle(s) in degrees
        
    Returns:
        Angle(s) in radians
    """
    return np.deg2rad(degrees)


def radians_to_degrees(radians: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert radians to degrees.
    
    Args:
        radians: Angle(s) in radians
        
    Returns:
        Angle(s) in degrees
    """
    return np.rad2deg(radians)


def normalize_angle_continuous(angles: np.ndarray) -> np.ndarray:
    """Normalize angle sequence to minimize discontinuities.
    
    Unwraps angle sequences to avoid large jumps due to 360-degree wrapping.
    Useful for continuous angle tracking where discontinuities should be minimized.
    
    Args:
        angles: Array of angles in degrees
        
    Returns:
        Unwrapped angle sequence
        
    Examples:
        >>> angles = np.array([350, 10, 20])  # 350 -> 10 is a small step, not 340
        >>> normalize_angle_continuous(angles)
        array([350., 370., 380.])
    """
    return np.rad2deg(np.unwrap(np.deg2rad(angles)))


def compute_angular_velocity(
    angles: np.ndarray, 
    dt: Union[float, np.ndarray], 
    smooth_window: int = 1
) -> np.ndarray:
    """Compute angular velocity from angle sequence.
    
    Args:
        angles: Array of angles in degrees
        dt: Time step(s) between measurements
        smooth_window: Window size for smoothing (default: 1, no smoothing)
        
    Returns:
        Angular velocity in degrees per time unit
    """
    # Unwrap angles to handle discontinuities
    unwrapped = normalize_angle_continuous(angles)
    
    # Compute velocity
    if isinstance(dt, (int, float)):
        velocity = np.gradient(unwrapped) / dt
    else:
        velocity = np.gradient(unwrapped, dt)
    
    # Apply smoothing if requested
    if smooth_window > 1:
        # Simple moving average
        velocity = np.convolve(
            velocity, 
            np.ones(smooth_window) / smooth_window, 
            mode='same'
        )
    
    return velocity