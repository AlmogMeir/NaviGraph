"""Coordinate transformation utilities for NaviGraph plugins.

Provides common transformation functions used by multiple plugins.
"""

import numpy as np
from typing import Tuple


def transform_coordinates(
    x_coords: np.ndarray, 
    y_coords: np.ndarray, 
    calibration_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform image coordinates to calibrated coordinates using homography matrix.
    
    This function applies a 3x3 homography transformation matrix to convert
    coordinates from image space to calibrated/map space. Handles NaN values
    properly and performs safe division for homogeneous coordinates.
    
    Args:
        x_coords: Array of x coordinates in image space
        y_coords: Array of y coordinates in image space  
        calibration_matrix: 3x3 homography transformation matrix
        
    Returns:
        Tuple of (transformed_x, transformed_y) arrays with same shape as input
        
    Note:
        - NaN input coordinates will remain NaN in output
        - Invalid transformations (w ≈ 0) will result in NaN output
        - Used by both map_location and graph_location plugins
    """
    # Handle NaN values
    valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords))
    
    # Initialize output arrays with NaN
    transformed_x = np.full_like(x_coords, np.nan, dtype=float)
    transformed_y = np.full_like(y_coords, np.nan, dtype=float)
    
    if not np.any(valid_mask):
        # No valid coordinates to transform
        return transformed_x, transformed_y
    
    # Extract valid coordinates
    valid_x = x_coords[valid_mask]
    valid_y = y_coords[valid_mask]
    
    # Create homogeneous coordinates [x, y, 1]
    ones = np.ones(len(valid_x))
    points = np.vstack([valid_x, valid_y, ones])
    
    # Apply transformation: [x', y', w'] = H * [x, y, 1]
    transformed_points = calibration_matrix @ points
    
    # Convert from homogeneous coordinates: x' = x'/w', y' = y'/w'
    # Handle potential division by zero
    w = transformed_points[2, :]
    valid_w_mask = np.abs(w) > 1e-10  # Avoid division by near-zero values
    
    # Initialize result arrays for valid coordinates
    result_x = np.full_like(valid_x, np.nan)
    result_y = np.full_like(valid_y, np.nan)
    
    # Perform division only for valid w values
    if np.any(valid_w_mask):
        result_x[valid_w_mask] = transformed_points[0, valid_w_mask] / w[valid_w_mask]
        result_y[valid_w_mask] = transformed_points[1, valid_w_mask] / w[valid_w_mask]
    
    # Place transformed coordinates back into full arrays
    transformed_x[valid_mask] = result_x
    transformed_y[valid_mask] = result_y
    
    return transformed_x, transformed_y


def validate_calibration_matrix(calibration_matrix: np.ndarray) -> None:
    """Validate that a calibration matrix is suitable for coordinate transformation.
    
    Args:
        calibration_matrix: Matrix to validate
        
    Raises:
        ValueError: If matrix is invalid
    """
    if calibration_matrix.shape != (3, 3):
        raise ValueError(
            f"Calibration matrix must be 3x3, got shape {calibration_matrix.shape}"
        )
    
    # Check if matrix is singular (determinant near zero)
    det = np.linalg.det(calibration_matrix)
    if np.abs(det) < 1e-10:
        raise ValueError(
            f"Calibration matrix is singular (determinant={det:.3e}). "
            f"This indicates an invalid transformation."
        )
    
    # Check for NaN or infinite values
    if not np.all(np.isfinite(calibration_matrix)):
        raise ValueError("Calibration matrix contains NaN or infinite values")


def apply_coordinate_transform_to_bodyparts(
    dataframe: "pd.DataFrame",
    bodyparts: list[str], 
    calibration_matrix: np.ndarray,
    output_suffix: str = "map"
) -> "pd.DataFrame":
    """Apply coordinate transformation to multiple bodyparts in a DataFrame.
    
    Convenience function that transforms all specified bodyparts using the
    calibration matrix and adds new columns with the specified suffix.
    
    Args:
        dataframe: DataFrame with bodypart coordinate columns
        bodyparts: List of bodypart names to transform
        calibration_matrix: 3x3 transformation matrix
        output_suffix: Suffix for output columns (e.g., "map" -> "bodypart_map_x")
        
    Returns:
        DataFrame with added transformed coordinate columns
        
    Raises:
        ValueError: If required columns are missing or matrix is invalid
    """
    import pandas as pd
    
    # Validate calibration matrix
    validate_calibration_matrix(calibration_matrix)
    
    # Create a copy to avoid modifying original
    result_df = dataframe.copy()
    
    for bodypart in bodyparts:
        x_col = f'{bodypart}_x'
        y_col = f'{bodypart}_y'
        
        # Check if required columns exist
        if x_col not in dataframe.columns or y_col not in dataframe.columns:
            raise ValueError(
                f"Missing required columns for bodypart '{bodypart}': {x_col}, {y_col}"
            )
        
        # Transform coordinates
        transformed_x, transformed_y = transform_coordinates(
            dataframe[x_col].values,
            dataframe[y_col].values,
            calibration_matrix
        )
        
        # Add transformed columns
        result_df[f'{bodypart}_{output_suffix}_x'] = transformed_x
        result_df[f'{bodypart}_{output_suffix}_y'] = transformed_y
    
    return result_df