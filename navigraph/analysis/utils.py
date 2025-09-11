"""Utility functions for analyzer metrics.

Generic helper functions for data processing and analysis that work with
any column names and data structures.
"""

from typing import Union, Any, List, Dict, Optional, Tuple
from numpy.typing import NDArray
import pandas as pd
import numpy as np


def count_value_transitions(dataframe: pd.DataFrame, start_idx: int, end_idx: int, column_name: str) -> int:
    """Count value transitions in any column (eliminating consecutive duplicates).
    
    Ignores None/NaN values in the transition count.
    
    Args:
        dataframe: DataFrame containing the data
        start_idx: Start index of segment
        end_idx: End index of segment (inclusive)
        column_name: Name of column to count transitions in
        
    Returns:
        Number of value transitions (changes from one value to another)
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column {column_name} not found in dataframe")
    
    segment = dataframe.loc[start_idx:end_idx, column_name]
    
    # Filter out None/NaN values first
    valid_segment = segment[segment.notna() & (segment != None)]
    
    if len(valid_segment) == 0:
        return 0
    
    # Remove consecutive duplicates from valid values
    transitions = []
    prev_val = None
    for val in valid_segment:
        if val != prev_val:
            transitions.append(val)
            prev_val = val
    
    # Count transitions (number of changes)
    return len(transitions) - 1 if len(transitions) > 0 else 0


def count_unique_values_in_segment(dataframe: pd.DataFrame, start_idx: int, end_idx: int, column_name: str) -> int:
    """Count unique values in any column within a dataframe segment.
    
    Excludes None and NaN values from the count.
    
    Args:
        dataframe: DataFrame containing the data
        start_idx: Start index of segment
        end_idx: End index of segment (inclusive)
        column_name: Name of column to count unique values in
        
    Returns:
        Number of unique non-null values in the segment
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column {column_name} not found in dataframe")
    
    segment = dataframe.loc[start_idx:end_idx, column_name]
    # Filter out both NaN and None values
    valid_values = segment[segment.notna() & (segment != None)]
    unique_values = valid_values.unique()
    
    return len(unique_values)


def get_path_sequence(dataframe: pd.DataFrame, start_idx: int, end_idx: int, 
                     column_name: str, remove_duplicates: bool = True) -> List[Any]:
    """Extract sequence of values from any column in a path.
    
    Args:
        dataframe: DataFrame containing the data
        start_idx: Start index of path
        end_idx: End index of path (inclusive)
        column_name: Name of column to extract values from
        remove_duplicates: If True, remove consecutive duplicate values
        
    Returns:
        List of values along the path
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column {column_name} not found in dataframe")
    
    segment = dataframe.loc[start_idx:end_idx, column_name]
    
    if remove_duplicates:
        # Remove consecutive duplicates
        values = []
        prev_val = None
        for val in segment:
            if pd.notna(val) and val != prev_val:
                values.append(val)
                prev_val = val
        return values
    else:
        return segment.dropna().tolist()


def calculate_path_distance(dataframe: pd.DataFrame, start_idx: int, end_idx: int,
                          x_column: str, y_column: str, 
                          pixel_to_meter: Optional[float] = None) -> float:
    """Calculate total distance traveled along a path using any coordinate columns.
    
    Skips segments where either coordinate is None/NaN.
    
    Args:
        dataframe: DataFrame containing coordinate data
        start_idx: Start index of path
        end_idx: End index of path (inclusive)
        x_column: Name of column containing x coordinates
        y_column: Name of column containing y coordinates
        pixel_to_meter: Optional conversion factor from pixels to meters.
                       If provided, returns distance in meters.
                       If None, returns distance in pixels.
        
    Returns:
        Total Euclidean distance traveled (in meters if pixel_to_meter provided,
        otherwise in pixels)
    """
    if x_column not in dataframe.columns or y_column not in dataframe.columns:
        raise ValueError(f"Coordinate columns {x_column}, {y_column} not found")
    
    x_coords = dataframe.loc[start_idx:end_idx, x_column].values
    y_coords = dataframe.loc[start_idx:end_idx, y_column].values
    
    # Calculate distances between consecutive points
    total_distance = 0.0
    for i in range(1, len(x_coords)):
        # Skip if any coordinate is NaN or None
        if not (np.isnan(x_coords[i-1]) or np.isnan(y_coords[i-1]) or 
                np.isnan(x_coords[i]) or np.isnan(y_coords[i]) or
                x_coords[i-1] is None or y_coords[i-1] is None or
                x_coords[i] is None or y_coords[i] is None):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            total_distance += np.sqrt(dx**2 + dy**2)
    
    # Convert to meters if pixel_to_meter ratio provided
    if pixel_to_meter is not None:
        total_distance *= pixel_to_meter
    
    return total_distance


def find_path_transitions(
    dataframe: pd.DataFrame,
    start_column: str,
    start_value: Union[str, int, tuple],
    end_column: str,
    end_value: Union[str, int, tuple],
    min_path_length: int = 0,
    min_unique_values_in_column: Optional[Dict[str, int]] = None,
    max_path_length: Optional[int] = None,
    first_occurrence_only: bool = True
) -> List[Tuple[int, int]]:
    """Find transitions between any values in any columns with flexible filtering.
    
    This function is completely generic - it works with ANY column names
    and value types, not limited to graph locations. Handles None/NaN values
    by skipping them when searching for transitions.
    
    Args:
        dataframe: DataFrame with data columns
        start_column: Name of column containing start location
        start_value: Value to search for in start column
        end_column: Name of column containing end location
        end_value: Value to search for in end column
        min_path_length: Minimum frames between start and end
        min_unique_values_in_column: Dict mapping column names to minimum unique values
                                     e.g., {'node_col': 3, 'edge_col': 2}
        max_path_length: Maximum frames allowed for path
        first_occurrence_only: If True, use first occurrence of end value after start.
                              If False, use last occurrence before next start
        
    Returns:
        List of (start_index, end_index) tuples for valid transitions
        
    Examples:
        # Graph-based transitions
        find_path_transitions(df, 'centroid_graph_node', 5, 'centroid_graph_node', 8)
        
        # Any column transitions
        find_path_transitions(df, 'state', 'exploring', 'state', 'rewarded')
        
        # With filtering
        find_path_transitions(df, 'zone', 'A', 'zone', 'B',
                            min_unique_values_in_column={'visited_nodes': 3})
    """
    
    # Validate columns exist
    missing = []
    if start_column not in dataframe.columns:
        missing.append(start_column)
    if end_column not in dataframe.columns:
        missing.append(end_column)
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    
    transitions = []
    df_remaining = dataframe.copy()
    
    while len(df_remaining) > 0:
        # Find start location (handle different types and skip None/NaN)
        if isinstance(start_value, (int, float)):
            # Skip NaN values for numeric comparisons
            start_mask = (df_remaining[start_column] == start_value) & df_remaining[start_column].notna()
        else:
            # For non-numeric, also skip None values
            start_mask = (df_remaining[start_column].astype(str) == str(start_value)) & \
                        df_remaining[start_column].notna()
        
        if not start_mask.any():
            break
        start_idx = df_remaining[start_mask].index[0]
        
        # Find end location after start (handling first_occurrence_only)
        if isinstance(end_value, (int, float)):
            end_mask = (df_remaining[end_column] == end_value) & \
                      (df_remaining.index > start_idx) & \
                      df_remaining[end_column].notna()
        else:
            end_mask = (df_remaining[end_column].astype(str) == str(end_value)) & \
                      (df_remaining.index > start_idx) & \
                      df_remaining[end_column].notna()
        
        end_candidates = df_remaining[end_mask]
        if end_candidates.empty:
            break
            
        if first_occurrence_only:
            # Use first occurrence of end value
            end_idx = end_candidates.index[0]
        else:
            # Find all occurrences until next start or end of dataframe
            next_start_mask = df_remaining[start_column] == start_value
            next_start_mask = next_start_mask & (df_remaining.index > start_idx)
            if next_start_mask.any():
                next_start_idx = df_remaining[next_start_mask].index[0]
                # Use last occurrence before next start
                valid_ends = end_candidates[end_candidates.index < next_start_idx]
                if not valid_ends.empty:
                    end_idx = valid_ends.index[-1]
                else:
                    end_idx = end_candidates.index[0]
            else:
                # No next start, use last occurrence
                end_idx = end_candidates.index[-1]
        
        # Check path length constraints
        path_length = end_idx - start_idx
        if path_length < min_path_length:
            df_remaining = df_remaining[df_remaining.index > end_idx]
            continue
        
        if max_path_length is not None and path_length > max_path_length:
            df_remaining = df_remaining[df_remaining.index > end_idx]
            continue
        
        # Check minimum unique values constraint
        if min_unique_values_in_column:
            valid_path = True
            for col, min_unique in min_unique_values_in_column.items():
                if col in dataframe.columns:
                    unique_count = count_unique_values_in_segment(dataframe, start_idx, end_idx, col)
                    if unique_count < min_unique:
                        valid_path = False
                        break
            
            if valid_path:
                transitions.append((start_idx, end_idx))
        else:
            transitions.append((start_idx, end_idx))
        
        # Continue after this end
        df_remaining = df_remaining[df_remaining.index > end_idx]
    
    return transitions