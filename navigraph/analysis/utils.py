"""Utility functions for analyzer plugins.

This module contains helper functions used by analyzer plugins for data processing
and analysis. These functions are migrated from the original analyzer/utils.py
to make the plugin system self-contained.
"""

from dataclasses import dataclass
from typing import Union, Any, Callable
from numpy.typing import NDArray
import pandas as pd
import numpy as np

# Constants for condition dictionary keys
CONDITIONS_DICT_COL_NAME_KEY = 'column_name'
CONDITIONS_DICT_FUNC_KEY = 'func'
CONDITIONS_DICT_THRESHOLD_KEY = 'threshold'
CONDITIONS_DICT_OPERATOR_KEY = 'operator'


@dataclass
class Condition:
    """Data class for defining analysis conditions.
    
    Used to specify filtering conditions for trajectory analysis,
    such as minimum path lengths or other constraints.
    """
    column_name: str
    func: Callable
    threshold: Union[str, float, int]
    operator: Callable


def count_node_visits_eliminating_sequences(ser: Union[pd.Series, NDArray]):
    """Count node visits while eliminating consecutive duplicates.
    
    This function counts how many times each node is visited, but consecutive
    visits to the same node are counted as a single visit.
    
    Args:
        ser: Series containing tree positions (nodes/edges)
        
    Returns:
        Total number of non-consecutive node visits
    """
    def int_nodes_only(x):
        if isinstance(x, frozenset):
            # Extract integer node from frozenset
            nodes = [v for v in x if isinstance(v, int)]
            return nodes[0] if nodes else None
        elif isinstance(x, int):
            return x
        else:
            return None

    # Extract only integer nodes
    ser = ser.apply(lambda x: int_nodes_only(x)).dropna()
    
    # Create shifted series to detect sequences
    ser_shifted = ser.shift(1)
    df_temp = pd.DataFrame(ser)
    df_temp['seq'] = ser != ser_shifted
    
    # Group by node and sum sequence flags
    visits_per_node = df_temp.groupby(ser.name).sum()
    
    return np.sum(visits_per_node.values)


def count_type_specific_objects(ser: Union[pd.Series, NDArray], dtype: Any):
    """Count objects of specific type(s) in a series.
    
    Args:
        ser: Series to count objects in
        dtype: Type or tuple of types to count
        
    Returns:
        Number of objects matching the specified type(s)
    """
    return np.sum([True if isinstance(obj, dtype) else False for obj in ser])


def count_unique_type_specific_objects(ser: pd.Series, dtype: Any):
    """Count unique objects of specific type(s) in a series.
    
    Args:
        ser: Series to count unique objects in
        dtype: Type or tuple of types to count
        
    Returns:
        Number of unique objects matching the specified type(s)
    """
    ser_unique = ser.unique()
    return count_type_specific_objects(ser_unique, dtype)


def a_to_b(df: pd.DataFrame,
           column_name: str,
           val_a: Union[int, float, str],
           val_b: Union[int, float, str],
           condition: Condition):
    """Find all trajectories from value A to value B in a DataFrame.
    
    This function identifies all instances where the subject moves from
    location/state A to location/state B, optionally filtered by conditions
    such as minimum path length.
    
    Args:
        df: DataFrame containing trajectory data
        column_name: Column to search for values (e.g., 'tile_id')
        val_a: Starting value (e.g., starting tile)
        val_b: Ending value (e.g., goal tile)
        condition: Optional condition to filter trajectories
        
    Returns:
        List of [start_index, end_index] pairs for each A→B trajectory
        
    Note:
        Currently only supports A before B order (not B before A)
    """
    # TODO: support both a before b and b before a
    a_b_pairs = []
    
    while len(df) > 0 and val_a in list(df[column_name]) and val_b in list(df[column_name]):
        # Find first occurrences of A and B
        first_a = df[column_name].eq(val_a).idxmax()
        first_b = df[column_name].eq(val_b).idxmax()
        
        # Only consider cases where A comes before B
        if first_a < first_b:
            if condition is None:
                # No condition - accept all A→B trajectories
                a_b_pairs.append([first_a, first_b])
            else:
                # Apply condition to filter trajectories
                value_to_threshold = condition.func(df.loc[first_a:first_b][condition.column_name])
                
                if condition.operator(value_to_threshold, condition.threshold):
                    a_b_pairs.append([first_a, first_b])
        
        # Continue searching after current B
        df = df[df.index > first_b]
    
    return a_b_pairs


def find_path_transitions(
    dataframe,
    bodypart: str,
    start_location_type: str,  # 'node' or 'edge'
    start_location_id: str,    # Always string (nodes: "5", edges: "(5,8)")
    end_location_type: str,    # 'node' or 'edge' 
    end_location_id: str,      # Always string (nodes: "8", edges: "(8,12)")
    min_path_length: int = 0
):
    """Find transitions between any graph locations (nodes/edges).
    
    Args:
        bodypart: Bodypart name (e.g., 'centroid')
        start_location_type: 'node' or 'edge'
        start_location_id: Starting location ID as string
        end_location_type: 'node' or 'edge'
        end_location_id: Target location ID as string
        min_path_length: Minimum frames between start and end
        
    Examples:
        # Node 5 to node 8
        find_path_transitions(df, 'centroid', 'node', '5', 'node', '8')
        
        # Node 5 to edge (5,8)  
        find_path_transitions(df, 'centroid', 'node', '5', 'edge', '(5,8)')
        
        # Edge (5,8) to node 12
        find_path_transitions(df, 'centroid', 'edge', '(5,8)', 'node', '12')
        
    Returns:
        List of (start_index, end_index) tuples for each transition
    """
    # Construct column names
    start_column = f"{bodypart}_graph_{start_location_type}"
    end_column = f"{bodypart}_graph_{end_location_type}"
    
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
        # Find start location (string comparison)
        start_mask = df_remaining[start_column].astype(str) == start_location_id
        if not start_mask.any():
            break
        start_idx = df_remaining[start_mask].index[0]
        
        # Find end location after start (string comparison)
        end_candidates = df_remaining[
            (df_remaining[end_column].astype(str) == end_location_id) & 
            (df_remaining.index > start_idx)
        ]
        if end_candidates.empty:
            break
        end_idx = end_candidates.index[0]
        
        # Check minimum path length
        if end_idx - start_idx >= min_path_length:
            transitions.append((start_idx, end_idx))
        
        # Continue after this end
        df_remaining = df_remaining[df_remaining.index > end_idx]
    
    return transitions