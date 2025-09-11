"""Session and cross-session metrics for NaviGraph analysis.

This module contains generic metric functions that work with any column names
and data structures. All metrics are registered via decorators for use by 
the Analyzer class.
"""

from typing import Any, List, Dict, Union, Optional
from ..core.registry import register_session_metric, register_cross_session_metric
from .utils import (
    find_path_transitions, 
    calculate_path_distance,
    count_unique_values_in_segment,
    count_value_transitions,
    get_path_sequence
)


@register_session_metric('time_a_to_b')
def time_a_to_b(
    session: Any,
    start_column: str,
    start_value: Union[str, int, tuple],
    end_column: str,
    end_value: Union[str, int, tuple],
    min_unique_values: Optional[Dict[str, int]] = None,
    **kwargs
) -> List[float]:
    """Calculate time to travel between any locations.
    
    This metric is completely generic - works with any column names and value types.
    
    Args:
        session: Session with integrated dataframe and shared resources
        start_column: Name of column containing start location
        start_value: Value to search for in start column
        end_column: Name of column containing end location
        end_value: Value to search for in end column
        min_unique_values: Optional dict of column_name: min_unique_count filters
        **kwargs: Additional arguments
        
    Returns:
        List of times in seconds for each transition
        
    Examples:
        # Graph-based locations
        time_a_to_b(session, 'centroid_graph_node', 5, 'centroid_graph_node', 8)
        
        # Any columns
        time_a_to_b(session, 'state', 'exploring', 'state', 'rewarded')
        
        # With filtering
        time_a_to_b(session, 'location', 'A', 'location', 'B',
                   min_unique_values={'visited_nodes': 3})
    """
    dataframe = session.get_integrated_dataframe()
    
    # Get fps from stream info if available
    stream_info = session.shared_resources.get('stream_info', {})
    fps = stream_info.get('fps', 30.0)  # Default to 30 fps if not available
    
    transitions = find_path_transitions(
        dataframe,
        start_column=start_column,
        start_value=start_value,
        end_column=end_column,
        end_value=end_value,
        min_unique_values_in_column=min_unique_values
    )
    
    times = [(end_idx - start_idx) / fps for start_idx, end_idx in transitions]
    return times


@register_session_metric('velocity_a_to_b')  
def velocity_a_to_b(
    session: Any,
    start_column: str,
    start_value: Union[str, int, tuple],
    end_column: str,
    end_value: Union[str, int, tuple],
    x_column: str,
    y_column: str,
    pixel_to_meter: Optional[float] = None,
    **kwargs
) -> List[float]:
    """Calculate velocity between any locations using any coordinate columns.
    
    Args:
        session: Session with dataframe and resources
        start_column: Name of column containing start location
        start_value: Value to search for in start column
        end_column: Name of column containing end location
        end_value: Value to search for in end column
        x_column: Name of column containing x coordinates
        y_column: Name of column containing y coordinates
        pixel_to_meter: Optional conversion factor from pixels to meters.
                       If provided, returns velocity in meters/second.
                       If None, returns velocity in pixels/second.
        **kwargs: Additional arguments
        
    Returns:
        List of velocities in meters/second (if pixel_to_meter provided) 
        or pixels/second (if not)
        
    Examples:
        # Using graph locations and centroid coordinates (pixels/sec)
        velocity_a_to_b(session, 'centroid_graph_node', 5, 'centroid_graph_node', 8,
                       'centroid_x', 'centroid_y')
        
        # With pixel to meter conversion (meters/sec)
        velocity_a_to_b(session, 'zone', 'start', 'zone', 'goal',
                       'position_x', 'position_y', pixel_to_meter=0.001)
    """
    dataframe = session.get_integrated_dataframe()
    
    # Get fps from stream info if available
    stream_info = session.shared_resources.get('stream_info', {})
    fps = stream_info.get('fps', 30.0)  # Default to 30 fps
    
    # Try to get pixel_to_meter from map metadata if not provided
    if pixel_to_meter is None:
        map_metadata = session.shared_resources.get('map_metadata', {})
        pixel_to_meter = map_metadata.get('pixel_to_meter')
    
    transitions = find_path_transitions(
        dataframe,
        start_column=start_column,
        start_value=start_value,
        end_column=end_column,
        end_value=end_value
    )
    
    velocities = []
    for start_idx, end_idx in transitions:
        time = (end_idx - start_idx) / fps
        if time > 0:
            distance = calculate_path_distance(
                dataframe, start_idx, end_idx, x_column, y_column, pixel_to_meter
            )
            velocities.append(distance / time)
        else:
            velocities.append(0.0)
    
    return velocities


@register_session_metric('exploration_percentage')
def exploration_percentage(
    session: Any,
    location_column: str,
    total_locations: Optional[int] = None,
    **kwargs
) -> float:
    """Calculate percentage of unique locations explored.
    
    Completely generic - works with any column representing locations.
    
    Args:
        session: Session with dataframe and resources
        location_column: Name of column containing location data
        total_locations: Total number of possible locations
                        If None, will try to get from graph resource
        **kwargs: Additional arguments
        
    Returns:
        Percentage of unique locations visited (0-100)
        
    Examples:
        # Graph nodes
        exploration_percentage(session, 'centroid_graph_node')
        
        # Any location column with known total
        exploration_percentage(session, 'room_id', total_locations=10)
        
        # Grid cells
        exploration_percentage(session, 'grid_cell', total_locations=100)
    """
    dataframe = session.get_integrated_dataframe()
    
    if location_column not in dataframe.columns:
        return 0.0  # Column doesn't exist, return 0
    
    # Count unique locations visited (excluding None/NaN)
    locations_visited = dataframe[location_column].dropna().unique()
    unique_locations = len([loc for loc in locations_visited if loc is not None])
    
    # Determine total locations
    if total_locations is None:
        # Try to get from graph if available
        graph = session.shared_resources.get('graph')
        if graph:
            if 'node' in location_column.lower():
                total_locations = graph.num_nodes
            elif 'edge' in location_column.lower():
                total_locations = graph.num_edges
            else:
                # Can't determine total, return count instead of percentage
                return float(unique_locations)
        else:
            # No graph available, return count
            return float(unique_locations)
    
    return (unique_locations / total_locations * 100) if total_locations > 0 else 0.0


@register_session_metric('num_values_in_path')
def num_values_in_path(
    session: Any,
    start_column: str,
    start_value: Union[str, int, tuple],
    end_column: str,
    end_value: Union[str, int, tuple],
    count_column: str,
    mode: str = 'unique',
    total_values: Optional[int] = None,
    min_unique_values: Optional[Dict[str, int]] = None,
    **kwargs
) -> List[Union[int, float]]:
    """Count values in paths between any locations.
    
    Generic function that counts values in any column along paths.
    
    Args:
        session: Session with dataframe
        start_column: Name of column containing start location
        start_value: Value to search for in start column
        end_column: Name of column containing end location
        end_value: Value to search for in end column
        count_column: Column to count values in
        mode: 'unique' (unique values), 'transitions' (eliminate sequences),
              or 'percentage' (exploration percentage)
        total_values: Total possible values (for percentage mode)
        min_unique_values: Filtering criteria
        **kwargs: Additional arguments
        
    Returns:
        List of counts or percentages
        
    Examples:
        # Count unique nodes visited
        num_values_in_path(session, 'state', 'start', 'state', 'goal',
                          'node_location', mode='unique')
        
        # Count transitions (eliminate sequences)
        num_values_in_path(session, 'zone', 'A', 'zone', 'B',
                          'current_node', mode='transitions')
        
        # Calculate exploration percentage
        num_values_in_path(session, 'trial_phase', 'begin', 'trial_phase', 'end',
                          'visited_cells', mode='percentage', total_values=100)
    """
    dataframe = session.get_integrated_dataframe()
    
    transitions = find_path_transitions(
        dataframe,
        start_column=start_column,
        start_value=start_value,
        end_column=end_column,
        end_value=end_value,
        min_unique_values_in_column=min_unique_values
    )
    
    results = []
    for start_idx, end_idx in transitions:
        if mode == 'unique':
            # Count unique values
            count = count_unique_values_in_segment(
                dataframe, start_idx, end_idx, count_column
            )
            results.append(count)
            
        elif mode == 'transitions':
            # Count transitions (eliminate sequences)
            count = count_value_transitions(
                dataframe, start_idx, end_idx, count_column
            )
            results.append(count)
            
        elif mode == 'percentage':
            # Calculate exploration percentage
            unique_count = count_unique_values_in_segment(
                dataframe, start_idx, end_idx, count_column
            )
            if total_values is None:
                # Try to infer from graph if available
                graph = session.shared_resources.get('graph')
                if graph and 'node' in count_column.lower():
                    total_values = graph.num_nodes
                elif graph and 'edge' in count_column.lower():
                    total_values = graph.num_edges
                else:
                    total_values = unique_count  # Fallback to 100%
            
            percentage = (unique_count / total_values * 100) if total_values > 0 else 0.0
            results.append(percentage)
    
    return results


@register_session_metric('shortest_path_efficiency')
def shortest_path_efficiency(
    session: Any,
    start_column: str,
    start_value: Union[str, int, tuple],
    end_column: str,
    end_value: Union[str, int, tuple],
    path_column: str,
    start_node: int,
    end_node: int,
    strike_levels: Optional[List[int]] = None,
    max_strikes: int = 0,
    **kwargs
) -> List[Dict[str, Any]]:
    """Calculate how efficiently paths follow the shortest route.
    
    Generic implementation that works with any columns and tracks
    progress along optimal paths.
    
    Args:
        session: Session with dataframe and graph
        start_column: Name of column containing start location
        start_value: Value to search for in start column
        end_column: Name of column containing end location
        end_value: Value to search for in end column
        path_column: Column containing values to track along path
        start_node: Node ID for shortest path calculation
        end_node: Node ID for shortest path calculation
        strike_levels: Optional levels where mistakes are allowed
        max_strikes: Maximum mistakes allowed per level
        **kwargs: Additional arguments
        
    Returns:
        List of dictionaries with path efficiency metrics
        
    Examples:
        # Track shortest path using node column
        shortest_path_efficiency(session, 'phase', 'start', 'phase', 'reward',
                                'current_node', start_node=0, end_node=15)
        
        # With strike allowance
        shortest_path_efficiency(session, 'state', 'exploring', 'state', 'found',
                                'node_id', 1, 10, strike_levels=[2, 3], max_strikes=1)
    """
    dataframe = session.get_integrated_dataframe()
    graph = session.shared_resources.get('graph')
    
    if not graph:
        # No graph available, return empty results
        return []
    
    transitions = find_path_transitions(
        dataframe,
        start_column=start_column,
        start_value=start_value,
        end_column=end_column,
        end_value=end_value
    )
    
    # Calculate optimal path once
    try:
        optimal_path = graph.get_shortest_path(start_node, end_node)
        optimal_length = len(optimal_path) - 1 if optimal_path else 0
    except:
        # Shortest path calculation failed
        return []
    
    results = []
    for start_idx, end_idx in transitions:
        # Get sequence of values along path
        path_sequence = get_path_sequence(
            dataframe, start_idx, end_idx, path_column, remove_duplicates=True
        )
        
        # Calculate efficiency metrics
        efficiency_metrics = _calculate_path_efficiency(
            path_sequence, optimal_path, strike_levels, max_strikes
        )
        
        efficiency_metrics['optimal_length'] = optimal_length
        efficiency_metrics['actual_length'] = len(path_sequence) - 1 if path_sequence else 0
        efficiency_metrics['efficiency'] = (
            (optimal_length / efficiency_metrics['actual_length'] * 100)
            if efficiency_metrics['actual_length'] > 0 else 0.0
        )
        
        results.append(efficiency_metrics)
    
    return results


def _calculate_path_efficiency(
    actual_path: List[Any],
    optimal_path: List[Any],
    strike_levels: Optional[List[int]] = None,
    max_strikes: int = 0
) -> Dict[str, Any]:
    """Helper to calculate path efficiency metrics.
    
    Returns:
        Dictionary with efficiency metrics:
        - progress: How far along optimal path
        - deviations: Number of deviations from optimal
        - strikes_used: Strikes used (if applicable)
        - completed: Whether path reached the goal
    """
    progress = 0
    deviations = 0
    strikes_used = 0
    current_strikes = 0
    
    for value in actual_path:
        if progress < len(optimal_path) - 1:
            # Check if on optimal path
            if value == optimal_path[progress + 1]:
                # On track
                progress += 1
                current_strikes = 0
            else:
                # Deviation
                deviations += 1
                
                # Check if strike is allowed
                if strike_levels and max_strikes > 0:
                    # Simple strike logic - can be customized
                    if current_strikes < max_strikes:
                        current_strikes += 1
                        strikes_used += 1
                    else:
                        # Reset progress - find where we are in optimal path
                        if value in optimal_path:
                            progress = optimal_path.index(value)
                        current_strikes = 0
    
    return {
        'progress': progress,
        'deviations': deviations,
        'strikes_used': strikes_used,
        'completed': progress == len(optimal_path) - 1
    }


@register_cross_session_metric('learning_progression')
def learning_progression(sessions: List[Dict], metric_name: str, **kwargs) -> List[float]:
    """Track metric progression across sessions.
    
    Args:
        sessions: List of session result dictionaries
        metric_name: Name of metric to track
        **kwargs: Additional arguments
        
    Returns:
        List of metric values across sessions
    """
    # Extract values from session results
    values = []
    for session_result in sessions:
        metrics = session_result.get('metrics', {})
        value = metrics.get(metric_name)
        if value is not None:
            values.append(value)
    
    return values