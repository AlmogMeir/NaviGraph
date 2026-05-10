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


@register_session_metric('detect_graph_jumps')
def detect_graph_jumps(
    session: Any,
    bodypart: str = 'headstage',
    min_jump_distance: int = 2,
    **kwargs
) -> Dict[str, Any]:
    """
    Detect jumps in graph navigation where tracking skips to non-adjacent nodes.
    
    A jump is defined as a transition between nodes that are not directly connected
    by an edge (i.e., not parent-child or sibling relationships).
    
    Parameters
    ----------
    session : Session
        Session object with integrated dataframe and graph structure
    bodypart : str
        Name of the tracked bodypart (default: 'headstage')
    min_jump_distance : int
        Minimum graph distance to count as a jump (default: 2, meaning skip at least 1 node)
    
    Returns
    -------
    dict
        {
            'total_jumps': int,
            'jump_rate': float (jumps per minute),
            'jump_events': list of dicts with {
                'frame': int,
                'time_sec': float,
                'from_node': str,
                'to_node': str,
                'distance': int (shortest path length),
                'missing_nodes': list (nodes skipped in shortest path)
            },
            'jump_statistics': {
                'mean_distance': float,
                'max_distance': int,
                'most_common_from_node': str,
                'most_common_to_node': str
            }
        }
    """
    import pandas as pd
    import numpy as np
    from loguru import logger
    
    df = session.get_integrated_dataframe()
    graph = session.shared_resources.get('graph')
    
    if graph is None:
        logger.warning("No graph structure found in shared resources")
        return {
            'total_jumps': 0,
            'jump_rate': 0.0,
            'jump_events': [],
            'jump_statistics': {}
        }
    
    node_col = f'{bodypart}_graph_node'
    
    if node_col not in df.columns:
        logger.warning(f"Column {node_col} not found in dataframe")
        return {
            'total_jumps': 0,
            'jump_rate': 0.0,
            'jump_events': [],
            'jump_statistics': {}
        }
    
    # Get node transitions
    nodes = df[node_col].values
    frames = df.index.values  # Frame numbers are in the index
    
    # Calculate time assuming 40 fps (adjust if needed from stream_info if available)
    stream_info = session.shared_resources.get('stream_info', {})
    fps = stream_info.get('fps', 40.0)
    times = frames / fps
    
    jump_events = []
    from_nodes = []
    to_nodes = []
    distances = []
    
    for i in range(1, len(nodes)):
        prev_node = nodes[i-1]
        curr_node = nodes[i]
        
        # Skip if either node is None/NaN
        if pd.isna(prev_node) or pd.isna(curr_node):
            continue
            
        # Skip if no transition occurred
        if prev_node == curr_node:
            continue
        
        # Check if nodes are directly connected
        if graph.has_edge(prev_node, curr_node):
            continue  # This is a normal transition
        
        # This is a jump! Calculate shortest path
        try:
            import networkx as nx
            shortest_path = nx.shortest_path(graph.graph, prev_node, curr_node)
            distance = len(shortest_path) - 1
            
            # Only count if distance exceeds minimum
            if distance < min_jump_distance:
                continue
            
            missing_nodes = shortest_path[1:-1]  # Nodes between start and end
            
            jump_events.append({
                'frame': int(frames[i]),
                'time_sec': float(times[i]),
                'from_node': str(prev_node),
                'to_node': str(curr_node),
                'distance': distance,
                'missing_nodes': [str(n) for n in missing_nodes]
            })
            
            from_nodes.append(prev_node)
            to_nodes.append(curr_node)
            distances.append(distance)
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Nodes are in disconnected components
            logger.debug(f"No path between {prev_node} and {curr_node}")
            jump_events.append({
                'frame': int(frames[i]),
                'time_sec': float(times[i]),
                'from_node': str(prev_node),
                'to_node': str(curr_node),
                'distance': float('inf'),
                'missing_nodes': []
            })
            from_nodes.append(prev_node)
            to_nodes.append(curr_node)
    
    # Calculate statistics
    total_jumps = len(jump_events)
    session_duration_min = (times[-1] - times[0]) / 60 if len(times) > 0 else 0
    jump_rate = total_jumps / session_duration_min if session_duration_min > 0 else 0
    
    statistics = {}
    if distances:
        finite_distances = [d for d in distances if d != float('inf')]
        if finite_distances:
            statistics = {
                'mean_distance': float(np.mean(finite_distances)),
                'max_distance': int(max(finite_distances)),
                'most_common_from_node': str(max(set(from_nodes), key=from_nodes.count)) if from_nodes else None,
                'most_common_to_node': str(max(set(to_nodes), key=to_nodes.count)) if to_nodes else None
            }
    
    return {
        'total_jumps': total_jumps,
        'jump_rate': jump_rate,
        'jump_events': jump_events,
        'jump_statistics': statistics
    }


@register_session_metric('node_transition_flow')
def node_transition_flow(
    session: Any,
    bodypart: str = 'headstage',
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze all node-to-node transitions in the session to understand movement patterns.
    
    This metric provides a complete view of how the animal moves through the graph,
    including transition counts, sequences, and validation of graph navigation.
    
    Parameters
    ----------
    session : Session
        Session object with integrated dataframe and graph structure
    bodypart : str
        Name of the tracked bodypart (default: 'headstage')
    
    Returns
    -------
    dict
        {
            'total_transitions': int,
            'unique_nodes_visited': int,
            'transition_matrix': dict of {(from_node, to_node): count},
            'most_common_transitions': list of tuples [(from, to, count), ...],
            'node_visit_counts': dict of {node: visit_count},
            'graph_edges_used': list of edges that were traversed,
            'non_edge_transitions': list of transitions not in graph (jumps),
            'edge_coverage': float (percentage of graph edges used),
            'transition_sequences': list of node sequences (first 100 transitions)
        }
    """
    import pandas as pd
    import numpy as np
    from collections import Counter, defaultdict
    from loguru import logger
    
    df = session.get_integrated_dataframe()
    graph = session.shared_resources.get('graph')
    
    if graph is None:
        logger.warning("No graph structure found in shared resources")
        return {
            'total_transitions': 0,
            'unique_nodes_visited': 0,
            'transition_matrix': {},
            'most_common_transitions': [],
            'node_visit_counts': {},
            'graph_edges_used': [],
            'non_edge_transitions': [],
            'edge_coverage': 0.0,
            'transition_sequences': []
        }
    
    node_col = f'{bodypart}_graph_node'
    
    if node_col not in df.columns:
        logger.warning(f"Column {node_col} not found in dataframe")
        return {
            'total_transitions': 0,
            'unique_nodes_visited': 0,
            'transition_matrix': {},
            'most_common_transitions': [],
            'node_visit_counts': {},
            'graph_edges_used': [],
            'non_edge_transitions': [],
            'edge_coverage': 0.0,
            'transition_sequences': []
        }
    
    # Get node data
    nodes = df[node_col].values
    frames = df.index.values  # Frame numbers are in the index
    
    # Count node visits
    valid_nodes = [n for n in nodes if not pd.isna(n)]
    node_visit_counts = dict(Counter(valid_nodes))
    unique_nodes = len(node_visit_counts)
    
    # Track transitions
    transition_counts = defaultdict(int)
    graph_edges_used = set()
    non_edge_transitions = []
    transition_sequence = []
    
    for i in range(1, len(nodes)):
        prev_node = nodes[i-1]
        curr_node = nodes[i]
        
        # Skip if either node is None/NaN
        if pd.isna(prev_node) or pd.isna(curr_node):
            continue
            
        # Skip if no transition occurred
        if prev_node == curr_node:
            continue
        
        # Record transition
        transition = (str(prev_node), str(curr_node))
        transition_counts[transition] += 1
        
        # Record sequence (limit to first 100 for memory)
        if len(transition_sequence) < 100:
            if not transition_sequence or transition_sequence[-1] != str(curr_node):
                transition_sequence.append(str(curr_node))
        
        # Check if this is a graph edge
        if graph.has_edge(prev_node, curr_node):
            graph_edges_used.add((str(prev_node), str(curr_node)))
        else:
            non_edge_transitions.append({
                'frame': int(frames[i]),
                'from_node': str(prev_node),
                'to_node': str(curr_node)
            })
    
    # Calculate edge coverage
    total_graph_edges = len(list(graph.graph.edges()))
    edge_coverage = (len(graph_edges_used) / total_graph_edges * 100) if total_graph_edges > 0 else 0.0
    
    # Sort transitions by count
    most_common = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    most_common_formatted = [(from_n, to_n, count) for (from_n, to_n), count in most_common]
    
    return {
        'total_transitions': sum(transition_counts.values()),
        'unique_nodes_visited': unique_nodes,
        'transition_matrix': dict(transition_counts),
        'most_common_transitions': most_common_formatted,
        'node_visit_counts': node_visit_counts,
        'graph_edges_used': [list(e) for e in graph_edges_used],
        'non_edge_transitions': non_edge_transitions[:100],  # Limit to first 100
        'edge_coverage': round(edge_coverage, 2),
        'transition_sequences': transition_sequence
    }


@register_session_metric('reconstruct_full_path')
def reconstruct_full_path(
    session: 'Session',
    bodypart: str = 'headstage',
    min_jump_distance: int = 2
) -> Dict[str, Any]:
    """
    Reconstruct the complete traversal path through the graph as a sequence of 
    node→edge→node transitions, independent of frame duration.
    
    This metric provides:
    1. Sequential path of unique graph positions (nodes/edges) visited
    2. Identifies jumps and infers the expected path taken during jumps
    3. Groups continuous segments between jumps
    4. Provides frame ranges and durations for each position
    
    Args:
        session: Session object containing tracking data
        bodypart: Which bodypart to analyze
        min_jump_distance: Minimum graph distance to consider a jump
        
    Returns:
        Dictionary containing:
        - traversal_sequence: List of dicts (node/edge, frames, duration, type)
        - path_segments: Continuous segments between jumps
        - jump_reconstructions: Inferred paths at jump locations
        - statistics: Summary statistics
    """
    import pandas as pd
    import networkx as nx
    from loguru import logger
    
    logger.info(f"Reconstructing full traversal path for bodypart: {bodypart}")
    
    # Get integrated dataframe and graph
    df = session.get_integrated_dataframe()
    graph = session.shared_resources.get('graph')
    
    if graph is None:
        logger.warning("No graph available for path reconstruction")
        return {
            'traversal_sequence': [],
            'path_segments': [],
            'jump_reconstructions': [],
            'statistics': {'error': 'No graph available'}
        }
    
    # Get node and edge columns
    node_col = f'{bodypart}_graph_node'
    edge_col = f'{bodypart}_graph_edge'
    
    if node_col not in df.columns:
        logger.warning(f"Column {node_col} not found in dataframe")
        logger.info(f"Available columns: {df.columns.tolist()}")
        return {
            'traversal_sequence': [],
            'path_segments': [],
            'jump_reconstructions': [],
            'statistics': {'error': f'Column {node_col} not found'}
        }
    
    # Check if edge column exists
    has_edge_col = edge_col in df.columns
    logger.info(f"Node column '{node_col}': found")
    logger.info(f"Edge column '{edge_col}': {'found' if has_edge_col else 'not found (will use nodes only)'}")
    
    # Initialize results
    traversal_sequence = []  # Node→edge→node sequence
    path_segments = []  # Continuous segments between jumps
    jump_reconstructions = []  # Details about inferred paths
    
    # Find likelihood column for trimming low-quality tracking at start/end
    likelihood_col = f'{bodypart}_likelihood'
    has_likelihood = likelihood_col in df.columns
    
    # Determine valid frame range based on likelihood
    start_frame_idx = 0
    end_frame_idx = len(df) - 1
    frames_skipped_start = 0
    frames_skipped_end = 0
    
    if has_likelihood:
        logger.info(f"Trimming start/end based on likelihood column: {likelihood_col}")
        
        # Find start: first frame where likelihood is consistently > 0.5
        likelihood_threshold = 0.6
        consistency_window = 20  # Require N consecutive frames above threshold
        
        for i in range(len(df) - consistency_window):
            window = df[likelihood_col].iloc[i:i+consistency_window]
            if (window > likelihood_threshold).all():
                start_frame_idx = i
                frames_skipped_start = i
                logger.info(f"  Start frame: {df.index[start_frame_idx]} (skipped {frames_skipped_start} frames)")
                break
        
        # Find end: last frame where likelihood is consistently > 0.5
        for i in range(len(df) - 1, consistency_window - 1, -1):
            window = df[likelihood_col].iloc[i-consistency_window+1:i+1]
            if (window > likelihood_threshold).all():
                end_frame_idx = i
                frames_skipped_end = len(df) - 1 - i
                logger.info(f"  End frame: {df.index[end_frame_idx]} (skipped {frames_skipped_end} frames)")
                break
    else:
        logger.info(f"No likelihood column found, using full frame range")
    
    # Trim dataframe to valid range
    df_trimmed = df.iloc[start_frame_idx:end_frame_idx+1]
    logger.info(f"Using frames {df_trimmed.index[0]} to {df_trimmed.index[-1]} ({len(df_trimmed)} frames)")
    
    # PASS 1: Extract raw tracked positions from dataframe
    raw_positions = []  # List of (type, position, start_frame, end_frame, duration)
    current_position = None
    position_start_frame = None
    
    for frame, row in df_trimmed.iterrows():
        current_node = row.get(node_col)
        current_edge = row.get(edge_col) if has_edge_col else None
        
        # Skip frames with no data
        if (pd.isna(current_node) or current_node is None) and (pd.isna(current_edge) or current_edge is None):
            continue
        
        # Determine position (edge takes priority)
        if current_edge is not None and not pd.isna(current_edge):
            new_position = ('edge', str(current_edge))
        elif current_node is not None and not pd.isna(current_node):
            new_position = ('node', str(current_node))
        else:
            continue
        
        # Track position changes
        if new_position != current_position:
            # Save previous position
            if current_position is not None and position_start_frame is not None:
                raw_positions.append({
                    'type': current_position[0],
                    'position': current_position[1],
                    'start_frame': position_start_frame,
                    'end_frame': frame - 1,
                    'duration': frame - position_start_frame
                })
            
            current_position = new_position
            position_start_frame = frame
    
    # Add final position
    if current_position is not None and position_start_frame is not None:
        raw_positions.append({
            'type': current_position[0],
            'position': current_position[1],
            'start_frame': position_start_frame,
            'end_frame': df_trimmed.index[-1],
            'duration': df_trimmed.index[-1] - position_start_frame + 1
        })
    
    logger.info(f"Pass 1: Extracted {len(raw_positions)} raw positions")
    
    # PASS 2: Fill in missing transitions and enforce alternation
    current_location = None  # Track current node location
    jump_count = 0
    
    for i, pos in enumerate(raw_positions):
        pos_type = pos['type']
        pos_value = pos['position']
        
        # Parse position to get node(s)
        if pos_type == 'node':
            pos_nodes = [pos_value]
        else:  # edge
            try:
                if pos_value.startswith("('") or pos_value.startswith('("'):
                    edge_tuple = eval(pos_value)
                    pos_nodes = [str(edge_tuple[0]), str(edge_tuple[1])]
                else:
                    pos_nodes = pos_value.split('_')
            except:
                logger.warning(f"Could not parse edge: {pos_value}")
                pos_nodes = []
        
        # Determine where we are now (for first position or after edge)
        if current_location is None:
            if pos_type == 'node':
                current_location = pos_value
            else:  # edge - pick first node
                current_location = pos_nodes[0] if pos_nodes else None
        
        # Determine where we need to go
        if pos_type == 'node':
            target_location = pos_value
        else:  # edge
            # Determine which node we're heading toward (which end of the edge we need to reach)
            if len(pos_nodes) == 2:
                if current_location == pos_nodes[0]:
                    target_location = pos_nodes[0]  # Already at start of edge
                elif current_location == pos_nodes[1]:
                    target_location = pos_nodes[1]  # Already at end of edge
                else:
                    # Not connected to either end - find closest endpoint to reach
                    try:
                        dist0 = nx.shortest_path_length(graph.graph, current_location, pos_nodes[0])
                        dist1 = nx.shortest_path_length(graph.graph, current_location, pos_nodes[1])
                        target_location = pos_nodes[0] if dist0 <= dist1 else pos_nodes[1]
                    except:
                        target_location = pos_nodes[0]
            else:
                target_location = current_location
        
        # Check if we need to add connecting node before this position
        # (e.g., after a tracked edge, before another edge)
        last_was_edge = len(traversal_sequence) > 0 and traversal_sequence[-1]['type'] == 'edge'
        
        if last_was_edge and current_location:
            # Add the connecting node between edges
            traversal_sequence.append({
                'type': 'node',
                'position': current_location,
                'start_frame': pos['start_frame'],
                'end_frame': pos['start_frame'],
                'duration': 0,
                'transition_type': 'inferred',
                'jump_id': None
            })
        
        # Check if we need to fill in path
        if current_location and target_location and current_location != target_location:
            # Check for direct connection or need to infer path
            has_direct_edge = graph.has_edge(current_location, target_location)
            need_path = not has_direct_edge
            
            # Special case: lowest level nodes (L4X-L5X, R4X-R5X) can be directly connected
            is_lowest_level_jump = False
            if current_location.startswith(('L4', 'L5', 'R4', 'R5')) and target_location.startswith(('L4', 'L5', 'R4', 'R5')):
                same_side = (current_location[0] == target_location[0])  # Both L or both R
                if same_side:
                    is_lowest_level_jump = True
                    need_path = False  # Allow direct node-to-node
            
            if need_path:
                try:
                    shortest_path = nx.shortest_path(graph.graph, current_location, target_location)
                    
                    # Add intermediate positions
                    for j in range(len(shortest_path) - 1):
                        from_n = shortest_path[j]
                        to_n = shortest_path[j + 1]
                        
                        # Add first node if not already present
                        if j == 0:
                            if not (len(traversal_sequence) > 0 and 
                                   traversal_sequence[-1]['type'] == 'node' and 
                                   traversal_sequence[-1]['position'] == from_n):
                                traversal_sequence.append({
                                    'type': 'node',
                                    'position': from_n,
                                    'start_frame': pos['start_frame'],
                                    'end_frame': pos['start_frame'],
                                    'duration': 0,
                                    'transition_type': 'inferred',
                                    'jump_id': jump_count
                                })
                        
                        # Add edge
                        traversal_sequence.append({
                            'type': 'edge',
                            'position': f"({from_n}, {to_n})",
                            'start_frame': pos['start_frame'],
                            'end_frame': pos['start_frame'],
                            'duration': 0,
                            'transition_type': 'inferred',
                            'jump_id': jump_count
                        })
                        
                        # Add node after edge
                        # Always add if this is the last edge in path, or if not the last edge
                        traversal_sequence.append({
                            'type': 'node',
                            'position': to_n,
                            'start_frame': pos['start_frame'],
                            'end_frame': pos['start_frame'],
                            'duration': 0,
                            'transition_type': 'inferred',
                            'jump_id': jump_count
                        })
                    
                    jump_count += 1
                    jump_reconstructions.append({
                        'jump_id': jump_count - 1,
                        'from_location': current_location,
                        'to_location': target_location,
                        'shortest_path': shortest_path
                    })
                    
                    # Update current location to target after inferring path
                    current_location = target_location
                    
                except nx.NetworkXNoPath:
                    logger.warning(f"No path from {current_location} to {target_location}")
            
            elif has_direct_edge and pos_type == 'edge':
                # Special case: Direct edge exists but next position is an EDGE (not a node)
                # We need to add the intermediate node before the tracked edge
                # e.g., L521 → L410 (missing) → edge('L35', 'L410')
                if not (len(traversal_sequence) > 0 and 
                       traversal_sequence[-1]['type'] == 'node' and 
                       traversal_sequence[-1]['position'] == target_location):
                    # Add the inferred edge from current to target
                    traversal_sequence.append({
                        'type': 'edge',
                        'position': f"({current_location}, {target_location})",
                        'start_frame': pos['start_frame'],
                        'end_frame': pos['start_frame'],
                        'duration': 0,
                        'transition_type': 'inferred',
                        'jump_id': None
                    })
                    
                    # Add the target node
                    traversal_sequence.append({
                        'type': 'node',
                        'position': target_location,
                        'start_frame': pos['start_frame'],
                        'end_frame': pos['start_frame'],
                        'duration': 0,
                        'transition_type': 'inferred',
                        'jump_id': None
                    })
                    
                    # Update current location
                    current_location = target_location
        
        # Add current tracked position
        traversal_sequence.append({
            'type': pos_type,
            'position': pos_value,
            'start_frame': pos['start_frame'],
            'end_frame': pos['end_frame'],
            'duration': pos['duration'],
            'transition_type': 'tracked'
        })
        
        # Update current location
        if pos_type == 'node':
            current_location = pos_value
        else:  # edge
            # After traversing edge, we're at the other end
            if len(pos_nodes) == 2:
                current_location = pos_nodes[1] if current_location == pos_nodes[0] else pos_nodes[0]
    
    # PASS 3: Detect and remove tracking outliers (brief jumps to unrealistic locations)
    logger.info("Pass 3: Detecting and removing tracking outliers...")
    outliers_removed = 0
    i = 0
    
    def get_node_from_position(pos_item):
        """Extract node from position (works for nodes and edges)."""
        if pos_item['type'] == 'node':
            return pos_item['position']
        else:  # edge
            try:
                edge_tuple = eval(pos_item['position'])
                return edge_tuple[0]  # Return first node of edge
            except:
                return None
    
    def calculate_graph_distance(node1, node2, graph):
        """Calculate shortest path distance between two nodes."""
        try:
            return nx.shortest_path_length(graph.graph, node1, node2)
        except:
            return float('inf')
    
    while i < len(traversal_sequence) - 2:
        current = traversal_sequence[i]
        middle = traversal_sequence[i + 1]
        after = traversal_sequence[i + 2]
        
        # Check for pattern: tracked → brief tracked (far away) → tracked
        if (current['transition_type'] == 'tracked' and
            middle['transition_type'] == 'tracked' and
            after['transition_type'] == 'tracked'):
            
            # Get nodes
            node_current = get_node_from_position(current)
            node_middle = get_node_from_position(middle)
            node_after = get_node_from_position(after)
            
            if node_current and node_middle and node_after:
                # Calculate distances
                dist_current_to_middle = calculate_graph_distance(node_current, node_middle, graph)
                dist_middle_to_after = calculate_graph_distance(node_middle, node_after, graph)
                dist_current_to_after = calculate_graph_distance(node_current, node_after, graph)
                
                # Check if middle is an outlier:
                # 1. Middle position is brief (≤ 5 frames)
                # 2. Jump to middle is large (≥ 3 nodes)
                # 3. Current and after are close (≤ 2 nodes apart)
                # 4. Direct path from current to after is shorter than going through middle
                is_brief = middle['duration'] <= 5
                is_far_jump = dist_current_to_middle >= 3
                are_close = dist_current_to_after <= 2
                is_detour = (dist_current_to_middle + dist_middle_to_after) > (dist_current_to_after + 2)
                
                if is_brief and is_far_jump and are_close and is_detour:
                    logger.debug(f"  Outlier detected: {node_current} -> {node_middle} ({middle['duration']}f) -> {node_after}")
                    logger.debug(f"    Distances: {dist_current_to_middle}, {dist_middle_to_after}, direct={dist_current_to_after}")
                    
                    # Remove the middle outlier and any inferred positions around it
                    # First, find all positions in the outlier sequence
                    outlier_start = i + 1
                    outlier_end = i + 1
                    
                    # Extend to include any inferred positions immediately after middle
                    while (outlier_end + 1 < len(traversal_sequence) and
                           traversal_sequence[outlier_end + 1]['transition_type'] == 'inferred' and
                           outlier_end - outlier_start < 10):  # Safety limit
                        outlier_end += 1
                    
                    # Extend to include any inferred positions immediately before middle
                    while (outlier_start - 1 > i and
                           traversal_sequence[outlier_start - 1]['transition_type'] == 'inferred'):
                        outlier_start -= 1
                    
                    # Extend current position to cover the outlier's frames
                    current['end_frame'] = traversal_sequence[outlier_end]['end_frame']
                    current['duration'] = current['end_frame'] - current['start_frame'] + 1
                    
                    # Adjust after position's start frame
                    after['start_frame'] = current['end_frame'] + 1
                    after['duration'] = after['end_frame'] - after['start_frame'] + 1
                    
                    # Remove all positions from outlier_start to outlier_end
                    num_removed = outlier_end - outlier_start + 1
                    for _ in range(num_removed):
                        traversal_sequence.pop(outlier_start)
                    
                    outliers_removed += 1
                    # Don't increment i, check the same position again
                    continue
        
        i += 1
    
    logger.info(f"Pass 3: Removed {outliers_removed} tracking outliers")
    
    # PASS 4: Fix unnecessary inferences (tracked edge → short inferred node → tracked node)
    logger.info("Pass 4: Fixing unnecessary inferences...")
    fixes_applied = 0
    i = 0
    
    while i < len(traversal_sequence) - 2:
        current = traversal_sequence[i]
        next_item = traversal_sequence[i + 1]
        after_next = traversal_sequence[i + 2]
        
        # Check for pattern: tracked edge → inferred node → tracked node (same position)
        if (current['type'] == 'edge' and 
            current['transition_type'] == 'tracked' and
            next_item['type'] == 'node' and 
            next_item['transition_type'] == 'inferred' and
            after_next['type'] == 'node' and
            after_next['transition_type'] == 'tracked' and
            next_item['position'] == after_next['position']):
            
            # Extract edge destination
            try:
                edge_tuple = eval(current['position'])
                edge_destination = edge_tuple[1]
                
                # Check if inferred node matches edge destination
                if next_item['position'] == edge_destination and next_item['duration'] <= 2:
                    logger.debug(f"  Fix: Extending edge {current['position']} to cover inferred node")
                    
                    # Extend the edge to include the inferred frame(s)
                    current['end_frame'] = next_item['end_frame']
                    current['duration'] = current['end_frame'] - current['start_frame'] + 1
                    
                    # Update the tracked node start frame
                    after_next['start_frame'] = next_item['end_frame'] + 1
                    after_next['duration'] = after_next['end_frame'] - after_next['start_frame'] + 1
                    
                    # Remove the inferred node
                    traversal_sequence.pop(i + 1)
                    
                    fixes_applied += 1
                    # Don't increment i, check the same position again
                    continue
            except:
                pass  # Edge parsing failed, skip
        
        i += 1
    
    logger.info(f"Pass 4: Applied {fixes_applied} fixes for unnecessary inferences")
    
    # PASS 5: Adjust durations for remaining inferred nodes
    for i in range(len(traversal_sequence)):
        if traversal_sequence[i]['transition_type'] == 'inferred' and traversal_sequence[i]['type'] == 'node':
            # Try to steal frames from adjacent positions
            can_steal_before = i > 0 and traversal_sequence[i-1]['duration'] > 1
            can_steal_after = i < len(traversal_sequence) - 1 and traversal_sequence[i+1]['duration'] > 1
            
            if can_steal_before and can_steal_after:
                # Steal 1 frame from both
                traversal_sequence[i-1]['end_frame'] -= 1
                traversal_sequence[i-1]['duration'] -= 1
                traversal_sequence[i]['start_frame'] = traversal_sequence[i-1]['end_frame'] + 1
                traversal_sequence[i+1]['start_frame'] += 1
                traversal_sequence[i]['end_frame'] = traversal_sequence[i+1]['start_frame'] - 1
                traversal_sequence[i+1]['duration'] -= 1
                traversal_sequence[i]['duration'] = 1
            elif can_steal_before:
                # Steal from before
                traversal_sequence[i-1]['end_frame'] -= 1
                traversal_sequence[i-1]['duration'] -= 1
                traversal_sequence[i]['start_frame'] = traversal_sequence[i-1]['end_frame'] + 1
                traversal_sequence[i]['end_frame'] = traversal_sequence[i]['start_frame']
                traversal_sequence[i]['duration'] = 1
            elif can_steal_after:
                # Steal from after
                traversal_sequence[i+1]['start_frame'] += 1
                traversal_sequence[i+1]['duration'] -= 1
                traversal_sequence[i]['end_frame'] = traversal_sequence[i+1]['start_frame'] - 1
                traversal_sequence[i]['start_frame'] = traversal_sequence[i]['end_frame']
                traversal_sequence[i]['duration'] = 1
    
    # Calculate statistics
    tracked_positions = sum(1 for p in traversal_sequence if p['transition_type'] == 'tracked')
    inferred_positions = sum(1 for p in traversal_sequence if p['transition_type'] == 'inferred')
    total_unique_positions = len(traversal_sequence)
    
    statistics = {
        'total_unique_positions': total_unique_positions,
        'tracked_positions': tracked_positions,
        'inferred_positions': inferred_positions,
        'jump_count': jump_count,
        'tracking_outliers_removed': outliers_removed,
        'unnecessary_inferences_fixed': fixes_applied,
        'frames_skipped_start': frames_skipped_start,
        'frames_skipped_end': frames_skipped_end,
        'total_frames_analyzed': len(df_trimmed),
        'inferred_ratio': inferred_positions / total_unique_positions if total_unique_positions > 0 else 0
    }
    
    logger.info(f"Path reconstruction complete:")
    logger.info(f"  Frames analyzed: {len(df_trimmed)} (skipped {frames_skipped_start} at start, {frames_skipped_end} at end)")
    logger.info(f"  Total positions: {total_unique_positions}")
    logger.info(f"  Tracked: {tracked_positions}, Inferred: {inferred_positions}")
    logger.info(f"  Jumps detected: {jump_count}")
    logger.info(f"  Tracking outliers removed: {outliers_removed}")
    logger.info(f"  Unnecessary inferences fixed: {fixes_applied}")
    
    return {
        'traversal_sequence': traversal_sequence,
        'jump_reconstructions': jump_reconstructions,
        'statistics': statistics
    }


@register_session_metric('traversal_sequence')
def traversal_sequence(
    session: 'Session',
    bodypart: str = 'headstage',
    min_jump_distance: int = 2
) -> Dict[str, Any]:
    """
    Simplified alias for reconstruct_full_path - provides the same functionality.
    
    Reconstructs complete graph traversal as node→edge→node sequence.
    
    Args:
        session: Session object containing tracking data
        bodypart: Which bodypart to analyze
        min_jump_distance: Minimum graph distance to consider a transition a jump
        
    Returns:
        Same as reconstruct_full_path
    """
    return reconstruct_full_path(session, bodypart, min_jump_distance)

