"""Session and cross-session metrics for NaviGraph analysis.

This module contains metric functions that are registered via decorators
and used by the Analyzer class for computation.
"""

from typing import Any, List, Dict
from ..core.registry import register_session_metric, register_cross_session_metric


@register_session_metric('time_a_to_b')
def time_a_to_b(session: Any, a: Any, b: Any, bodypart: str = 'centroid', **kwargs) -> float:
    """Compute time to travel from location A to location B.
    
    Args:
        session: Session object with get_integrated_dataframe() method
        a: Starting location (node or tile ID)
        b: Target location (node or tile ID)
        bodypart: Bodypart to track (default: centroid)
        **kwargs: Additional arguments
        
    Returns:
        Time in seconds (dummy value for now)
    """
    # Dummy implementation - return a fixed value for testing
    return 42.0


@register_session_metric('velocity_a_to_b')  
def velocity_a_to_b(session: Any, a: Any, b: Any, bodypart: str = 'centroid', **kwargs) -> float:
    """Compute velocity from location A to location B.
    
    Args:
        session: Session object
        a: Starting location
        b: Target location  
        bodypart: Bodypart to track
        **kwargs: Additional arguments
        
    Returns:
        Velocity in units/second (dummy value for now)
    """
    # Dummy implementation
    return 10.0


@register_session_metric('exploration_percentage')
def exploration_percentage(session: Any, bodypart: str = 'centroid', **kwargs) -> float:
    """Compute percentage of graph explored.
    
    Args:
        session: Session object
        bodypart: Bodypart to track
        **kwargs: Additional arguments
        
    Returns:
        Exploration percentage (dummy value for now)
    """
    # Dummy implementation
    return 75.0


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