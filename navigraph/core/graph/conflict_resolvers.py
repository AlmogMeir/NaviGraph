"""Conflict resolution strategies for spatial mapping.

This module provides strategies for resolving conflicts when a point
falls within multiple spatial regions (e.g., both node and edge regions).
"""

from typing import List, Tuple, Optional, Any, Callable
import numpy as np


class ConflictResolvers:
    """Registry of conflict resolution strategies."""
    _strategies = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy function."""
        def decorator(func):
            cls._strategies[name] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name_or_callable):
        """Get strategy by name or return callable as-is."""
        if callable(name_or_callable):
            # Advanced users can pass custom functions
            return name_or_callable
        elif isinstance(name_or_callable, str):
            # Config users use string names
            if name_or_callable not in cls._strategies:
                available = list(cls._strategies.keys())
                raise ValueError(f"Unknown strategy '{name_or_callable}'. Available: {available}")
            return cls._strategies[name_or_callable]
        else:
            raise ValueError(f"Strategy must be a string name or callable, got {type(name_or_callable)}")
    
    @classmethod
    def list_strategies(cls):
        """List available strategies for documentation."""
        return list(cls._strategies.keys())


# Built-in strategies
@ConflictResolvers.register("node_priority")
def node_priority(nodes: List[Tuple], edges: List[Tuple], point: Tuple[float, float]) -> Tuple[Optional[Any], Optional[Tuple]]:
    """Always prefer nodes over edges (DEFAULT)."""
    return (nodes[0][0], None) if nodes else ((None, edges[0][0]) if edges else (None, None))


@ConflictResolvers.register("edge_priority") 
def edge_priority(nodes: List[Tuple], edges: List[Tuple], point: Tuple[float, float]) -> Tuple[Optional[Any], Optional[Tuple]]:
    """Always prefer edges over nodes."""
    return (None, edges[0][0]) if edges else ((nodes[0][0], None) if nodes else (None, None))


@ConflictResolvers.register("smallest_region")
def smallest_region(nodes: List[Tuple], edges: List[Tuple], point: Tuple[float, float]) -> Tuple[Optional[Any], Optional[Tuple]]:
    """Choose element with smallest region area."""
    all_matches = [(n, r, 'node') for n, r in nodes] + [(e, r, 'edge') for e, r in edges]
    if all_matches:
        smallest = min(all_matches, key=lambda x: x[1].get_area())
        return (smallest[0], None) if smallest[2] == 'node' else (None, smallest[0])
    return (None, None)


@ConflictResolvers.register("largest_region")
def largest_region(nodes: List[Tuple], edges: List[Tuple], point: Tuple[float, float]) -> Tuple[Optional[Any], Optional[Tuple]]:
    """Choose element with largest region area."""
    all_matches = [(n, r, 'node') for n, r in nodes] + [(e, r, 'edge') for e, r in edges]
    if all_matches:
        largest = max(all_matches, key=lambda x: x[1].get_area())
        return (largest[0], None) if largest[2] == 'node' else (None, largest[0])
    return (None, None)


@ConflictResolvers.register("nearest_center")
def nearest_center(nodes: List[Tuple], edges: List[Tuple], point: Tuple[float, float]) -> Tuple[Optional[Any], Optional[Tuple]]:
    """Choose element whose region center is nearest to the point."""
    all_matches = [(n, r, 'node') for n, r in nodes] + [(e, r, 'edge') for e, r in edges]
    if all_matches:
        pt = np.array(point)
        def distance_to_center(match):
            center = match[1].get_center()
            return np.linalg.norm(pt - np.array([center.x, center.y]))
        
        nearest = min(all_matches, key=distance_to_center)
        return (nearest[0], None) if nearest[2] == 'node' else (None, nearest[0])
    return (None, None)


@ConflictResolvers.register("first_found")
def first_found(nodes: List[Tuple], edges: List[Tuple], point: Tuple[float, float]) -> Tuple[Optional[Any], Optional[Tuple]]:
    """Return first match found (fastest performance)."""
    return (nodes[0][0], None) if nodes else ((None, edges[0][0]) if edges else (None, None))


@ConflictResolvers.register("raise_error")
def raise_error(nodes: List[Tuple], edges: List[Tuple], point: Tuple[float, float]) -> Tuple[Optional[Any], Optional[Tuple]]:
    """Raise exception on conflicts - forces explicit region mapping."""
    if len(nodes) + len(edges) > 1:
        raise ValueError(f"Point {point} matches multiple regions: "
                       f"nodes={[n for n,_ in nodes]}, edges={[e for e,_ in edges]}")
    return (nodes[0][0], None) if nodes else ((None, edges[0][0]) if edges else (None, None))