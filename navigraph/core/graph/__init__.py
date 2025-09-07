"""NaviGraph graph infrastructure module.

This module provides flexible graph-based spatial mapping infrastructure
for NaviGraph, enabling researchers to define arbitrary graph structures
and map them to spatial regions with support for both nodes and edges.

Core Components:
- structures: GraphStructure class wrapping NetworkX graphs
- builders: Graph construction functions for common topologies
- regions: Spatial region classes for different geometric shapes
- mapping: Enhanced SpatialMapping system linking regions to nodes/edges
- storage: Unified persistence system supporting multiple formats
- setup_gui: Dual-view interactive GUI for creating mappings

Quick Start:
    from navigraph.core.graph import (
        GraphStructure, GraphBuilder,
        SpatialMapping, launch_setup_gui
    )
    
    # Create a graph
    builder = GraphBuilder.from_config({'type': 'binary_tree', 'height': 4})
    graph = GraphStructure(builder)
    
    # Create mapping with setup GUI
    mapping = launch_setup_gui(graph, map_image)
"""

from .structures import GraphStructure
from .builders import (
    GraphBuilder,
    register_graph_builder,
    get_graph_builder,
    list_graph_builders
)
from .regions import (
    SpatialRegion, Point, ContourRegion, RectangleRegion,
    CircleRegion, GridCell, HexagonalCell, EllipseRegion
)
from .mapping import SpatialMapping, MappingStatistics, OverlapInfo, NodeConflictInfo
from .storage import MappingStorage
from .setup_gui_qt import launch_setup_gui, GridConfig
from .gui_utils import is_gui_available, print_backend_info

__all__ = [
    # Core structures
    'GraphStructure',
    
    # Builder system
    'GraphBuilder',
    'register_graph_builder',
    'get_graph_builder',
    'list_graph_builders',
    
    # Regions
    'SpatialRegion',
    'Point',
    'ContourRegion',
    'RectangleRegion',
    'CircleRegion',
    'GridCell',
    'HexagonalCell',
    'EllipseRegion',
    
    # Mapping
    'SpatialMapping',
    'MappingStatistics', 
    'OverlapInfo',
    'NodeConflictInfo',
    
    # Storage
    'MappingStorage',
    
    # Interactive tools
    'launch_setup_gui',
    'GridConfig',
    
    # GUI utilities
    'is_gui_available',
    'print_backend_info',
]