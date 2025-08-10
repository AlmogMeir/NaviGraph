"""Spatial mapping system for NaviGraph.

This module provides the core mapping functionality that links spatial regions
to graph nodes and edges, enabling spatial navigation analysis.
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from .regions import SpatialRegion
from .structures import GraphStructure


@dataclass
class OverlapInfo:
    """Information about overlapping regions."""
    region1_id: str
    region2_id: str
    element1: Tuple[str, Any]  # ('node', id) or ('edge', (id1, id2))
    element2: Tuple[str, Any]
    overlap_severity: float = 0.0
    
    def __str__(self) -> str:
        type1, id1 = self.element1
        type2, id2 = self.element2
        return f"Overlap between {self.region1_id} ({type1} {id1}) and {self.region2_id} ({type2} {id2})"


@dataclass
class NodeConflictInfo:
    """Information about nodes mapped to same region."""
    region_id: str
    node1: Any
    node2: Any
    
    def __str__(self) -> str:
        return f"Conflict: Nodes {self.node1} and {self.node2} both mapped to region {self.region_id}"


@dataclass
class MappingStatistics:
    """Statistics about a spatial mapping."""
    total_nodes: int
    mapped_nodes: int
    unmapped_nodes: int
    total_edges: int
    mapped_edges: int
    unmapped_edges: int
    total_regions: int
    overlapping_regions: int
    coverage_percentage: float
    overlaps: List[OverlapInfo] = field(default_factory=list)
    node_conflicts: List[NodeConflictInfo] = field(default_factory=list)
    
    @property
    def node_mapping_completeness(self) -> float:
        """Percentage of nodes that have been mapped."""
        if self.total_nodes == 0:
            return 100.0
        return (self.mapped_nodes / self.total_nodes) * 100.0
    
    @property
    def edge_mapping_completeness(self) -> float:
        """Percentage of edges that have been mapped."""
        if self.total_edges == 0:
            return 100.0
        return (self.mapped_edges / self.total_edges) * 100.0
    
    def __str__(self) -> str:
        return f"""Mapping Statistics:
  Nodes:
    Total: {self.total_nodes}
    Mapped: {self.mapped_nodes} ({self.node_mapping_completeness:.1f}%)
    Unmapped: {self.unmapped_nodes}
  Edges:
    Total: {self.total_edges}
    Mapped: {self.mapped_edges} ({self.edge_mapping_completeness:.1f}%)
    Unmapped: {self.unmapped_edges}
  Regions:
    Total: {self.total_regions}
    Overlapping: {self.overlapping_regions}
  Coverage: {self.coverage_percentage:.1f}%
  Overlaps found: {len(self.overlaps)}
  Node conflicts: {len(self.node_conflicts)}"""


class SpatialMapping:
    """Maps spatial regions to graph nodes and edges.
    
    This class manages the relationship between spatial regions (areas on a map)
    and graph elements (nodes and edges), providing efficient point-to-element
    lookup and validation. Supports multiple regions per element and ensures
    no two nodes map to the same region.
    """
    
    def __init__(self, graph: Optional[GraphStructure] = None, 
                 unmapped_value: Any = None):
        """Initialize spatial mapping.
        
        Args:
            graph: Optional graph structure for validation
            unmapped_value: Value to return for unmapped points (default: None)
        """
        self.graph = graph
        self.unmapped_value = unmapped_value
        
        # Core mapping data structures
        # Each element can have multiple regions (lists of contours)
        self._node_to_regions: Dict[Any, List[str]] = {}  # node_id -> [region_ids]
        self._edge_to_regions: Dict[Tuple[Any, Any], List[str]] = {}  # (node1, node2) -> [region_ids]
        self._region_to_element: Dict[str, Tuple[str, Any]] = {}  # region_id -> ('node', id) or ('edge', (id1, id2))
        self._regions: Dict[str, SpatialRegion] = {}  # region_id -> SpatialRegion
        
        # Track which regions contain nodes (for conflict detection)
        self._node_regions: Set[str] = set()  # Region IDs that contain nodes
        
        # Cache for performance
        self._point_cache: Dict[Tuple[float, float], Tuple[Optional[Any], Optional[Tuple[Any, Any]]]] = {}
        self._cache_enabled = True
        self._max_cache_size = 10000
    
    def add_node_region(self, region: SpatialRegion, node_id: Any, allow_multiple: bool = True):
        """Add a region mapping for a node.
        
        Args:
            region: Spatial region to map
            node_id: Graph node to associate with the region
            allow_multiple: Whether to allow multiple regions per node
            
        Raises:
            ValueError: If node doesn't exist or region conflicts with another node
        """
        # Validate node exists in graph if provided
        if self.graph and not self.graph.has_node(node_id):
            raise ValueError(f"Node {node_id} does not exist in the graph")
        
        # Check for node conflicts
        if region.region_id in self._node_regions:
            existing_element = self._region_to_element.get(region.region_id)
            if existing_element and existing_element[0] == 'node' and existing_element[1] != node_id:
                raise ValueError(f"Region {region.region_id} already mapped to node {existing_element[1]}")
        
        region_id = region.region_id
        
        # Store region
        self._regions[region_id] = region
        self._region_to_element[region_id] = ('node', node_id)
        self._node_regions.add(region_id)
        
        # Update node mapping (supports multiple regions per node)
        if node_id not in self._node_to_regions:
            self._node_to_regions[node_id] = []
        if not allow_multiple:
            self._node_to_regions[node_id] = [region_id]
        else:
            self._node_to_regions[node_id].append(region_id)
        
        self._clear_cache()
    
    def add_edge_region(self, region: SpatialRegion, edge: Tuple[Any, Any], allow_multiple: bool = True):
        """Add a region mapping for an edge.
        
        Args:
            region: Spatial region to map
            edge: Edge tuple (node1, node2) to associate with the region
            allow_multiple: Whether to allow multiple regions per edge
            
        Raises:
            ValueError: If edge doesn't exist in graph
        """
        # Normalize edge representation
        edge = tuple(sorted(edge))
        
        # Validate edge exists in graph if provided
        if self.graph and not self.graph.has_edge(edge[0], edge[1]):
            raise ValueError(f"Edge {edge} does not exist in the graph")
        
        region_id = region.region_id
        
        # Store region
        self._regions[region_id] = region
        self._region_to_element[region_id] = ('edge', edge)
        
        # Update edge mapping (supports multiple regions per edge)
        if edge not in self._edge_to_regions:
            self._edge_to_regions[edge] = []
        if not allow_multiple:
            self._edge_to_regions[edge] = [region_id]
        else:
            self._edge_to_regions[edge].append(region_id)
        
        self._clear_cache()
    
    def remove_region(self, region_id: str):
        """Remove a region mapping.
        
        Args:
            region_id: ID of region to remove
        """
        if region_id not in self._regions:
            return
        
        # Get associated element
        element = self._region_to_element.get(region_id)
        
        if element:
            element_type, element_id = element
            
            if element_type == 'node':
                # Remove from node mapping
                if element_id in self._node_to_regions:
                    self._node_to_regions[element_id].remove(region_id)
                    if not self._node_to_regions[element_id]:
                        del self._node_to_regions[element_id]
                self._node_regions.discard(region_id)
                
            elif element_type == 'edge':
                # Remove from edge mapping
                if element_id in self._edge_to_regions:
                    self._edge_to_regions[element_id].remove(region_id)
                    if not self._edge_to_regions[element_id]:
                        del self._edge_to_regions[element_id]
        
        # Remove from data structures
        del self._regions[region_id]
        del self._region_to_element[region_id]
        
        self._clear_cache()
    
    def map_point_to_elements(self, x: float, y: float) -> Tuple[Optional[Any], Optional[Tuple[Any, Any]]]:
        """Map a coordinate point to graph elements.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Tuple of (node_id, edge_tuple) where either or both can be None
        """
        # Check cache first
        if self._cache_enabled:
            cache_key = (round(x, 6), round(y, 6))
            if cache_key in self._point_cache:
                return self._point_cache[cache_key]
        
        node_id = None
        edge_tuple = None
        
        # Find which regions contain the point
        for region_id, region in self._regions.items():
            if region.contains_point(x, y):
                element = self._region_to_element.get(region_id)
                if element:
                    element_type, element_id = element
                    if element_type == 'node':
                        node_id = element_id
                    elif element_type == 'edge':
                        edge_tuple = element_id
        
        # Cache result
        if self._cache_enabled and len(self._point_cache) < self._max_cache_size:
            self._point_cache[cache_key] = (node_id, edge_tuple)
        
        return (node_id, edge_tuple)
    
    def map_point_to_node(self, x: float, y: float) -> Any:
        """Map a coordinate point to a graph node.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Node ID if point is in a node region, unmapped_value otherwise
        """
        node_id, _ = self.map_point_to_elements(x, y)
        return node_id if node_id is not None else self.unmapped_value
    
    def map_point_to_edge(self, x: float, y: float) -> Optional[Tuple[Any, Any]]:
        """Map a coordinate point to a graph edge.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Edge tuple if point is in an edge region, None otherwise
        """
        _, edge_tuple = self.map_point_to_elements(x, y)
        return edge_tuple
    
    def get_node_regions(self, node_id: Any) -> List[SpatialRegion]:
        """Get all regions associated with a node.
        
        Args:
            node_id: Graph node ID
            
        Returns:
            List of spatial regions associated with the node
        """
        region_ids = self._node_to_regions.get(node_id, [])
        return [self._regions[rid] for rid in region_ids]
    
    def get_edge_regions(self, edge: Tuple[Any, Any]) -> List[SpatialRegion]:
        """Get all regions associated with an edge.
        
        Args:
            edge: Edge tuple (node1, node2)
            
        Returns:
            List of spatial regions associated with the edge
        """
        edge = tuple(sorted(edge))
        region_ids = self._edge_to_regions.get(edge, [])
        return [self._regions[rid] for rid in region_ids]
    
    def get_mapped_nodes(self) -> Set[Any]:
        """Get set of all nodes that have been mapped to regions.
        
        Returns:
            Set of node IDs that have spatial mappings
        """
        return set(self._node_to_regions.keys())
    
    def get_mapped_edges(self) -> Set[Tuple[Any, Any]]:
        """Get set of all edges that have been mapped to regions.
        
        Returns:
            Set of edge tuples that have spatial mappings
        """
        return set(self._edge_to_regions.keys())
    
    def get_unmapped_nodes(self) -> Set[Any]:
        """Get set of nodes that have no spatial mapping.
        
        Returns:
            Set of node IDs with no spatial mappings
        """
        if not self.graph:
            return set()
        
        all_nodes = set(self.graph.nodes)
        mapped_nodes = self.get_mapped_nodes()
        return all_nodes - mapped_nodes
    
    def get_unmapped_edges(self) -> Set[Tuple[Any, Any]]:
        """Get set of edges that have no spatial mapping.
        
        Returns:
            Set of edge tuples with no spatial mappings
        """
        if not self.graph:
            return set()
        
        all_edges = set(tuple(sorted(e)) for e in self.graph.edges)
        mapped_edges = self.get_mapped_edges()
        return all_edges - mapped_edges
    
    def get_all_regions(self) -> Dict[str, SpatialRegion]:
        """Get all regions in the mapping.
        
        Returns:
            Dictionary mapping region IDs to SpatialRegion objects
        """
        return self._regions.copy()
    
    def find_overlaps(self, tolerance: float = 0.01) -> List[OverlapInfo]:
        """Find overlapping regions.
        
        Args:
            tolerance: Minimum overlap to consider significant
            
        Returns:
            List of OverlapInfo objects describing overlaps
        """
        overlaps = []
        region_items = list(self._regions.items())
        
        for i in range(len(region_items)):
            for j in range(i + 1, len(region_items)):
                region1_id, region1 = region_items[i]
                region2_id, region2 = region_items[j]
                
                if region1.overlaps_with(region2):
                    element1 = self._region_to_element[region1_id]
                    element2 = self._region_to_element[region2_id]
                    
                    overlap_info = OverlapInfo(
                        region1_id=region1_id,
                        region2_id=region2_id,
                        element1=element1,
                        element2=element2,
                        overlap_severity=1.0  # Could calculate actual overlap
                    )
                    overlaps.append(overlap_info)
        
        return overlaps
    
    def find_node_conflicts(self) -> List[NodeConflictInfo]:
        """Find cases where multiple nodes map to same region.
        
        Returns:
            List of NodeConflictInfo objects describing conflicts
        """
        conflicts = []
        region_to_nodes: Dict[str, List[Any]] = {}
        
        # Build reverse mapping of regions to nodes
        for node_id, region_ids in self._node_to_regions.items():
            for region_id in region_ids:
                if region_id not in region_to_nodes:
                    region_to_nodes[region_id] = []
                region_to_nodes[region_id].append(node_id)
        
        # Find conflicts
        for region_id, nodes in region_to_nodes.items():
            if len(nodes) > 1:
                # Create conflict info for each pair
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        conflict = NodeConflictInfo(
                            region_id=region_id,
                            node1=nodes[i],
                            node2=nodes[j]
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    def validate_mapping(self) -> MappingStatistics:
        """Validate the current mapping and return statistics.
        
        Returns:
            MappingStatistics object with validation results
        """
        # Basic counts
        total_nodes = len(self.graph.nodes) if self.graph else 0
        mapped_nodes = len(self.get_mapped_nodes())
        unmapped_nodes = len(self.get_unmapped_nodes())
        
        total_edges = len(self.graph.edges) if self.graph else 0
        mapped_edges = len(self.get_mapped_edges())
        unmapped_edges = len(self.get_unmapped_edges())
        
        total_regions = len(self._regions)
        
        # Find overlaps and conflicts
        overlaps = self.find_overlaps()
        node_conflicts = self.find_node_conflicts()
        
        overlapping_regions = len(set(
            [overlap.region1_id for overlap in overlaps] +
            [overlap.region2_id for overlap in overlaps]
        ))
        
        # Calculate coverage
        coverage_percentage = ((mapped_nodes + mapped_edges) / 
                             (total_nodes + total_edges) * 100) if (total_nodes + total_edges) > 0 else 0
        
        return MappingStatistics(
            total_nodes=total_nodes,
            mapped_nodes=mapped_nodes,
            unmapped_nodes=unmapped_nodes,
            total_edges=total_edges,
            mapped_edges=mapped_edges,
            unmapped_edges=unmapped_edges,
            total_regions=total_regions,
            overlapping_regions=overlapping_regions,
            coverage_percentage=coverage_percentage,
            overlaps=overlaps,
            node_conflicts=node_conflicts
        )
    
    def get_region_by_id(self, region_id: str) -> Optional[SpatialRegion]:
        """Get a region by its ID.
        
        Args:
            region_id: Region identifier
            
        Returns:
            SpatialRegion object or None if not found
        """
        return self._regions.get(region_id)
    
    def get_element_for_region(self, region_id: str) -> Optional[Tuple[str, Any]]:
        """Get the element associated with a region.
        
        Args:
            region_id: Region identifier
            
        Returns:
            Tuple of (element_type, element_id) or None
        """
        return self._region_to_element.get(region_id)
    
    def set_cache_enabled(self, enabled: bool):
        """Enable or disable point lookup caching.
        
        Args:
            enabled: Whether to enable caching
        """
        self._cache_enabled = enabled
        if not enabled:
            self._clear_cache()
    
    def _clear_cache(self):
        """Clear the point lookup cache."""
        self._point_cache.clear()
    
    def clear_all_mappings(self):
        """Clear all mappings and regions."""
        self._node_to_regions.clear()
        self._edge_to_regions.clear()
        self._region_to_element.clear()
        self._regions.clear()
        self._node_regions.clear()
        self._clear_cache()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert mapping to dictionary format.
        
        Returns:
            Dictionary with all mapping data
        """
        return {
            'node_to_regions': {str(k): v for k, v in self._node_to_regions.items()},
            'edge_to_regions': {str(k): v for k, v in self._edge_to_regions.items()},
            'region_to_element': self._region_to_element,
            'unmapped_value': self.unmapped_value
        }
    
    def __len__(self) -> int:
        """Return number of mapped regions."""
        return len(self._regions)
    
    def __str__(self) -> str:
        """String representation of mapping."""
        stats = self.validate_mapping()
        return str(stats)