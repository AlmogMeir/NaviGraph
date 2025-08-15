"""Spatial mapping system for NaviGraph.

This module provides the core mapping functionality that links spatial regions
to graph nodes and edges, enabling spatial navigation analysis.
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import json
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
                 unmapped_value: Any = None,
                 conflict_strategy: str = "node_priority"):
        """Initialize spatial mapping.
        
        Args:
            graph: Optional graph structure for validation
            unmapped_value: Value to return for unmapped points (default: None)
            conflict_strategy: Strategy for resolving conflicts when point is in multiple regions
        """
        self.graph = graph
        self.unmapped_value = unmapped_value
        
        # Set up conflict resolution strategy
        from .conflict_resolvers import ConflictResolvers
        self.conflict_resolver = ConflictResolvers.get(conflict_strategy)
        
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
        # Keep edge representation as provided by the graph
        
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
        """Map a coordinate point to graph elements with conflict resolution.
        
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
        
        # Find all matching regions
        matching_nodes = []
        matching_edges = []
        
        for region_id, region in self._regions.items():
            if region.contains_point(x, y):
                element = self._region_to_element.get(region_id)
                if element:
                    element_type, element_id = element
                    if element_type == 'node':
                        matching_nodes.append((element_id, region))
                    elif element_type == 'edge':
                        matching_edges.append((element_id, region))
        
        # Apply conflict resolution strategy
        if not matching_nodes and not matching_edges:
            result = (self.unmapped_value, None)
        else:
            # Use resolver
            node_id, edge_id = self.conflict_resolver(matching_nodes, matching_edges, (x, y))
            result = (node_id if node_id is not None else self.unmapped_value, edge_id)
        
        # Cache result
        if self._cache_enabled and len(self._point_cache) < self._max_cache_size:
            self._point_cache[cache_key] = result
        
        return result
    
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
        # For undirected graphs, try both edge directions
        if self.graph and not self.graph.graph.is_directed():
            region_ids = self._edge_to_regions.get(edge, [])
            if not region_ids:
                # Try reversed edge
                reversed_edge = (edge[1], edge[0])
                region_ids = self._edge_to_regions.get(reversed_edge, [])
        else:
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
    
    def to_simple_format(self) -> Dict[str, Any]:
        """Convert mapping to simple x,y point format.
        
        Returns:
            Dictionary with simple point lists for nodes and edges
        """
        simple_mapping = {
            'nodes': {},
            'edges': {}
        }
        
        # Convert node regions to point lists
        for node_id in self.get_mapped_nodes():
            regions = self.get_node_regions(node_id)
            contours = []
            for region in regions:
                # Extract contour points from any region type
                points = self._extract_contour_points(region)
                if points is not None and len(points) > 0:
                    contours.append(points.tolist())  # Convert numpy to list
            if contours:
                simple_mapping['nodes'][str(node_id)] = contours
        
        # Convert edge regions to point lists with consistent format
        for edge in self.get_mapped_edges():
            regions = self.get_edge_regions(edge)
            contours = []
            for region in regions:
                points = self._extract_contour_points(region)
                if points is not None and len(points) > 0:
                    contours.append(points.tolist())  # Convert numpy to list
            if contours:
                # Always use consistent edge string format
                edge_str = f"{edge[0]}_{edge[1]}"
                simple_mapping['edges'][edge_str] = contours
        
        return simple_mapping
    
    def from_simple_format(self, simple_mapping: Dict[str, Any]):
        """Load mapping from simple x,y point format.
        
        Args:
            simple_mapping: Dictionary with point lists
        """
        from .regions import ContourRegion
        
        # Clear existing mappings
        self.clear_all_mappings()
        
        # Load node mappings
        for node_id_str, contour_lists in simple_mapping.get('nodes', {}).items():
            # Convert string back to appropriate type
            node_id = self._parse_node_id(node_id_str)
            
            for i, contour_points in enumerate(contour_lists):
                region = ContourRegion(
                    region_id=f"node_{node_id}_region_{i}",
                    contour_points=np.array(contour_points)
                )
                self.add_node_region(region, node_id)
        
        # Load edge mappings
        for edge_str, contour_lists in simple_mapping.get('edges', {}).items():
            # Parse edge string to tuple
            edge = self._parse_edge_string(edge_str)
            
            for i, contour_points in enumerate(contour_lists):
                region = ContourRegion(
                    region_id=f"edge_{edge_str}_region_{i}",
                    contour_points=np.array(contour_points)
                )
                self.add_edge_region(region, edge)
    
    def _extract_contour_points(self, region) -> Optional[np.ndarray]:
        """Extract contour points from any region type.
        
        Args:
            region: SpatialRegion object
            
        Returns:
            Array of contour points or None
        """
        from .regions import ContourRegion, GridCell, RectangleRegion
        
        if isinstance(region, ContourRegion):
            # Already a contour - use as-is
            return region.contour_points
        elif isinstance(region, (GridCell, RectangleRegion)):
            # Convert rectangle/grid cell to 4-point contour
            bounds = region.get_bounds()  # Assuming bounds method exists
            if bounds:
                x_min, y_min, x_max, y_max = bounds
                return np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ])
        return None
    
    def _parse_node_id(self, node_id_str: str) -> Any:
        """Parse node ID from string format.
        
        Args:
            node_id_str: String representation of node ID
            
        Returns:
            Parsed node ID (int if numeric, otherwise string)
        """
        try:
            # Try to convert to int if it looks numeric
            return int(node_id_str)
        except ValueError:
            # Return as string if not numeric
            return node_id_str
    
    def _parse_edge_string(self, edge_str: str) -> Tuple[Any, Any]:
        """Parse edge string to tuple.
        
        Args:
            edge_str: String in format "node1_node2"
            
        Returns:
            Tuple of (node1, node2)
        """
        parts = edge_str.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid edge string format: {edge_str}")
        
        node1 = self._parse_node_id(parts[0])
        node2 = self._parse_node_id(parts[1])
        return (node1, node2)
    
    def save_with_builder_info(self, file_path, setup_mode_state: Optional[Dict[str, Any]] = None):
        """Save mapping with builder information for reconstruction.
        
        Args:
            file_path: Path object or string for save location
            setup_mode_state: Optional state from GUI (mode, grid config, progress)
        """
        from pathlib import Path
        import pickle
        import json
        from datetime import datetime
        
        file_path = Path(file_path)
        
        # Get builder info from graph structure
        builder_info = self.graph.metadata if self.graph else None
        if not builder_info:
            raise ValueError("Cannot save mapping without graph builder information")
        
        # Create complete mapping data
        data = {
            'format_version': '3.0',
            'graph_builder': {
                'type': self._get_builder_registry_name(builder_info['builder_type']),
                'config': builder_info.get('parameters', {})
            },
            'mappings': self.to_simple_format(),
            'setup_mode': setup_mode_state or {},
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_nodes': len(self.get_mapped_nodes()),
                'total_edges': len(self.get_mapped_edges()),
                'total_regions': len(self._regions)
            }
        }
        
        # Save as JSON for readability (or pickle for compatibility)
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, cls=self._JSONEncoder)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
    
    @classmethod
    def load_with_builder_reconstruction(cls, file_path) -> 'SpatialMapping':
        """Load mapping and reconstruct graph using builder information.
        
        Args:
            file_path: Path object or string for mapping file
            
        Returns:
            SpatialMapping with reconstructed graph
        """
        from pathlib import Path
        import pickle
        import json
        from .structures import GraphStructure
        
        file_path = Path(file_path)
        
        # Load data
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        
        # Check format version
        version = data.get('format_version', '1.0')
        if version < '3.0':
            raise ValueError(
                f"Unsupported mapping format version: {version}. "
                f"Please recreate this mapping using the updated GUI."
            )
        
        # Reconstruct graph from builder info
        builder_info = data['graph_builder']
        graph_structure = GraphStructure.from_config(
            builder_info['type'],
            builder_info['config']
        )
        
        # Create mapping with reconstructed graph
        mapping = cls(graph_structure)
        
        # Load the simple format mappings
        mapping.from_simple_format(data['mappings'])
        
        # Store setup mode state for GUI loading
        mapping._setup_mode_state = data.get('setup_mode', {})
        
        return mapping
    
    def _get_builder_registry_name(self, builder_class_name: str) -> str:
        """Convert builder class name to registry name.
        
        Args:
            builder_class_name: Class name like 'BinaryTreeBuilder'
            
        Returns:
            Registry name like 'binary_tree'
        """
        # Remove 'Builder' suffix and convert to snake_case
        name = builder_class_name.replace('Builder', '')
        
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name
    
    class _JSONEncoder(json.JSONEncoder):
        """Custom JSON encoder for numpy arrays and other types."""
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)
    
    def get_setup_mode_state(self) -> Dict[str, Any]:
        """Get setup mode state for GUI continuation.
        
        Returns:
            Dictionary with setup mode state or empty dict
        """
        return getattr(self, '_setup_mode_state', {})
    
    def query_point(self, x: float, y: float) -> Tuple[Optional[Any], Optional[Any]]:
        """Find which graph element (node or edge) contains the given point.
        
        Args:
            x, y: Point coordinates
            
        Returns:
            Tuple of (node_id, edge_id) where one is None
            Returns (None, None) if no mapping found
        """
        # Check node regions first (nodes have priority over edges)
        for node_id in self.get_mapped_nodes():
            regions = self.get_node_regions(node_id)
            for region in regions:
                if region.contains_point(x, y):
                    return (node_id, None)
        
        # Check edge regions
        for edge_id in self.get_mapped_edges():
            regions = self.get_edge_regions(edge_id)
            for region in regions:
                if region.contains_point(x, y):
                    return (None, edge_id)
                    
        return (None, None)
    
    def __len__(self) -> int:
        """Return number of mapped regions."""
        return len(self._regions)
    
    def __str__(self) -> str:
        """String representation of mapping."""
        stats = self.validate_mapping()
        return str(stats)