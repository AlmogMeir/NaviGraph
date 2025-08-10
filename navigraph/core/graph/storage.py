"""Storage system for spatial mappings.

This module provides functionality to save and load spatial mappings
in various formats, enabling persistence and sharing of graph-space mappings.
Supports unified format for both node and edge mappings with multiple regions.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import pickle
import json
import csv
import numpy as np
from dataclasses import asdict

from .mapping import SpatialMapping, MappingStatistics
from .structures import GraphStructure
from .regions import (SpatialRegion, ContourRegion, RectangleRegion, 
                     CircleRegion, GridCell, HexagonalCell, EllipseRegion)


class MappingStorage:
    """Handles saving and loading of spatial mappings with unified format."""
    
    @staticmethod
    def save_mapping(mapping: SpatialMapping, filepath: Union[str, Path], 
                    format: str = 'pickle', include_graph: bool = True) -> bool:
        """Save spatial mapping to file.
        
        Args:
            mapping: SpatialMapping to save
            filepath: Output file path
            format: File format ('pickle', 'json', 'h5', 'csv')
            include_graph: Whether to include graph structure in saved data
            
        Returns:
            True if save successful, False otherwise
        """
        filepath = Path(filepath)
        
        try:
            if format == 'pickle':
                return MappingStorage._save_pickle(mapping, filepath, include_graph)
            elif format == 'json':
                return MappingStorage._save_json(mapping, filepath, include_graph)
            elif format == 'h5':
                return MappingStorage._save_h5(mapping, filepath, include_graph)
            elif format == 'csv':
                return MappingStorage._save_csv(mapping, filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            print(f"Error saving mapping: {e}")
            return False
    
    @staticmethod
    def load_mapping(filepath: Union[str, Path], 
                    graph: Optional[GraphStructure] = None) -> Optional[SpatialMapping]:
        """Load spatial mapping from file.
        
        Args:
            filepath: Path to mapping file
            graph: Optional graph to associate with mapping
            
        Returns:
            Loaded SpatialMapping or None if load failed
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        try:
            # Determine format from extension
            if filepath.suffix == '.pkl':
                return MappingStorage._load_pickle(filepath, graph)
            elif filepath.suffix == '.json':
                return MappingStorage._load_json(filepath, graph)
            elif filepath.suffix in ['.h5', '.hdf5']:
                return MappingStorage._load_h5(filepath, graph)
            elif filepath.suffix == '.csv':
                return MappingStorage._load_csv(filepath, graph)
            else:
                # Try to detect format from content
                try:
                    return MappingStorage._load_pickle(filepath, graph)
                except:
                    try:
                        return MappingStorage._load_json(filepath, graph)
                    except:
                        return None
        except Exception as e:
            print(f"Error loading mapping: {e}")
            return None
    
    @staticmethod
    def _save_pickle(mapping: SpatialMapping, filepath: Path, 
                    include_graph: bool) -> bool:
        """Save mapping in pickle format (preserves all data types)."""
        data = {
            'format_version': '2.0',  # New unified format
            'regions': {},
            'node_to_regions': mapping._node_to_regions.copy(),
            'edge_to_regions': {str(k): v for k, v in mapping._edge_to_regions.items()},
            'region_to_element': mapping._region_to_element.copy(),
            'unmapped_value': mapping.unmapped_value,
            'graph': mapping.graph.to_dict() if include_graph and mapping.graph else None
        }
        
        # Serialize regions
        for region_id, region in mapping._regions.items():
            data['regions'][region_id] = MappingStorage._serialize_region(region)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        return True
    
    @staticmethod
    def _load_pickle(filepath: Path, graph: Optional[GraphStructure]) -> SpatialMapping:
        """Load mapping from pickle format."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Check format version
        format_version = data.get('format_version', '1.0')
        
        if format_version == '2.0':
            # New unified format
            return MappingStorage._load_unified_format(data, graph)
        else:
            # Legacy format (backward compatibility)
            return MappingStorage._load_legacy_format(data, graph)
    
    @staticmethod
    def _load_unified_format(data: Dict, graph: Optional[GraphStructure]) -> SpatialMapping:
        """Load mapping from unified format."""
        # Reconstruct graph if provided in data and not overridden
        if graph is None and data.get('graph'):
            graph = GraphStructure.from_dict(data['graph'])
        
        # Create mapping
        mapping = SpatialMapping(graph, data.get('unmapped_value'))
        
        # Reconstruct regions and mappings
        for region_id, region_data in data['regions'].items():
            region = MappingStorage._deserialize_region(region_id, region_data)
            element = data['region_to_element'].get(region_id)
            
            if element:
                elem_type, elem_id = element
                if elem_type == 'node':
                    # Add directly to avoid validation issues during loading
                    mapping._regions[region_id] = region
                    mapping._region_to_element[region_id] = element
                    mapping._node_regions.add(region_id)
                    if elem_id not in mapping._node_to_regions:
                        mapping._node_to_regions[elem_id] = []
                    mapping._node_to_regions[elem_id].append(region_id)
                elif elem_type == 'edge':
                    # Convert edge ID back to tuple if needed
                    if isinstance(elem_id, str):
                        elem_id = eval(elem_id)  # Convert string representation to tuple
                    mapping._regions[region_id] = region
                    mapping._region_to_element[region_id] = ('edge', tuple(elem_id))
                    if elem_id not in mapping._edge_to_regions:
                        mapping._edge_to_regions[tuple(elem_id)] = []
                    mapping._edge_to_regions[tuple(elem_id)].append(region_id)
        
        return mapping
    
    @staticmethod
    def _load_legacy_format(data: Dict, graph: Optional[GraphStructure]) -> SpatialMapping:
        """Load mapping from legacy format (backward compatibility)."""
        # Reconstruct graph if provided in data and not overridden
        if graph is None and data.get('graph'):
            graph = GraphStructure.from_dict(data['graph'])
        
        # Create mapping
        mapping = SpatialMapping(graph, data.get('unmapped_value'))
        
        # Legacy format only had node mappings
        for region_id, region_data in data['regions'].items():
            region = MappingStorage._deserialize_region(region_id, region_data)
            node_id = data.get('region_to_node', {}).get(region_id)
            if node_id:
                mapping.add_node_region(region, node_id)
        
        return mapping
    
    @staticmethod
    def _save_json(mapping: SpatialMapping, filepath: Path, 
                  include_graph: bool) -> bool:
        """Save mapping in JSON format (human-readable)."""
        data = {
            'format_version': '2.0',
            'regions': {},
            'node_to_regions': {},
            'edge_to_regions': {},
            'region_to_element': {},
            'unmapped_value': mapping.unmapped_value,
            'graph': mapping.graph.to_dict() if include_graph and mapping.graph else None
        }
        
        # Convert node mappings (handle various node ID types)
        for node_id, region_ids in mapping._node_to_regions.items():
            data['node_to_regions'][str(node_id)] = region_ids
        
        # Convert edge mappings
        for edge, region_ids in mapping._edge_to_regions.items():
            edge_key = f"{edge[0]}_{edge[1]}"
            data['edge_to_regions'][edge_key] = region_ids
        
        # Convert region to element mappings
        for region_id, element in mapping._region_to_element.items():
            elem_type, elem_id = element
            if elem_type == 'edge':
                elem_id = f"{elem_id[0]}_{elem_id[1]}"
            data['region_to_element'][region_id] = [elem_type, str(elem_id)]
        
        # Serialize regions
        for region_id, region in mapping._regions.items():
            region_data = MappingStorage._serialize_region(region)
            # Convert numpy arrays to lists for JSON
            if 'contour_points' in region_data:
                region_data['contour_points'] = [list(p) for p in region_data['contour_points']]
            data['regions'][region_id] = region_data
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    @staticmethod
    def _load_json(filepath: Path, graph: Optional[GraphStructure]) -> SpatialMapping:
        """Load mapping from JSON format."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        format_version = data.get('format_version', '1.0')
        
        # Reconstruct graph if provided
        if graph is None and data.get('graph'):
            graph = GraphStructure.from_dict(data['graph'])
        
        # Create mapping
        mapping = SpatialMapping(graph, data.get('unmapped_value'))
        
        # Load regions
        for region_id, region_data in data['regions'].items():
            # Convert lists back to numpy arrays if needed
            if 'contour_points' in region_data:
                region_data['contour_points'] = np.array(region_data['contour_points'])
            
            region = MappingStorage._deserialize_region(region_id, region_data)
            
            # Find element for this region
            if format_version == '2.0':
                element_data = data['region_to_element'].get(region_id)
                if element_data:
                    elem_type, elem_id = element_data
                    if elem_type == 'node':
                        # Try to convert back to original type
                        try:
                            elem_id = int(elem_id)
                        except:
                            pass  # Keep as string
                        mapping.add_node_region(region, elem_id)
                    elif elem_type == 'edge':
                        # Parse edge ID
                        parts = elem_id.split('_')
                        if len(parts) == 2:
                            edge = (parts[0], parts[1])
                            try:
                                edge = (int(parts[0]), int(parts[1]))
                            except:
                                pass  # Keep as strings
                            mapping.add_edge_region(region, edge)
        
        return mapping
    
    @staticmethod
    def _save_h5(mapping: SpatialMapping, filepath: Path, 
                include_graph: bool) -> bool:
        """Save mapping in HDF5 format (efficient for large datasets)."""
        try:
            import h5py
        except ImportError:
            print("h5py not installed. Install with: pip install h5py")
            return False
        
        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs['format_version'] = '2.0'
            f.attrs['unmapped_value'] = str(mapping.unmapped_value) if mapping.unmapped_value else 'None'
            
            # Save regions
            regions_group = f.create_group('regions')
            for region_id, region in mapping._regions.items():
                region_group = regions_group.create_group(region_id)
                region_data = MappingStorage._serialize_region(region)
                
                for key, value in region_data.items():
                    if isinstance(value, (list, np.ndarray)):
                        region_group.create_dataset(key, data=np.array(value))
                    else:
                        region_group.attrs[key] = value
            
            # Save mappings
            mappings_group = f.create_group('mappings')
            
            # Node to regions
            node_group = mappings_group.create_group('node_to_regions')
            for node_id, region_ids in mapping._node_to_regions.items():
                node_group.create_dataset(str(node_id), data=np.array(region_ids, dtype='S'))
            
            # Edge to regions
            edge_group = mappings_group.create_group('edge_to_regions')
            for edge, region_ids in mapping._edge_to_regions.items():
                edge_key = f"{edge[0]}_{edge[1]}"
                edge_group.create_dataset(edge_key, data=np.array(region_ids, dtype='S'))
            
            # Region to element
            element_group = mappings_group.create_group('region_to_element')
            for region_id, element in mapping._region_to_element.items():
                elem_type, elem_id = element
                if elem_type == 'edge':
                    elem_id = f"{elem_id[0]}_{elem_id[1]}"
                element_group.attrs[region_id] = f"{elem_type}:{elem_id}"
            
            # Save graph if requested
            if include_graph and mapping.graph:
                graph_group = f.create_group('graph')
                graph_data = mapping.graph.to_dict()
                graph_group.attrs['directed'] = graph_data.get('directed', False)
                # Store nodes and edges as datasets
                # (simplified - full implementation would handle attributes)
        
        return True
    
    @staticmethod
    def _load_h5(filepath: Path, graph: Optional[GraphStructure]) -> SpatialMapping:
        """Load mapping from HDF5 format."""
        try:
            import h5py
        except ImportError:
            print("h5py not installed. Install with: pip install h5py")
            return None
        
        mapping = SpatialMapping(graph)
        
        with h5py.File(filepath, 'r') as f:
            # Load metadata
            unmapped_value = f.attrs.get('unmapped_value', 'None')
            if unmapped_value != 'None':
                mapping.unmapped_value = unmapped_value
            
            # Load regions and reconstruct mappings
            regions_group = f['regions']
            for region_id in regions_group.keys():
                region_group = regions_group[region_id]
                
                # Reconstruct region data
                region_data = dict(region_group.attrs)
                for dataset_name in region_group.keys():
                    region_data[dataset_name] = region_group[dataset_name][()]
                
                region = MappingStorage._deserialize_region(region_id, region_data)
                
                # Get element mapping
                element_str = f['mappings/region_to_element'].attrs.get(region_id)
                if element_str:
                    elem_type, elem_id = element_str.split(':')
                    if elem_type == 'node':
                        try:
                            elem_id = int(elem_id)
                        except:
                            pass
                        mapping.add_node_region(region, elem_id)
                    elif elem_type == 'edge':
                        parts = elem_id.split('_')
                        if len(parts) == 2:
                            edge = (parts[0], parts[1])
                            try:
                                edge = (int(parts[0]), int(parts[1]))
                            except:
                                pass
                            mapping.add_edge_region(region, edge)
        
        return mapping
    
    @staticmethod
    def _save_csv(mapping: SpatialMapping, filepath: Path) -> bool:
        """Save mapping summary in CSV format (limited information)."""
        rows = []
        
        # Add headers
        headers = ['region_id', 'element_type', 'element_id', 'region_type', 
                  'center_x', 'center_y', 'area']
        
        for region_id, region in mapping._regions.items():
            element = mapping._region_to_element.get(region_id)
            if element:
                elem_type, elem_id = element
                if elem_type == 'edge':
                    elem_id = f"{elem_id[0]}-{elem_id[1]}"
                
                center = region.get_center()
                area = region.get_area()
                
                row = {
                    'region_id': region_id,
                    'element_type': elem_type,
                    'element_id': str(elem_id),
                    'region_type': type(region).__name__,
                    'center_x': center.x,
                    'center_y': center.y,
                    'area': area
                }
                rows.append(row)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        
        return True
    
    @staticmethod
    def _load_csv(filepath: Path, graph: Optional[GraphStructure]) -> SpatialMapping:
        """Load mapping from CSV format (limited reconstruction)."""
        print("Warning: CSV format has limited information. Some data may be lost.")
        
        mapping = SpatialMapping(graph)
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CSV only stores basic info, cannot fully reconstruct regions
                print(f"CSV row: {row['region_id']} -> {row['element_type']} {row['element_id']}")
        
        return mapping
    
    @staticmethod
    def _serialize_region(region: SpatialRegion) -> Dict[str, Any]:
        """Serialize a region to dictionary."""
        data = {
            'type': type(region).__name__,
            'metadata': region.metadata
        }
        
        if isinstance(region, ContourRegion):
            data['contour_points'] = region.contour_points
        elif isinstance(region, RectangleRegion):
            data['x'] = region.x
            data['y'] = region.y
            data['width'] = region.width
            data['height'] = region.height
        elif isinstance(region, CircleRegion):
            data['center_x'] = region.center_x
            data['center_y'] = region.center_y
            data['radius'] = region.radius
        elif isinstance(region, GridCell):
            data['row'] = region.row
            data['col'] = region.col
            data['cell_width'] = region.cell_width
            data['cell_height'] = region.cell_height
            data['grid_origin'] = region.grid_origin
        elif isinstance(region, HexagonalCell):
            data['center_x'] = region.center_x
            data['center_y'] = region.center_y
            data['radius'] = region.radius
        elif isinstance(region, EllipseRegion):
            data['center_x'] = region.center_x
            data['center_y'] = region.center_y
            data['semi_major'] = region.semi_major
            data['semi_minor'] = region.semi_minor
            data['rotation'] = region.rotation
        
        return data
    
    @staticmethod
    def _deserialize_region(region_id: str, data: Dict[str, Any]) -> SpatialRegion:
        """Deserialize a region from dictionary."""
        region_type = data.get('type', 'RectangleRegion')
        metadata = data.get('metadata', {})
        
        if region_type == 'ContourRegion':
            return ContourRegion(region_id, data['contour_points'], metadata)
        elif region_type == 'RectangleRegion':
            return RectangleRegion(region_id, data['x'], data['y'], 
                                 data['width'], data['height'], metadata)
        elif region_type == 'CircleRegion':
            return CircleRegion(region_id, data['center_x'], data['center_y'],
                              data['radius'], metadata)
        elif region_type == 'GridCell':
            return GridCell(region_id, data['row'], data['col'],
                          data['cell_width'], data['cell_height'],
                          data['grid_origin'], metadata)
        elif region_type == 'HexagonalCell':
            return HexagonalCell(region_id, data['center_x'], data['center_y'],
                               data['radius'], metadata)
        elif region_type == 'EllipseRegion':
            return EllipseRegion(region_id, data['center_x'], data['center_y'],
                               data['semi_major'], data['semi_minor'],
                               data.get('rotation', 0), metadata)
        else:
            # Default to rectangle
            return RectangleRegion(region_id, 0, 0, 10, 10, metadata)
    
    @staticmethod
    def export_mapping_report(mapping: SpatialMapping, filepath: Union[str, Path],
                             format: str = 'txt') -> bool:
        """Export a human-readable report of the mapping.
        
        Args:
            mapping: SpatialMapping to report on
            filepath: Output file path
            format: Report format ('txt', 'html', 'md')
            
        Returns:
            True if export successful
        """
        filepath = Path(filepath)
        stats = mapping.validate_mapping()
        
        try:
            if format == 'txt':
                with open(filepath, 'w') as f:
                    f.write("NAVIGRAPH MAPPING REPORT\n")
                    f.write("="*50 + "\n\n")
                    f.write(str(stats) + "\n\n")
                    
                    if stats.node_conflicts:
                        f.write("NODE CONFLICTS:\n")
                        f.write("-"*30 + "\n")
                        for conflict in stats.node_conflicts:
                            f.write(f"  • {conflict}\n")
                        f.write("\n")
                    
                    if stats.overlaps:
                        f.write("OVERLAPPING REGIONS:\n")
                        f.write("-"*30 + "\n")
                        for overlap in stats.overlaps:
                            f.write(f"  • {overlap}\n")
                        f.write("\n")
                    
                    f.write("DETAILED MAPPINGS:\n")
                    f.write("-"*30 + "\n")
                    
                    # Node mappings
                    f.write("\nNodes:\n")
                    for node_id, region_ids in mapping._node_to_regions.items():
                        f.write(f"  Node {node_id}: {len(region_ids)} regions\n")
                    
                    # Edge mappings
                    f.write("\nEdges:\n")
                    for edge, region_ids in mapping._edge_to_regions.items():
                        f.write(f"  Edge {edge}: {len(region_ids)} regions\n")
            
            elif format == 'md':
                with open(filepath, 'w') as f:
                    f.write("# NaviGraph Mapping Report\n\n")
                    f.write("## Summary Statistics\n\n")
                    f.write("```\n")
                    f.write(str(stats))
                    f.write("\n```\n\n")
                    
                    if stats.node_conflicts:
                        f.write("## ⚠️ Node Conflicts\n\n")
                        for conflict in stats.node_conflicts:
                            f.write(f"- {conflict}\n")
                        f.write("\n")
                    
                    if stats.overlaps:
                        f.write("## ⚠️ Overlapping Regions\n\n")
                        for overlap in stats.overlaps:
                            f.write(f"- {overlap}\n")
                        f.write("\n")
            
            elif format == 'html':
                with open(filepath, 'w') as f:
                    f.write("<html><head><title>NaviGraph Mapping Report</title></head>\n")
                    f.write("<body>\n")
                    f.write("<h1>NaviGraph Mapping Report</h1>\n")
                    f.write("<h2>Summary</h2>\n")
                    f.write("<pre>" + str(stats) + "</pre>\n")
                    
                    if stats.node_conflicts:
                        f.write("<h2>Node Conflicts</h2>\n<ul>\n")
                        for conflict in stats.node_conflicts:
                            f.write(f"<li>{conflict}</li>\n")
                        f.write("</ul>\n")
                    
                    f.write("</body></html>\n")
            
            return True
            
        except Exception as e:
            print(f"Error exporting report: {e}")
            return False