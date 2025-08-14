# NaviGraph Graph Architecture Refactoring Plan

## Overview
Transform NaviGraph from a binary tree-specific system to a generic NetworkX graph framework with pluggable builders and visualizations, with self-contained mapping files.

**Key Principles:**
- **Self-Contained Mappings**: Mapping files include graph builder info for automatic reconstruction
- **Simple Data Format**: Store only x,y points, not complex objects
- **No Backward Compatibility**: Clean break from old system
- **Test What Ships**: Test mode uses actual SpatialMapping, not parallel structures
- **Consistent Edge Representation**: Use "node1_node2" string format for edges

---

## ✅ Phase 1: Core Graph Builder Architecture - COMPLETED

**Summary of Changes:**
- Created abstract `GraphBuilder` base class with `build_graph()` and `get_visualization()` methods
- Implemented decorator-based registry system for graph builders
- Created `BinaryTreeBuilder` with automatic parameter validation via constructor
- Added builder metadata extraction using Python introspection
- Integrated with `GraphStructure` wrapper class

**Files Created/Modified:**
- `navigraph/core/graph/builders/base.py` - Abstract base class
- `navigraph/core/graph/builders/registry.py` - Registry system
- `navigraph/core/graph/builders/binary_tree.py` - Binary tree implementation
- `navigraph/core/graph/structures.py` - GraphStructure wrapper

---

## ✅ Phase 2: GUI Builder Integration - COMPLETED

**Summary of Changes:**
- Added dynamic parameter discovery for builders in GUI
- Implemented automatic UI generation based on builder constructor parameters
- Created builder selection dropdown and configuration panels
- Added edge width support in graph visualization

**Files Modified:**
- `navigraph/core/graph/setup_gui_qt.py` - Added builder configuration UI
- `navigraph/core/graph/builders/binary_tree.py` - Added edge width parameter

---

## ✅ Phase 3: Edge Ordering Fix - COMPLETED

**Summary of Changes:**
- Fixed edge tuple ordering mismatch between NetworkX and saved mappings
- Added bidirectional edge lookup for undirected graphs
- Fixed contour display in test mode for mapped edges
- Documented issue in EDGE_ORDERING_ISSUE.md

**Files Modified:**
- `navigraph/core/graph/setup_gui_qt.py` - Added bidirectional string lookup
- `navigraph/core/graph/mapping.py` - Added bidirectional edge region lookup
- Created `EDGE_ORDERING_ISSUE.md` documentation

---

## 🚧 Phase 4: Self-Contained Mapping System

### Goals
1. Make mapping files self-contained with graph builder information
2. Store simple x,y point lists instead of complex objects
3. Remove standardized_contours (test mode should use real mapping)
4. Ensure consistent edge representation across the system

### 4.1 Enhanced Mapping Format

**New mapping file structure:**
```python
{
    'format_version': '3.0',
    'graph_builder': {
        'type': 'binary_tree',     # Registry name
        'config': {'height': 7}     # Builder parameters
    },
    'mappings': {
        'nodes': {
            0: [[x1,y1], [x2,y2], ...],     # List of contour points
            10: [[x1,y1], [x2,y2], ...],
            # ...
        },
        'edges': {
            '10_20': [[x1,y1], [x2,y2], ...],  # Consistent string format
            '20_30': [[x1,y1], [x2,y2], ...],
            # ...
        }
    },
    'setup_mode': {
        'mode': 'grid',  # or 'manual'
        'grid_config': {
            'structure_type': 'rectangle',
            'rows': 8,
            'cols': 8,
            'cell_width': 50.0,
            'cell_height': 50.0,
            'origin_x': 100,
            'origin_y': 100
        },
        'element_queue_index': 45,  # Where we left off in grid mode
        'manual_state': {
            'current_element': 'edge_10_20',
            'completed_elements': ['node_0', 'node_10', ...]
        }
    },
    'metadata': {
        'created_at': '2024-01-01T00:00:00',
        'map_image_hash': 'abc123...',
        'total_nodes': 127,
        'total_edges': 126
    }
}
```

### 4.2 Conflict Resolution Strategies

When a pixel falls within multiple regions (node+edge, 2 nodes, 2 edges), we need a strategy to decide what to return.

**Registry-Based Strategy System:**

**File**: `navigraph/core/graph/conflict_resolvers.py`

```python
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
            return name_or_callable
        elif isinstance(name_or_callable, str):
            if name_or_callable not in cls._strategies:
                available = list(cls._strategies.keys())
                raise ValueError(f"Unknown strategy '{name_or_callable}'. Available: {available}")
            return cls._strategies[name_or_callable]
        else:
            raise ValueError(f"Strategy must be a string name or callable")
    
    @classmethod
    def list_strategies(cls):
        """List available strategies for documentation."""
        return list(cls._strategies.keys())

# Built-in strategies
@ConflictResolvers.register("node_priority")
def node_priority(nodes, edges, point):
    """Always prefer nodes over edges (DEFAULT)."""
    return (nodes[0][0], None) if nodes else ((None, edges[0][0]) if edges else (None, None))

@ConflictResolvers.register("edge_priority") 
def edge_priority(nodes, edges, point):
    """Always prefer edges over nodes."""
    return (None, edges[0][0]) if edges else ((nodes[0][0], None) if nodes else (None, None))

@ConflictResolvers.register("smallest_region")
def smallest_region(nodes, edges, point):
    """Choose element with smallest region area."""
    all_matches = [(n, r, 'node') for n, r in nodes] + [(e, r, 'edge') for e, r in edges]
    if all_matches:
        smallest = min(all_matches, key=lambda x: x[1].area())
        return (smallest[0], None) if smallest[2] == 'node' else (None, smallest[0])
    return (None, None)

@ConflictResolvers.register("largest_region")
def largest_region(nodes, edges, point):
    """Choose element with largest region area."""
    all_matches = [(n, r, 'node') for n, r in nodes] + [(e, r, 'edge') for e, r in edges]
    if all_matches:
        largest = max(all_matches, key=lambda x: x[1].area())
        return (largest[0], None) if largest[2] == 'node' else (None, largest[0])
    return (None, None)

@ConflictResolvers.register("nearest_center")
def nearest_center(nodes, edges, point):
    """Choose element whose region center is nearest to the point."""
    all_matches = [(n, r, 'node') for n, r in nodes] + [(e, r, 'edge') for e, r in edges]
    if all_matches:
        pt = np.array(point)
        nearest = min(all_matches, key=lambda x: np.linalg.norm(x[1].center() - pt))
        return (nearest[0], None) if nearest[2] == 'node' else (None, nearest[0])
    return (None, None)

@ConflictResolvers.register("first_found")
def first_found(nodes, edges, point):
    """Return first match found (fastest performance)."""
    return (nodes[0][0], None) if nodes else ((None, edges[0][0]) if edges else (None, None))

@ConflictResolvers.register("raise_error")
def raise_error(nodes, edges, point):
    """Raise exception on conflicts - forces explicit region mapping."""
    if len(nodes) + len(edges) > 1:
        raise ValueError(f"Point {point} matches multiple regions: "
                       f"nodes={[n for n,_ in nodes]}, edges={[e for e,_ in edges]}")
    return (nodes[0][0], None) if nodes else ((None, edges[0][0]) if edges else (None, None))
```

**Configuration (Primary Method):**
```yaml
# config.yaml  
shared_resources:
  - name: graph_provider
    enable: true
    config:
      graph_path: "resources/graph_mapping.pkl"
      conflict_strategy: "node_priority"  # OPTIONAL, defaults to node_priority
      # Available: node_priority, edge_priority, smallest_region, largest_region,
      #           nearest_center, first_found, raise_error
```

**CLI Commands:**
```bash
# List available conflict resolvers
navigraph list-conflict-resolvers

# Override config strategy via CLI (temporary override)
navigraph run config.yaml --conflict-strategy edge_priority
```

### 4.3 New CLI Command for Listing Conflict Resolvers

**File**: `navigraph/cli/main.py`

```python
@cli.command('list-conflict-resolvers')
@click.option('--format', '-f', 
              type=click.Choice(['table', 'json', 'simple']),
              default='table',
              help='Output format')
def list_conflict_resolvers(format: str):
    """List all available conflict resolution strategies.
    
    Display registered conflict resolvers that can be used in spatial mapping
    when pixels fall within multiple regions.
    
    Examples:
      navigraph list-conflict-resolvers
      navigraph list-conflict-resolvers --format json
    """
    try:
        from ..core.graph.conflict_resolvers import ConflictResolvers
        
        strategies = ConflictResolvers._strategies
        
        if not strategies:
            click.echo("⚠️  No conflict resolvers found")
            return
        
        if format == 'simple':
            for name in strategies.keys():
                click.echo(name)
        
        elif format == 'json':
            import json
            output = {
                'available_resolvers': [
                    {
                        'name': name,
                        'description': func.__doc__.strip() if func.__doc__ else ''
                    }
                    for name, func in strategies.items()
                ],
                'total_count': len(strategies),
                'default': 'node_priority'
            }
            click.echo(json.dumps(output, indent=2))
        
        else:  # table format
            click.echo()
            click.echo("🔀 Available Conflict Resolution Strategies")
            click.echo("=" * 55)
            
            for name, func in strategies.items():
                click.echo(f"\n🔹 {name}")
                if func.__doc__:
                    doc = func.__doc__.strip().split('\n')[0]
                    click.echo(f"   {doc}")
            
            click.echo()
            click.echo(f"Total: {len(strategies)} strategies available")
            click.echo(f"Default: node_priority")
            click.echo()
            click.echo("Usage in config:")
            click.echo("  shared_resources:")
            click.echo("    - name: graph_provider")
            click.echo("      config:")
            click.echo("        conflict_strategy: <resolver_name>")
            click.echo()
            click.echo("CLI override:")
            click.echo("  navigraph run config.yaml --conflict-strategy <resolver_name>")
            
    except Exception as e:
        click.echo(f"❌ Failed to list conflict resolvers: {str(e)}", err=True)
        sys.exit(1)
```

### 4.4 SpatialMapping Enhancements

**File**: `navigraph/core/graph/mapping.py`

Updated SpatialMapping with conflict resolution and new methods:

```python
class SpatialMapping:
    def to_simple_format(self) -> Dict[str, Any]:
        """Convert mapping to simple x,y point format.
        
        Returns:
            Dictionary with graph builder info and point lists
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
                if points:
                    contours.append(points)
            if contours:
                simple_mapping['nodes'][node_id] = contours
        
        # Convert edge regions to point lists with consistent format
        for edge in self.get_mapped_edges():
            regions = self.get_edge_regions(edge)
            contours = []
            for region in regions:
                points = self._extract_contour_points(region)
                if points:
                    contours.append(points)
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
        # Clear existing mappings
        self.clear_all_mappings()
        
        # Load node mappings
        for node_id, contour_lists in simple_mapping.get('nodes', {}).items():
            # Convert string back to appropriate type if needed
            node_id = self._parse_node_id(node_id)
            
            for contour_points in contour_lists:
                region = ContourRegion(
                    region_id=f"node_{node_id}_region_{len(self._regions)}",
                    contour_points=np.array(contour_points)
                )
                self.add_node_region(region, node_id)
        
        # Load edge mappings
        for edge_str, contour_lists in simple_mapping.get('edges', {}).items():
            # Parse edge string to tuple
            edge = self._parse_edge_string(edge_str)
            
            for contour_points in contour_lists:
                region = ContourRegion(
                    region_id=f"edge_{edge_str}_region_{len(self._regions)}",
                    contour_points=np.array(contour_points)
                )
                self.add_edge_region(region, edge)
    
    def save_with_builder_info(self, file_path: Path):
        """Save mapping with builder information for reconstruction."""
        
        # Create complete mapping data
        data = {
            'format_version': '3.0',
            'graph_builder': {
                'type': self._get_builder_type(),
                'config': self.graph.metadata['config']
            },
            'mappings': self.to_simple_format(),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_nodes': len(self.get_mapped_nodes()),
                'total_edges': len(self.get_mapped_edges())
            }
        }
        
        # Save as JSON for readability (or pickle for compatibility)
        if file_path.suffix == '.json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
    
    @classmethod
    def load_with_builder_reconstruction(cls, file_path: Path) -> 'SpatialMapping':
        """Load mapping and reconstruct graph using builder information."""
        
        # Load data
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        
        # Check format version
        version = data.get('format_version', '1.0')
        if version < '3.0':
            raise ValueError(f"Unsupported mapping format version: {version}")
        
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
        
        return mapping
```

### 4.3 GUI Updates

**File**: `navigraph/core/graph/setup_gui_qt.py`

Remove all standardized_contours code and update save/load:

```python
def save_mapping(self):
    """Save mapping using new self-contained format."""
    if not self.mapping:
        QMessageBox.warning(self, "Warning", "No mapping to save")
        return
    
    file_path, _ = QFileDialog.getSaveFileName(
        self, "Save Mapping", "", 
        "Pickle files (*.pkl);;JSON files (*.json)"
    )
    
    if file_path:
        try:
            self.mapping.save_with_builder_info(Path(file_path))
            QMessageBox.information(self, "Success", "Mapping saved successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

def load_mapping(self):
    """Load mapping using new self-contained format."""
    file_path, _ = QFileDialog.getOpenFileName(
        self, "Load Mapping", "",
        "Mapping files (*.pkl *.json)"
    )
    
    if file_path:
        try:
            # Load mapping with automatic graph reconstruction
            self.mapping = SpatialMapping.load_with_builder_reconstruction(Path(file_path))
            self.graph = self.mapping.graph
            
            # Update UI to reflect loaded mapping
            self._update_ui_from_mapping()
            
            QMessageBox.information(self, "Success", "Mapping loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")

def _test_mode_highlight(self, element_type: str, element_id: Any):
    """Highlight element in test mode using real mapping."""
    if element_type == 'node':
        regions = self.mapping.get_node_regions(element_id)
    else:  # edge
        regions = self.mapping.get_edge_regions(element_id)
    
    # Draw regions directly from mapping
    for region in regions:
        contour = self._extract_contour_points(region)
        self._draw_contour(contour, highlight=True)
```

### 4.4 Remove Standardized Contours

Tasks:
1. Remove all `standardized_contours` references from setup_gui_qt.py
2. Remove `_create_standardized_contour_mapping()` method
3. Update test mode to use `mapping.get_node_regions()` and `mapping.get_edge_regions()`
4. Remove backward compatibility code for old format versions

---

## 📋 Phase 5: Pipeline Integration

### Goals
1. Update pipeline to load self-contained mappings
2. Add compatibility checking between loaded mapping and config
3. Provide clear error messages for mismatches

### 5.1 Graph Provider Plugin Update

**File**: `navigraph/plugins/shared_resources/graph_provider.py`

```python
class GraphProvider(BasePlugin):
    def process(self, session_data):
        """Load graph and mapping from self-contained file."""
        
        mapping_path = self.config.get('mapping_file')
        if not mapping_path:
            raise ValueError("No mapping file specified")
        
        # Load self-contained mapping
        mapping = SpatialMapping.load_with_builder_reconstruction(Path(mapping_path))
        
        # Store in session
        session_data.shared_data['graph'] = mapping.graph
        session_data.shared_data['mapping'] = mapping
        
        return {
            'status': 'success',
            'graph_type': mapping.graph.metadata['builder_class'],
            'graph_config': mapping.graph.metadata['config']
        }
```

---

## 📋 Phase 6: Testing and Validation

### Goals
1. Ensure test mode tests actual production code
2. Verify self-contained mappings work correctly
3. Test backward compatibility handling

### Test Cases

1. **Self-contained mapping test**:
   - Create mapping with GUI
   - Save as JSON and pickle
   - Load in new session without graph
   - Verify graph reconstruction

2. **Test mode validation**:
   - Verify test mode uses real SpatialMapping
   - Check point-to-node/edge mapping
   - Validate contour display

3. **Edge consistency test**:
   - Test undirected graph edge lookup
   - Verify bidirectional edge finding
   - Check edge string format consistency

4. **Migration test**:
   - Load old format mapping
   - Get clear error message
   - Successfully recreate with new system

---

## Implementation Order

1. **First**: Implement `to_simple_format()` and `from_simple_format()` in SpatialMapping
2. **Second**: Add save/load with builder info methods
3. **Third**: Update GUI to use new save/load methods
4. **Fourth**: Remove standardized_contours completely
5. **Fifth**: Update test mode to use real mapping
6. **Sixth**: Update pipeline integration
7. **Finally**: Comprehensive testing

---

## Migration Guide for Users

### For Existing Mappings
1. Old mappings (.pkl files) are **not compatible** with the new system
2. You must recreate mappings using the updated GUI
3. New mappings are self-contained and portable

### Benefits of New System
- **Portable**: Share mapping files without separate graph configs
- **Readable**: JSON format option for human inspection
- **Reliable**: Test mode uses exact same code as production
- **Consistent**: Edge representation standardized throughout

### Configuration Changes
No changes needed - the pipeline automatically uses the self-contained mappings.