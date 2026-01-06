# Dual-Rooted Binary Tree - Quick Reference

## Configuration

```yaml
graph:
  builder:
    type: dual_root_binary_tree
    params:
      left_height: 3   # Required: 1 or greater
      right_height: 3  # Required: 1 or greater
```

## Node IDs

- **Roots**: `L0`, `R0`
- **Children**: `[L|R][level][position]`
  - Example: `L12` = Left tree, level 1, position 2

## Key Commands

```bash
# Setup spatial mapping
navigraph setup graph config.yaml

# Test mapping
navigraph test graph config.yaml

# Run analysis
navigraph run config.yaml
```

## Programmatic Usage

```python
from navigraph.core.graph.builders import get_graph_builder

# Get builder
Builder = get_graph_builder("dual_root_binary_tree")

# Create tree
builder = Builder(left_height=3, right_height=3)
graph = builder.build_graph()

# Get info
metadata = builder.get_metadata()
print(f"Nodes: {metadata['total_nodes']}")
print(f"Edges: {metadata['total_edges']}")
```

## Example Metrics

```yaml
analyze:
  metrics:
    # Track tree preference
    - name: left_tree_visits
      type: node_visit_count
      filter: {tree: left}
    
    # Count root crossings
    - name: root_crossings
      type: edge_transitions
      filter: {edge_type: root_connection}
    
    # Measure exploration depth
    - name: max_depth
      type: max_level
```

## Node Properties

Each node has these attributes:
- `level`: Depth in tree (0 = root)
- `tree`: 'left' or 'right'
- `position_in_level`: Position within level
- `pos`: (x, y) coordinates for visualization

## Edge Properties

- **Root connection**: Connects L0 and R0
  - Attribute: `edge_type='root_connection'`
- **Parent-child edges**: Binary tree edges
  - Attribute: `child_type='left'` or `child_type='right'`

## File Locations

- **Builder Class**: `navigraph/core/graph/builders/dual_root_binary_tree.py`
- **Function**: `navigraph/core/graph/builders.py` → `build_dual_root_binary_tree()`
- **Example Config**: `examples/dual_root_tree/config_dual_root.yaml`
- **Test Script**: `examples/dual_root_tree/test_dual_root_builder.py`
- **Documentation**: `examples/dual_root_tree/README.md`

## Tree Sizes

| Config | Nodes | Edges | Description |
|--------|-------|-------|-------------|
| (1, 1) | 2 | 1 | Just roots |
| (2, 2) | 6 | 5 | Minimal symmetric |
| (3, 3) | 14 | 13 | Small symmetric |
| (4, 4) | 30 | 29 | Medium symmetric |
| (4, 2) | 17 | 16 | Asymmetric |
| (5, 1) | 32 | 31 | Very asymmetric |

Formula: nodes = 2^(h) - 1 for each tree, plus connection

## Common Patterns

### Symmetric Exploration
```yaml
params:
  left_height: 3
  right_height: 3  # Same as left
```

### Biased/Rewarded Side
```yaml
params:
  left_height: 4   # Deeper exploration
  right_height: 2  # Less exploration
```

### Simple Choice
```yaml
params:
  left_height: 2   # Minimal branches
  right_height: 2
```

## Visualization Colors

- **Light Blue**: Left tree nodes
- **Light Coral**: Right tree nodes  
- **Light Green**: Root nodes (if using custom coloring)

## Integration

Works with all NaviGraph features:
- ✅ Pose tracking (DLC, custom)
- ✅ Neural activity (Minian, zarr)
- ✅ Head direction (quaternions, IMU)
- ✅ Custom plugins
- ✅ Video visualization
- ✅ Cross-session analysis
- ✅ Interactive GUI setup
