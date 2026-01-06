# Dual-Rooted Binary Tree Example

This example demonstrates the new `dual_root_binary_tree` graph builder, which creates two binary trees connected at their root nodes. This structure is particularly useful for:

- **Choice paradigms** (e.g., left vs. right choice experiments)
- **T-maze variants** with complex branch structures
- **Bidirectional navigation** experiments
- **Symmetric and asymmetric exploration** tasks

## Structure

The dual-rooted binary tree consists of:
- Two root nodes (`L0` and `R0`) connected by an edge
- Independent binary trees extending from each root
- Flexible heights for each tree (can be symmetric or asymmetric)

### Node Naming Convention

- **Root nodes**: `L0` (left root), `R0` (right root)
- **Child nodes**: `[L|R][level][position]`
  - `L` or `R`: indicates which tree (left or right)
  - `[level]`: depth level (1, 2, 3, ...)
  - `[position]`: position within the level (0, 1, 2, ...)
  
**Examples:**
- `L12`: Left tree, level 1, position 2
- `R20`: Right tree, level 2, position 0

## Configuration

### Basic Configuration

```yaml
graph:
  builder:
    type: dual_root_binary_tree
    params:
      left_height: 3   # Height of left tree
      right_height: 3  # Height of right tree
```

### Symmetric Tree
For balanced, symmetric structures (equal exploration on both sides):

```yaml
graph:
  builder:
    type: dual_root_binary_tree
    params:
      left_height: 4
      right_height: 4
```

### Asymmetric Tree
For biased or asymmetric structures:

```yaml
graph:
  builder:
    type: dual_root_binary_tree
    params:
      left_height: 5   # Deep left side
      right_height: 2  # Shallow right side
```

## Usage

### 1. Test the Builder

Run the test script to validate the builder and visualize different configurations:

```bash
# Using UV (recommended)
uv run python examples/dual_root_tree/test_dual_root_builder.py

# Or with activated environment
python examples/dual_root_tree/test_dual_root_builder.py
```

This will:
- Verify the builder is registered
- Test symmetric and asymmetric configurations
- Generate visualizations
- Validate edge cases

### 2. Setup Graph Mapping

Use the interactive GUI to map your experimental arena to the dual-tree structure:

```bash
# Create resources directory
mkdir -p examples/dual_root_tree/resources

# Copy your maze image to resources/maze_map.png

# Launch interactive graph setup
uv run navigraph setup graph examples/dual_root_tree/config_dual_root.yaml
```

**GUI Controls:**
- **V key**: Place vertices (nodes)
- **E key**: Create edges
- **S key**: Save mapping
- **Mouse wheel**: Zoom
- **Click and drag**: Pan

### 3. Run Analysis

Once you have your data and mapping configured:

```bash
uv run navigraph run examples/dual_root_tree/config_dual_root.yaml
```

## Programmatic Usage

### Creating a Dual-Rooted Tree

```python
from navigraph.core.graph.builders import get_graph_builder

# Get the builder class
DualRootBuilder = get_graph_builder("dual_root_binary_tree")

# Create symmetric tree
symmetric_tree = DualRootBuilder(left_height=3, right_height=3)
graph = symmetric_tree.build_graph()

# Create asymmetric tree
asymmetric_tree = DualRootBuilder(left_height=4, right_height=2)
graph = asymmetric_tree.build_graph()

# Get metadata
metadata = symmetric_tree.get_metadata()
print(f"Total nodes: {metadata['total_nodes']}")
print(f"Left nodes: {metadata['left_nodes']}")
print(f"Right nodes: {metadata['right_nodes']}")
```

### Using the Functional Interface

```python
from navigraph.core.graph.builders import build_dual_root_binary_tree

# Create tree structure
structure = build_dual_root_binary_tree(
    left_height=3,
    right_height=3,
    left_root_id=0,   # Optional: custom root IDs
    right_root_id=1
)

# Access the graph
graph = structure.graph

# Access node positions
positions = structure.get_node_positions()
```

## Example Analyses

### Track Tree Preference

```yaml
analyze:
  metrics:
    - name: left_tree_time
      type: custom
      filter_nodes:
        tree: left
    
    - name: right_tree_time
      type: custom
      filter_nodes:
        tree: right
    
    - name: preference_ratio
      type: computed
      formula: "left_tree_time / right_tree_time"
```

### Count Root Crossings

```yaml
analyze:
  metrics:
    - name: root_crossings
      type: edge_transitions
      filter_edges:
        edge_type: root_connection
```

### Depth Analysis

```yaml
analyze:
  metrics:
    - name: max_depth_left
      type: max_level
      filter_nodes:
        tree: left
    
    - name: max_depth_right
      type: max_level
      filter_nodes:
        tree: right
```

## Visualization

The builder includes built-in visualization that color-codes nodes by tree:
- **Light blue**: Left tree nodes
- **Light coral**: Right tree nodes
- **Light green**: Root nodes

## Properties

### Symmetric Tree (height 3, 3)
- Total nodes: 14 (2 roots + 6 left + 6 right)
- Total edges: 13
- Perfectly balanced structure

### Asymmetric Tree (height 4, 2)
- Total nodes: 17 (2 roots + 14 left + 1 right)
- Total edges: 16
- Biased toward left exploration

### Minimal Tree (height 1, 1)
- Total nodes: 2 (just the two roots)
- Total edges: 1 (connecting the roots)
- No child nodes

## Use Cases

1. **T-Maze with Branches**: Map each arm to a tree, allow complex branching
2. **Choice Paradigm**: Study preference between two hierarchical options
3. **Bidirectional Navigation**: Animal can traverse between two spatial regions
4. **Asymmetric Exploration**: Test behavior in balanced vs. biased environments
5. **Learning Paradigms**: Track depth of exploration in rewarded vs. unrewarded sides

## Files

- `config_dual_root.yaml`: Example configuration file
- `test_dual_root_builder.py`: Test script with visualizations
- `README.md`: This file
- `resources/`: Directory for maze images and mappings (create as needed)

## Next Steps

1. **Add your data**: Place pose tracking files in a `data/` or `sessions/` directory
2. **Customize configuration**: Adjust heights, metrics, and visualizations
3. **Map your arena**: Use the interactive GUI to create spatial mappings
4. **Run analysis**: Process your experiments with the dual-tree structure
5. **Extend**: Add custom metrics specific to your research questions
