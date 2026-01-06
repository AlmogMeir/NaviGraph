# Dual-Rooted Binary Tree Builder - Implementation Summary

## Overview

Added a new graph builder for **dual-rooted binary trees** to the NaviGraph framework. This structure consists of two binary trees connected at their root nodes, with each tree capable of having different heights.

## What Was Added

### 1. Core Builder Function
**File:** `navigraph/core/graph/builders.py`

Added the `build_dual_root_binary_tree()` function:
```python
def build_dual_root_binary_tree(left_height: int, right_height: int, 
                                left_root_id: int = 0, right_root_id: int = 1) -> GraphStructure
```

**Features:**
- Creates two independent binary trees
- Connects them at the root level with a single edge
- Supports symmetric and asymmetric configurations
- Automatic positioning for visualization
- Returns a `GraphStructure` with metadata

### 2. Builder Class (Registry-Compatible)
**File:** `navigraph/core/graph/builders/dual_root_binary_tree.py`

Created `DualRootBinaryTreeBuilder` class that:
- Inherits from `GraphBuilder` base class
- Registered as `"dual_root_binary_tree"` in the builder registry
- Auto-discovered by the NaviGraph plugin system
- Includes custom visualization with color-coded nodes

**Key Methods:**
- `build_graph()`: Constructs the dual tree structure
- `get_visualization()`: Generates colored visualization (blue=left, coral=right)
- `get_metadata()`: Returns tree statistics and properties

### 3. Example Configuration
**File:** `examples/dual_root_tree/config_dual_root.yaml`

Complete working configuration demonstrating:
- Builder configuration syntax
- Plugin setup for pose tracking and graph location
- Analysis metrics specific to dual-tree structures
- Visualization layers

### 4. Test Suite
**File:** `examples/dual_root_tree/test_dual_root_builder.py`

Comprehensive test script that:
- Validates builder registration
- Tests symmetric trees (3, 3)
- Tests asymmetric trees (4, 2)
- Tests edge cases (1, 1) and extreme asymmetry
- Generates visualization comparisons
- Outputs to `examples/dual_root_tree/resources/dual_tree_visualization.png`

### 5. Documentation
**File:** `examples/dual_root_tree/README.md`

Complete usage guide including:
- Structure explanation and node naming conventions
- Configuration examples
- Step-by-step usage instructions
- Programmatic API examples
- Example analyses (tree preference, root crossings, depth)
- Use cases for research applications

## Structure Details

### Node Naming Convention

**Root Nodes:**
- `L0`: Left root node
- `R0`: Right root node

**Child Nodes:**
- Format: `[L|R][level][position]`
- Examples:
  - `L12`: Left tree, level 1, position 2
  - `R20`: Right tree, level 2, position 0

### Tree Properties

**Symmetric Tree (height 3, 3):**
- 14 nodes total (2 roots + 12 children)
- 13 edges
- Perfectly balanced

**Asymmetric Tree (height 4, 2):**
- 17 nodes total (2 roots + 15 children)
- 16 edges  
- Biased structure

### Edge Types

1. **root_connection**: Edge connecting L0 and R0
2. **left/right child edges**: Standard binary tree child edges
   - Marked with `child_type='left'` or `child_type='right'`

## Usage

### In Configuration Files

```yaml
graph:
  builder:
    type: dual_root_binary_tree
    params:
      left_height: 3   # Can be any value >= 1
      right_height: 3  # Independent of left_height
```

### Programmatic Usage

```python
from navigraph.core.graph.builders import get_graph_builder

# Get builder class
DualRootBuilder = get_graph_builder("dual_root_binary_tree")

# Create instance
builder = DualRootBuilder(left_height=3, right_height=3)

# Build graph
graph = builder.build_graph()

# Get metadata
metadata = builder.get_metadata()
```

### Functional Interface

```python
from navigraph.core.graph.builders import build_dual_root_binary_tree

structure = build_dual_root_binary_tree(
    left_height=4,
    right_height=2,
    left_root_id='A',  # Optional custom IDs
    right_root_id='B'
)
```

## Integration with NaviGraph

The dual-rooted binary tree integrates seamlessly with all NaviGraph features:

1. **Interactive GUI Setup**: Use `navigraph setup graph` to map nodes to your arena
2. **Graph Location Plugin**: Track animal position in the dual-tree structure  
3. **Analysis Metrics**: Measure tree preference, root crossings, depth exploration
4. **Visualization**: Overlay graph structure on video with active node highlighting
5. **Cross-Session Analysis**: Compare behavior across multiple experimental sessions

## Research Applications

1. **Choice Paradigms**: Left vs. right choice with complex branching
2. **T-Maze Variants**: Traditional T-maze extended with hierarchical branches
3. **Bidirectional Navigation**: Movement between two connected spatial regions
4. **Asymmetric Exploration**: Study behavior in balanced vs. biased environments
5. **Learning Studies**: Track exploration depth in rewarded vs. unrewarded sides
6. **Decision Making**: Hierarchical choice trees for multi-step decisions

## Testing

To test the implementation (requires dependencies installed):

```bash
# Run test suite
python examples/dual_root_tree/test_dual_root_builder.py

# Or with uv
uv run python examples/dual_root_tree/test_dual_root_builder.py
```

## Files Modified/Created

**Modified:**
- `navigraph/core/graph/builders.py`: Added `build_dual_root_binary_tree()` function

**Created:**
- `navigraph/core/graph/builders/dual_root_binary_tree.py`: Builder class
- `examples/dual_root_tree/config_dual_root.yaml`: Example configuration
- `examples/dual_root_tree/test_dual_root_builder.py`: Test script
- `examples/dual_root_tree/README.md`: Usage documentation
- `examples/dual_root_tree/IMPLEMENTATION.md`: This file

## Auto-Discovery

The new builder is automatically discovered and registered when NaviGraph imports the builders module. No manual registration needed - just import and use!

## Next Steps for Users

1. **Install dependencies** (if not already done):
   ```bash
   ./install.sh  # Linux/Mac
   install.bat   # Windows
   ```

2. **Test the builder**:
   ```bash
   python examples/dual_root_tree/test_dual_root_builder.py
   ```

3. **Create your experiment**:
   - Copy `examples/dual_root_tree` folder
   - Customize `config_dual_root.yaml` for your needs
   - Add your maze image to `resources/maze_map.png`
   - Run interactive setup to map nodes
   - Process your data!

## Flexibility

The builder supports:
- ✅ Symmetric trees (same height on both sides)
- ✅ Asymmetric trees (different heights)
- ✅ Minimal trees (height 1 = just root nodes)
- ✅ Large trees (tested up to height 5+)
- ✅ Custom root IDs
- ✅ Custom node attributes
- ✅ Custom edge attributes
- ✅ Integration with all NaviGraph plugins

## Compatibility

Works with all existing NaviGraph features:
- ✅ Interactive graph setup GUI
- ✅ Pose tracking plugins
- ✅ Neural activity analysis
- ✅ Video visualization
- ✅ Cross-session analysis
- ✅ Custom metrics
- ✅ File export (GraphML, GEXF, GML)
