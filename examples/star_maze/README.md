# Star Maze Example

This example demonstrates the custom `StarGraphBuilder` with a circular maze layout.

## Graph Structure

The star graph consists of:
- **1 central node** (node 0) - represents the center of the circular maze
- **18 peripheral nodes** (nodes 1-18) - represent the 18 holes around the perimeter
- **18 edges** - each peripheral node connects to the center node
- **Optional peripheral ring** - adjacent peripheral nodes can be connected

## Configuration Options

### Basic Star (default)
```yaml
graph_structure:
  type: star
  parameters:
    n_peripheral_nodes: 18
    # connect_periphery: false  # default
```

### Star with Peripheral Ring
```yaml
graph_structure:
  type: star
  parameters:
    n_peripheral_nodes: 18
    connect_periphery: true  # adds ring of connections between adjacent peripheral nodes
```

## Custom Visualization

The `StarGraphBuilder` includes custom visualization:
- **Center node**: Extra large size (1200), light blue color
- **Peripheral nodes**: Large size (600), same light blue color, arranged in perfect circle
- **Layout**: Polar coordinates with nodes evenly spaced at 20° intervals

## Usage

### Basic Star (default - no peripheral connections)
```bash
poetry run navigraph setup graph examples/star_maze/config_star.yaml
```

### Star with Peripheral Ring
```bash  
poetry run navigraph setup graph examples/star_maze/config_star_connected.yaml
```

## Expected Behavior

- **Left panel**: Shows the circular maze image with center region and 18 holes
- **Right panel**: Shows star-shaped graph with prominent center node
- **Mapping**: Map the center region to node 0, and each hole to nodes 1-18

## Graph Properties

- **Total nodes**: 19 (1 center + 18 peripheral)
- **Total edges**: 18 (star topology)
- **Diameter**: 2 (all peripheral nodes are 2 hops apart via center)
- **Connectivity**: All paths go through the center node

This structure is perfect for behavioral experiments where animals start at the center and choose between peripheral locations.