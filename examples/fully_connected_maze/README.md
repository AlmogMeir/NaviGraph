# Fully Connected Maze Example

This example demonstrates the **FullyConnectedGraphBuilder**, which creates a complete graph where every node is connected to every other node.

## Graph Structure

- **Nodes**: 18 nodes (0-17) arranged in a perfect circle
- **Connectivity**: Every node is connected to every other node (complete graph)
- **No Center**: Unlike the star graph, there is no central node - all nodes are on the perimeter
- **Total Edges**: 153 edges (n*(n-1)/2 where n=18)

## Key Features

- **Complete Connectivity**: Animals can move directly between any two locations
- **Circular Layout**: Nodes positioned in a perfect circle for clear visualization
- **Dense Edge Display**: Uses lighter edge colors and thinner lines for visual clarity
- **Reusable Map**: Uses the same circular maze map as the star example

## Usage

### Setup Graph Mapping
```bash
# Navigate to the example directory
cd examples/fully_connected_maze

# Launch the interactive setup GUI
navigraph setup graph config_fully_connected.yaml
```

### Test Graph Structure
```bash
# Test the graph mapping (opens visualization)
navigraph test graph config_fully_connected.yaml
```

## Configuration

The example uses a single parameter:
- `n_nodes`: Number of nodes in the graph (default: 18)

To create a different sized fully connected graph, modify the configuration:

```yaml
graph_structure:
  type: fully_connected
  parameters:
    n_nodes: 12  # Creates 12-node complete graph
```

## Graph Properties

For a fully connected graph with n nodes:
- **Nodes**: n
- **Edges**: n × (n-1) / 2
- **Diameter**: 1 (maximum shortest path between any two nodes)
- **Clustering Coefficient**: 1.0 (perfect clustering)
- **Degree**: Every node has degree n-1

## Visualization

The builder creates a circular layout where:
- All nodes are evenly spaced around a circle
- All possible edges are drawn between nodes
- Edge colors are lightened to improve readability with dense connectivity
- Node labels show the node IDs (0 to n-1)

This creates a "web-like" appearance with nodes on the perimeter and edges forming a dense interconnected pattern in the center.