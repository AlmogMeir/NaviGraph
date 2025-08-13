# NaviGraph

A flexible framework for spatial navigation analysis with pluggable graph architecture.

## Installation

```bash
poetry install
```

## Quick Start

```bash
# Setup graph mapping interactively
navigraph setup graph config.yaml

# Test graph mapping
navigraph test graph config.yaml

# Run full analysis
navigraph run config.yaml

# List available graph builders
navigraph list-graph-builders
```

## Graph Architecture

NaviGraph uses a flexible, pluggable graph builder system that allows you to create and use different graph structures for your spatial navigation analysis.

### Using Built-in Graph Builders

NaviGraph comes with several built-in graph builders:

```python
from navigraph.core.graph import GraphStructure, get_graph_builder

# Get a builder from the registry
BinaryTreeBuilder = get_graph_builder("binary_tree")

# Create builder instance with parameters
builder = BinaryTreeBuilder(height=5)

# Create graph structure
graph = GraphStructure(builder)

# Use the graph
print(f"Nodes: {graph.num_nodes}")
print(f"Edges: {graph.num_edges}")

# Get visualization
image_array = graph.get_visualization()  # Returns numpy array (H, W, 3)
```

### Creating Custom Graph Builders

You have two options for creating custom graphs:

#### Option 1: Create a Graph Builder Class (Recommended)

This is the recommended approach as it integrates seamlessly with NaviGraph's configuration system and GUI.

```python
from navigraph.core.graph.builders import GraphBuilder, register_graph_builder
import networkx as nx
import numpy as np

@register_graph_builder("my_custom_graph")
class MyCustomGraphBuilder(GraphBuilder):
    """Custom graph builder for my specific topology."""
    
    def __init__(self, size: int, connectivity: float = 0.3):
        """Initialize with custom parameters.
        
        Args:
            size: Number of nodes
            connectivity: Probability of edge creation (0-1)
        """
        if size < 1:
            raise ValueError("Size must be >= 1")
        if not 0 <= connectivity <= 1:
            raise ValueError("Connectivity must be between 0 and 1")
            
        self.size = size
        self.connectivity = connectivity
    
    def build_graph(self) -> nx.Graph:
        """Build and return the custom graph.
        
        This is the only required method to implement.
        """
        # Create your custom graph using NetworkX
        graph = nx.erdos_renyi_graph(self.size, self.connectivity, seed=42)
        
        # Optionally add node positions for visualization
        pos = nx.spring_layout(graph, seed=42)
        nx.set_node_attributes(graph, pos, 'pos')
        
        return graph
    
    # Optional: Override visualization for custom appearance
    def get_visualization(self, **kwargs) -> np.ndarray:
        """Custom visualization with special styling."""
        # You can provide your own visualization logic
        # or use the default with custom parameters
        return super().get_visualization(
            node_color='lightgreen',
            edge_color='blue',
            node_size=500,
            **kwargs
        )
```

Once registered, your custom builder is automatically available:

```python
# Use via registry
MyBuilder = get_graph_builder("my_custom_graph")
builder = MyBuilder(size=20, connectivity=0.4)

# Or use directly if imported
builder = MyCustomGraphBuilder(size=20, connectivity=0.4)

# Create graph structure
graph = GraphStructure(builder)
```

Your custom builder will also be available in configuration files:

```yaml
# config.yaml
graph_structure:
  type: my_custom_graph
  parameters:
    size: 30
    connectivity: 0.25
```

#### Option 2: Create Graph Offline and Load from File

If you prefer not to create a builder class, you can create your graph externally and load it:

```python
import networkx as nx
from navigraph.core.graph import GraphStructure, get_graph_builder

# Create your custom graph using NetworkX or any other tool
my_graph = nx.karate_club_graph()

# Save it to a file
nx.write_graphml(my_graph, "my_custom_graph.graphml")

# Load it using the file_loader builder
FileLoader = get_graph_builder("file_loader")
loader = FileLoader(filepath="my_custom_graph.graphml", format="graphml")

# Create graph structure
graph = GraphStructure(loader)
```

Supported file formats:
- `graphml` - GraphML format (recommended for preserving attributes)
- `gexf` - GEXF format
- `gml` - GML format
- `edgelist` - Simple edge list
- `adjlist` - Adjacency list

You can also configure file loading in YAML:

```yaml
# config.yaml
graph_structure:
  type: file_loader
  parameters:
    filepath: "./graphs/my_custom_graph.graphml"
    format: "graphml"
```

### Custom Visualization

The default visualization uses matplotlib to render graphs, but you can create any visualization you want:

```python
class MyArtisticGraphBuilder(GraphBuilder):
    def build_graph(self) -> nx.Graph:
        # Your graph building logic
        return nx.cycle_graph(10)
    
    def get_visualization(self, **kwargs) -> np.ndarray:
        """Create a custom artistic visualization."""
        import matplotlib.pyplot as plt
        import matplotlib
        
        # Save and restore matplotlib backend
        original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        
        try:
            graph = self.build_graph()
            
            # Create custom visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Use a circular layout
            pos = nx.circular_layout(graph)
            
            # Draw with custom artistic style
            nx.draw_networkx_nodes(
                graph, pos, ax=ax,
                node_color=range(len(graph.nodes())),
                node_size=1000,
                cmap=plt.cm.rainbow,
                alpha=0.8
            )
            
            nx.draw_networkx_edges(
                graph, pos, ax=ax,
                edge_color='gray',
                width=3,
                alpha=0.5,
                style='dashed'
            )
            
            # Add labels with custom font
            nx.draw_networkx_labels(
                graph, pos, ax=ax,
                font_size=16,
                font_family='serif',
                font_weight='bold'
            )
            
            ax.set_facecolor('#f0f0f0')
            ax.axis('off')
            
            # Convert to numpy array
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return image
            
        finally:
            matplotlib.use(original_backend)
```

### Builder Discovery and Introspection

List all available graph builders:

```python
from navigraph.core.graph.builders import list_graph_builders, get_graph_builder_info

# Get list of all registered builders
builders = list_graph_builders()
print(f"Available builders: {builders}")

# Get detailed information about a builder
info = get_graph_builder_info("binary_tree")
print(f"Builder: {info['name']}")
print(f"Class: {info['class_name']}")
print(f"Parameters: {info['parameters']}")
# Output:
# Parameters: {
#     'height': {
#         'type': <class 'int'>,
#         'required': True,
#         'default': None
#     }
# }
```

### Integration with NaviGraph Pipeline

Once you've created a graph (either through a custom builder or loaded from file), it integrates seamlessly with NaviGraph's spatial mapping and analysis pipeline:

```python
from navigraph.core.graph import GraphStructure, SpatialMapping
from navigraph.core.graph.setup_gui_qt import launch_setup_gui

# Create your graph
builder = MyCustomGraphBuilder(size=20)
graph = GraphStructure(builder)

# Launch GUI to create spatial mapping
mapping = launch_setup_gui(graph, map_image)

# Save the mapping (includes builder information)
mapping.save("my_mapping.pkl")

# Later, load and recreate the exact same graph
loaded_mapping = SpatialMapping.load("my_mapping.pkl")
# The mapping knows which builder was used and can recreate it
```

### Best Practices

1. **Use Builder Classes for Reusability**: If you'll use the same graph structure multiple times or want to share it, create a builder class.

2. **Use File Loading for One-off Graphs**: If you have a specific graph for a single experiment, save it to a file and use the file_loader.

3. **Validate Parameters in __init__**: Always validate parameters in the constructor - this provides automatic validation when loaded from config.

4. **Add Node Positions**: If your graph has a natural layout, add 'pos' attributes to nodes for better default visualization.

5. **Keep build_graph() Pure**: The build_graph() method should be deterministic and not have side effects.

6. **Document Parameters**: Use clear docstrings to document what each parameter does - this helps with GUI integration.

### Example: Complete Custom Builder

Here's a complete example of a custom maze graph builder:

```python
from navigraph.core.graph.builders import GraphBuilder, register_graph_builder
import networkx as nx
import numpy as np

@register_graph_builder("maze_graph")
class MazeGraphBuilder(GraphBuilder):
    """Creates a grid-based maze graph with random walls removed."""
    
    def __init__(self, width: int, height: int, wall_removal_prob: float = 0.3):
        """Initialize maze builder.
        
        Args:
            width: Width of the maze grid
            height: Height of the maze grid  
            wall_removal_prob: Probability of removing a wall (0-1)
        """
        if width < 2 or height < 2:
            raise ValueError("Maze must be at least 2x2")
        if not 0 <= wall_removal_prob <= 1:
            raise ValueError("Wall removal probability must be between 0 and 1")
            
        self.width = width
        self.height = height
        self.wall_removal_prob = wall_removal_prob
    
    def build_graph(self) -> nx.Graph:
        """Build maze graph with some walls randomly removed."""
        # Start with a grid
        graph = nx.grid_2d_graph(self.width, self.height)
        
        # Randomly remove some edges (walls)
        edges_to_remove = []
        for edge in graph.edges():
            if np.random.random() < self.wall_removal_prob:
                edges_to_remove.append(edge)
        
        graph.remove_edges_from(edges_to_remove)
        
        # Add positions for visualization
        pos = {(x, y): (x, -y) for x in range(self.width) for y in range(self.height)}
        nx.set_node_attributes(graph, pos, 'pos')
        
        # Relabel nodes to single integers
        mapping = {(x, y): y * self.width + x for x in range(self.width) for y in range(self.height)}
        graph = nx.relabel_nodes(graph, mapping)
        
        return graph
    
    def get_visualization(self, **kwargs) -> np.ndarray:
        """Visualize as a maze with grid layout."""
        kwargs.setdefault('node_size', 300)
        kwargs.setdefault('node_color', 'lightblue')
        kwargs.setdefault('edge_color', 'black')
        kwargs.setdefault('width', 3)  # Edge width
        kwargs.setdefault('figsize', (12, 12))
        
        return super().get_visualization(**kwargs)
```

Use it in your experiments:

```yaml
# config.yaml
graph_structure:
  type: maze_graph
  parameters:
    width: 10
    height: 10
    wall_removal_prob: 0.2

# The graph builder will be automatically created and used
# when you run: navigraph setup graph config.yaml
```

## Contributing

To add new graph builders to NaviGraph:

1. Create a new Python file in `navigraph/core/graph/builders/`
2. Implement your builder class inheriting from `GraphBuilder`
3. Add the `@register_graph_builder("name")` decorator
4. The builder will be automatically discovered and available

## License

MIT