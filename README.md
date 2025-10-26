<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/NaviGraph_logo_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/NaviGraph_logo_light.png">
  <img src="docs/images/NaviGraph_logo_light.png" alt="NaviGraph Logo" width="400">
</picture>

A flexible framework for multi-session behavioral experiments that enables researchers to integrate, synchronize, and analyze diverse data sources within unified spatial and temporal domains. NaviGraph provides the freedom to combine any data streams, implement custom metrics, and visualize results across time and graph-based spatial representations - all within a single, coherent analysis environment.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

<p align="left">
  <img src="docs/images/software_pipeline.png" alt="NaviGraph pipeline"/> 
</p>

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [Graph Architecture](#graph-architecture)
- [Plugin System](#plugin-system)
- [Spatial Calibration](#spatial-calibration)
- [Analysis Pipeline](#analysis-pipeline)
- [Visualization](#visualization)
- [Examples](#examples)
- [Extending NaviGraph](#extending-navigraph)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

NaviGraph solves a critical challenge in behavioral research: **integrating uncorrelated, unsynchronized data from multiple sources across multi-session experiments**. Researchers often work with diverse outputs from different tools and pipelines, making it difficult to conduct unified analysis. NaviGraph provides a single arena where you can:

### Core Capabilities

- **Universal Data Integration**: Connect any data source - pose tracking, neural recordings, physiological signals, environmental sensors, custom measurements
- **Cross-Session Synchronization**: Align and synchronize data streams across multiple experimental sessions in shared spatial environments
- **Flexible Metric Implementation**: Define custom analysis metrics that operate across any combination of your data sources
- **Multi-Domain Visualization**: Traverse and visualize data across time, spatial coordinates, and graph-based representations
- **Plugin-Based Architecture**: Complete freedom to extend the system with your specific data types and analysis needs
- **Interactive Spatial Mapping**: Map your experimental environment to graph structures for spatial analysis

### Research Freedom & Flexibility

**Multi-Repository Integration**: Work with outputs from different research tools and repositories - each producing uncorrelated data streams. NaviGraph becomes your central integration point.

**Domain Traversal**: Seamlessly move between and analyze data across three key domains:
- **Temporal Domain**: Time-series analysis, event detection, sequence patterns
- **Spatial Domain**: Coordinate-based analysis, trajectory mapping, location patterns  
- **Graph Domain**: Network topology analysis, path optimization, connectivity patterns

**Custom Analysis Pipeline**: Build analysis workflows specific to your research questions by combining any data sources with custom metrics and visualizations.

### Typical Workflow

1. **Configure Data Sources**: Define plugins for each data stream (pose tracking, neural activity, environmental sensors, custom measurements)
2. **Spatial Calibration**: Map your experimental environment to coordinate systems and graph structures
3. **Multi-Session Processing**: Process multiple experimental sessions with synchronized data alignment
4. **Custom Analysis**: Implement metrics that traverse time, space, and graph domains as needed
5. **Integrated Visualization**: Generate visualizations that combine multiple data sources and domains
6. **Cross-Session Insights**: Compare and aggregate results across experimental sessions

## Installation

### Requirements
- Python 3.9, 3.10, 3.11, or 3.12
- OpenCV, NetworkX, PyQt5, Pandas, NumPy, Matplotlib

### Quick Install

```bash
# Linux/Mac:
./install.sh

# Windows:
install.bat
```

The script automatically installs UV (if needed) and runs `uv sync` to set up everything.

### Manual Installation

If you already have UV installed:

```bash
uv sync  # Creates venv, installs dependencies, sets up CLI
```

To use a specific Python version:

```bash
uv sync --python 3.10  # Use Python 3.10
uv sync --python 3.12  # Use Python 3.12
```

## Quick Start

1. **Prepare your data**: Pose tracking files (`.h5`), optional calcium imaging (`zarr`)
2. **Create configuration**: Use example YAML configs as templates
3. **Set up spatial mapping**: Use interactive GUI to calibrate coordinates and graph
4. **Run analysis**: Execute the full pipeline

```bash
# Setup graph mapping (interactive GUI)
uv run navigraph setup graph config.yaml

# Test mapping (validation GUI)  
uv run navigraph test graph config.yaml

# Run full analysis pipeline
uv run navigraph run config.yaml
```

## CLI Usage

NaviGraph provides a command-line interface with multiple operation modes:

### Command Structure

```bash
# Option 1: UV Run (No activation needed)
uv run navigraph [command] [options]

# Option 2: Traditional activation
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate      # Windows
navigraph [command] [options]
```

### Available Commands

#### Setup Commands

```bash
# Interactive graph setup - launches PyQt5 GUI for spatial calibration
navigraph setup graph config.yaml

# Interactive calibration setup - camera-to-world coordinate transformation
navigraph setup calibration config.yaml
```

#### Testing Commands

```bash
# Test graph mapping - validates spatial mapping with visual feedback
navigraph test graph config.yaml

# Test calibration - validates camera transformation
navigraph test calibration config.yaml
```

#### Analysis Commands

```bash
# Run full analysis pipeline
navigraph run config.yaml

# Validate configuration file
navigraph validate config.yaml
```

#### Help and Information

```bash
# General help
navigraph --help

# Command-specific help
navigraph setup --help
navigraph run --help
```

### Interactive GUI Controls

When using setup or test commands, the GUI provides these controls:

**Graph Setup Mode:**
- **V key**: Vertex/node placement mode
- **E key**: Edge creation mode  
- **R key**: Region definition mode
- **S key**: Save graph mapping
- **Mouse wheel**: Zoom in/out
- **Click and drag**: Pan around image

**Calibration Mode:**
- **Click**: Place calibration points
- **Right-click**: Remove points
- **S key**: Save calibration matrix
- **R key**: Reset points

## Configuration

NaviGraph uses YAML configuration files with a structured plugin system:

### Basic Configuration Structure

```yaml
# Logging and paths
log_level: info
experiment_path: .
experiment_output_path: "{PROJECT_ROOT}/output"

# Setup configuration for GUI tools
setup:
  map_path: ./resources/maze_map.png
  spatial_image_for_calibration: ./demo_session/video.avi
  calibration_matrix: ./resources/transform_matrix.npy

# Plugin configuration
plugins:
  - name: pose_data
    type: pose_tracking
    file_pattern: '.*DLC.*\.h5$'
    config:
      bodypart: 'Nose'
      likelihood_threshold: 0.3

# Graph structure
graph:
  builder:
    type: binary_tree
    config:
      height: 7
  mapping_file: ./resources/graph_mapping.pkl
  conflict_strategy: node_priority

# Analysis metrics
analyze:
  metrics:
    time_to_reward: 
      func_name: 'time_a_to_b'
      args: 
        a: 145  # Start node
        b: 273  # Goal node
  save_as_csv: true
  save_as_pkl: true

# Visualization pipeline
visualizations:
  output:
    enabled: true
    format: mp4
    fps: 30
  pipeline:
    - name: bodypart_display
      type: bodyparts
      config:
        bodyparts: ['Nose', 'Tail_base']
        colors:
          Nose: [255, 0, 0]
          Tail_base: [0, 255, 0]
```

### Plugin System

NaviGraph's plugin architecture provides complete flexibility for researchers to integrate any data source. The system includes built-in plugins for common research scenarios, but is designed to accommodate any data type through custom plugin development.

#### Built-in Data Source Plugins

**Behavioral Tracking:**
- **pose_tracking**: DeepLabCut H5 files, custom pose estimation outputs
- **calibration**: Camera transformation matrices, coordinate calibrations

**Spatial Analysis:**
- **map_location**: Coordinate transformation to spatial maps
- **graph_location**: Mapping to graph nodes and edges for network analysis
- **stream_info**: Video metadata, temporal synchronization data

**Neural/Physiological:**
- **neural_activity**: Minian zarr files, calcium imaging data, any neural time series
- **head_direction**: Quaternion data, orientation sensors, IMU outputs

**Research Flexibility**: These are starting points - researchers commonly create custom plugins for:
- Custom sensor outputs (pressure, temperature, light, sound)
- Behavioral scoring data from different tools
- Physiological measurements (heart rate, respiration, EMG)
- Environmental monitoring data
- Custom computed features from other analysis pipelines

#### Plugin Configuration

```yaml
plugins:
  - name: pose_data           # Instance name
    type: pose_tracking       # Plugin type
    file_pattern: '.*DLC.*\.h5$'  # File matching pattern
    shared: false             # Per-session (false) or shared (true)
    config:                   # Plugin-specific configuration
      bodypart: 'Nose'
      likelihood_threshold: 0.3
      bodyparts: 'all'
```

### Creating Custom Plugins

#### Data Source Plugin

```python
from navigraph.core.navigraph_plugin import NaviGraphPlugin
from navigraph.core.registry import register_data_source_plugin
import pandas as pd

@register_data_source_plugin("custom_data")
class CustomDataPlugin(NaviGraphPlugin):
    """Custom data source plugin."""
    
    def __init__(self, config):
        super().__init__(config)
        self.custom_param = config.get('custom_param', 'default')
    
    def provide(self, shared_resources):
        """Provide shared resources (optional)."""
        # Add shared data to resources
        shared_resources['custom_resource'] = self.load_custom_resource()
    
    def augment_data(self, dataframe, shared_resources):
        """Add columns to the main dataframe."""
        # Load and process your custom data
        custom_data = self.load_custom_data()
        
        # Add new columns to dataframe
        dataframe['custom_metric'] = custom_data
        
        return dataframe
    
    def load_custom_data(self):
        """Load and process custom data files."""
        # Your custom loading logic here
        pass
```

#### Adding Custom Metrics

```python
# In your plugin's augment_data method
def augment_data(self, dataframe, shared_resources):
    # Calculate custom metric
    dataframe['speed'] = np.sqrt(
        dataframe['x'].diff()**2 + dataframe['y'].diff()**2
    )
    
    # Calculate rolling mean
    dataframe['speed_smoothed'] = dataframe['speed'].rolling(
        window=10, center=True
    ).mean()
    
    return dataframe
```

## Graph Architecture

NaviGraph supports flexible graph structures through a builder system:

### Built-in Graph Builders

- **binary_tree**: Binary tree structures with configurable height
- **fully_connected**: Complete graphs where every node connects to every other
- **star**: Star topology with central hub node
- **random**: Random graph generation with configurable connectivity
- **file_loader**: Load graphs from GraphML, GEXF, GML formats

### Using Built-in Builders

```yaml
graph:
  builder:
    type: binary_tree
    config:
      height: 7
  mapping_file: ./resources/graph_mapping.pkl
```

### Creating Custom Graph Builders

#### Option 1: Builder Class (Recommended)

```python
from navigraph.core.graph.builders import GraphBuilder, register_graph_builder
import networkx as nx
import numpy as np

@register_graph_builder("maze_graph")
class MazeGraphBuilder(GraphBuilder):
    """Grid-based maze with configurable walls."""
    
    def __init__(self, width: int, height: int, wall_removal_prob: float = 0.3):
        if width < 2 or height < 2:
            raise ValueError("Maze must be at least 2x2")
        if not 0 <= wall_removal_prob <= 1:
            raise ValueError("Wall removal probability must be between 0 and 1")
            
        self.width = width
        self.height = height
        self.wall_removal_prob = wall_removal_prob
    
    def build_graph(self) -> nx.Graph:
        """Build the maze graph."""
        # Create grid graph
        graph = nx.grid_2d_graph(self.width, self.height)
        
        # Randomly remove walls
        edges_to_remove = [
            edge for edge in graph.edges() 
            if np.random.random() < self.wall_removal_prob
        ]
        graph.remove_edges_from(edges_to_remove)
        
        # Add positions for visualization
        pos = {(x, y): (x, -y) for x in range(self.width) for y in range(self.height)}
        nx.set_node_attributes(graph, pos, 'pos')
        
        # Relabel to integer nodes
        mapping = {(x, y): y * self.width + x 
                  for x in range(self.width) for y in range(self.height)}
        return nx.relabel_nodes(graph, mapping)
```

Use in configuration:
```yaml
graph:
  builder:
    type: maze_graph
    config:
      width: 10
      height: 10
      wall_removal_prob: 0.2
```

#### Option 2: Load from File

```python
# Create graph externally
import networkx as nx
my_graph = nx.karate_club_graph()
nx.write_graphml(my_graph, "custom_graph.graphml")
```

```yaml
graph:
  builder:
    type: file_loader
    config:
      filepath: "./graphs/custom_graph.graphml"
      format: "graphml"
```

## Random Walks on Graphs

NaviGraph supports flexible random walk generation on any graph structure. Random walks are useful for analyzing graph exploration patterns, path characteristics, navigation strategies, and agent-based simulations.

### Overview

The `random_walks()` method generates multiple random walks with configurable parameters:

- **Target-directed or fixed-length walks**: Walk to a specific target or for a fixed number of steps
- **Backtracking control**: Configure the probability of returning to previous nodes
- **Weighted transitions**: Use edge weights for transition probabilities (optional)
- **Parallel processing**: Leverage multiple CPU cores for large-scale simulations
- **Statistical analysis**: Optional summary statistics and success rate reporting
- **Reproducibility**: Seed-based reproducible walks

### Basic Usage

```python
from navigraph.core.graph import GraphStructure

# Create a binary tree graph
graph = GraphStructure.from_config('binary_tree', {'height': 7})

# Generate 100 random walks of length 20 from root node
paths = graph.random_walks(
    start_node=0,  # Root of binary tree
    n_walks=100,
    max_steps=20
)

print(f"Generated {len(paths)} walks")
print(f"First walk: {paths[0]}")
```

### Target-Directed Walks with Statistics

```python
# Walk from root to a leaf node with statistics
paths, stats = graph.random_walks(
    start_node=0,
    target_node=127,  # Target leaf node
    n_walks=1000,
    max_steps=50,
    backtrack_prob=0.0,  # No backtracking
    return_stats=True
)

print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Average path length: {stats['mean_length']:.2f} steps")
print(f"Median path length: {stats['median_length']:.1f} steps")
print(f"Standard deviation: {stats['std_length']:.2f} steps")

# Compare to shortest path
shortest = graph.get_shortest_path(0, 127)
print(f"Shortest path: {len(shortest)-1} steps")
print(f"Efficiency: {(len(shortest)-1) / stats['mean_length']:.1%}")
```

### Backtracking Modes

NaviGraph supports four backtracking modes to model different navigation behaviors:

```python
# Mode 1: No backtracking (default - goal-directed navigation)
paths = graph.random_walks(
    start_node=0,
    max_steps=20,
    n_walks=1000,
    backtrack_prob=0.0  # Never return to previous node
)

# Mode 2: Uniform distribution (unbiased exploration)
# Treats previous node like any other neighbor - SIMPLEST MODE
paths = graph.random_walks(
    start_node=0,
    max_steps=20,
    n_walks=1000,
    backtrack_prob=-1  # Equal probability to ALL neighbors
)

# Mode 3: Probabilistic backtracking (modeling uncertainty)
paths = graph.random_walks(
    start_node=0,
    max_steps=20,
    n_walks=1000,
    backtrack_prob=0.3  # 30% chance to backtrack, 70% to other neighbors
)

# Mode 4: Always backtrack (testing/debugging)
paths = graph.random_walks(
    start_node=0,
    max_steps=20,
    n_walks=1000,
    backtrack_prob=1.0  # Always return if possible
)
```

**Which mode should you use?**

| Mode | `backtrack_prob` | When to Use |
|------|-----------------|-------------|
| No Backtracking | `0.0` | Goal-directed navigation, preventing immediate returns |
| **Uniform** | **`-1`** | **Unbiased exploration without manually calculating probabilities** |
| Probabilistic | `0.0-1.0` | Modeling hesitation, uncertainty, or specific behaviors |
| Always Backtrack | `1.0` | Testing, debugging, or validating graph structure |

**Example: Comparing backtracking modes**

```python
# Compare different backtracking strategies
modes = [
    ("No Backtrack", 0.0),
    ("Uniform", -1),
    ("30% Backtrack", 0.3),
    ("50% Backtrack", 0.5),
]

for name, prob in modes:
    paths, stats = graph.random_walks(
        start_node=0,
        target_node=127,
        n_walks=1000,
        max_steps=50,
        backtrack_prob=prob,
        return_stats=True,
        seed=42
    )
    print(f"{name:15} | Success: {stats['success_rate']:.1%} | Avg: {stats['mean_length']:.1f} steps")
```

### Parallel Processing

```python
# Use multiprocessing for large-scale simulations
paths = graph.random_walks(
    start_node=0,
    n_walks=100000,  # Many walks
    max_steps=30,
    n_jobs=-1,  # Use all CPU cores
    seed=42     # Still reproducible with parallel execution
)

# Or specify exact number of processes
paths = graph.random_walks(
    start_node=0,
    n_walks=10000,
    max_steps=20,
    n_jobs=4  # Use 4 processes
)
```

**Performance Guidelines:**
- Use `n_jobs=1` (default) for small numbers of walks (< 100)
- Use `n_jobs=-1` for large simulations (> 1000 walks)
- Multiprocessing overhead means it's not always faster for small tasks
- Typical speedup: 4-8x on modern multi-core systems

### Weighted Random Walks

```python
# If your graph has edge weights, use them for transition probabilities
# (weights represent preference/cost for taking that edge)
paths = graph.random_walks(
    start_node=0,
    n_walks=500,
    max_steps=25,
    use_edge_weights=True  # Use edge weights if present
)

# Otherwise, uniform probability among neighbors (default)
paths = graph.random_walks(
    start_node=0,
    n_walks=500,
    max_steps=25,
    use_edge_weights=False  # Equal probability (default)
)
```

### Reproducible Walks

```python
# Same seed produces identical walks
paths1 = graph.random_walks(
    start_node=0, n_walks=10, max_steps=5, seed=123
)
paths2 = graph.random_walks(
    start_node=0, n_walks=10, max_steps=5, seed=123
)

assert paths1 == paths2  # True - identical walks

# Works with parallel execution too
paths_parallel = graph.random_walks(
    start_node=0, n_walks=10000, max_steps=5,
    n_jobs=-1, seed=123
)
# Produces same results as serial execution with seed=123
```

### Complete Example

See [`examples/multimodal_demo/random_walk_example.py`](examples/multimodal_demo/random_walk_example.py) for a comprehensive example including:

1. **Basic fixed-length walks** from root node
2. **Target-directed walks** with success rate analysis
3. **Backtracking comparison** across different probabilities
4. **Performance benchmarks** comparing serial vs parallel execution
5. **Path length distribution** analysis and visualization
6. **Reproducibility demonstration** with seed management

Run the example:
```bash
# Activate virtual environment first
uv run python examples/multimodal_demo/random_walk_example.py
```

Expected output includes:
- Statistical summaries for each scenario
- Performance comparisons (serial vs parallel)
- Distribution plots (if matplotlib available)
- Efficiency metrics comparing random walks to shortest paths

### CLI Usage

You can also run random walks directly from the command line:

```bash
# Get help and see all options
uv run navigraph walk --help

# Basic: 100 walks of 20 steps from node 0 (no backtracking)
uv run navigraph walk config.yaml --start 0 --max-steps 20

# Uniform random walk: backtracking allowed with equal probability
uv run navigraph walk config.yaml -s 0 -m 20 -b -1 -n 1000

# Target-directed: walk from node 0 to node 127
uv run navigraph walk config.yaml -s 0 -t 127 -m 50 -n 1000

# With backtracking: 30% chance to return to previous node
uv run navigraph walk config.yaml -s 0 -m 15 -b 0.3 -n 500

# Parallel execution on all CPU cores (major speedup!)
uv run navigraph walk config.yaml -s 0 -t 127 -m 50 -n 10000 -j -1

# Save paths to JSON file for later analysis
uv run navigraph walk config.yaml -s 0 -m 20 -n 100 --save-paths walks.json

# With seed for reproducibility
uv run navigraph walk config.yaml -s 0 -t 127 -m 50 -n 1000 --seed 42

# Verbose output with path details and statistics
uv run navigraph walk config.yaml -s 0 -t 127 -m 50 -n 100 -v
```

**Key CLI Options:**
- `-s, --start`: Starting node (required)
- `-t, --target`: Target node (optional)
- `-n, --n-walks`: Number of walks [default: 100]
- `-m, --max-steps`: Maximum steps per walk
- `-b, --backtrack-prob`: Backtracking behavior [default: 0.0]
  - `-1`: Uniform distribution (treat previous node like any other neighbor)
  - `0.0`: No backtracking (previous node excluded)
  - `0.0-1.0`: Explicit backtrack probability (e.g., 0.3 = 30% chance)
  - `1.0`: Always backtrack if possible
- `-j, --n-jobs`: Parallel processes (1=serial, -1=all cores) [default: 1]
- `--seed`: Random seed for reproducibility
- `--save-paths`: Save paths to JSON file
- `-v, --verbose`: Detailed output
- `--help`: Show complete help message with all options

**Understanding Backtracking Modes:**

| Mode | `backtrack_prob` | Behavior | Use Case |
|------|-----------------|----------|----------|
| **No Backtracking** | `0.0` (default) | Never return to previous node | Goal-directed navigation |
| **Uniform** | `-1` | Equal probability to all neighbors | Unbiased exploration |
| **Probabilistic** | `0.0-1.0` | Explicit backtrack chance | Modeling uncertainty |
| **Always Backtrack** | `1.0` | Always return if possible | Testing/debugging |

**Example CLI output:**
```
🚶 Running random walks...
   Start node: 0, Target node: 127, Walks: 1000, Max steps: 50, Backtrack prob: 0.00, Seed: 42
✅ Completed in 0.45s

📊 Summary Statistics:
   Mean path length: 15.23 steps
   Median path length: 14.0 steps
   Std deviation: 8.45 steps
   Min-Max: 6-50 steps
   Success rate: 68.5%
   Successful walks: 685/1000
   Shortest path: 6 steps
   Efficiency: 39.4%
```

### API Reference

**Main Method:**
```python
GraphStructure.random_walks(
    start_node,                    # Required: starting node
    n_walks=1,                     # Number of walks to generate
    max_steps=None,                # Max steps per walk (or None if target provided)
    target_node=None,              # Optional target node
    terminate_on_target=True,      # Stop when target reached?
    backtrack_prob=0.0,            # Probability (0-1) of backtracking
    use_edge_weights=False,        # Use edge weights for transitions?
    return_stats=False,            # Return statistics dictionary?
    seed=None,                     # Random seed for reproducibility
    n_jobs=1                       # Number of processes (1=serial, -1=all cores)
)
```

**Returns:**
- `List[List[Any]]`: List of walks (each walk is a list of nodes)
- OR `Tuple[List[List[Any]], Dict]`: Walks + statistics if `return_stats=True`

**Statistics Dictionary:**
- `'mean_length'`: Mean path length across all walks
- `'median_length'`: Median path length
- `'std_length'`: Standard deviation
- `'min_length'`: Minimum path length
- `'max_length'`: Maximum path length
- `'success_rate'`: Fraction reaching target (if target provided)
- `'successful_walks'`: List of successful walk indices (if target provided)

## Spatial Calibration

NaviGraph provides interactive tools for spatial calibration:

### Camera Calibration

Transform camera pixel coordinates to real-world coordinates:

```bash
navigraph setup calibration config.yaml
```

This launches an interactive GUI where you:
1. Click corresponding points in video and real-world coordinates
2. Build transformation matrix
3. Save calibration for use in experiments

### Graph Mapping

Map real-world coordinates to graph nodes and edges:

```bash
navigraph setup graph config.yaml
```

Interactive GUI controls:
- **V**: Place nodes on the maze map
- **E**: Create edges between nodes  
- **R**: Define spatial regions
- **S**: Save the spatial mapping

### Testing Calibration

Validate your calibration and mapping:

```bash
navigraph test graph config.yaml
navigraph test calibration config.yaml
```

## Analysis Pipeline

### Built-in Metrics

NaviGraph provides standard navigation metrics:

```yaml
analyze:
  metrics:
    # Time to reach target
    time_to_reward: 
      func_name: 'time_a_to_b'
      args: 
        a: 145  # Start node
        b: 273  # Goal node
    
    # Average velocity between points
    velocity_to_reward:
      func_name: 'velocity_a_to_b'
      args:
        a: 145
        b: 273
    
    # Exploration behavior
    exploration_percentage:
      func_name: 'exploration_percentage'
```

### Custom Analysis

Add custom analysis functions:

```python
def custom_metric(dataframe, **kwargs):
    """Calculate custom navigation metric."""
    # Your analysis logic
    return metric_value

# Register in configuration
analyze:
  metrics:
    my_metric:
      func_name: 'custom_metric'
      args:
        param1: value1
```

## Visualization

NaviGraph provides a flexible visualization pipeline:

### Visualization Components

```yaml
visualizations:
  output:
    enabled: true
    format: mp4
    fps: 30
    
  pipeline:
    # Show body part tracking
    - name: bodypart_display
      type: bodyparts
      config:
        bodyparts: ['Nose', 'Tail_base']
        colors:
          Nose: [255, 0, 0]      # Red
          Tail_base: [0, 255, 0] # Green
        radius: 6
    
    # Overlay maze map
    - name: map_overlay
      type: map_overlay
      config:
        position: 'bottom_right'
        size: 0.25
        opacity: 0.8
        show_trajectory: true
        trail_length: 100
    
    # Show graph structure
    - name: graph_overlay
      type: graph_overlay
      config:
        mode: 'overlay'
        highlight_node_color: 'red'
        highlight_edge_color: 'blue'
    
    # Display metrics
    - name: metrics_display
      type: text_display
      config:
        columns: ['speed', 'current_node']
        position: 'top_left'
        font_scale: 0.7
```

### Creating Custom Visualizers

```python
@register_visualizer_plugin("custom_viz")
class CustomVisualizerPlugin(NaviGraphPlugin):
    """Custom visualization plugin."""
    
    def augment_data(self, dataframe, shared_resources):
        # Add visualization-specific data
        return dataframe
    
    def render_frame(self, frame, frame_data, shared_resources):
        # Custom rendering logic
        return modified_frame
```

## Examples

NaviGraph includes examples demonstrating different research scenarios and data integration patterns:

### Basic Maze - Single Data Source
```bash
cd examples/basic_maze
uv run navigraph setup graph config.yaml  # Interactive spatial mapping
uv run navigraph run config.yaml          # Single-session analysis
```

**Use Case**: Simple pose tracking analysis with spatial mapping. Good starting point for researchers with single data streams.

### Multimodal Demo - Multi-Source Integration
```bash
cd examples/multimodal_demo
uv run navigraph run config.yaml
```

**Use Case**: Demonstrates integration of multiple uncorrelated data sources:
- Pose tracking (behavioral)
- Neural activity (calcium imaging) 
- Head direction (orientation sensors)

Shows how NaviGraph synchronizes and analyzes diverse data streams in unified temporal and spatial domains.

### Custom Graph Types - Flexible Spatial Analysis
```bash
cd examples/star_maze
uv run navigraph run config_star.yaml

cd examples/graph_file_loader  
uv run navigraph run config_file_loader.yaml
```

**Use Case**: Different spatial representations for the same environment. Demonstrates how researchers can experiment with various graph topologies to find optimal spatial analysis frameworks.

### Research Scenarios

**Multi-Session Studies**: Each example can be extended to process multiple sessions with shared spatial mappings, enabling longitudinal analysis and cross-session comparisons.

**Custom Data Integration**: Examples serve as templates for integrating your own data sources - replace the example plugins with ones that match your research tools and data formats.

**Domain Traversal**: All examples demonstrate analysis across time (temporal patterns), space (coordinate analysis), and graph domains (network topology), showing how researchers can freely traverse these domains for comprehensive insights.

## Extending NaviGraph

### Adding Custom Graph Builders

1. Create a new file in `navigraph/core/graph/builders/`
2. Inherit from `GraphBuilder` 
3. Add `@register_graph_builder("name")` decorator
4. Implement `build_graph()` method

### Adding Custom Plugins

1. Create plugin in appropriate `navigraph/plugins/` subdirectory
2. Inherit from `NaviGraphPlugin`
3. Add appropriate registration decorator
4. Implement required methods (`provide()`, `augment_data()`)

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/navigraph.git
cd navigraph

# Install in development mode
uv sync --dev

# Run tests
uv run pytest

# Code formatting
uv run black navigraph/
uv run ruff navigraph/
```

## Contributing

We welcome community involvement in NaviGraph development:

- **Bug Reports**: Found an issue? Please report it via [GitHub Issues](https://github.com/your-username/navigraph/issues)
- **Feature Requests**: Have ideas for improvements? Open a feature request on GitHub Issues
- **Documentation**: Help improve documentation, examples, and tutorials
- **Testing**: Test NaviGraph with your data and report compatibility issues

**Note**: We're currently defining our contribution guidelines and review process. For now, please use GitHub Issues to discuss potential contributions before submitting pull requests.

## Citation

If you use NaviGraph in your research, please cite:

> Iton, A. K., Iton, E., Michaelson, D. M., & Blinder, P. (2024). NaviGraph: A graph-based framework for multimodal analysis of spatial decision-making. *Journal of Neuroscience Methods*, Volume(Issue), Pages. [DOI]

## License

NaviGraph is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Support

- **Documentation**: Full documentation available in the `docs/` directory
- **Issues**: Report bugs and request features via GitHub Issues
- **Examples**: Complete examples in `examples/` directory
- **Help**: Use `navigraph --help` for command-line assistance