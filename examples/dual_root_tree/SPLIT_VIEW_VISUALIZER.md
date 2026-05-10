# Split View Visualizer

The `split_view` visualizer creates a side-by-side display:
- **Left side**: Original video with mapping overlay (node/edge bounding boxes with adjustable transparency)
- **Right side**: Graph visualization using actual node positions from the mapping file, with current location highlighted

## Features

### Left Side (Video + Mapping)
- Displays the original video feed
- Overlays mapping bounding boxes with configurable transparency (default: 0.3)
- Shows node and/or edge bounding boxes
- Customizable colors for nodes and edges

### Right Side (Graph Visualization)
- Uses actual spatial positions from the mapping file
- Automatically scales and centers the graph to fit the canvas
- Highlights current node position in real-time
- Shows edges connecting nodes
- Optional node labels
- Customizable colors, sizes, and styling

## Configuration

### Basic Usage

Add the `split_view` visualizer to your config file:

```yaml
visualizations:
  pipeline:
    - name: split_visualization
      type: split_view
      config:
        mapping_file: ./resources/maze_mappings/your_mapping.pkl
        bodypart: headstage
        mapping_alpha: 0.3
        graph_width_ratio: 0.5
```

### Full Configuration Options

#### Mapping Overlay (Left Side)
```yaml
mapping_file: ./resources/maze_mappings/your_mapping.pkl  # Required
mapping_alpha: 0.3  # Transparency (0.0-1.0)
show_node_boxes: true  # Show node bounding boxes
show_edge_boxes: false  # Show edge bounding boxes
node_box_color: [0, 255, 0]  # BGR color for node boxes
edge_box_color: [255, 0, 0]  # BGR color for edge boxes
```

#### Graph Visualization (Right Side)
```yaml
graph_bg_color: [40, 40, 40]  # Background color (dark gray)
node_radius: 8  # Circle radius for nodes
node_color: [200, 200, 200]  # Color for inactive nodes
current_node_color: [0, 255, 0]  # Color for current position
edge_color: [100, 100, 100]  # Color for edges
edge_thickness: 1  # Line thickness for edges
show_node_labels: true  # Display node IDs
label_font_scale: 0.4  # Font size for labels
label_color: [255, 255, 255]  # Text color for labels
```

#### Tracking & Layout
```yaml
bodypart: headstage  # Which bodypart to track
graph_width_ratio: 0.5  # Graph width as ratio of total (0.0-1.0)
padding: 20  # Pixels of padding around graph
```

## Example Configurations

### Example 1: Default Split View
```yaml
- name: split_viz
  type: split_view
  config:
    mapping_file: ./resources/maze_mappings/my_maze.pkl
    bodypart: headstage
```

### Example 2: Customized Appearance
```yaml
- name: custom_split_viz
  type: split_view
  config:
    # Video overlay
    mapping_file: ./resources/maze_mappings/my_maze.pkl
    mapping_alpha: 0.2  # More subtle overlay
    show_node_boxes: true
    show_edge_boxes: true
    node_box_color: [0, 255, 255]  # Cyan
    edge_box_color: [255, 128, 0]  # Orange
    
    # Graph styling
    graph_bg_color: [20, 20, 20]  # Darker background
    node_radius: 10  # Larger nodes
    current_node_color: [255, 0, 0]  # Red for current position
    edge_thickness: 2  # Thicker edges
    label_font_scale: 0.5  # Larger labels
    
    # Layout
    graph_width_ratio: 0.6  # Graph takes 60% of width
    padding: 30
```

### Example 3: Minimal (No Labels, No Boxes)
```yaml
- name: minimal_split_viz
  type: split_view
  config:
    mapping_file: ./resources/maze_mappings/my_maze.pkl
    mapping_alpha: 0.0  # No mapping overlay
    show_node_boxes: false
    show_edge_boxes: false
    show_node_labels: false
    bodypart: headstage
```

## Usage

1. **Create your config file** using `config_dual_root_split_view.yaml` as a template
2. **Run visualization**:
   ```bash
   cd /path/to/NaviGraph/examples/dual_root_tree
   uv run navigraph run config_dual_root_split_view.yaml
   ```

3. **Output**: Video will be saved to the specified output directory with both views combined

## Tips

- **Transparency**: Use `mapping_alpha: 0.3` for subtle overlay, `0.5` for balanced, `0.7` for prominent
- **Graph Width**: Adjust `graph_width_ratio` based on your video aspect ratio (0.4-0.6 works well)
- **Node Labels**: Disable `show_node_labels: false` for cleaner look with many nodes
- **Performance**: The visualizer caches node positions, so performance is good even with complex graphs

## Combining with Other Visualizers

You can stack multiple visualizers. The split view creates the base layout, then other visualizers can add overlays:

```yaml
pipeline:
  - name: split_base
    type: split_view
    config:
      mapping_file: ./resources/maze_mappings/my_maze.pkl
      bodypart: headstage
  
  - name: text_overlay
    type: text_display
    config:
      columns:
        - frame
        - headstage_graph_node
      position: top_left
```

## Comparison with Other Visualizers

| Feature | `split_view` | `map_overlay` | `graph_overlay` |
|---------|-------------|---------------|-----------------|
| Video display | ✅ Left side | ✅ Full frame | ✅ Full frame |
| Mapping overlay | ✅ Configurable | ❌ | ❌ |
| Graph visualization | ✅ Right side | ❌ | ✅ Overlay |
| Actual node positions | ✅ | ❌ | Depends on builder |
| Side-by-side layout | ✅ | Optional | ❌ |

## Troubleshooting

**Graph not showing up?**
- Check that `mapping_file` path is correct
- Ensure mapping file has 'mappings' → 'nodes' and 'edges' keys
- Verify bodypart name matches your tracking data

**Current position not updating?**
- Check that `bodypart` matches your config (e.g., 'headstage')
- Ensure graph location plugin is loaded
- Verify column name: `{bodypart}_graph_node` exists in data

**Mapping overlay too strong/weak?**
- Adjust `mapping_alpha` (0.0 = invisible, 1.0 = fully opaque)
- Recommended range: 0.2-0.4 for subtle overlay

**Graph too small/large?**
- Adjust `graph_width_ratio` (0.0-1.0)
- Increase `padding` if graph is clipped at edges
- Adjust `node_radius` and `label_font_scale` for better visibility
