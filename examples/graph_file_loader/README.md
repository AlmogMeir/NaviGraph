# Graph File Loader Example

This example demonstrates loading graphs from files using the `FileGraphBuilder` and testing the GUI with file-based graphs.

## Purpose

- Test that `FileGraphBuilder` works correctly with the GUI
- Verify that graphs loaded from files use the base visualization properly
- Ensure spatial mapping works with arbitrary graph structures
- Test different file formats (GraphML, GEXF, GML)

## Files Structure

```
examples/graph_file_loader/
├── README.md                      # This file
├── generate_test_graphs.py        # Script to create test graphs
├── config_file_loader.yaml        # Main config (Erdős-Rényi graph)
├── config_barabasi_albert.yaml    # Barabási-Albert graph config  
├── config_gexf_format.yaml        # GEXF format test config
├── graphs/                        # Generated graph files
│   ├── *.graphml                  # GraphML format files
│   ├── *.gexf                     # GEXF format files
│   ├── *.gml                      # GML format files
│   └── graph_summary.txt          # Summary of all graphs
└── resources/
    ├── test_map.png              # Map image for spatial mapping
    └── *.pkl                     # Saved spatial mappings (created by GUI)
```

## Generated Test Graphs

The script creates 8 different types of graphs:

1. **random_erdos_renyi_small** - 15 nodes, Erdős-Rényi model
2. **random_erdos_renyi_medium** - 25 nodes, Erdős-Rényi model  
3. **random_barabasi_albert** - 20 nodes, Barabási-Albert model
4. **random_watts_strogatz** - 20 nodes, Watts-Strogatz model
5. **complete_graph** - 8 nodes, complete graph
6. **grid_graph** - 4x5 grid structure
7. **path_graph** - 12 nodes in a path
8. **cycle_graph** - 10 nodes in a cycle

Each graph is saved in 3 formats: `.graphml`, `.gexf`, and `.gml`

## Manual Testing Instructions

### 1. Test Basic File Loading
```bash
cd examples/graph_file_loader
poetry run navigraph setup graph config_file_loader.yaml
```

**Expected behavior:**
- GUI opens showing the test map
- Graph panel displays Erdős-Rényi graph with 15 nodes using base visualization
- Can create spatial mapping by placing nodes on the map
- Can save mapping to `resources/graph_mapping.pkl`

### 2. Test Different Graph Types
```bash
# Test Barabási-Albert graph
poetry run navigraph setup graph config_barabasi_albert.yaml

# Test GEXF format with grid graph
poetry run navigraph setup graph config_gexf_format.yaml
```

### 3. Verify Base Visualization
All graphs should display correctly using the inherited `_default_visualization()` method from `GraphBuilder` base class, confirming that:
- FileGraphBuilder doesn't need custom visualization
- Base class provides adequate default rendering
- Different graph structures display properly

### 4. Test Spatial Mapping
- Place graph nodes/edges on the map image
- Save the mapping
- Reload the configuration to test mapping persistence

## What This Tests

✅ **FileGraphBuilder Integration**: Loads graphs from files correctly  
✅ **Base Visualization Inheritance**: Uses default GraphBuilder visualization  
✅ **Multiple File Formats**: GraphML, GEXF, GML all work  
✅ **GUI Compatibility**: File-loaded graphs work with setup GUI  
✅ **Spatial Mapping**: Can map arbitrary graphs to spatial regions  
✅ **Different Graph Types**: Various structures (random, grid, path, etc.)  

## Troubleshooting

If you encounter issues:

1. **Graph doesn't display**: Check that the file path in config is correct
2. **File format error**: Verify the format parameter matches the file extension
3. **Import errors**: Ensure you're using `poetry run` for correct environment
4. **GUI crashes**: Check the console for error messages about graph loading

## Next Steps

After confirming file loading works:
- Test with your own graph files
- Try different NetworkX-supported formats
- Create custom graphs and save them for spatial analysis