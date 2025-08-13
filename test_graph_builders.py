#!/usr/bin/env python3
"""Test script for the new graph builder architecture."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from navigraph.core.graph.structures import GraphStructure
from navigraph.core.graph.builders import (
    list_graph_builders,
    get_graph_builder,
    get_graph_builder_info
)


def test_binary_tree_builder():
    """Test binary tree builder."""
    print("Testing Binary Tree Builder...")
    
    # Get builder class from registry
    BinaryTreeBuilder = get_graph_builder("binary_tree")
    
    # Create builder
    builder = BinaryTreeBuilder(height=4)
    
    # Create graph structure
    graph_struct = GraphStructure(builder)
    
    # Test properties
    print(f"  Nodes: {graph_struct.num_nodes}")
    print(f"  Edges: {graph_struct.num_edges}")
    print(f"  Metadata: {graph_struct.metadata}")
    
    # Test visualization
    viz = graph_struct.get_visualization()
    print(f"  Visualization shape: {viz.shape}")
    
    # Test graph operations
    print(f"  Node 00 neighbors: {graph_struct.get_neighbors(0)}")
    print(f"  Shortest path 00->37: {graph_struct.get_shortest_path(0, 37)}")
    print("  ✓ Binary tree builder works!")


def test_registry():
    """Test builder registry."""
    print("\nTesting Builder Registry...")
    
    # List all builders
    builders = list_graph_builders()
    print(f"  Available builders: {builders}")
    
    # Get builder info
    for builder_name in builders:
        info = get_graph_builder_info(builder_name)
        print(f"\n  {builder_name}:")
        print(f"    Class: {info['class_name']}")
        print(f"    Parameters: {info['parameters']}")
    
    print("  ✓ Registry works!")


def test_from_config():
    """Test creating builder from config."""
    print("\nTesting from_config...")
    
    # Get builder class
    BinaryTreeClass = get_graph_builder("binary_tree")
    
    # Create from config
    config = {"height": 5}
    builder = BinaryTreeClass.from_config(config)
    
    # Create graph
    graph_struct = GraphStructure(builder)
    print(f"  Created graph with {graph_struct.num_nodes} nodes")
    print("  ✓ from_config works!")


def test_file_loader():
    """Test file loader builder."""
    print("\nTesting File Loader Builder...")
    
    # Get builder class from registry
    BinaryTreeBuilder = get_graph_builder("binary_tree")
    
    # First create and save a graph
    builder = BinaryTreeBuilder(height=3)
    graph_struct = GraphStructure(builder)
    
    # Save it using edgelist format (simpler, no attributes)
    test_file = Path("/tmp/test_graph.edgelist")
    graph_struct.save(test_file, format='edgelist')
    print(f"  Saved graph to {test_file}")
    
    # Now load it using file loader
    FileGraphBuilder = get_graph_builder("file_loader")
    loader = FileGraphBuilder(str(test_file), format='edgelist')
    loaded_struct = GraphStructure(loader)
    
    print(f"  Loaded graph with {loaded_struct.num_nodes} nodes")
    print("  ✓ File loader works!")
    
    # Clean up
    test_file.unlink()


if __name__ == "__main__":
    print("=" * 50)
    print("Testing New Graph Builder Architecture")
    print("=" * 50)
    
    test_binary_tree_builder()
    test_registry()
    test_from_config()
    test_file_loader()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)