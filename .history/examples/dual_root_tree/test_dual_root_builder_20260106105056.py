"""Test script for the dual-rooted binary tree builder.

This script demonstrates the new dual_root_binary_tree graph builder
and validates its functionality.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from navigraph.core.graph.builders import get_graph_builder, list_graph_builders
import networkx as nx
import matplotlib.pyplot as plt


def test_dual_root_builder():
    """Test the dual-rooted binary tree builder."""
    
    print("=" * 60)
    print("Testing Dual-Rooted Binary Tree Builder")
    print("=" * 60)
    
    # List all available builders
    print("\nAvailable graph builders:")
    builders = list_graph_builders()
    for builder in builders:
        print(f"  - {builder}")
    
    # Verify our new builder is registered
    assert "dual_root_binary_tree" in builders, "dual_root_binary_tree not registered!"
    print("\n✓ dual_root_binary_tree builder is registered")
    
    # Test symmetric dual tree
    print("\n" + "-" * 60)
    print("Test 1: Symmetric dual tree (3 levels on each side)")
    print("-" * 60)
    
    builder_class = get_graph_builder("dual_root_binary_tree")
    symmetric_builder = builder_class(left_height=3, right_height=3)
    symmetric_graph = symmetric_builder.build_graph()
    
    print(f"Number of nodes: {symmetric_graph.number_of_nodes()}")
    print(f"Number of edges: {symmetric_graph.number_of_edges()}")
    print(f"Root nodes: L0, R0")
    print(f"Roots connected: {symmetric_graph.has_edge('L0', 'R0')}")
    
    # Check metadata
    metadata = symmetric_builder.get_metadata()
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    assert symmetric_graph.has_edge('L0', 'R0'), "Roots should be connected!"
    print("\n✓ Symmetric tree built successfully")
    
    # Test asymmetric dual tree
    print("\n" + "-" * 60)
    print("Test 2: Asymmetric dual tree (4 levels left, 2 levels right)")
    print("-" * 60)
    
    asymmetric_builder = builder_class(left_height=4, right_height=2)
    asymmetric_graph = asymmetric_builder.build_graph()
    
    print(f"Number of nodes: {asymmetric_graph.number_of_nodes()}")
    print(f"Number of edges: {asymmetric_graph.number_of_edges()}")
    
    # Count nodes on each side
    left_nodes = [n for n, d in asymmetric_graph.nodes(data=True) if d.get('tree') == 'left']
    right_nodes = [n for n, d in asymmetric_graph.nodes(data=True) if d.get('tree') == 'right']
    
    print(f"Left tree nodes: {len(left_nodes)}")
    print(f"Right tree nodes: {len(right_nodes)}")
    
    metadata = asymmetric_builder.get_metadata()
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    assert len(left_nodes) > len(right_nodes), "Left tree should have more nodes!"
    print("\n✓ Asymmetric tree built successfully")
    
    # Test visualization
    print("\n" + "-" * 60)
    print("Test 3: Visualization")
    print("-" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Visualize symmetric tree
    positions = nx.get_node_attributes(symmetric_graph, 'pos')
    node_colors = []
    for node in symmetric_graph.nodes():
        tree_side = symmetric_graph.nodes[node].get('tree', '')
        if tree_side == 'left':
            node_colors.append('lightblue')
        elif tree_side == 'right':
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightgreen')
    
    nx.draw(symmetric_graph, positions, ax=axes[0], 
           node_color=node_colors, node_size=500, 
           with_labels=True, font_size=8, font_weight='bold',
           edge_color='gray', width=2)
    axes[0].set_title('Symmetric Dual Tree (3, 3)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Visualize asymmetric tree
    positions = nx.get_node_attributes(asymmetric_graph, 'pos')
    node_colors = []
    for node in asymmetric_graph.nodes():
        tree_side = asymmetric_graph.nodes[node].get('tree', '')
        if tree_side == 'left':
            node_colors.append('lightblue')
        elif tree_side == 'right':
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightgreen')
    
    nx.draw(asymmetric_graph, positions, ax=axes[1],
           node_color=node_colors, node_size=500,
           with_labels=True, font_size=8, font_weight='bold',
           edge_color='gray', width=2)
    axes[1].set_title('Asymmetric Dual Tree (4, 2)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = Path(__file__).parent / "resources"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "dual_tree_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    print("\n✓ Visualization test completed")
    
    # Test edge cases
    print("\n" + "-" * 60)
    print("Test 4: Edge cases")
    print("-" * 60)
    
    # Minimal tree (height 1 on both sides)
    minimal_builder = builder_class(left_height=1, right_height=1)
    minimal_graph = minimal_builder.build_graph()
    print(f"Minimal tree (1, 1) - Nodes: {minimal_graph.number_of_nodes()}, Edges: {minimal_graph.number_of_edges()}")
    assert minimal_graph.number_of_nodes() == 2, "Should have only 2 root nodes"
    assert minimal_graph.number_of_edges() == 1, "Should have only 1 edge connecting roots"
    
    # Very asymmetric tree
    extreme_builder = builder_class(left_height=1, right_height=5)
    extreme_graph = extreme_builder.build_graph()
    print(f"Extreme asymmetry (1, 5) - Nodes: {extreme_graph.number_of_nodes()}, Edges: {extreme_graph.number_of_edges()}")
    
    print("\n✓ Edge cases handled correctly")
    
    # Summary
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nThe dual-rooted binary tree builder is ready to use.")
    print("Example configuration: examples/dual_root_tree/config_dual_root.yaml")


if __name__ == "__main__":
    test_dual_root_builder()
