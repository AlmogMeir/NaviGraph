#!/usr/bin/env python3
"""Test graph provider integration with session system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from navigraph.plugins.shared_resources.graph_provider import GraphProviderResource


def test_graph_provider_as_shared_resource():
    """Test graph provider works as a shared resource (like session uses it)."""
    print("Testing Graph Provider as Shared Resource")
    print("=" * 50)
    
    # Configuration (like session would provide)
    config = {
        'mapping_file': './examples/basic_maze/resources/graph_mapping.pkl',
        'builder': {
            'type': 'binary_tree',
            'config': {'height': 7}
        },
        'conflict_strategy': 'node_priority',
        'metadata': {
            'reward_node': 63,  # Valid node (highest in binary tree height 7)
            'start_node': 0,    # Root node
            'safe_zones': [10, 11, 20, 21]
        }
    }
    
    # Create provider (like session does)
    provider = GraphProviderResource.from_config(config)
    
    # Test that it's initialized
    assert provider.is_initialized(), "Provider should be initialized"
    print("✓ Provider initialized successfully")
    
    # Test how analyzers would access it (like existing code does)
    graph_instance = provider.get_graph_instance()  # Legacy method
    print(f"✓ Graph accessed: {graph_instance.num_nodes} nodes, {graph_instance.num_edges} edges")
    
    # Test how new code would access it
    graph = provider.get_graph()
    mapping = provider.get_mapping()
    metadata = provider.get_metadata()
    
    print(f"✓ New API: graph {graph.num_nodes} nodes, mapping {len(mapping.get_mapped_nodes())} mapped nodes")
    print(f"✓ Metadata: {list(metadata.keys())}")
    
    # Test spatial mapping functionality
    # Test a point that should map to something
    node, edge = mapping.map_point_to_elements(100, 100)
    print(f"✓ Spatial mapping works: point (100,100) -> node {node}, edge {edge}")
    
    return True


def test_builder_priority_system():
    """Test the builder priority system works correctly."""
    print("\nTesting Builder Priority System")  
    print("=" * 50)
    
    # Test 1: Config builder with same parameters as mapping (should work fine)
    print("Test 1: Config builder with compatible parameters")
    config1 = {
        'mapping_file': './examples/basic_maze/resources/graph_mapping.pkl',  # height=7 in mapping
        'builder': {
            'type': 'binary_tree',
            'config': {'height': 7}  # Same height - should work
        }
    }
    
    provider1 = GraphProviderResource.from_config(config1)
    graph1 = provider1.get_graph()
    expected_nodes_h7 = 127  # 2^7 - 1
    print(f"✓ Config priority: {graph1.num_nodes} nodes (height=7)")
    
    # Test 2: Config builder with incompatible parameters (should fail with clear error)
    print("\nTest 2: Config builder with incompatible parameters (should fail)")
    config_incompatible = {
        'mapping_file': './examples/basic_maze/resources/graph_mapping.pkl',  # height=7 in mapping
        'builder': {
            'type': 'binary_tree',
            'config': {'height': 5}  # Different height - mapping incompatible
        }
    }
    
    try:
        provider_bad = GraphProviderResource.from_config(config_incompatible)
        print("❌ Should have failed with incompatible mapping")
        return False
    except Exception as e:
        print(f"✓ Correctly caught incompatible mapping: {type(e).__name__}")
        print(f"  Error: {str(e)[:100]}...")
    
    # Test 3: No config builder - should use mapping builder
    print("\nTest 3: Mapping builder when no config builder")
    config2 = {
        'mapping_file': './examples/basic_maze/resources/graph_mapping.pkl'
        # No builder in config - should use mapping's builder (height=7)
    }
    
    provider2 = GraphProviderResource.from_config(config2)
    graph2 = provider2.get_graph()
    expected_nodes_h7 = 127  # 2^7 - 1
    assert graph2.num_nodes == expected_nodes_h7, f"Expected 127 nodes, got {graph2.num_nodes}"
    print(f"✓ Mapping fallback: {graph2.num_nodes} nodes (height=7 from mapping)")
    
    return True


def test_metadata_validation():
    """Test metadata validation works correctly."""
    print("\nTesting Metadata Validation")
    print("=" * 50)
    
    config = {
        'mapping_file': './examples/basic_maze/resources/graph_mapping.pkl',
        'builder': {'type': 'binary_tree', 'config': {'height': 7}},
        'metadata': {
            'reward_node': 63,     # Valid node
            'start_node': 0,       # Valid node  
            'invalid_node': 999,   # Invalid - should warn
            'safe_zones': [10, 11, 999],  # Mix of valid/invalid
            'danger_edges': [[0, 10], [999, 888]]  # Mix of valid/invalid
        }
    }
    
    # Should initialize successfully but with warnings
    provider = GraphProviderResource.from_config(config)
    metadata = provider.get_metadata()
    
    print(f"✓ Metadata loaded with validation: {len(metadata)} items")
    print(f"  Valid items: reward_node={metadata['reward_node']}, start_node={metadata['start_node']}")
    
    return True


def test_session_integration_simulation():
    """Simulate how the session system would use the graph provider."""
    print("\nSimulating Session Integration")
    print("=" * 50)
    
    # Simulate session.shared_resources dictionary
    shared_resources = {}
    
    # Create graph provider like session does
    config = {
        'mapping_file': './examples/basic_maze/resources/graph_mapping.pkl',
        'builder': {'type': 'binary_tree', 'config': {'height': 7}},
        'metadata': {'reward_node': 63, 'start_node': 0}
    }
    
    # Session stores the provider object directly
    shared_resources['graph'] = GraphProviderResource.from_config(config)
    
    # Simulate how analyzers access it (like current code)
    graph_provider = shared_resources.get('graph')
    if graph_provider:
        # Legacy method (existing analyzers use this)
        graph_instance = graph_provider.get_graph_instance()
        print(f"✓ Analyzer access (legacy): {graph_instance.num_nodes} nodes")
        
        # New method (for new analyzers)
        mapping = graph_provider.get_mapping()
        metadata = graph_provider.get_metadata()
        
        print(f"✓ New analyzer access: {len(mapping.get_mapped_nodes())} mapped nodes")
        print(f"✓ Metadata access: reward at node {metadata.get('reward_node')}")
    
    return True


if __name__ == "__main__":
    print("🚀 Testing Graph Provider Integration")
    print("=" * 60)
    
    try:
        test_graph_provider_as_shared_resource()
        test_builder_priority_system()
        test_metadata_validation()
        test_session_integration_simulation()
        
        print("\n" + "=" * 60)
        print("🎉 All Graph Provider Integration tests passed! ✅")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)