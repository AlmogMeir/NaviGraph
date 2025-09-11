#!/usr/bin/env python3
"""Test script for migrated metrics."""

import pandas as pd
import numpy as np
from navigraph.analysis.utils import (
    find_path_transitions,
    calculate_path_distance,
    count_unique_values_in_segment,
    count_value_transitions,
    get_path_sequence
)

def test_find_path_transitions():
    """Test the generic find_path_transitions function."""
    # Create test dataframe
    data = {
        'state': ['exploring', 'exploring', 'moving', 'moving', 'rewarded', 'rewarded'],
        'node_id': [1, 2, 2, 3, 3, 4],
        'zone': ['A', 'A', 'A', 'B', 'B', 'B']
    }
    df = pd.DataFrame(data)
    
    # Test basic transition finding
    transitions = find_path_transitions(df, 'state', 'exploring', 'state', 'rewarded')
    print(f"State transitions (exploring -> rewarded): {transitions}")
    assert len(transitions) == 1
    assert transitions[0] == (0, 4)
    
    # Test with different columns
    transitions = find_path_transitions(df, 'zone', 'A', 'zone', 'B')
    print(f"Zone transitions (A -> B): {transitions}")
    assert len(transitions) == 1
    assert transitions[0] == (0, 3)
    
    # Test with filtering
    transitions = find_path_transitions(
        df, 'state', 'exploring', 'state', 'rewarded',
        min_unique_values_in_column={'node_id': 3}
    )
    print(f"Filtered transitions (min 3 unique nodes): {transitions}")
    assert len(transitions) == 1  # Should still find it as we have nodes 1,2,3
    
    print("✓ find_path_transitions tests passed")

def test_count_functions():
    """Test counting utility functions."""
    data = {
        'node_id': [1, 1, 2, 2, 3, 3, 3, 4],
        'x': [0, 1, 2, 3, 4, 5, 6, 7],
        'y': [0, 0, 1, 1, 2, 2, 3, 3]
    }
    df = pd.DataFrame(data)
    
    # Test unique value counting
    unique_count = count_unique_values_in_segment(df, 0, 7, 'node_id')
    print(f"Unique nodes in segment: {unique_count}")
    assert unique_count == 4  # nodes 1, 2, 3, 4
    
    # Test transition counting (eliminate sequences)
    transition_count = count_value_transitions(df, 0, 7, 'node_id')
    print(f"Node transitions: {transition_count}")
    assert transition_count == 3  # 1->2, 2->3, 3->4
    
    # Test path sequence extraction
    sequence = get_path_sequence(df, 0, 7, 'node_id', remove_duplicates=True)
    print(f"Path sequence: {sequence}")
    assert sequence == [1, 2, 3, 4]
    
    # Test distance calculation
    distance = calculate_path_distance(df, 0, 3, 'x', 'y')
    print(f"Path distance: {distance:.2f}")
    # Should be sqrt(1^2 + 0^2) + sqrt(1^2 + 1^2) + sqrt(1^2 + 0^2) ≈ 1 + 1.41 + 1 = 3.41
    assert abs(distance - 3.41) < 0.1
    
    print("✓ Counting and utility functions tests passed")

def test_with_mock_session():
    """Test metrics with a mock session object."""
    # Create mock session
    class MockSession:
        def __init__(self):
            self.shared_resources = {
                'stream_info': {'fps': 30.0},
                'graph': MockGraph()
            }
            self.dataframe = pd.DataFrame({
                'centroid_graph_node': [1, 2, 2, 3, 3, 4, 4, 5, 5, 5],
                'centroid_x': [0, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                'centroid_y': [0, 0, 5, 5, 10, 10, 15, 15, 20, 20],
                'state': ['start'] * 3 + ['moving'] * 4 + ['goal'] * 3
            })
        
        def get_integrated_dataframe(self):
            return self.dataframe
    
    class MockGraph:
        @property
        def num_nodes(self):
            return 10
        
        @property
        def num_edges(self):
            return 15
        
        def get_shortest_path(self, start, end):
            # Simple mock path
            return list(range(start, end + 1))
    
    session = MockSession()
    
    # Test time_a_to_b
    from navigraph.analysis.metrics import time_a_to_b
    times = time_a_to_b(session, 'state', 'start', 'state', 'goal')
    print(f"Time from start to goal: {times}")
    assert len(times) == 1
    assert times[0] == 7 / 30.0  # 7 frames at 30 fps
    
    # Test velocity_a_to_b
    from navigraph.analysis.metrics import velocity_a_to_b
    velocities = velocity_a_to_b(
        session, 'state', 'start', 'state', 'goal',
        'centroid_x', 'centroid_y'
    )
    print(f"Velocity from start to goal: {velocities}")
    assert len(velocities) == 1
    
    # Test exploration_percentage
    from navigraph.analysis.metrics import exploration_percentage
    exploration = exploration_percentage(session, 'centroid_graph_node')
    print(f"Exploration percentage: {exploration:.1f}%")
    assert exploration == 50.0  # 5 unique nodes out of 10 total
    
    # Test num_values_in_path
    from navigraph.analysis.metrics import num_values_in_path
    node_counts = num_values_in_path(
        session, 'state', 'start', 'state', 'goal',
        'centroid_graph_node', mode='unique'
    )
    print(f"Unique nodes in path: {node_counts}")
    assert len(node_counts) == 1
    assert node_counts[0] == 5  # nodes 1,2,3,4,5
    
    print("✓ Mock session tests passed")

if __name__ == "__main__":
    print("Testing migrated metrics implementation...\n")
    
    test_find_path_transitions()
    print()
    
    test_count_functions()
    print()
    
    test_with_mock_session()
    print()
    
    print("\n✅ All tests passed! Migration successful.")