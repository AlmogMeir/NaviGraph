#!/usr/bin/env python3
"""Comprehensive test script for migrated metrics with edge cases."""

import pandas as pd
import numpy as np
from navigraph.analysis.utils import (
    find_path_transitions,
    calculate_path_distance,
    count_unique_values_in_segment,
    count_value_transitions,
    get_path_sequence
)

def test_none_handling():
    """Test that functions properly handle None and NaN values."""
    print("Testing None/NaN handling...")
    
    # Create dataframe with None and NaN values
    data = {
        'state': ['start', None, 'moving', np.nan, 'moving', 'goal'],
        'node_id': [1, None, 2, np.nan, 3, 4],
        'x': [0, np.nan, 10, None, 20, 30],
        'y': [0, None, 5, np.nan, 10, 15]
    }
    df = pd.DataFrame(data)
    
    # Test find_path_transitions skips None/NaN
    transitions = find_path_transitions(df, 'state', 'start', 'state', 'goal')
    print(f"  Transitions with None/NaN: {transitions}")
    assert len(transitions) == 1
    assert transitions[0] == (0, 5)  # Should find start at 0 and goal at 5
    
    # Test counting functions skip None/NaN
    unique_count = count_unique_values_in_segment(df, 0, 5, 'node_id')
    print(f"  Unique nodes (excluding None/NaN): {unique_count}")
    assert unique_count == 4  # Should only count 1,2,3,4, not None/NaN
    
    # Test transitions skip None/NaN
    transition_count = count_value_transitions(df, 0, 5, 'node_id')
    print(f"  Node transitions (excluding None/NaN): {transition_count}")
    assert transition_count == 3  # 1->2, 2->3, 3->4
    
    # Test distance calculation skips None/NaN segments
    distance = calculate_path_distance(df, 0, 5, 'x', 'y')
    print(f"  Distance (skipping None/NaN): {distance:.2f}")
    # Only valid consecutive pairs: (20,10) to (30,15)
    # Distance = sqrt((30-20)^2 + (15-10)^2) = sqrt(100+25) = sqrt(125)
    expected = np.sqrt(125)
    assert abs(distance - expected) < 0.1
    
    print("  ✓ None/NaN handling tests passed")

def test_duplicate_handling():
    """Test handling of duplicate values with first_occurrence_only."""
    print("Testing duplicate handling...")
    
    # Create dataframe with duplicate goal values
    data = {
        'state': ['start', 'moving', 'goal', 'goal', 'moving', 'goal'],
        'node_id': [1, 2, 3, 3, 4, 5],
        'time': [0, 1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)
    
    # Test first occurrence only (default)
    transitions_first = find_path_transitions(
        df, 'state', 'start', 'state', 'goal', 
        first_occurrence_only=True
    )
    print(f"  First occurrence only: {transitions_first}")
    assert len(transitions_first) == 1
    assert transitions_first[0] == (0, 2)  # Should use first 'goal' at index 2
    
    # Test last occurrence
    transitions_last = find_path_transitions(
        df, 'state', 'start', 'state', 'goal',
        first_occurrence_only=False
    )
    print(f"  Last occurrence: {transitions_last}")
    assert len(transitions_last) == 1
    assert transitions_last[0] == (0, 5)  # Should use last 'goal' at index 5
    
    print("  ✓ Duplicate handling tests passed")

def test_pixel_to_meter_conversion():
    """Test pixel to meter conversion in distance calculation."""
    print("Testing pixel to meter conversion...")
    
    data = {
        'x': [0, 100, 200, 300],
        'y': [0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    
    # Test without conversion (pixels)
    distance_pixels = calculate_path_distance(df, 0, 3, 'x', 'y')
    print(f"  Distance in pixels: {distance_pixels}")
    assert distance_pixels == 300  # 300 pixels total
    
    # Test with conversion (meters)
    pixel_to_meter = 0.01  # 1 pixel = 0.01 meters (1cm)
    distance_meters = calculate_path_distance(df, 0, 3, 'x', 'y', pixel_to_meter)
    print(f"  Distance in meters: {distance_meters}")
    assert distance_meters == 3.0  # 300 pixels * 0.01 = 3 meters
    
    print("  ✓ Pixel to meter conversion tests passed")

def test_with_realistic_data():
    """Test with realistic navigation data including None values."""
    print("Testing with realistic navigation data...")
    
    # Create realistic dataframe with some missing graph locations
    np.random.seed(42)
    n_frames = 100
    
    # Simulate animal moving with some frames where location is unknown
    node_sequence = []
    x_coords = []
    y_coords = []
    
    current_node = 1
    current_x, current_y = 0, 0
    
    for i in range(n_frames):
        # 10% chance of unknown location (None)
        if np.random.random() < 0.1:
            node_sequence.append(None)
            x_coords.append(np.nan)
            y_coords.append(np.nan)
        else:
            # Move to next node occasionally
            if i > 0 and i % 20 == 0:
                current_node += 1
            node_sequence.append(current_node)
            # Add some movement
            current_x += np.random.randn() * 5
            current_y += np.random.randn() * 5
            x_coords.append(current_x)
            y_coords.append(current_y)
    
    # Add clear start and goal states
    state_sequence = ['exploring'] * 30 + ['moving'] * 40 + ['rewarded'] * 30
    
    df = pd.DataFrame({
        'frame': range(n_frames),
        'state': state_sequence,
        'node_id': node_sequence,
        'x': x_coords,
        'y': y_coords
    })
    
    # Test finding transitions
    transitions = find_path_transitions(df, 'state', 'exploring', 'state', 'rewarded')
    print(f"  Found {len(transitions)} transition(s)")
    assert len(transitions) == 1
    
    # Test counting unique nodes (should skip None)
    start_idx, end_idx = transitions[0]
    unique_nodes = count_unique_values_in_segment(df, start_idx, end_idx, 'node_id')
    print(f"  Unique nodes visited: {unique_nodes}")
    assert unique_nodes > 0  # Should have some valid nodes
    
    # Test distance calculation (should skip NaN coordinates)
    distance = calculate_path_distance(df, start_idx, end_idx, 'x', 'y')
    print(f"  Distance traveled: {distance:.2f} pixels")
    assert distance > 0  # Should have some valid movement
    
    print("  ✓ Realistic data tests passed")

def test_metrics_with_none_values():
    """Test metrics functions with None values."""
    print("Testing metrics with None values...")
    
    # Create mock session with None values
    class MockSession:
        def __init__(self):
            self.shared_resources = {
                'stream_info': {'fps': 30.0},
                'graph': MockGraph(),
                'map_metadata': {'pixel_to_meter': 0.001}  # 1mm per pixel
            }
            # Include None values in data
            self.dataframe = pd.DataFrame({
                'centroid_graph_node': [1, None, 2, 2, None, 3, 3, 4, None, 5],
                'centroid_x': [0, np.nan, 10, 15, None, 20, 25, 30, np.nan, 40],
                'centroid_y': [0, None, 0, 5, np.nan, 5, 10, 10, None, 15],
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
            return list(range(start, end + 1))
    
    session = MockSession()
    
    # Test time_a_to_b with None values
    from navigraph.analysis.metrics import time_a_to_b
    times = time_a_to_b(session, 'state', 'start', 'state', 'goal')
    print(f"  Time from start to goal (with Nones): {times}")
    assert len(times) == 1
    assert times[0] == 7 / 30.0  # 7 frames at 30 fps
    
    # Test velocity_a_to_b with None values and pixel_to_meter
    from navigraph.analysis.metrics import velocity_a_to_b
    velocities = velocity_a_to_b(
        session, 'state', 'start', 'state', 'goal',
        'centroid_x', 'centroid_y'  # Will use pixel_to_meter from map_metadata
    )
    print(f"  Velocity (m/s) from start to goal: {velocities}")
    assert len(velocities) == 1
    assert velocities[0] > 0  # Should have positive velocity
    
    # Test exploration_percentage with None values
    from navigraph.analysis.metrics import exploration_percentage
    exploration = exploration_percentage(session, 'centroid_graph_node')
    print(f"  Exploration percentage (excluding Nones): {exploration:.1f}%")
    # Should count only non-None unique nodes: 1,2,3,4,5 = 5 out of 10
    assert exploration == 50.0
    
    # Test num_values_in_path with None values
    from navigraph.analysis.metrics import num_values_in_path
    node_counts = num_values_in_path(
        session, 'state', 'start', 'state', 'goal',
        'centroid_graph_node', mode='unique'
    )
    print(f"  Unique nodes in path (excluding Nones): {node_counts}")
    assert len(node_counts) == 1
    assert node_counts[0] == 4  # nodes 1,2,3,4 (5 is after goal state starts)
    
    print("  ✓ Metrics with None values tests passed")

if __name__ == "__main__":
    print("Testing comprehensive metrics implementation...\n")
    
    test_none_handling()
    print()
    
    test_duplicate_handling()
    print()
    
    test_pixel_to_meter_conversion()
    print()
    
    test_with_realistic_data()
    print()
    
    test_metrics_with_none_values()
    print()
    
    print("\n✅ All comprehensive tests passed! Migration handles edge cases correctly.")