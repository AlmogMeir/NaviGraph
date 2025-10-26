#!/usr/bin/env python3
"""Validate NaviGraph plugin produces same first 50 values as working script."""

import pandas as pd
import numpy as np
from pathlib import Path
from navigraph.core.conversion_utils import quaternions_to_euler

# Your expected values (first 50)
expected_values = [
    (-34.061630, 37.764337, 36.925976),
    (-33.649713, 37.196262, 37.163060),
    (-33.638529, 38.760348, 37.191981),
    (-34.033832, 38.434760, 36.204460),
    (-33.723874, 38.771000, 36.345520),
    (-33.166253, 39.357184, 37.264891),
    (-32.691302, 39.257774, 37.547164),
    (-32.004844, 39.558370, 38.094695),
    (-30.600212, 40.227806, 39.850278),
    (-28.786402, 41.252775, 41.906370),
    (-27.522541, 42.089209, 42.861369),
    (-27.168308, 41.009586, 42.494445),
    (-27.928283, 36.627018, 39.243463),
    (-28.533548, 34.588069, 35.936133),
    (-29.062685, 35.813728, 34.183801),
    (-28.525161, 35.082384, 31.720884),
    (-28.088771, 30.201920, 26.484420),
    (-29.692529, 25.227042, 21.633544),
    (-23.023220, 32.356009, 29.094473),
    (-13.085570, 36.392196, 37.235090),
    (-8.238539, 33.856195, 35.007584),
    (-5.862129, 32.648051, 32.571309),
    (-3.779536, 34.928118, 34.350029),
    (4.304188, 40.515450, 41.716298),
    (11.940066, 41.166035, 42.376512),
    (21.966482, 37.670022, 42.809391),
    (34.992866, 27.286046, 54.596461),
    (39.193590, 24.158645, 57.220077),
    (41.016764, 24.935913, 60.318469),
    (42.061099, 24.759379, 63.203093),
    (40.203963, 23.200106, 58.761151),
    (40.247857, 24.881125, 52.230312),
    (39.928719, 25.870052, 52.999766),
    (36.078112, 27.630566, 48.841206),
    (31.410229, 27.575667, 41.009693),
    (30.564639, 24.323960, 43.422880),
    (26.860695, 20.332184, 42.979529),
    (19.095972, 18.336899, 31.965465),
    (17.872713, 18.310373, 36.346092),
    (16.062404, 20.262835, 45.431674),
    (16.320440, 19.446164, 45.074937),
    (13.990256, 20.051922, 43.101287),
    (14.684341, 18.182462, 42.291168),
    (15.299344, 16.481388, 42.115249),
    (15.536357, 15.587278, 41.344461),
    (21.546294, 11.985931, 41.216935),
    (27.339528, 9.622366, 42.726044),
    (31.473786, 6.405858, 42.294462),
    (29.761473, 5.698505, 40.185982),
    (31.438613, 4.710428, 37.731276)
]

# Load CSV and test NaviGraph implementation
csv_path = Path("examples/multimodal_demo/demo_session/headOrientation.csv")
data = pd.read_csv(csv_path)

print("=== Validating First 50 Values ===")
print(f"CSV loaded: {data.shape[0]} samples")

# Get first 50 samples
test_data = data.head(50)

# Run NaviGraph conversion
navigraph_euler = quaternions_to_euler(
    test_data,
    yaw_offset=-167,
    positive_direction=-1
)

print(f"\nNaviGraph produced {len(navigraph_euler)} values")
print(f"Expected {len(expected_values)} values")

# Compare each value
print(f"\n{'Index':<5} {'Expected Yaw':<12} {'NaviGraph Yaw':<13} {'Δ Yaw':<8} {'Expected Pitch':<14} {'NaviGraph Pitch':<15} {'Δ Pitch':<9} {'Expected Roll':<13} {'NaviGraph Roll':<14} {'Δ Roll':<8} {'Match'}")
print("="*140)

all_match = True
for i in range(min(50, len(navigraph_euler), len(expected_values))):
    exp_yaw, exp_pitch, exp_roll = expected_values[i]
    ng_yaw, ng_pitch, ng_roll = navigraph_euler[i]

    diff_yaw = abs(ng_yaw - exp_yaw)
    diff_pitch = abs(ng_pitch - exp_pitch)
    diff_roll = abs(ng_roll - exp_roll)

    # Consider match if difference < 0.000001 (6 decimal places)
    is_match = diff_yaw < 0.000001 and diff_pitch < 0.000001 and diff_roll < 0.000001
    match_str = "✓" if is_match else "❌"

    if not is_match:
        all_match = False

    print(f"{i:<5} {exp_yaw:<12.6f} {ng_yaw:<13.6f} {diff_yaw:<8.6f} {exp_pitch:<14.6f} {ng_pitch:<15.6f} {diff_pitch:<9.6f} {exp_roll:<13.6f} {ng_roll:<14.6f} {diff_roll:<8.6f} {match_str}")

print("="*140)
if all_match:
    print("🎉 SUCCESS: All 50 values match perfectly!")
else:
    print("❌ MISMATCH: Some values don't match")

# Summary statistics
yaw_diffs = [abs(navigraph_euler[i][0] - expected_values[i][0]) for i in range(min(50, len(navigraph_euler), len(expected_values)))]
pitch_diffs = [abs(navigraph_euler[i][1] - expected_values[i][1]) for i in range(min(50, len(navigraph_euler), len(expected_values)))]
roll_diffs = [abs(navigraph_euler[i][2] - expected_values[i][2]) for i in range(min(50, len(navigraph_euler), len(expected_values)))]

print(f"\nSummary Statistics:")
print(f"  Max Yaw difference:   {max(yaw_diffs):.9f}°")
print(f"  Max Pitch difference: {max(pitch_diffs):.9f}°")
print(f"  Max Roll difference:  {max(roll_diffs):.9f}°")
print(f"  Mean Yaw difference:  {np.mean(yaw_diffs):.9f}°")
print(f"  Mean Pitch difference:{np.mean(pitch_diffs):.9f}°")
print(f"  Mean Roll difference: {np.mean(roll_diffs):.9f}°")

# Test the frame mapping with skip_index=2
print(f"\n=== Frame Mapping Test (skip_index=2) ===")
print("Testing that frame_idx * 2 gives correct IMU sample:")
for frame_idx in range(10):
    imu_idx = frame_idx * 2
    if imu_idx < len(navigraph_euler):
        ng_yaw = navigraph_euler[imu_idx][0]
        exp_yaw = expected_values[imu_idx][0]
        diff = abs(ng_yaw - exp_yaw)
        match = "✓" if diff < 0.000001 else "❌"
        print(f"  Frame {frame_idx:2d} → IMU[{imu_idx:2d}] → Expected: {exp_yaw:8.3f}°, NaviGraph: {ng_yaw:8.3f}°, Δ: {diff:.6f}° {match}")
    else:
        print(f"  Frame {frame_idx:2d} → IMU[{imu_idx:2d}] → OUT OF RANGE")

print(f"\n=== Configuration Validation ===")
print("Current config values:")
print("  yaw_offset: -167")
print("  positive_direction: -1")
print("  skip_index: 2")
print("\nThese should match your working script values ✓")