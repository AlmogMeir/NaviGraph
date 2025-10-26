#!/usr/bin/env python3
"""Test script to validate head direction conversion matches working implementation."""

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from navigraph.core.conversion_utils import quaternions_to_euler

# Test with some sample quaternion data
test_data = pd.DataFrame({
    'qw': [1.0, 0.9239, 0.7071, 0.0, -0.7071],
    'qx': [0.0, 0.0, 0.0, 0.0, 0.0],
    'qy': [0.0, 0.0, 0.0, 1.0, 0.0],
    'qz': [0.0, 0.3827, 0.7071, 0.0, 0.7071]
})

print("=== Testing Head Direction Conversion ===")
print(f"Test data shape: {test_data.shape}")
print("\nQuaternion data:")
print(test_data)

# Method 1: NaviGraph implementation
print("\n=== NaviGraph Implementation ===")
try:
    navigraph_euler = quaternions_to_euler(
        test_data,
        yaw_offset=-167,
        positive_direction=-1
    )
    print("NaviGraph Euler angles (yaw, pitch, roll):")
    for i, angles in enumerate(navigraph_euler):
        print(f"  Row {i}: yaw={angles[0]:.2f}°, pitch={angles[1]:.2f}°, roll={angles[2]:.2f}°")
except Exception as e:
    print(f"NaviGraph implementation failed: {e}")

# Method 2: Working implementation (from your code)
print("\n=== Working Implementation (Your Code) ===")
try:
    quaternions = test_data[['qw', 'qx', 'qy', 'qz']].values

    # Convert quaternions to Euler angles (ZYX: Yaw, Pitch, Roll)
    euler_angles = R.from_quat(quaternions[:, [1, 2, 3, 0]]).as_euler('zyx', degrees=True)

    # Create temporary DataFrame for processing
    temp_data = pd.DataFrame()
    temp_data['Yaw (degrees)'] = euler_angles[:, 0]
    temp_data['Pitch (degrees)'] = euler_angles[:, 1]
    temp_data['Roll (degrees)'] = euler_angles[:, 2]

    # Normalize and unwrap angles - Convert to Cartesian (unit circle)
    temp_data['Yaw_x'] = np.cos(np.radians(temp_data['Yaw (degrees)']))
    temp_data['Yaw_y'] = np.sin(np.radians(temp_data['Yaw (degrees)']))
    temp_data['Yaw (degrees)'] = np.degrees(np.arctan2(temp_data['Yaw_y'], temp_data['Yaw_x']))

    temp_data['Pitch_x'] = np.cos(np.radians(temp_data['Pitch (degrees)']))
    temp_data['Pitch_y'] = np.sin(np.radians(temp_data['Pitch (degrees)']))
    temp_data['Pitch (degrees)'] = np.degrees(np.arctan2(temp_data['Pitch_y'], temp_data['Pitch_x']))

    temp_data['Roll_x'] = np.cos(np.radians(temp_data['Roll (degrees)']))
    temp_data['Roll_y'] = np.sin(np.radians(temp_data['Roll (degrees)']))
    temp_data['Roll (degrees)'] = np.degrees(np.arctan2(temp_data['Roll_y'], temp_data['Roll_x']))

    # Define Yaw calibration
    yaw_offset = -167
    temp_data['Yaw (degrees)'] = (temp_data['Yaw (degrees)'] - yaw_offset + 180) % 360 - 180

    define_positive_direction = -1
    temp_data['Yaw (degrees)'] *= define_positive_direction

    print("Working implementation Euler angles (yaw, pitch, roll):")
    for i in range(len(temp_data)):
        yaw = temp_data.iloc[i]['Yaw (degrees)']
        pitch = temp_data.iloc[i]['Pitch (degrees)']
        roll = temp_data.iloc[i]['Roll (degrees)']
        print(f"  Row {i}: yaw={yaw:.2f}°, pitch={pitch:.2f}°, roll={roll:.2f}°")

except Exception as e:
    print(f"Working implementation failed: {e}")

print("\n=== Comparison ===")
try:
    if 'navigraph_euler' in locals() and 'temp_data' in locals():
        print("Differences (NaviGraph - Working):")
        for i in range(min(len(navigraph_euler), len(temp_data))):
            ng_yaw = navigraph_euler[i][0]
            ng_pitch = navigraph_euler[i][1]
            ng_roll = navigraph_euler[i][2]

            work_yaw = temp_data.iloc[i]['Yaw (degrees)']
            work_pitch = temp_data.iloc[i]['Pitch (degrees)']
            work_roll = temp_data.iloc[i]['Roll (degrees)']

            yaw_diff = ng_yaw - work_yaw
            pitch_diff = ng_pitch - work_pitch
            roll_diff = ng_roll - work_roll

            print(f"  Row {i}: Δyaw={yaw_diff:.2f}°, Δpitch={pitch_diff:.2f}°, Δroll={roll_diff:.2f}°")

            if abs(yaw_diff) > 0.01 or abs(pitch_diff) > 0.01 or abs(roll_diff) > 0.01:
                print(f"    ^^^^ MISMATCH DETECTED ^^^^")
    else:
        print("Could not compare - one implementation failed")
except Exception as e:
    print(f"Comparison failed: {e}")

print("\n=== Test Frame Mapping ===")
# Test skip_index behavior
print("Testing skip_index=2 behavior:")
for frame_idx in range(5):
    imu_idx = frame_idx * 2
    print(f"  Video frame {frame_idx} → IMU sample {imu_idx}")