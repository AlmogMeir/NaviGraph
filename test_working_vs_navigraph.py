#!/usr/bin/env python3
"""Compare working script vs NaviGraph on actual data."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from navigraph.core.conversion_utils import quaternions_to_euler

# Load the actual CSV file
csv_path = Path("examples/multimodal_demo/demo_session/headOrientation.csv")
data = pd.read_csv(csv_path)

print(f"=== Loaded CSV: {data.shape[0]} samples ===")
print(f"Time range: {data['Time Stamp (ms)'].min()} to {data['Time Stamp (ms)'].max()} ms")

# Test first 20 samples
test_samples = 20
sample_data = data.head(test_samples)

print(f"\n=== NaviGraph Implementation (first {test_samples}) ===")
navigraph_euler = quaternions_to_euler(
    sample_data,
    yaw_offset=-167,
    positive_direction=-1
)
for i, angles in enumerate(navigraph_euler):
    timestamp = sample_data.iloc[i]['Time Stamp (ms)']
    print(f"  {i:2d} (t={timestamp:4.0f}ms): yaw={angles[0]:6.2f}°, pitch={angles[1]:6.2f}°, roll={angles[2]:6.2f}°")

print(f"\n=== Working Implementation (first {test_samples}) ===")
# Replicate your exact working code
quaternions = sample_data[['qw', 'qx', 'qy', 'qz']].values
euler_angles = R.from_quat(quaternions[:, [1, 2, 3, 0]]).as_euler('zyx', degrees=True)

# Create temporary DataFrame for processing (like your code)
temp_data = sample_data.copy()
temp_data['Yaw (degrees)'] = euler_angles[:, 0]
temp_data['Pitch (degrees)'] = euler_angles[:, 1]
temp_data['Roll (degrees)'] = euler_angles[:, 2]

# Cartesian normalization
temp_data['Yaw_x'] = np.cos(np.radians(temp_data['Yaw (degrees)']))
temp_data['Yaw_y'] = np.sin(np.radians(temp_data['Yaw (degrees)']))
temp_data['Yaw (degrees)'] = np.degrees(np.arctan2(temp_data['Yaw_y'], temp_data['Yaw_x']))

temp_data['Pitch_x'] = np.cos(np.radians(temp_data['Pitch (degrees)']))
temp_data['Pitch_y'] = np.sin(np.radians(temp_data['Pitch (degrees)']))
temp_data['Pitch (degrees)'] = np.degrees(np.arctan2(temp_data['Pitch_y'], temp_data['Pitch_x']))

temp_data['Roll_x'] = np.cos(np.radians(temp_data['Roll (degrees)']))
temp_data['Roll_y'] = np.sin(np.radians(temp_data['Roll (degrees)']))
temp_data['Roll (degrees)'] = np.degrees(np.arctan2(temp_data['Roll_y'], temp_data['Roll_x']))

# Yaw calibration
yaw_offset = -167
temp_data['Yaw (degrees)'] = (temp_data['Yaw (degrees)'] - yaw_offset + 180) % 360 - 180
define_positive_direction = -1
temp_data['Yaw (degrees)'] *= define_positive_direction

for i in range(test_samples):
    timestamp = temp_data.iloc[i]['Time Stamp (ms)']
    yaw = temp_data.iloc[i]['Yaw (degrees)']
    pitch = temp_data.iloc[i]['Pitch (degrees)']
    roll = temp_data.iloc[i]['Roll (degrees)']
    print(f"  {i:2d} (t={timestamp:4.0f}ms): yaw={yaw:6.2f}°, pitch={pitch:6.2f}°, roll={roll:6.2f}°")

print(f"\n=== Comparison ===")
print("Differences (NaviGraph - Working):")
for i in range(test_samples):
    ng_angles = navigraph_euler[i]
    work_yaw = temp_data.iloc[i]['Yaw (degrees)']
    work_pitch = temp_data.iloc[i]['Pitch (degrees)']
    work_roll = temp_data.iloc[i]['Roll (degrees)']

    diff_yaw = ng_angles[0] - work_yaw
    diff_pitch = ng_angles[1] - work_pitch
    diff_roll = ng_angles[2] - work_roll

    if abs(diff_yaw) > 0.01 or abs(diff_pitch) > 0.01 or abs(diff_roll) > 0.01:
        print(f"  {i:2d}: Δyaw={diff_yaw:6.2f}°, Δpitch={diff_pitch:6.2f}°, Δroll={diff_roll:6.2f}° ❌")
    else:
        print(f"  {i:2d}: MATCH ✓")

print(f"\n=== Frame Mapping with skip_index=2 ===")
print("Video frame → IMU sample → Yaw value:")
for frame_idx in range(10):
    imu_idx = frame_idx * 2
    if imu_idx < test_samples:
        ng_yaw = navigraph_euler[imu_idx][0]
        work_yaw = temp_data.iloc[imu_idx]['Yaw (degrees)']
        timestamp = temp_data.iloc[imu_idx]['Time Stamp (ms)']
        print(f"  Frame {frame_idx:2d} → IMU[{imu_idx:2d}] (t={timestamp:4.0f}ms) → NaviGraph: {ng_yaw:6.2f}°, Working: {work_yaw:6.2f}°")
    else:
        print(f"  Frame {frame_idx:2d} → IMU[{imu_idx:2d}] → OUT OF RANGE")