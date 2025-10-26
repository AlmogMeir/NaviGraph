#!/usr/bin/env python3
"""Test with actual data from the demo session."""

import pandas as pd
import numpy as np
from pathlib import Path
from navigraph.core.conversion_utils import quaternions_to_euler

# Try to find the actual CSV file
demo_path = Path("examples/multimodal_demo/demo_session")
csv_files = list(demo_path.glob("*headOrientation*.csv"))

if csv_files:
    csv_path = csv_files[0]
    print(f"Found CSV file: {csv_path}")

    # Load the data
    data = pd.read_csv(csv_path)
    print(f"CSV shape: {data.shape}")
    print(f"CSV columns: {list(data.columns)}")
    print("\nFirst 5 rows:")
    print(data.head())

    # Check for quaternion columns
    quat_cols = ['qw', 'qx', 'qy', 'qz']
    missing_cols = [col for col in quat_cols if col not in data.columns]
    if missing_cols:
        print(f"\nMissing quaternion columns: {missing_cols}")
        print("Available columns that might be quaternions:")
        for col in data.columns:
            if any(q in col.lower() for q in ['qw', 'qx', 'qy', 'qz', 'quaternion']):
                print(f"  {col}")
    else:
        print(f"\nAll quaternion columns found: {quat_cols}")

        # Test conversion on first 10 samples
        sample_data = data.head(10)
        print(f"\nTesting conversion on first 10 samples...")

        try:
            euler_result = quaternions_to_euler(
                sample_data,
                yaw_offset=-167,
                positive_direction=-1
            )
            print("NaviGraph conversion results (first 10):")
            for i, angles in enumerate(euler_result):
                print(f"  Sample {i}: yaw={angles[0]:.2f}°, pitch={angles[1]:.2f}°, roll={angles[2]:.2f}°")

            # Test frame mapping for skip_index=2
            print(f"\nFrame mapping test (skip_index=2):")
            for frame_idx in range(5):
                imu_idx = frame_idx * 2
                if imu_idx < len(euler_result):
                    angles = euler_result[imu_idx]
                    print(f"  Video frame {frame_idx} → IMU[{imu_idx}] → yaw={angles[0]:.2f}°")
                else:
                    print(f"  Video frame {frame_idx} → IMU[{imu_idx}] → OUT OF RANGE")

        except Exception as e:
            print(f"Conversion failed: {e}")

else:
    print("No CSV file found in examples/multimodal_demo/demo_session/")
    print("Available files:")
    if demo_path.exists():
        for f in demo_path.iterdir():
            print(f"  {f.name}")
    else:
        print(f"Demo session path does not exist: {demo_path}")