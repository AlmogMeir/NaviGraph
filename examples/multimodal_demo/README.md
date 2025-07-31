# NaviGraph Multimodal Demo

This example demonstrates NaviGraph's ability to integrate multiple data modalities for comprehensive behavioral analysis. The demo uses real experimental data from an apoE4 female RSC 4m AAV1 GCaMP6s memory task (021123).

## Data Modalities

### 1. Behavioral Tracking (DeepLabCut)
- **Source**: Pose estimation from video using DeepLabCut
- **File**: `memory5-11022023_withrewardafter8minDLC_resnet50_apoE_naive_4m_females_CA1_automated_1222Dec12shuffle1_225500.h5`
- **Contains**: Keypoint coordinates for multiple body parts (Nose, Midline_top, Midline_middle, Tail_base)
- **Purpose**: Primary temporal index and spatial position tracking

### 2. Neural Activity (Calcium Imaging)
- **Source**: Two-photon calcium imaging processed with Minian
- **Directory**: `demo_session/minian/` (zarr format)
- **Contains**: 
  - `C.zarr`: Calcium traces (df/f values) for 43 neurons
  - `A.zarr`: Spatial footprints for each neuron
  - `S.zarr`: Smoothed calcium traces
- **Purpose**: Neural activity correlation with behavior

### 3. Head Direction (IMU Sensor)
- **Source**: Inertial measurement unit (IMU) sensor
- **File**: `demo_session/headOrientation.csv`
- **Contains**: Quaternion data (qw, qx, qy, qz) sampled at ~30Hz
- **Purpose**: Head orientation tracking (yaw, pitch, roll angles)

### 4. Spatial Navigation
- **Source**: Camera-to-maze coordinate transformation
- **Files**: 
  - `../basic_maze/resources/maze_map.png`: Maze layout
  - `demo_session/calibration/transform_matrix.npy`: Calibration matrix
- **Purpose**: Spatial tile detection and navigation analysis

## Usage

### Run the Demo
```bash
# Navigate to demo directory
cd examples/multimodal_demo

# Run analysis pipeline
navigraph run config.yaml

# Or run with specific mode
navigraph run config.yaml --system_running_mode=analyze
```

### Available Running Modes
- `analyze`: Extract behavioral and neural metrics
- `visualize`: Create visualization outputs
- `calibrate`: Recalibrate camera-maze transformation
- `test`: Validate calibration quality

## Output Data Structure

After running the analysis, the session DataFrame will contain:

### Behavioral Columns
- `scorer_DeepLabCut_*`: Original DeepLabCut keypoint data
- `x`, `y`: Primary tracking coordinates
- `tile_id`: Current maze tile (0-288)
- `graph_node`: Binary tree node representation

### Neural Activity Columns
- `neuron_0` to `neuron_42`: df/f values for 43 neurons
- Each column contains calcium activity for that neuron at each frame

### Head Direction Columns
- `yaw`: Head yaw angle in degrees (-180 to 180)
- `pitch`: Head pitch angle in degrees
- `roll`: Head roll angle in degrees

### Example Data Access
```python
import pandas as pd

# Load processed session data
df = pd.read_pickle('output/session_data.pkl')

# Access behavioral data
position = df[['x', 'y']]
current_tile = df['tile_id']

# Access neural data
neural_activity = df[[col for col in df.columns if col.startswith('neuron_')]]

# Access head direction
head_angles = df[['yaw', 'pitch', 'roll']]

# Multimodal analysis example
import numpy as np

# Find frames where animal is at reward location
reward_frames = df[df['tile_id'] == 273]

# Get neural activity during reward
reward_neural = reward_frames[[col for col in df.columns if col.startswith('neuron_')]]

# Compute mean neural activity at reward
mean_reward_activity = reward_neural.mean()
```

## Configuration Highlights

The `config.yaml` file demonstrates several key features:

### Plugin Configuration
```yaml
data_sources:
  - name: neural_activity
    type: neural_activity
    config:
      minian_path: "./demo_session/minian"
      merge_mode: "left"
      unit_prefix: "neuron_"

  - name: head_direction
    type: head_direction
    config:
      head_direction_path: "./demo_session/headOrientation.csv"
      yaw_offset: -167
      positive_direction: -1
      skip_index: 2
```

### Data Synchronization
All data modalities are synchronized by frame index:
- Video: 30 fps baseline
- Neural: Matched to video frames
- Head direction: Downsampled using `skip_index: 2`

## Analysis Capabilities

This multimodal setup enables advanced analyses such as:

1. **Neural-Behavioral Correlations**
   - Neural activity patterns during specific behaviors
   - Spatial tuning of individual neurons
   - Head direction cell identification

2. **Temporal Dynamics**
   - Neural activity sequences during navigation
   - Head direction stability analysis
   - Behavioral state transitions

3. **Spatial Analysis**
   - Place cell detection and characterization
   - Head direction cell analysis
   - Neural ensemble dynamics in different maze regions

## Custom Analysis Development

To create custom analyzers that leverage multimodal data:

```python
from navigraph.core import IAnalyzer, AnalysisResult

class NeuralBehavioralAnalyzer(IAnalyzer):
    def analyze_session(self, session):
        df = session.get_dataframe()
        
        # Access all data modalities
        neural_cols = [col for col in df.columns if col.startswith('neuron_')]
        neural_data = df[neural_cols]
        position_data = df[['x', 'y']]
        head_direction = df[['yaw', 'pitch', 'roll']]
        
        # Perform multimodal analysis
        results = self.compute_neural_behavioral_metrics(
            neural_data, position_data, head_direction
        )
        
        return AnalysisResult(
            session_id=session.session_id,
            analyzer_name="neural_behavioral",
            metrics=results
        )
```

## File Structure
```
multimodal_demo/
├── README.md                     # This file
├── config.yaml                   # Main configuration
├── demo_session/                 # Session data
│   ├── memory5-11022023_withrewardafter8min.avi
│   ├── memory5-11022023_withrewardafter8minDLC_...h5
│   ├── headOrientation.csv
│   ├── minian/                   # Neural data (zarr format)
│   │   ├── A.zarr/              # Spatial footprints
│   │   ├── C.zarr/              # Calcium traces
│   │   ├── S.zarr/              # Smoothed traces
│   │   └── ...
│   └── calibration/
│       └── transform_matrix.npy
└── resources/
    └── README.md                 # Resources documentation
```

## Data Specifications

### Video Data
- **Format**: AVI
- **Resolution**: Variable (will be resized for processing)
- **Frame Rate**: ~30 fps
- **Duration**: ~8 minutes of memory task

### Neural Data (Minian Format)
- **Neurons**: 43 identified units
- **Sampling**: Matched to video frame rate
- **Format**: Zarr arrays with xarray structure
- **Dimensions**: frame × unit_id for calcium traces

### Head Direction Data
- **Format**: CSV with columns: Time Stamp (ms), qw, qx, qy, qz
- **Sampling**: ~30 Hz (before downsampling)
- **Conversion**: Quaternions → Euler angles with calibration

This demo provides a complete example of multimodal NaviGraph analysis, suitable as a template for similar experimental setups combining behavioral tracking, neural recording, and orientation sensing.