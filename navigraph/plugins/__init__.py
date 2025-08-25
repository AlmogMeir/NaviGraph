"""Data source plugins for NaviGraph unified architecture.

This package contains all unified NaviGraph plugins that implement the
NaviGraphPlugin interface. Each plugin provides or augments data in the
session DataFrame through provide() and augment_data() methods.

Available unified plugins:
- pose_tracking: DeepLabCut pose estimation data with bodypart tracking
- calibration: Camera calibration transformation matrix
- stream_info: Video stream metadata (fps, frame count, duration)
- map_location: Spatial coordinate transformation using calibration
- graph_location: Graph node/edge mapping from coordinates
- neural_activity: Neural activity data from Minian zarr format
- head_direction: Head orientation from quaternion data with Euler conversion
"""

# Import all unified plugins to trigger registration
from . import pose_tracking
from . import calibration
from . import stream_info
from . import map_location
from . import graph_location
from . import neural_activity
from . import head_direction

__all__ = [
    'pose_tracking',
    'calibration',
    'stream_info',
    'map_location',
    'graph_location',
    'neural_activity',
    'head_direction'
]