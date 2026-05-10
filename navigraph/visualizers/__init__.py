"""Visualizer functions for NaviGraph.

Functions that process video frames for visualization.
Use @register_visualizer decorator to auto-register.
"""

# Import all visualizers to trigger registration
from . import bodyparts
from . import graph_overlay
from . import map_overlay
from . import neural_segmentation
from . import text_display
from . import time_series
from . import split_view