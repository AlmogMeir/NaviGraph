"""Camera-to-map calibration module for NaviGraph.

This module provides interactive tools for establishing spatial correspondence
between camera view and map coordinates through point-based calibration.
"""

from .interactive_calibrator import InteractiveCalibrator
from .point_selector import PointSelector, Point
from .transform_calculator import TransformCalculator, TransformMethod, CalibrationResult

__all__ = [
    "InteractiveCalibrator",
    "PointSelector", 
    "Point",
    "TransformCalculator",
    "TransformMethod",
    "CalibrationResult"
]