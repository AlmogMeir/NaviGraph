"""Camera-to-map calibration module for NaviGraph.

This module provides interactive tools for establishing spatial correspondence
between camera view and map coordinates through point-based calibration.
"""

from .interactive_calibrator import InteractiveCalibrator
from .calibration_tester import CalibrationTester
from .point_selector import PointSelector, Point
from .map_point_sets import (
    MapPointSet,
    load_map_point_set,
    load_map_point_sets_from_path,
    load_map_point_sets_from_config,
    find_map_point_set,
    DEFAULT_POINT_SETS_DIR,
)
from .transform_calculator import TransformCalculator, TransformMethod, CalibrationResult

__all__ = [
    "InteractiveCalibrator",
    "CalibrationTester",
    "PointSelector",
    "Point",
    "MapPointSet",
    "load_map_point_set",
    "load_map_point_sets_from_path",
    "load_map_point_sets_from_config",
    "find_map_point_set",
    "DEFAULT_POINT_SETS_DIR",
    "TransformCalculator",
    "TransformMethod",
    "CalibrationResult"
]