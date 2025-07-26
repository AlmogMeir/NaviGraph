"""Calibrator utilities for NaviGraph calibration plugin.

This module contains utility classes migrated from the original calibrator/utils.py,
providing interactive point capture functionality for camera calibration.
"""

import cv2
from typing import List, Tuple, Dict, Any


class PointCapture:
    """Interactive point capture utility for calibration.
    
    This class provides an interactive interface for capturing calibration points
    from video frames or images. Users can click on points in an image window
    to select corresponding locations for calibration matrix calculation.
    
    Migrated from calibrator/utils.py with full functionality preservation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize point capture with configuration.
        
        Args:
            config: Configuration dictionary containing calibrator_parameters
        """
        self._config = config
        self.points: List[Tuple[int, int]] = []
        self.frame = None
        # TODO: add counter and put text in click_event function
        # self.count = 0
    
    @property 
    def config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        return self._config
    
    def click_event(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse click events for point capture.
        
        Args:
            event: OpenCV mouse event type
            x: X coordinate of click
            y: Y coordinate of click  
            flags: Event flags
            param: Additional parameters (unused)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get calibration parameters for visualization
            calib_params = self._config.get('calibrator_parameters', {})
            point_params = calib_params.get('points_capture_parameters', {})
            
            radius = point_params.get('radius', 9)
            color = point_params.get('color', (208, 224, 64))
            thickness = point_params.get('thickness', 3)
            
            # Convert color string to tuple if needed (preserving original eval logic)
            if isinstance(color, str):
                try:
                    color = eval(color)
                except:
                    color = (208, 224, 64)  # Default turquoise
            
            # Draw circle at clicked point
            cv2.circle(self.frame, (x, y), radius, color, thickness)
            
            # Add point to list
            self.points.append((x, y))
    
    def capture_points(self, frame) -> List[Tuple[int, int]]:
        """Capture calibration points interactively from a frame.
        
        Args:
            frame: Input frame/image for point capture
            
        Returns:
            List of captured point coordinates as (x, y) tuples
        """
        # TODO: add window name options and move destroy all windows
        self.frame = frame.copy()
        self.points = []
        
        # Create window and set mouse callback
        cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Capture', self.click_event)
        cv2.imshow('Capture', self.frame)
        
        # Wait for user input
        while True:
            cv2.imshow('Capture', self.frame)
            key = cv2.waitKey(1)
            if key & 0xFF == 13:  # Enter button
                cv2.destroyAllWindows()
                break
        
        return self.points