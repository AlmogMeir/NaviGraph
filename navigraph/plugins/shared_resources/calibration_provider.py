"""Calibration provider shared resource for NaviGraph.

This plugin provides camera calibration and coordinate transformation utilities,
making transformation matrices available to data sources that need to convert
between image coordinates and spatial coordinates.

Includes full MazeCalibrator functionality migrated from calibrator/maze_calibrator.py.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import os
import cv2

from ...core.interfaces import ISharedResource, SharedResourceError  
from ...core.registry import register_shared_resource_plugin
from .calibrator_utils import PointCapture


@register_shared_resource_plugin("calibration_provider")
class CalibrationProviderResource(ISharedResource):
    """Provides camera calibration and coordinate transformation utilities.
    
    This shared resource loads and manages camera calibration data needed for
    coordinate transformation between image space and spatial coordinates.
    It preserves the existing calibration logic from the MapLabeler class.
    """
    
    def __init__(self):
        """Initialize empty calibration provider."""
        self._transformation_matrix = None
        self._calibration_config = None
        self._initialized = False
        self._transform_matrix_path = None
        self._point_capture_utility = None
    
    def initialize_resource(
        self, 
        resource_config: Dict[str, Any], 
        logger
    ) -> None:
        """Initialize calibration provider with configuration.
        
        Args:
            resource_config: Configuration containing calibration parameters
            logger: Logger for initialization messages
            
        Raises:
            SharedResourceError: If initialization fails
        """
        try:
            logger.info("Initializing calibration provider resource")
            
            # Get transformation matrix
            transform_matrix = resource_config.get('transform_matrix')
            pre_calculated_path = resource_config.get('pre_calculated_transform_matrix_path')
            
            if transform_matrix is not None:
                # Use provided transformation matrix
                self._transformation_matrix = np.array(transform_matrix)
                logger.info("Using provided transformation matrix")
                
            elif pre_calculated_path is not None:
                # Load from file (preserving existing logic)
                if not os.path.isfile(pre_calculated_path):
                    raise SharedResourceError(
                        f"Calibration file not found at path: {pre_calculated_path}"
                    )
                
                self._transformation_matrix = np.load(pre_calculated_path)
                logger.info(f"Loaded transformation matrix from: {Path(pre_calculated_path).name}")
                
            else:
                raise SharedResourceError(
                    "Calibration provider requires either 'transform_matrix' or "
                    "'pre_calculated_transform_matrix_path' in configuration"
                )
            
            # Validate transformation matrix
            if self._transformation_matrix.shape != (3, 3):
                raise SharedResourceError(
                    f"Transformation matrix must be 3x3, got shape: {self._transformation_matrix.shape}"
                )
            
            self._calibration_config = resource_config.copy()
            
            # Initialize point capture utility for interactive calibration
            self._point_capture_utility = PointCapture(self._calibration_config)
            
            self._initialized = True
            
            logger.info(
                f"✓ Calibration provider initialized: 3x3 transformation matrix loaded"
            )
            
        except Exception as e:
            raise SharedResourceError(
                f"Failed to initialize calibration provider: {str(e)}"
            ) from e
    
    def cleanup_resource(self, logger) -> None:
        """Clean up calibration provider resources."""
        logger.debug("Cleaning up calibration provider resource")
        self._transformation_matrix = None
        self._calibration_config = None
        self._transform_matrix_path = None
        self._point_capture_utility = None
        self._initialized = False
    
    def is_initialized(self) -> bool:
        """Check if calibration provider is initialized."""
        return self._initialized
    
    def get_required_config_keys(self) -> list:
        """Return required configuration keys."""
        return []  # Either transform_matrix OR pre_calculated_transform_matrix_path
    
    def get_transformation_matrix(self) -> np.ndarray:
        """Get the camera calibration transformation matrix.
        
        Returns:
            3x3 transformation matrix for coordinate conversion
            
        Raises:
            SharedResourceError: If not initialized
        """
        if not self._initialized:
            raise SharedResourceError("Calibration provider not initialized")
        return self._transformation_matrix.copy()
    
    def get_calibration_configuration(self) -> Dict[str, Any]:
        """Get calibration configuration.
        
        Returns:
            Dictionary with calibration settings
            
        Raises:
            SharedResourceError: If not initialized
        """
        if not self._initialized:
            raise SharedResourceError("Calibration provider not initialized")
        return self._calibration_config.copy()
    
    def transform_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Transform coordinates using the calibration matrix.
        
        Args:
            points: Array of points to transform, shape (N, 2) for N points
            
        Returns:
            Transformed points array
            
        Raises:
            SharedResourceError: If not initialized or invalid input
        """
        if not self._initialized:
            raise SharedResourceError("Calibration provider not initialized")
        
        try:
            # Ensure points are in the right format
            points_array = np.array(points, dtype='float32')
            
            if points_array.ndim == 1:
                # Single point (x, y)
                if len(points_array) != 2:
                    raise SharedResourceError("Single point must have 2 coordinates (x, y)")
                points_array = points_array.reshape(1, 1, 2)
            elif points_array.ndim == 2:
                # Multiple points
                if points_array.shape[1] != 2:
                    raise SharedResourceError("Points must have 2 coordinates per point (x, y)")
                points_array = points_array.reshape(-1, 1, 2)
            else:
                raise SharedResourceError("Points array must be 1D or 2D")
            
            # Apply perspective transformation (preserving existing cv2 logic)
            import cv2
            transformed = cv2.perspectiveTransform(points_array, self._transformation_matrix)
            
            # Return in same format as input
            if len(transformed) == 1:
                return transformed.ravel().astype(int)
            else:
                return transformed.reshape(-1, 2).astype(int)
                
        except Exception as e:
            raise SharedResourceError(
                f"Coordinate transformation failed: {str(e)}"
            ) from e
    
    def capture_points(self, video_path: str) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Capture calibration points from a video frame.
        
        Args:
            video_path: Path to video file for point capture
            
        Returns:
            Tuple of (points_array, frame_shape)
            
        Raises:
            SharedResourceError: If video cannot be loaded or point capture fails
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap:
                raise SharedResourceError(f"Problem loading video path: {video_path}")
            
            ret, frame = cap.read()
            if not ret:
                raise SharedResourceError(f"Problem reading frame from: {video_path}")
            
            cap.release()
            
            print("Hello, Maze-Master. Please choose a minimum of 4 points. Press 'Enter' when done:")
            points_captured = np.round(np.array(self._point_capture_utility.capture_points(frame)))
            print(f"Points captured: {points_captured}")
            
            return points_captured, frame.shape
            
        except Exception as e:
            raise SharedResourceError(
                f"Point capture failed: {str(e)}"
            ) from e
    
    def find_transform_matrix(
        self, 
        video_path: str, 
        map_path: str,
        full_config: Dict[str, Any]
    ) -> np.ndarray:
        """Find transformation matrix using interactive point capture.
        
        Args:
            video_path: Path to video file
            map_path: Path to map image file  
            full_config: Full experiment configuration
            
        Returns:
            Calculated transformation matrix
            
        Raises:
            SharedResourceError: If calibration process fails
        """
        try:
            # Capture points from video and map
            frame_points, frame_shape = self.capture_points(video_path)
            map_points, map_shape = self.capture_points(map_path)
            
            # Get calibration parameters
            calib_params = full_config.get('calibrator_parameters', {})
            registration_method = calib_params.get('registration_method', 'homography&ransac')
            
            # Calculate transformation matrix based on method
            if registration_method == 'affine':
                self._transformation_matrix = cv2.getPerspectiveTransform(
                    frame_points.astype(np.float32),
                    map_points.astype(np.float32)
                )
            
            elif registration_method == 'homography':
                self._transformation_matrix, _ = cv2.findHomography(
                    frame_points.astype(np.float32),
                    map_points.astype(np.float32),
                    method=0
                )
            
            elif registration_method == 'homography&ransac':
                self._transformation_matrix, _ = cv2.findHomography(
                    frame_points.astype(np.float32),
                    map_points.astype(np.float32),
                    method=cv2.RANSAC,
                    ransacReprojThreshold=30
                )
            
            else:
                raise SharedResourceError(f'Registration method not supported: {registration_method}')
            
            print(f'Calculated transform matrix: {self._transformation_matrix}')
            
            # Save transformation matrix if requested
            if calib_params.get('save_transform_matrix', False):
                save_path = calib_params.get('path_to_save_calibration_files', '.')
                calibration_dir = os.path.join(save_path, 'calibration_files')
                os.makedirs(calibration_dir, exist_ok=True)
                
                self._transform_matrix_path = os.path.join(calibration_dir, 'transform_matrix.npy')
                np.save(self._transform_matrix_path, self._transformation_matrix)
                print(f'Transform matrix saved at: {self._transform_matrix_path}')
            
            return self._transformation_matrix
            
        except Exception as e:
            raise SharedResourceError(
                f"Transform matrix calculation failed: {str(e)}"
            ) from e
    
    def test_calibration(
        self,
        video_path: str,
        map_path: str, 
        matrix_path_or_array,
        full_config: Dict[str, Any]
    ) -> None:
        """Test calibration quality using interactive point selection.
        
        Args:
            video_path: Path to video file
            map_path: Path to map image file
            matrix_path_or_array: Either path to saved matrix or matrix array
            full_config: Full experiment configuration
            
        Raises:
            SharedResourceError: If testing fails
        """
        try:
            # Load map image
            map_img = cv2.imread(map_path)
            if map_img is None:
                raise SharedResourceError(f"Could not load map image: {map_path}")
            
            # Load video
            cap = cv2.VideoCapture(video_path)
            if not cap:
                raise SharedResourceError(f"Problem loading video path: {video_path}")
            
            ret, frame = cap.read()
            if not ret:
                raise SharedResourceError(f"Problem reading frame from: {video_path}")
            
            cap.release()
            
            # Get transformation matrix
            if isinstance(matrix_path_or_array, np.ndarray):
                matrix = matrix_path_or_array
            else:
                if not os.path.isfile(matrix_path_or_array):
                    raise SharedResourceError(f"Matrix file not found: {matrix_path_or_array}")
                matrix = np.load(matrix_path_or_array)
            
            print("Hello, Maze-Master. Please choose new points to test. Press 'Enter' when done:")
            
            # Capture test points
            points = self._point_capture_utility.capture_points(frame)
            points_on_map = cv2.perspectiveTransform(
                np.array([points], dtype='float32'), matrix
            )[0]
            
            # Get test visualization parameters
            calib_params = full_config.get('calibrator_parameters', {})
            test_params = calib_params.get('map_test_parameters', {})
            
            radius = test_params.get('radius', 9)
            color = test_params.get('color', (255, 0, 255))
            thickness = test_params.get('thickness', 3)
            
            # Convert color string to tuple if needed (preserving original eval logic)
            if isinstance(color, str):
                try:
                    color = eval(color)
                except:
                    color = (255, 0, 255)  # Default pink
            
            # Draw test points on map
            for x, y in points_on_map:
                cv2.circle(
                    map_img, 
                    (x.astype(int), y.astype(int)),
                    radius,
                    color,
                    thickness
                )
            
            # Display test result
            print('Press Enter to close window . . . ')
            while True:
                cv2.imshow('test', map_img)
                key = cv2.waitKey(1)
                if key & 0xFF == 13:  # Enter
                    cv2.destroyAllWindows()
                    break
                    
        except Exception as e:
            raise SharedResourceError(
                f"Calibration test failed: {str(e)}"
            ) from e
    
    def get_transform_matrix_path(self) -> Optional[str]:
        """Get path where transformation matrix was saved.
        
        Returns:
            Path to saved transformation matrix file, or None if not saved
        """
        return self._transform_matrix_path