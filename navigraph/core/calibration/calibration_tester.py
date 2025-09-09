"""Interactive calibration tester for NaviGraph.

Tests existing calibration matrices by allowing users to click on images and see
the corresponding transformed points on maps, and vice versa.
"""

from typing import Union, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from loguru import logger

from .transform_calculator import TransformMethod


class CalibrationTester:
    """Interactive tester for validating calibration matrices."""
    
    # Visual styling constants
    POINT_RADIUS = 8
    POINT_THICKNESS = 3
    LINE_THICKNESS = 2
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    
    # Colors (BGR format for OpenCV)
    COLOR_VALID_POINT = (0, 255, 0)      # Green for valid points
    COLOR_INVALID_POINT = (0, 0, 255)    # Red for out-of-bounds points
    COLOR_CONNECTION_LINE = (255, 255, 0) # Cyan for connection lines
    COLOR_TEXT = (255, 255, 255)         # White text
    COLOR_TEXT_BG = (0, 0, 0)            # Black background
    
    def __init__(self, logger_instance: Optional = None):
        """Initialize calibration tester.
        
        Args:
            logger_instance: Optional logger instance (uses global if None)
        """
        self.logger = logger_instance or logger
        self.calibration_matrix = None
        self.inverse_matrix = None
        self.spatial_image = None
        self.map_image = None
        self.window_name = "Calibration Test"
        
        # Test points storage
        self.test_points = []  # List of (source_point, target_point, is_valid)
        
    def test_calibration(
        self,
        spatial_image_path: Union[str, Path],
        map_image_path: Path,
        calibration_matrix_path: Path,
        method: str = "homography_ransac"
    ) -> None:
        """Launch interactive calibration test interface.
        
        Args:
            spatial_image_path: Path to spatial image (camera frame or video)
            map_image_path: Path to map image
            calibration_matrix_path: Path to calibration matrix (.npy file)
            method: Transformation method used (for inverse calculation)
        """
        self.logger.info("🧪 Starting interactive calibration test")
        
        # Load calibration matrix
        self.calibration_matrix = self._load_calibration_matrix(calibration_matrix_path)
        
        # Calculate inverse matrix for reverse transformation
        try:
            self.inverse_matrix = np.linalg.inv(self.calibration_matrix)
            self.logger.info("✓ Inverse calibration matrix calculated")
        except np.linalg.LinAlgError:
            self.logger.error("❌ Cannot calculate inverse matrix - matrix is singular")
            raise ValueError("Calibration matrix is not invertible")
        
        # Load images
        self.spatial_image = self._load_spatial_image(spatial_image_path)
        self.map_image = self._load_image(map_image_path)
        
        self.logger.info("✓ Images loaded successfully")
        
        # Show calibration info
        self._display_calibration_info()
        
        # Launch interactive interface
        self._run_interactive_test()
    
    def _load_calibration_matrix(self, matrix_path: Path) -> np.ndarray:
        """Load calibration matrix from file.
        
        Args:
            matrix_path: Path to .npy calibration matrix file
            
        Returns:
            Loaded calibration matrix
            
        Raises:
            ValueError: If matrix is invalid
        """
        try:
            matrix = np.load(matrix_path)
            
            if matrix.shape != (3, 3):
                raise ValueError(f"Expected 3x3 matrix, got {matrix.shape}")
            
            # Check if matrix is singular
            det = np.linalg.det(matrix)
            if abs(det) < 1e-10:
                raise ValueError(f"Matrix is singular (determinant={det:.3e})")
            
            self.logger.info(f"✓ Calibration matrix loaded: det={det:.3f}")
            return matrix
            
        except Exception as e:
            raise ValueError(f"Failed to load calibration matrix: {e}")
    
    def _load_spatial_image(self, source_path: Union[str, Path]) -> np.ndarray:
        """Load spatial image from file (image or first frame of video).
        
        Args:
            source_path: Path to image file or video file
            
        Returns:
            Spatial image as numpy array
            
        Raises:
            ValueError: If file can't be opened or no frame captured
        """
        source_path = Path(source_path)
        
        # Check if it's an image file first
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        if source_path.suffix.lower() in image_extensions:
            frame = cv2.imread(str(source_path))
            if frame is None:
                raise ValueError(f"Failed to load image: {source_path}")
            return frame
        
        # Try as video file
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {source_path}")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            raise ValueError(f"Failed to read first frame from video: {source_path}")
        
        return frame
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image
    
    def _display_calibration_info(self) -> None:
        """Display calibration matrix information."""
        det = np.linalg.det(self.calibration_matrix)
        condition = np.linalg.cond(self.calibration_matrix)
        
        self.logger.info(f"📊 Calibration Matrix Info:")
        self.logger.info(f"   Determinant: {det:.3f}")
        self.logger.info(f"   Condition number: {condition:.3f}")
        self.logger.info(f"   Spatial image: {self.spatial_image.shape[1]}x{self.spatial_image.shape[0]}")
        self.logger.info(f"   Map image: {self.map_image.shape[1]}x{self.map_image.shape[0]}")
    
    def _run_interactive_test(self) -> None:
        """Run the interactive test interface."""
        self.logger.info("🖱️  Interactive Test Controls:")
        self.logger.info("   • Click on spatial image (left) to see transformed point on map")
        self.logger.info("   • Click on map (right) to see inverse transformed point on image")
        self.logger.info("   • Press 'r' to reset all test points")
        self.logger.info("   • Press 's' to save screenshot")
        self.logger.info("   • Press ESC to exit")
        
        # Create side-by-side display
        self._create_display_window()
        
        while True:
            # Create display with current test points
            display = self._create_test_display()
            
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC - Exit
                self.logger.info("Calibration test completed")
                break
            elif key == ord('r') or key == ord('R'):  # Reset
                self.test_points.clear()
                self.logger.info("Reset all test points")
            elif key == ord('s') or key == ord('S'):  # Save screenshot
                self._save_screenshot(display)
        
        cv2.destroyAllWindows()
    
    def _create_display_window(self) -> None:
        """Create OpenCV window and set up mouse callback."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _create_test_display(self) -> np.ndarray:
        """Create side-by-side display with test results.
        
        Returns:
            Combined display image
        """
        # Get image dimensions
        spatial_h, spatial_w = self.spatial_image.shape[:2]
        map_h, map_w = self.map_image.shape[:2]
        
        # Resize map to match spatial image height
        if map_h != spatial_h:
            aspect_ratio = map_w / map_h
            new_width = int(spatial_h * aspect_ratio)
            map_resized = cv2.resize(self.map_image, (new_width, spatial_h))
        else:
            map_resized = self.map_image.copy()
            new_width = map_w
        
        # Create combined image
        combined_width = spatial_w + new_width + 10  # 10px separator
        combined = np.zeros((spatial_h, combined_width, 3), dtype=np.uint8)
        
        # Place images
        combined[:, :spatial_w] = self.spatial_image
        combined[:, spatial_w + 10:] = map_resized
        
        # Draw separator line
        cv2.line(combined, (spatial_w + 5, 0), (spatial_w + 5, spatial_h), (100, 100, 100), 2)
        
        # Calculate scaling factor for map coordinates
        map_scale_x = new_width / map_w
        map_scale_y = spatial_h / map_h
        
        # Draw test points and connections
        for i, (src_pt, dst_pt, is_valid) in enumerate(self.test_points):
            src_color = self.COLOR_VALID_POINT if is_valid else self.COLOR_INVALID_POINT
            
            # Draw source point (left side)
            cv2.circle(combined, src_pt, self.POINT_RADIUS, src_color, self.POINT_THICKNESS)
            
            # Scale and offset destination point (right side)
            scaled_dst = (
                int(dst_pt[0] * map_scale_x) + spatial_w + 10,
                int(dst_pt[1] * map_scale_y)
            )
            
            # Draw destination point
            cv2.circle(combined, scaled_dst, self.POINT_RADIUS, src_color, self.POINT_THICKNESS)
            
            # Draw connection line
            cv2.line(combined, src_pt, scaled_dst, self.COLOR_CONNECTION_LINE, self.LINE_THICKNESS)
            
            # Draw point numbers
            self._draw_point_number(combined, src_pt, i + 1)
            self._draw_point_number(combined, scaled_dst, i + 1)
        
        # Add instructions and info
        self._draw_instructions(combined, spatial_w, new_width)
        
        return combined
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point testing.
        
        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Mouse flags
            param: User parameter (unused)
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        # Get image dimensions for determining which side was clicked
        spatial_w = self.spatial_image.shape[1]
        map_w = self.map_image.shape[1]
        map_h = self.map_image.shape[0]
        
        # Calculate actual map dimensions in display
        spatial_h = self.spatial_image.shape[0]
        if map_h != spatial_h:
            aspect_ratio = map_w / map_h
            display_map_w = int(spatial_h * aspect_ratio)
        else:
            display_map_w = map_w
        
        map_scale_x = display_map_w / map_w
        map_scale_y = spatial_h / map_h
        
        separator_x = spatial_w + 10
        
        if x < spatial_w:
            # Clicked on spatial image (left side) - transform to map
            self._test_spatial_to_map(x, y)
        elif x > separator_x:
            # Clicked on map (right side) - transform to spatial
            # Convert display coordinates back to original map coordinates
            map_x = (x - separator_x) / map_scale_x
            map_y = y / map_scale_y
            self._test_map_to_spatial(map_x, map_y)
    
    def _test_spatial_to_map(self, x: int, y: int) -> None:
        """Test transformation from spatial image to map coordinates.
        
        Args:
            x, y: Coordinates on spatial image
        """
        # Transform point
        src_point = np.array([[x, y, 1]], dtype=np.float32).T
        transformed = self.calibration_matrix @ src_point
        
        # Convert from homogeneous coordinates
        if abs(transformed[2, 0]) > 1e-10:
            map_x = transformed[0, 0] / transformed[2, 0]
            map_y = transformed[1, 0] / transformed[2, 0]
        else:
            self.logger.warning("Invalid transformation: w coordinate is zero")
            return
        
        # Check if result is within map bounds
        map_h, map_w = self.map_image.shape[:2]
        is_valid = (0 <= map_x <= map_w) and (0 <= map_y <= map_h)
        
        # Store test point
        src_pos = (x, y)
        dst_pos = (map_x, map_y)
        self.test_points.append((src_pos, dst_pos, is_valid))
        
        # Log result
        status = "✓ VALID" if is_valid else "✗ OUT OF BOUNDS"
        self.logger.info(f"Spatial→Map: ({x}, {y}) → ({map_x:.1f}, {map_y:.1f}) {status}")
    
    def _test_map_to_spatial(self, x: float, y: float) -> None:
        """Test inverse transformation from map to spatial coordinates.
        
        Args:
            x, y: Coordinates on map image
        """
        # Transform point using inverse matrix
        map_point = np.array([[x, y, 1]], dtype=np.float32).T
        transformed = self.inverse_matrix @ map_point
        
        # Convert from homogeneous coordinates
        if abs(transformed[2, 0]) > 1e-10:
            spatial_x = transformed[0, 0] / transformed[2, 0]
            spatial_y = transformed[1, 0] / transformed[2, 0]
        else:
            self.logger.warning("Invalid inverse transformation: w coordinate is zero")
            return
        
        # Check if result is within spatial image bounds
        spatial_h, spatial_w = self.spatial_image.shape[:2]
        is_valid = (0 <= spatial_x <= spatial_w) and (0 <= spatial_y <= spatial_h)
        
        # Store test point (reversed: spatial is source, map is destination)
        src_pos = (int(spatial_x), int(spatial_y))
        dst_pos = (x, y)
        self.test_points.append((src_pos, dst_pos, is_valid))
        
        # Log result
        status = "✓ VALID" if is_valid else "✗ OUT OF BOUNDS"
        self.logger.info(f"Map→Spatial: ({x:.1f}, {y:.1f}) → ({spatial_x:.1f}, {spatial_y:.1f}) {status}")
    
    def _draw_point_number(self, image: np.ndarray, point: Tuple[int, int], number: int) -> None:
        """Draw point number with background.
        
        Args:
            image: Image to draw on
            point: Point coordinates
            number: Point number to display
        """
        text = str(number)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        text_size = cv2.getTextSize(text, font, self.FONT_SCALE, self.FONT_THICKNESS)[0]
        
        # Position text above point
        text_x = point[0] - text_size[0] // 2
        text_y = point[1] - self.POINT_RADIUS - 10
        
        # Draw background
        cv2.rectangle(
            image,
            (text_x - 3, text_y - text_size[1] - 3),
            (text_x + text_size[0] + 3, text_y + 3),
            self.COLOR_TEXT_BG,
            -1
        )
        
        # Draw text
        cv2.putText(image, text, (text_x, text_y), font, self.FONT_SCALE, 
                   self.COLOR_TEXT, self.FONT_THICKNESS)
    
    def _draw_instructions(self, image: np.ndarray, spatial_w: int, map_w: int) -> None:
        """Draw instructions and status information.
        
        Args:
            image: Image to draw on
            spatial_w: Width of spatial image portion
            map_w: Width of map image portion
        """
        instructions = [
            "Left: Spatial Image | Right: Map Image",
            "Click anywhere to test transformation",
            f"Test points: {len(self.test_points)} | ESC: Exit | R: Reset | S: Save"
        ]
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], 80), self.COLOR_TEXT_BG, -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        # Draw instructions
        for i, instruction in enumerate(instructions):
            y_pos = 20 + i * 20
            cv2.putText(image, instruction, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, self.COLOR_TEXT, 1, cv2.LINE_AA)
        
        # Draw calibration info
        if len(self.test_points) > 0:
            valid_count = sum(1 for _, _, is_valid in self.test_points if is_valid)
            invalid_count = len(self.test_points) - valid_count
            
            status_text = f"Valid: {valid_count} | Invalid: {invalid_count}"
            cv2.putText(image, status_text, (image.shape[1] - 200, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       self.COLOR_VALID_POINT if invalid_count == 0 else self.COLOR_INVALID_POINT, 
                       1, cv2.LINE_AA)
    
    def _save_screenshot(self, display: np.ndarray) -> None:
        """Save screenshot of current test display.
        
        Args:
            display: Display image to save
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_test_{timestamp}.png"
        
        cv2.imwrite(filename, display)
        self.logger.info(f"📸 Screenshot saved: {filename}")