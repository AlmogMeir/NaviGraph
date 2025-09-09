"""Interactive point selection UI for camera-to-map calibration."""

from typing import List, Tuple, Optional, NamedTuple
import cv2
import numpy as np
from loguru import logger


class Point(NamedTuple):
    """Represents a calibration point with coordinates."""
    x: float
    y: float
    
    def as_tuple(self) -> Tuple[int, int]:
        """Return point as integer tuple for OpenCV."""
        return (int(self.x), int(self.y))


class PointSelector:
    """Interactive point selection with visual feedback and keyboard controls."""
    
    # Visual styling constants
    POINT_RADIUS = 8
    POINT_THICKNESS = 3
    LINE_THICKNESS = 2
    FONT_SCALE = 0.8
    FONT_THICKNESS = 2
    
    # Colors (BGR format for OpenCV)
    COLOR_SOURCE_POINT = (255, 100, 0)    # Blue
    COLOR_TARGET_POINT = (0, 200, 0)      # Green  
    COLOR_CONNECTION_LINE = (255, 255, 0) # Cyan
    COLOR_TEXT = (255, 255, 255)          # White
    COLOR_TEXT_BG = (0, 0, 0)             # Black
    
    def __init__(self):
        """Initialize point selector."""
        self.source_points: List[Point] = []
        self.target_points: List[Point] = []
        self.current_stage = "source"  # "source" or "target"
        self.display_image = None
        self.original_source = None
        self.original_target = None
        self.window_name = "Calibration Point Selection"
        
    def select_corresponding_points(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        min_points: int = 4,
        window_title: str = "NaviGraph Calibration"
    ) -> Tuple[List[Point], List[Point]]:
        """Select corresponding points between source and target images.
        
        Args:
            source_image: Source image (camera frame)
            target_image: Target image (map)
            min_points: Minimum number of points required
            window_title: Window title for display
            
        Returns:
            Tuple of (source_points, target_points)
            
        Raises:
            ValueError: If user cancels or insufficient points selected
        """
        self.source_points = []
        self.target_points = []
        self.original_source = source_image.copy()
        self.original_target = target_image.copy()
        self.window_name = window_title
        
        logger.info(f"Starting interactive calibration with minimum {min_points} points")
        logger.info("Controls: Click to add point, Right-click to remove last, 'r' to reset, Enter to confirm, ESC to cancel")
        
        # First select points on source image  
        self.current_stage = "source"
        self._select_points_on_image(self.original_source, "image (at least 4 points)", min_points)
        
        if not self.source_points:
            raise ValueError("Calibration cancelled by user")
            
        # Then select EXACTLY the same number of corresponding points on target image
        self.current_stage = "target" 
        actual_points_needed = len(self.source_points)
        self._select_points_on_image(self.original_target, f"map ({actual_points_needed} points to match)", actual_points_needed, exact_points=True)
        
        if len(self.target_points) != len(self.source_points):
            raise ValueError("Calibration cancelled or point count mismatch")
        
        logger.info(f"Successfully selected {len(self.source_points)} point pairs")
        return self.source_points, self.target_points
    
    def _select_points_on_image(self, image: np.ndarray, image_type: str, required_points: int, exact_points: bool = False) -> None:
        """Handle point selection on a single image.
        
        Args:
            image: Image to select points on
            image_type: Description for user ("image" or "map")
            required_points: Number of points needed
            exact_points: If True, user must select exactly this many points (no more)
        """
        self.display_image = image.copy()
        
        # Setup window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        current_points = self.source_points if self.current_stage == "source" else self.target_points
        
        while True:
            # Create display with current points and instructions
            display = self.display_image.copy()
            self._draw_points(display, current_points, self.current_stage)
            self._draw_instructions(display, image_type, required_points, len(current_points), exact_points)
            
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC - Cancel
                logger.info("Calibration cancelled by user")
                cv2.destroyAllWindows()
                if self.current_stage == "source":
                    self.source_points = []
                else:
                    self.target_points = []
                return
                
            elif key == 13 or key == 10:  # Enter - Confirm
                if exact_points:
                    # Must have exactly the required number
                    if len(current_points) == required_points:
                        logger.info(f"Confirmed {len(current_points)} points on {image_type}")
                        break
                    else:
                        logger.warning(f"Need exactly {required_points} points, have {len(current_points)}")
                else:
                    # Must have at least the required number
                    if len(current_points) >= required_points:
                        logger.info(f"Confirmed {len(current_points)} points on {image_type}")
                        break
                    else:
                        logger.warning(f"Need at least {required_points} points, have {len(current_points)}")
                    
            elif key == ord('r') or key == ord('R'):  # Reset
                logger.info(f"Reset all points on {image_type}")
                current_points.clear()
                self.display_image = image.copy()
                
        cv2.destroyAllWindows()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point selection."""
        current_points = self.source_points if self.current_stage == "source" else self.target_points
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if we're in exact mode and already have enough points
            exact_mode = self.current_stage == "target"  # Target stage requires exact points
            if exact_mode:
                target_count = len(self.source_points)  # Must match source count
                if len(current_points) >= target_count:
                    logger.debug(f"Already have {target_count} points, cannot add more")
                    return
            
            # Add point
            new_point = Point(x, y)
            current_points.append(new_point)
            logger.debug(f"Added point {len(current_points)}: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last point
            if current_points:
                removed = current_points.pop()
                logger.debug(f"Removed point: ({removed.x}, {removed.y})")
                
                # Redraw image without removed point
                if self.current_stage == "source":
                    self.display_image = self.original_source.copy()
                else:
                    self.display_image = self.original_target.copy()
    
    def _draw_points(self, image: np.ndarray, points: List[Point], stage: str) -> None:
        """Draw points on image with numbering."""
        color = self.COLOR_SOURCE_POINT if stage == "source" else self.COLOR_TARGET_POINT
        
        for i, point in enumerate(points):
            # Draw point circle
            cv2.circle(image, point.as_tuple(), self.POINT_RADIUS, color, self.POINT_THICKNESS)
            
            # Draw point number
            text = str(i + 1)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.FONT_THICKNESS)[0]
            text_x = point.as_tuple()[0] - text_size[0] // 2
            text_y = point.as_tuple()[1] - self.POINT_RADIUS - 10
            
            # Draw text background
            cv2.rectangle(
                image,
                (text_x - 3, text_y - text_size[1] - 3),
                (text_x + text_size[0] + 3, text_y + 3),
                self.COLOR_TEXT_BG,
                -1
            )
            
            # Draw text
            cv2.putText(
                image, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.COLOR_TEXT, self.FONT_THICKNESS
            )
    
    def _draw_instructions(
        self, 
        image: np.ndarray, 
        image_type: str, 
        required_points: int, 
        current_count: int,
        exact_points: bool = False
    ) -> None:
        """Draw minimal status information on image."""
        # Create status text based on mode
        if exact_points:
            if current_count < required_points:
                status_text = f"Select point {current_count + 1}/{required_points} on {image_type}"
            else:
                status_text = f"Selected {current_count}/{required_points} points on {image_type}"
        else:
            status_text = f"Selected {current_count} points on {image_type}"
        
        # Draw small status bar at top
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (image.shape[1], text_size[1] + 20),
            self.COLOR_TEXT_BG,
            -1
        )
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Status text
        cv2.putText(
            image, status_text, (10, text_size[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_TEXT, 2
        )
        
        # Ready indicator based on requirements
        ready = False
        ready_text = ""
        
        if exact_points:
            if current_count == required_points:
                ready = True
                ready_text = "Press Enter to confirm"
        else:
            if current_count >= required_points:
                ready = True
                ready_text = "Press Enter to confirm"
        
        if ready:
            ready_size = cv2.getTextSize(ready_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(
                image, ready_text, 
                (image.shape[1] - ready_size[0] - 10, text_size[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TARGET_POINT, 1
            )
    
    def show_correspondence_preview(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_points: List[Point],
        target_points: List[Point]
    ) -> bool:
        """Show side-by-side preview of point correspondences.
        
        Args:
            source_image: Source image
            target_image: Target image  
            source_points: Points from source
            target_points: Corresponding points from target
            
        Returns:
            True if user confirms, False if user wants to redo
        """
        if len(source_points) != len(target_points):
            return True  # Skip preview if mismatch
            
        # Create side-by-side display
        h1, w1 = source_image.shape[:2]
        h2, w2 = target_image.shape[:2]
        
        # Calculate scaling for target image if needed
        target_height = h1
        scale_x = scale_y = 1.0
        
        if h2 != h1:
            aspect_ratio = w2 / h2
            target_width = int(target_height * aspect_ratio)
            target_resized = cv2.resize(target_image, (target_width, target_height))
            # Calculate scaling factors
            scale_x = target_width / w2
            scale_y = target_height / h2
        else:
            target_resized = target_image.copy()
            target_width = w2
        
        # Create combined image
        combined = np.zeros((target_height, w1 + target_width, 3), dtype=np.uint8)
        combined[:, :w1] = source_image
        combined[:, w1:] = target_resized
        
        # Draw points and connections
        display = combined.copy()
        
        for i, (src_pt, tgt_pt) in enumerate(zip(source_points, target_points)):
            # Draw source point
            cv2.circle(display, src_pt.as_tuple(), self.POINT_RADIUS, self.COLOR_SOURCE_POINT, self.POINT_THICKNESS)
            
            # Scale target point to match resized image and offset by source width
            scaled_target_x = int(tgt_pt.x * scale_x) + w1
            scaled_target_y = int(tgt_pt.y * scale_y)
            target_pos = (scaled_target_x, scaled_target_y)
            
            cv2.circle(display, target_pos, self.POINT_RADIUS, self.COLOR_TARGET_POINT, self.POINT_THICKNESS)
            
            # Draw connection line
            cv2.line(display, src_pt.as_tuple(), target_pos, self.COLOR_CONNECTION_LINE, self.LINE_THICKNESS)
            
            # Draw numbers with black background for visibility
            text = str(i + 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Source point number
            src_text_pos = (src_pt.as_tuple()[0] - 10, src_pt.as_tuple()[1] - 15)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            cv2.rectangle(display, 
                         (src_text_pos[0] - 3, src_text_pos[1] - text_size[1] - 3),
                         (src_text_pos[0] + text_size[0] + 3, src_text_pos[1] + 3),
                         self.COLOR_TEXT_BG, -1)
            cv2.putText(display, text, src_text_pos, font, font_scale, self.COLOR_TEXT, thickness)
            
            # Target point number  
            tgt_text_pos = (target_pos[0] - 10, target_pos[1] - 15)
            cv2.rectangle(display,
                         (tgt_text_pos[0] - 3, tgt_text_pos[1] - text_size[1] - 3),
                         (tgt_text_pos[0] + text_size[0] + 3, tgt_text_pos[1] + 3),
                         self.COLOR_TEXT_BG, -1)
            cv2.putText(display, text, tgt_text_pos, font, font_scale, self.COLOR_TEXT, thickness)
        
        # Add instructions
        cv2.putText(display, "Point Correspondences - Enter to confirm, 'r' to redo, ESC to cancel",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_TEXT, 2)
        
        # Show preview
        cv2.namedWindow("Calibration Preview", cv2.WINDOW_NORMAL)
        cv2.imshow("Calibration Preview", display)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13 or key == 10:  # Enter - Confirm
                cv2.destroyAllWindows()
                return True
            elif key == ord('r') or key == ord('R'):  # Redo
                cv2.destroyAllWindows() 
                return False
            elif key == 27:  # ESC - Cancel
                cv2.destroyAllWindows()
                return False