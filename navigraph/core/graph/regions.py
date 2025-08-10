"""Spatial region classes for NaviGraph.

This module provides classes for defining spatial regions that can be
mapped to graph nodes. Supports various region types including contours,
rectangles, circles, and grid cells.
"""

from typing import List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class Point:
    """Represents a 2D point."""
    x: float
    y: float
    
    def __iter__(self):
        """Allow tuple unpacking."""
        yield self.x
        yield self.y
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)


class SpatialRegion(ABC):
    """Abstract base class for spatial regions.
    
    A spatial region represents an area in 2D space that can be associated
    with a graph node. Each region must be able to determine if a point
    lies within its boundaries.
    """
    
    def __init__(self, region_id: str, metadata: Optional[dict] = None):
        """Initialize spatial region.
        
        Args:
            region_id: Unique identifier for this region
            metadata: Optional metadata about the region
        """
        self.region_id = region_id
        self.metadata = metadata or {}
    
    @abstractmethod
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this region.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is inside the region
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of the region.
        
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        pass
    
    @abstractmethod
    def get_center(self) -> Point:
        """Get center point of the region.
        
        Returns:
            Center point of the region
        """
        pass
    
    @abstractmethod
    def get_area(self) -> float:
        """Get area of the region.
        
        Returns:
            Area of the region
        """
        pass
    
    def overlaps_with(self, other: 'SpatialRegion', num_test_points: int = 100) -> bool:
        """Check if this region overlaps with another region.
        
        Uses a sampling-based approach to detect overlaps.
        
        Args:
            other: Other spatial region
            num_test_points: Number of points to test for overlap
            
        Returns:
            True if regions overlap
        """
        # Get bounds of both regions
        x1_min, y1_min, x1_max, y1_max = self.get_bounds()
        x2_min, y2_min, x2_max, y2_max = other.get_bounds()
        
        # Check if bounding boxes overlap
        if (x1_max < x2_min or x2_max < x1_min or 
            y1_max < y2_min or y2_max < y1_min):
            return False
        
        # Sample points in the overlapping bounding box
        overlap_x_min = max(x1_min, x2_min)
        overlap_y_min = max(y1_min, y2_min)
        overlap_x_max = min(x1_max, x2_max)
        overlap_y_max = min(y1_max, y2_max)
        
        # Generate test points
        x_points = np.linspace(overlap_x_min, overlap_x_max, int(np.sqrt(num_test_points)))
        y_points = np.linspace(overlap_y_min, overlap_y_max, int(np.sqrt(num_test_points)))
        
        for x in x_points:
            for y in y_points:
                if self.contains_point(x, y) and other.contains_point(x, y):
                    return True
        
        return False
    
    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance from point to region boundary.
        
        If point is inside, returns 0. Otherwise returns minimum distance
        to region boundary.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Distance to region boundary
        """
        if self.contains_point(x, y):
            return 0.0
        
        # For base implementation, return distance to center
        center = self.get_center()
        return np.sqrt((x - center.x) ** 2 + (y - center.y) ** 2)


class ContourRegion(SpatialRegion):
    """Region defined by a polygon contour.
    
    Uses OpenCV's pointPolygonTest for efficient point-in-polygon testing.
    """
    
    def __init__(self, region_id: str, contour_points: List[Tuple[float, float]],
                 metadata: Optional[dict] = None):
        """Initialize contour region.
        
        Args:
            region_id: Unique identifier
            contour_points: List of (x, y) points defining the polygon
            metadata: Optional metadata
        """
        super().__init__(region_id, metadata)
        
        if len(contour_points) < 3:
            raise ValueError("Contour must have at least 3 points")
        
        # Store contour as numpy array for OpenCV
        self.contour_points = contour_points
        self.contour = np.array(contour_points, dtype=np.float32)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside the contour."""
        # Use OpenCV's point-in-polygon test
        result = cv2.pointPolygonTest(self.contour, (x, y), False)
        return result >= 0  # >= 0 means inside or on boundary
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of the contour."""
        x_coords = [p[0] for p in self.contour_points]
        y_coords = [p[1] for p in self.contour_points]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def get_center(self) -> Point:
        """Get centroid of the contour."""
        # Calculate centroid using OpenCV moments
        moments = cv2.moments(self.contour)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            # Fallback to geometric center
            x_coords = [p[0] for p in self.contour_points]
            y_coords = [p[1] for p in self.contour_points]
            cx = sum(x_coords) / len(x_coords)
            cy = sum(y_coords) / len(y_coords)
        
        return Point(cx, cy)
    
    def get_area(self) -> float:
        """Get area of the contour."""
        return cv2.contourArea(self.contour)
    
    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance from point to contour boundary."""
        # Use OpenCV's pointPolygonTest with distance calculation
        distance = cv2.pointPolygonTest(self.contour, (x, y), True)
        return abs(distance)  # Return absolute distance


class RectangleRegion(SpatialRegion):
    """Rectangular region defined by position and size."""
    
    def __init__(self, region_id: str, x: float, y: float, 
                 width: float, height: float, metadata: Optional[dict] = None):
        """Initialize rectangle region.
        
        Args:
            region_id: Unique identifier
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Width of rectangle
            height: Height of rectangle
            metadata: Optional metadata
        """
        super().__init__(region_id, metadata)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside the rectangle."""
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (same as rectangle bounds)."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def get_center(self) -> Point:
        """Get center of the rectangle."""
        return Point(self.x + self.width / 2, self.y + self.height / 2)
    
    def get_area(self) -> float:
        """Get area of the rectangle."""
        return self.width * self.height
    
    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance from point to rectangle boundary."""
        if self.contains_point(x, y):
            return 0.0
        
        # Distance to rectangle boundary
        dx = max(0, max(self.x - x, x - (self.x + self.width)))
        dy = max(0, max(self.y - y, y - (self.y + self.height)))
        return np.sqrt(dx * dx + dy * dy)


class CircleRegion(SpatialRegion):
    """Circular region defined by center and radius."""
    
    def __init__(self, region_id: str, center_x: float, center_y: float,
                 radius: float, metadata: Optional[dict] = None):
        """Initialize circle region.
        
        Args:
            region_id: Unique identifier
            center_x: X coordinate of center
            center_y: Y coordinate of center
            radius: Radius of circle
            metadata: Optional metadata
        """
        super().__init__(region_id, metadata)
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside the circle."""
        distance_sq = (x - self.center_x) ** 2 + (y - self.center_y) ** 2
        return distance_sq <= self.radius ** 2
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of the circle."""
        return (self.center_x - self.radius, self.center_y - self.radius,
                self.center_x + self.radius, self.center_y + self.radius)
    
    def get_center(self) -> Point:
        """Get center of the circle."""
        return Point(self.center_x, self.center_y)
    
    def get_area(self) -> float:
        """Get area of the circle."""
        return np.pi * self.radius ** 2
    
    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance from point to circle boundary."""
        center_distance = np.sqrt((x - self.center_x) ** 2 + (y - self.center_y) ** 2)
        return abs(center_distance - self.radius)


class GridCell(RectangleRegion):
    """Grid cell region with row/column indices.
    
    Extends RectangleRegion with grid-specific functionality.
    """
    
    def __init__(self, region_id: str, row: int, col: int,
                 cell_width: float, cell_height: float,
                 origin: Tuple[float, float] = (0, 0),
                 metadata: Optional[dict] = None):
        """Initialize grid cell.
        
        Args:
            region_id: Unique identifier
            row: Row index in grid
            col: Column index in grid
            cell_width: Width of grid cell
            cell_height: Height of grid cell
            origin: Origin point (top-left of grid)
            metadata: Optional metadata
        """
        # Calculate position from grid indices
        x = origin[0] + col * cell_width
        y = origin[1] + row * cell_height
        
        super().__init__(region_id, x, y, cell_width, cell_height, metadata)
        
        self.row = row
        self.col = col
        self.origin = origin
    
    def get_grid_position(self) -> Tuple[int, int]:
        """Get grid position as (row, col)."""
        return (self.row, self.col)
    
    def get_neighbors(self, total_rows: int, total_cols: int,
                     connectivity: int = 4) -> List[Tuple[int, int]]:
        """Get neighboring grid cells.
        
        Args:
            total_rows: Total number of rows in grid
            total_cols: Total number of columns in grid
            connectivity: 4 for von Neumann, 8 for Moore neighborhood
            
        Returns:
            List of (row, col) tuples for valid neighbors
        """
        neighbors = []
        
        if connectivity == 4:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif connectivity == 8:
            deltas = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
        else:
            raise ValueError(f"Connectivity must be 4 or 8, got {connectivity}")
        
        for dr, dc in deltas:
            new_row = self.row + dr
            new_col = self.col + dc
            
            if 0 <= new_row < total_rows and 0 <= new_col < total_cols:
                neighbors.append((new_row, new_col))
        
        return neighbors


class HexagonalCell(SpatialRegion):
    """Hexagonal cell region for hexagonal grids."""
    
    def __init__(self, region_id: str, center_x: float, center_y: float,
                 radius: float, metadata: Optional[dict] = None):
        """Initialize hexagonal cell.
        
        Args:
            region_id: Unique identifier
            center_x: X coordinate of center
            center_y: Y coordinate of center
            radius: Radius (distance from center to vertex)
            metadata: Optional metadata
        """
        super().__init__(region_id, metadata)
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        
        # Generate hexagon vertices
        self.vertices = self._generate_vertices()
        
        # Create contour for point-in-polygon testing
        self.contour = np.array(self.vertices, dtype=np.float32)
    
    def _generate_vertices(self) -> List[Tuple[float, float]]:
        """Generate vertices of regular hexagon."""
        vertices = []
        for i in range(6):
            angle = i * np.pi / 3  # 60 degrees
            x = self.center_x + self.radius * np.cos(angle)
            y = self.center_y + self.radius * np.sin(angle)
            vertices.append((x, y))
        return vertices
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside the hexagon."""
        result = cv2.pointPolygonTest(self.contour, (x, y), False)
        return result >= 0
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of the hexagon."""
        # For regular hexagon, bounds are based on radius
        width = 2 * self.radius
        height = self.radius * np.sqrt(3)
        return (self.center_x - width/2, self.center_y - height/2,
                self.center_x + width/2, self.center_y + height/2)
    
    def get_center(self) -> Point:
        """Get center of the hexagon."""
        return Point(self.center_x, self.center_y)
    
    def get_area(self) -> float:
        """Get area of the hexagon."""
        return (3 * np.sqrt(3) / 2) * self.radius ** 2
    
    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance from point to hexagon boundary."""
        if self.contains_point(x, y):
            return 0.0
        
        # Use OpenCV's pointPolygonTest with distance calculation
        distance = cv2.pointPolygonTest(self.contour, (x, y), True)
        return abs(distance)


class EllipseRegion(SpatialRegion):
    """Elliptical region defined by center and radii."""
    
    def __init__(self, region_id: str, center_x: float, center_y: float,
                 radius_x: float, radius_y: float, angle: float = 0.0,
                 metadata: Optional[dict] = None):
        """Initialize ellipse region.
        
        Args:
            region_id: Unique identifier
            center_x: X coordinate of center
            center_y: Y coordinate of center
            radius_x: Radius along x-axis
            radius_y: Radius along y-axis
            angle: Rotation angle in radians
            metadata: Optional metadata
        """
        super().__init__(region_id, metadata)
        self.center_x = center_x
        self.center_y = center_y
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.angle = angle
        
        # Precompute rotation matrix
        self.cos_angle = np.cos(angle)
        self.sin_angle = np.sin(angle)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside the ellipse."""
        # Translate to center
        dx = x - self.center_x
        dy = y - self.center_y
        
        # Rotate to align with axes
        rx = dx * self.cos_angle + dy * self.sin_angle
        ry = -dx * self.sin_angle + dy * self.cos_angle
        
        # Check ellipse equation
        return (rx / self.radius_x) ** 2 + (ry / self.radius_y) ** 2 <= 1
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of the ellipse."""
        # For rotated ellipse, calculate exact bounds
        cos_a = abs(self.cos_angle)
        sin_a = abs(self.sin_angle)
        
        half_width = np.sqrt((self.radius_x * cos_a) ** 2 + (self.radius_y * sin_a) ** 2)
        half_height = np.sqrt((self.radius_x * sin_a) ** 2 + (self.radius_y * cos_a) ** 2)
        
        return (self.center_x - half_width, self.center_y - half_height,
                self.center_x + half_width, self.center_y + half_height)
    
    def get_center(self) -> Point:
        """Get center of the ellipse."""
        return Point(self.center_x, self.center_y)
    
    def get_area(self) -> float:
        """Get area of the ellipse."""
        return np.pi * self.radius_x * self.radius_y