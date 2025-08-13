"""PyQt5-based interactive dual-view GUI for graph-space mapping setup.

This module provides a professional Qt-based interface for creating spatial mappings
between graph nodes/edges and regions on a map, with both grid-based and manual
contour drawing modes.
"""

from typing import Dict, Any, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
import pickle

from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QLabel, QComboBox, QSpinBox, QSlider,
    QListWidget, QListWidgetItem, QTabWidget, QGroupBox, QMessageBox,
    QProgressBar, QToolBar, QStatusBar, QDockWidget, QTextEdit,
    QRadioButton, QButtonGroup, QStackedWidget, QDoubleSpinBox,
    QCheckBox, QFileDialog, QSizePolicy, QGridLayout
)
from PyQt5.QtCore import Qt, QPointF, QRectF, QPoint, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPolygonF, QPainter, QCursor

import networkx as nx
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .structures import GraphStructure
from .mapping import SpatialMapping
from .regions import (SpatialRegion, ContourRegion, RectangleRegion,
                     CircleRegion, GridCell, HexagonalCell)
from .storage import MappingStorage


@dataclass
class GridConfig:
    """Configuration for grid-based mapping."""
    structure_type: str = 'rectangle'  # 'rectangle', 'hexagon'
    rows: int = 8
    cols: int = 8
    cell_width: float = 50.0
    cell_height: float = 50.0
    origin_x: float = 0.0
    origin_y: float = 0.0


class MapWidget(QWidget):
    """Widget for displaying and interacting with the map image.
    
    Features:
    - Mouse scroll wheel to zoom in/out (0.1x to 10x)
    - Right mouse button drag to pan the image
    - Press 'R' key to reset zoom and center image
    - Left mouse button for grid placement and cell selection
    """
    
    gridPlaced = pyqtSignal(float, float)  # Emitted when grid origin is placed
    cellClicked = pyqtSignal(str)  # Emitted when a grid cell is clicked
    contourDrawn = pyqtSignal(list)  # Emitted when a contour is completed
    
    def __init__(self, map_image: np.ndarray, parent=None):
        super().__init__(parent)
        self.map_image = map_image
        self.original_image = map_image.copy()
        
        # Display state
        self.scale_factor = 1.0
        self.user_scale_factor = 1.0  # User-controlled zoom
        self.base_scale_factor = 1.0  # Auto-calculated fit-to-widget scale
        self.offset_x = 0
        self.offset_y = 0
        
        # Panning state
        self.panning = False
        self.last_pan_point = QPoint()
        
        # Grid state
        self.grid_config = GridConfig()
        self.grid_enabled = False
        self.grid_cells = {}  # cell_id -> QRectF
        self.selected_cells = set()
        self.cell_colors = {}  # cell_id -> QColor
        self.cell_mappings = {}  # cell_id -> (elem_type, elem_id) for mapped cells
        
        # Drawing state
        self.drawing_mode = False
        self.current_contour = []
        self.completed_contours = []  # List of (contour_points, region_id, color, elem_info)
        self.contour_mappings = {}  # region_id -> (elem_type, elem_id)
        
        # Interaction state
        self.interaction_mode = 'none'  # 'place_grid', 'select_cells', 'draw_contour'
        self.show_all_mappings = True  # Toggle for showing all mapped regions
        self.show_cell_labels = True  # Toggle for showing cell labels
        self.adaptive_font_size = True  # Toggle for adaptive font sizing
        self.current_element_type = None
        self.current_element_id = None
        
        # Contour highlighting state
        self.highlighted_contour_id = None
        self.original_contour_colors = {}  # Store original colors for unhighlighting
        
        self.setMouseTracking(True)
        self.setMinimumSize(800, 600)
        self.setFocusPolicy(Qt.StrongFocus)  # Enable keyboard focus
        
        # Enable tooltips with faster response
        from PyQt5.QtWidgets import QToolTip
        self.setToolTip("")
        self.current_tooltip = ""  # Track tooltip state for immediate updates
        
    def set_interaction_mode(self, mode: str):
        """Set the current interaction mode."""
        self.interaction_mode = mode
        self.current_contour = []
        self.update()
        
    def set_grid_config(self, config: GridConfig):
        """Update grid configuration."""
        self.grid_config = config
        if self.grid_enabled:
            self._generate_grid()
            
    def enable_grid(self, origin_x: float, origin_y: float):
        """Enable and place the grid at the specified origin."""
        self.grid_config.origin_x = origin_x
        self.grid_config.origin_y = origin_y
        self.grid_enabled = True
        self._generate_grid()
        self.update()
        
    def _generate_grid(self):
        """Generate grid cells based on current configuration."""
        self.grid_cells.clear()
        
        for row in range(self.grid_config.rows):
            for col in range(self.grid_config.cols):
                cell_id = f"cell_{row}_{col}"
                
                x = self.grid_config.origin_x + col * self.grid_config.cell_width
                y = self.grid_config.origin_y + row * self.grid_config.cell_height
                
                if self.grid_config.structure_type == 'rectangle':
                    rect = QRectF(x, y, self.grid_config.cell_width, self.grid_config.cell_height)
                    self.grid_cells[cell_id] = rect
                # TODO: Add hexagon support
                
    def select_cells(self, cell_ids: Set[str], color: QColor = QColor(0, 255, 0, 100)):
        """Select and highlight specific cells."""
        for cell_id in cell_ids:
            if cell_id in self.grid_cells:
                self.selected_cells.add(cell_id)
                self.cell_colors[cell_id] = color
        self.update()
        
    def clear_selection(self):
        """Clear all selected cells."""
        self.selected_cells.clear()
        # Don't clear cell_colors as they track mapped cells
        self.update()
    
    def reset_all(self):
        """Reset all visualizations and mappings."""
        self.grid_enabled = False
        self.grid_cells.clear()
        self.selected_cells.clear()
        self.cell_colors.clear()
        self.cell_mappings.clear()
        self.current_contour.clear()
        self.completed_contours.clear()
        self.contour_mappings.clear()
        self.interaction_mode = 'none'
        self.update()
    
    def set_current_element(self, elem_type: str, elem_id):
        """Set the current element being mapped."""
        self.current_element_type = elem_type
        self.current_element_id = elem_id
    
    def add_cell_mapping(self, cell_id: str, elem_type: str, elem_id, color: QColor):
        """Add a mapping for a cell."""
        self.cell_mappings[cell_id] = (elem_type, elem_id)
        self.cell_colors[cell_id] = color
        
    def add_contour(self, points: List[Tuple[float, float]], region_id: str, 
                   elem_type: str, elem_id, color: QColor = QColor(0, 0, 255, 100)):
        """Add a completed contour to the map."""
        self.completed_contours.append((points, region_id, color, elem_type, elem_id))
        self.contour_mappings[region_id] = (elem_type, elem_id)
        self.update()
        
    def clear_contours(self):
        """Clear all contours."""
        self.completed_contours.clear()
        self.current_contour.clear()
        self.update()
        
    def paintEvent(self, event):
        """Paint the map, grid, and contours."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw map image centered
        if self.map_image is not None:
            height, width = self.map_image.shape[:2]
            bytes_per_line = 3 * width
            
            if len(self.map_image.shape) == 2:
                # Grayscale
                image = cv2.cvtColor(self.map_image, cv2.COLOR_GRAY2RGB)
            else:
                # Color (BGR to RGB)
                image = cv2.cvtColor(self.map_image, cv2.COLOR_BGR2RGB)
                
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Calculate optimal scaling to fit widget while preserving aspect ratio
            widget_size = self.size()
            # Leave some padding (20px on each side)
            available_width = widget_size.width() - 40
            available_height = widget_size.height() - 40
            
            # Calculate base scale factor to fit both dimensions
            scale_x = available_width / width if width > 0 else 1.0
            scale_y = available_height / height if height > 0 else 1.0
            self.base_scale_factor = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            # Calculate combined scale factor (base * user zoom)
            self.scale_factor = self.base_scale_factor * self.user_scale_factor
            
            # Calculate final dimensions
            final_width = int(width * self.scale_factor)
            final_height = int(height * self.scale_factor)
            
            # Scale the pixmap
            scaled_pixmap = pixmap.scaled(final_width, final_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # If user hasn't zoomed yet, center the image
            if self.user_scale_factor == 1.0:
                self.offset_x = (widget_size.width() - final_width) // 2
                self.offset_y = (widget_size.height() - final_height) // 2
            
            painter.drawPixmap(int(self.offset_x), int(self.offset_y), scaled_pixmap)
            
        # Draw grid if enabled
        if self.grid_enabled:
            # Calculate adaptive font size based on cell size
            cell_size = self.grid_config.cell_width * self.scale_factor
            if self.adaptive_font_size:
                if cell_size > 80:
                    font_size = 9
                elif cell_size > 60:
                    font_size = 8
                elif cell_size > 40:
                    font_size = 7
                elif cell_size > 30:
                    font_size = 6
                else:
                    font_size = 5
            else:
                font_size = 8
            
            font = painter.font()
            font.setPointSize(font_size)
            painter.setFont(font)
            
            for cell_id, rect in self.grid_cells.items():
                scaled_rect = QRectF(
                    self.offset_x + rect.x() * self.scale_factor,
                    self.offset_y + rect.y() * self.scale_factor,
                    rect.width() * self.scale_factor,
                    rect.height() * self.scale_factor
                )
                
                # Check if cell is mapped or selected
                is_mapped = cell_id in self.cell_mappings
                is_selected = cell_id in self.selected_cells
                
                # Draw cell based on state
                if is_mapped and self.show_all_mappings:
                    # Mapped cell - show with persistent color
                    color = self.cell_colors.get(cell_id, QColor(200, 200, 200, 100))
                    painter.fillRect(scaled_rect, color)
                    painter.setPen(QPen(color.darker(), 2))
                    painter.drawRect(scaled_rect)
                    
                    # Draw element ID if labels are enabled
                    if self.show_cell_labels:
                        elem_type, elem_id = self.cell_mappings[cell_id]
                        if elem_type == 'node':
                            label = f"N:{elem_id}"
                            painter.setPen(QPen(QColor(0, 100, 0), 1))
                        else:
                            if isinstance(elem_id, tuple):
                                label = f"E:{elem_id[0]},{elem_id[1]}"
                            else:
                                label = f"E:{elem_id}"
                            painter.setPen(QPen(QColor(150, 75, 0), 1))  # Orange-brown for edges
                        
                        # Smart text positioning to fit in cell
                        text_rect = scaled_rect.adjusted(2, 2, -2, -2)
                        painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, label)
                    
                elif is_selected:
                    # Currently selected for mapping - bright green
                    painter.fillRect(scaled_rect, QColor(0, 255, 0, 100))
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                    painter.drawRect(scaled_rect)
                else:
                    # Empty cell
                    painter.setPen(QPen(QColor(100, 100, 100), 1))
                    painter.drawRect(scaled_rect)
                
        # Draw completed contours
        if self.show_all_mappings:
            for contour_data in self.completed_contours:
                points, region_id, color = contour_data[:3]
                if len(contour_data) > 3:
                    elem_type, elem_id = contour_data[3:5]
                else:
                    elem_type, elem_id = self.contour_mappings.get(region_id, (None, None))
                    
                if len(points) > 2:
                    poly_points = [QPointF(self.offset_x + p[0] * self.scale_factor, 
                                         self.offset_y + p[1] * self.scale_factor) 
                                 for p in points]
                    polygon = QPolygonF(poly_points)
                    
                    painter.setPen(QPen(color.darker(), 2))
                    painter.setBrush(QBrush(color))
                    painter.drawPolygon(polygon)
                    
                    # Draw element label at centroid with adaptive font size
                    if elem_type and elem_id is not None and self.show_cell_labels:
                        bounding_rect = polygon.boundingRect()
                        centroid = bounding_rect.center()
                        
                        # Calculate adaptive font size based on contour size
                        contour_area = bounding_rect.width() * bounding_rect.height()
                        if self.adaptive_font_size:
                            # Scale-aware font sizing
                            scale_adjusted_area = contour_area / (self.scale_factor * self.scale_factor)
                            if scale_adjusted_area > 5000:
                                font_size = 14
                            elif scale_adjusted_area > 2000:
                                font_size = 12
                            elif scale_adjusted_area > 1000:
                                font_size = 10
                            elif scale_adjusted_area > 500:
                                font_size = 9
                            elif scale_adjusted_area > 200:
                                font_size = 8
                            elif scale_adjusted_area > 100:
                                font_size = 7
                            else:
                                font_size = 6
                        else:
                            font_size = 9
                        
                        font = painter.font()
                        font.setPointSize(font_size)
                        painter.setFont(font)
                        
                        if elem_type == 'node':
                            label = f"N:{elem_id}"
                        else:
                            label = f"E:{elem_id[0]},{elem_id[1]}" if isinstance(elem_id, tuple) else f"E:{elem_id}"
                        
                        # Calculate text size for centering
                        text_rect = painter.fontMetrics().boundingRect(label)
                        text_center = QPointF(
                            centroid.x() - text_rect.width() / 2,
                            centroid.y() + text_rect.height() / 4  # Adjust for vertical centering
                        )
                        
                        painter.setPen(QPen(color.darker().darker(), 1))
                        painter.drawText(text_center, label)
                
        # Draw current contour being drawn
        if self.current_contour:
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            scaled_points = [(self.offset_x + p[0] * self.scale_factor, 
                            self.offset_y + p[1] * self.scale_factor) 
                           for p in self.current_contour]
            for i in range(len(scaled_points) - 1):
                painter.drawLine(QPointF(*scaled_points[i]), QPointF(*scaled_points[i+1]))
                
            # Draw points
            painter.setBrush(QBrush(QColor(255, 0, 0)))
            for point in scaled_points:
                painter.drawEllipse(QPointF(*point), 3, 3)
        
        # Draw test mode highlights
        if hasattr(self, 'test_highlights'):
            try:
                for region_id, highlight_color in self.test_highlights:
                    # Find the region in completed contours or grid cells
                    found_region = False
                    
                    # Check completed contours first
                    for contour_data in self.completed_contours:
                        if contour_data[1] == region_id:  # region_id matches
                            points = contour_data[0]
                            if len(points) > 2:
                                poly_points = [QPointF(self.offset_x + p[0] * self.scale_factor, 
                                                     self.offset_y + p[1] * self.scale_factor) 
                                             for p in points]
                                polygon = QPolygonF(poly_points)
                                
                                # Draw highlight with transparent color
                                painter.setPen(QPen(QColor(*highlight_color[:3]), 3))
                                painter.setBrush(QBrush(QColor(*highlight_color)))
                                painter.drawPolygon(polygon)
                                found_region = True
                                break
                    
                    # Check grid cells if not found in contours
                    if not found_region and region_id in self.grid_cells:
                        rect = self.grid_cells[region_id]
                        scaled_rect = QRectF(
                            self.offset_x + rect.x() * self.scale_factor,
                            self.offset_y + rect.y() * self.scale_factor,
                            rect.width() * self.scale_factor,
                            rect.height() * self.scale_factor
                        )
                        
                        # Draw highlight with transparent color
                        painter.setPen(QPen(QColor(*highlight_color[:3]), 3))
                        painter.setBrush(QBrush(QColor(*highlight_color)))
                        painter.drawRect(scaled_rect)
            except Exception as e:
                print(f"Error drawing test highlights: {e}")
                # Clear the problematic highlights to prevent repeated crashes
                self.test_highlights = []
        
        # Draw test highlight contours (for selected elements in test mode)
        if hasattr(self, 'test_highlight_contours'):
            try:
                for highlight_data in self.test_highlight_contours:
                    points = highlight_data['points']
                    color = highlight_data['color']
                    
                    if len(points) > 2:
                        # Convert points to scaled polygon
                        poly_points = [QPointF(self.offset_x + p[0] * self.scale_factor,
                                             self.offset_y + p[1] * self.scale_factor)
                                     for p in points]
                        polygon = QPolygonF(poly_points)
                        
                        # Draw with highlight color
                        painter.setPen(QPen(color, 3))
                        painter.setBrush(QBrush(color))
                        painter.drawPolygon(polygon)
            except Exception as e:
                print(f"Error drawing test highlight contours: {e}")
                # Clear problematic data
                self.test_highlight_contours = []
        
        # Draw click position indicator in test mode
        if hasattr(self, 'test_click_position') and self.test_click_position:
            x, y = self.test_click_position
            # Convert to screen coordinates
            screen_x = self.offset_x + x * self.scale_factor
            screen_y = self.offset_y + y * self.scale_factor
            
            # Draw a circle at the click position with selected color (coral/pink)
            painter.setPen(QPen(QColor(255, 150, 150), 3))  # Light coral/pink for selection
            painter.setBrush(Qt.NoBrush)  # No fill, just outline
            painter.drawEllipse(QPointF(screen_x, screen_y), 8, 8)  # 8 pixel radius circle
            
            # Draw a smaller filled center point
            painter.setPen(QPen(QColor(255, 150, 150), 1))
            painter.setBrush(QBrush(QColor(255, 150, 150)))
            painter.drawEllipse(QPointF(screen_x, screen_y), 3, 3)  # 3 pixel radius filled circle
                
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.RightButton:
            # Start panning
            self.panning = True
            self.last_pan_point = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
            
        if event.button() == Qt.LeftButton:
            # Convert to image coordinates
            x = (event.x() - self.offset_x) / self.scale_factor
            y = (event.y() - self.offset_y) / self.scale_factor
            
            if self.interaction_mode == 'place_grid':
                self.enable_grid(x, y)
                self.gridPlaced.emit(x, y)
                self.interaction_mode = 'select_cells'
                
            elif self.interaction_mode == 'select_cells' and self.grid_enabled:
                # Find clicked cell
                for cell_id, rect in self.grid_cells.items():
                    if rect.contains(QPointF(x, y)):
                        if cell_id in self.selected_cells:
                            self.selected_cells.remove(cell_id)
                            if cell_id in self.cell_colors:
                                del self.cell_colors[cell_id]
                        else:
                            self.selected_cells.add(cell_id)
                            self.cell_colors[cell_id] = QColor(0, 255, 0, 100)
                        self.cellClicked.emit(cell_id)
                        self.update()
                        break
                        
            elif self.interaction_mode == 'draw_contour':
                self.current_contour.append((x, y))
                self.update()
    
    def add_contour_point(self, x: float, y: float):
        """Add a point to the current contour."""
        if self.interaction_mode == 'draw_contour':
            self.current_contour.append((x, y))
            self.update()
    
    def remove_last_contour_point(self):
        """Remove the last point from current contour."""
        if self.current_contour:
            self.current_contour.pop()
            self.update()
    
    def finish_current_contour(self):
        """Finish and emit the current contour."""
        if len(self.current_contour) > 2:
            self.contourDrawn.emit(self.current_contour.copy())
        self.current_contour.clear()
        self.update()
    
    def cancel_current_contour(self):
        """Cancel the current contour drawing."""
        self.current_contour.clear()
        self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.RightButton and self.panning:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for panning and tooltips."""
        if self.panning:
            # Pan the image
            delta = event.pos() - self.last_pan_point
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_pan_point = event.pos()
            self.update()
            return
            
        # Convert to image coordinates
        x = (event.x() - self.offset_x) / self.scale_factor
        y = (event.y() - self.offset_y) / self.scale_factor
        tooltip_text = ""
        
        if self.grid_enabled:
            # Check if mouse is over a mapped cell
            for cell_id, rect in self.grid_cells.items():
                if rect.contains(QPointF(x, y)) and cell_id in self.cell_mappings:
                    elem_type, elem_id = self.cell_mappings[cell_id]
                    if elem_type == 'node':
                        tooltip_text = f"Node: {elem_id}\nCell: {cell_id}"
                    else:
                        if isinstance(elem_id, tuple):
                            tooltip_text = f"Edge: {elem_id[0]} → {elem_id[1]}\nCell: {cell_id}"
                        else:
                            tooltip_text = f"Edge: {elem_id}\nCell: {cell_id}"
                    break
        else:
            # Check if mouse is over a contour (manual mode)
            from PyQt5.QtGui import QPainterPath
            point = QPointF(x, y)
            found_contour = False
            
            for contour_data in self.completed_contours:
                if len(contour_data) >= 5:
                    points, region_id, color, elem_type, elem_id = contour_data[:5]
                elif len(contour_data) >= 3:
                    points, region_id, color = contour_data[:3]
                    elem_type, elem_id = self.contour_mappings.get(region_id, (None, None))
                else:
                    continue
                    
                if elem_type and elem_id is not None and len(points) > 2:
                    # Create path from contour points
                    path = QPainterPath()
                    path.moveTo(points[0][0], points[0][1])
                    for pt in points[1:]:
                        path.lineTo(pt[0], pt[1])
                    path.closeSubpath()
                    
                    # Check if point is inside contour
                    if path.contains(point):
                        if elem_type == 'node':
                            tooltip_text = f"Node: {elem_id}\nRegion: {region_id}"
                        else:
                            if isinstance(elem_id, tuple):
                                tooltip_text = f"Edge: {elem_id[0]} → {elem_id[1]}\nRegion: {region_id}"
                            else:
                                tooltip_text = f"Edge: {elem_id}\nRegion: {region_id}"
                        found_contour = True
                        break
            
            # Clear tooltip if not over any contour
            if not found_contour:
                tooltip_text = ""
                        
        # Only update tooltip if it changed (improves responsiveness)
        if tooltip_text != self.current_tooltip:
            self.current_tooltip = tooltip_text
            self.setToolTip(tooltip_text)
            
            # Force immediate hide when clearing tooltip
            if not tooltip_text:
                from PyQt5.QtWidgets import QToolTip
                QToolTip.hideText()
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        # Get mouse position relative to widget
        mouse_x = event.x()
        mouse_y = event.y()
        
        # Get mouse position in image coordinates before zoom
        image_x = (mouse_x - self.offset_x) / self.scale_factor
        image_y = (mouse_y - self.offset_y) / self.scale_factor
        
        # Calculate zoom factor
        zoom_delta = event.angleDelta().y() / 120.0  # Standard wheel delta is 120
        zoom_factor = 1.1 ** zoom_delta
        
        # Apply zoom with limits (0.1x to 10x)
        new_user_scale = self.user_scale_factor * zoom_factor
        new_user_scale = max(0.1, min(10.0, new_user_scale))
        
        # Update user scale factor
        self.user_scale_factor = new_user_scale
        
        # Calculate new combined scale factor
        self.scale_factor = self.base_scale_factor * self.user_scale_factor
        
        # Adjust offsets to keep the mouse position fixed during zoom
        new_offset_x = mouse_x - image_x * self.scale_factor
        new_offset_y = mouse_y - image_y * self.scale_factor
        
        self.offset_x = new_offset_x
        self.offset_y = new_offset_y
        
        self.update()
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_R:
            # Reset zoom and center image
            self.reset_zoom()
        else:
            super().keyPressEvent(event)
    
    def reset_zoom(self):
        """Reset zoom to fit image in widget."""
        self.user_scale_factor = 1.0
        self.update()
    
    def highlight_contour(self, region_id: str):
        """Highlight a specific contour on the map persistently."""
        # First unhighlight any currently highlighted contour
        self.unhighlight_current_contour()
        
        # Find and highlight the new contour
        for i, contour_data in enumerate(self.completed_contours):
            if contour_data[1] == region_id:  # region_id is at index 1
                # Store original color before highlighting
                original_color = contour_data[2]
                self.original_contour_colors[region_id] = original_color
                
                # Apply persistent highlight
                highlight_color = QColor(255, 255, 0, 150)  # Yellow highlight
                self.completed_contours[i] = (contour_data[0], contour_data[1], highlight_color) + contour_data[3:]
                
                # Track currently highlighted contour
                self.highlighted_contour_id = region_id
                self.update()
                break
    
    def unhighlight_current_contour(self):
        """Remove highlighting from currently highlighted contour."""
        if self.highlighted_contour_id and self.highlighted_contour_id in self.original_contour_colors:
            # Restore original color
            original_color = self.original_contour_colors[self.highlighted_contour_id]
            for i, contour_data in enumerate(self.completed_contours):
                if contour_data[1] == self.highlighted_contour_id:
                    self.completed_contours[i] = (contour_data[0], contour_data[1], original_color) + contour_data[3:]
                    break
            
            # Clean up tracking
            del self.original_contour_colors[self.highlighted_contour_id]
            self.highlighted_contour_id = None
            self.update()
    
    
    def remove_contour(self, region_id: str):
        """Remove a contour from the map."""
        self.completed_contours = [
            contour for contour in self.completed_contours 
            if contour[1] != region_id
        ]
        if region_id in self.contour_mappings:
            del self.contour_mappings[region_id]
        self.update()

    def clear_highlights(self):
        """Clear all test mode highlights."""
        try:
            if hasattr(self, 'test_highlights'):
                self.test_highlights = []
            self.update()
        except Exception as e:
            print(f"Error clearing highlights: {e}")
            # Don't crash, just continue
    
    def highlight_region(self, region_id: str, color_type='highlight'):
        """Highlight a specific region in test mode."""
        try:
            if not hasattr(self, 'test_highlights'):
                self.test_highlights = []
            
            # Define color schemes (avoiding green/orange of mapping contours and blue nodes)
            colors = {
                'selected': (255, 150, 150, 180),    # Light coral/pink for selection
                'highlight': (160, 100, 255, 140)    # Darker purple for better visibility
            }
            
            color = colors.get(color_type, colors['highlight'])
            
            # Remove existing highlight for this region
            self.test_highlights = [h for h in self.test_highlights if h[0] != region_id]
            
            # Add new highlight
            self.test_highlights.append((region_id, color))
            self.update()
        except Exception as e:
            print(f"Error highlighting region {region_id}: {e}")
            # Don't crash, just continue


class GraphWidget(FigureCanvas):
    """Widget for displaying the graph structure using matplotlib."""
    
    nodeClicked = pyqtSignal(object)  # Emitted when a node is clicked
    edgeClicked = pyqtSignal(tuple)  # Emitted when an edge is clicked
    
    def __init__(self, graph: GraphStructure, parent=None):
        self.graph = graph
        self.figure = Figure(figsize=(24, 8))
        super().__init__(self.figure)
        self.setParent(parent)
        
        self.ax = self.figure.add_subplot(111)
        self.node_colors = {}
        self.edge_colors = {}
        self.highlighted_nodes = set()
        self.highlighted_edges = set()
        
        self._compute_layout()
        self.draw_graph()
        
        # Note: Graph click detection removed - using dropdown selection in test mode instead

    def _compute_layout(self):
        """Compute graph layout for visualization."""
        # Use stored positions if available, otherwise compute
        if self.graph.node_positions:
            self.pos = self.graph.node_positions
        else:
            # For binary trees, create a hierarchical layout manually
            if hasattr(self.graph, 'graph') and nx.is_tree(self.graph.graph):
                try:
                    # Try graphviz first - it creates the best tree layout
                    self.pos = nx.nx_agraph.graphviz_layout(self.graph.graph, prog='dot')
                    # Scale horizontally to make it wider
                    if self.pos:
                        scale_factor = 4.0  # Make it much wider to spread leaf nodes
                        self.pos = {node: (x * scale_factor, y) for node, (x, y) in self.pos.items()}
                except:
                    # Manual hierarchical layout for binary trees
                    self.pos = self._create_binary_tree_layout()
            else:
                # Use spring layout with stronger spacing for better node separation
                self.pos = nx.spring_layout(self.graph.graph, seed=42, k=15.0, iterations=300)
    
    def _create_binary_tree_layout(self):
        """Create a hierarchical layout for binary trees."""
        pos = {}
        nodes = list(self.graph.graph.nodes())
        
        # Simple level-based layout
        levels = {}
        queue = [nodes[0]] if nodes else []  # Start with root
        visited = set()
        level = 0
        
        while queue:
            next_queue = []
            for node in queue:
                if node not in visited:
                    if level not in levels:
                        levels[level] = []
                    levels[level].append(node)
                    visited.add(node)
                    
                    # Add children
                    for neighbor in self.graph.graph.neighbors(node):
                        if neighbor not in visited:
                            next_queue.append(neighbor)
            
            queue = next_queue
            level += 1
        
        # Position nodes with better spacing
        max_width = max(len(level_nodes) for level_nodes in levels.values()) if levels else 1
        for level, level_nodes in levels.items():
            y = -level * 5  # Increased vertical spacing
            for i, node in enumerate(level_nodes):
                # Spread nodes horizontally with much better spacing
                if len(level_nodes) == 1:
                    x = 0
                else:
                    # Much wider horizontal spacing, especially for bottom levels
                    base_spacing = max(4, max_width / len(level_nodes)) * 5
                    # Increase spacing exponentially for lower levels to handle leaf nodes
                    level_multiplier = 1 + (level * 1.0)
                    spacing_factor = base_spacing * level_multiplier
                    x = (i - (len(level_nodes) - 1) / 2) * spacing_factor
                pos[node] = (x, y)
        
        return pos
                
    def draw_graph(self):
        """Draw or redraw the graph."""
        self.ax.clear()
        
        # Prepare node colors
        node_colors = []
        for node in self.graph.nodes:
            if node in self.node_colors:
                node_colors.append(self.node_colors[node])
            else:
                node_colors.append('lightblue')
                
        # Draw graph with appropriate sizing for large trees
        node_count = len(self.graph.nodes)
        if node_count > 100:
            # Very small nodes for very large graphs
            node_size = 250
            font_size = 7
        elif node_count > 50:
            # Small nodes for large graphs
            node_size = 400
            font_size = 8
        elif node_count > 20:
            # Medium nodes for medium graphs
            node_size = 600
            font_size = 10
        else:
            # Large nodes for small graphs
            node_size = 800
            font_size = 12
            
        nx.draw(self.graph.graph, pos=self.pos, ax=self.ax,
               node_color=node_colors, node_size=node_size,
               with_labels=True, font_size=font_size, font_weight='bold',
               edge_color='gray', linewidths=1, font_color='black',
               node_shape='o', alpha=0.9)
               
        # Highlight specific edges if needed
        if self.highlighted_edges:
            edge_list = list(self.highlighted_edges)
            # Get colors for highlighted edges
            edge_colors_list = [self.edge_colors.get(edge, 'orange') for edge in edge_list]
            nx.draw_networkx_edges(self.graph.graph, pos=self.pos, ax=self.ax,
                                  edgelist=edge_list, edge_color=edge_colors_list, width=2)
                                  
        self.ax.set_title(f"Graph Structure ({len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges)")
        
        # Adjust subplot margins to prevent title truncation
        self.figure.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
        
        self.draw()
        
    def highlight_nodes(self, nodes: Set, color: str = 'lightgreen'):
        """Highlight specific nodes."""
        self.highlighted_nodes = nodes
        # Map color names to actual colors (avoiding green/orange of mapping contours and blue nodes)
        color_map = {
            'selected': 'lightcoral',
            'highlight': 'mediumorchid',  # Darker purple for better visibility
            'lightgreen': 'lightgreen'
        }
        actual_color = color_map.get(color, color)
        
        # Store colors for highlighted nodes
        for node in nodes:
            self.node_colors[node] = actual_color
        self.draw_graph()
        
    def highlight_edges(self, edges: Set[Tuple], color: str = 'orange'):
        """Highlight specific edges."""
        self.highlighted_edges = edges
        # Map color names to actual colors (avoiding green/orange of mapping contours and blue nodes)
        color_map = {
            'selected': 'lightcoral',
            'highlight': 'mediumorchid',  # Darker purple for better visibility
            'orange': 'orange'
        }
        actual_color = color_map.get(color, color)
        
        # Store colors for highlighted edges
        for edge in edges:
            self.edge_colors[edge] = actual_color
        self.draw_graph()
        
    def clear_highlights(self):
        """Clear all highlights."""
        self.highlighted_nodes.clear()
        self.highlighted_edges.clear()
        self.node_colors.clear()
        self.edge_colors.clear()
        self.draw_graph()


class GraphSetupWindow(QMainWindow):
    """Main window for graph-space mapping setup."""
    
    def __init__(self, graph: GraphStructure, map_image: np.ndarray):
        super().__init__()
        
        self.graph = graph
        self.map_image = map_image
        self.mapping = SpatialMapping(graph)
        
        # Setup state
        self.setup_mode = None  # 'grid' or 'manual'
        self.current_element = None  # ('node', id) or ('edge', (id1, id2))
        self.element_queue = []  # Queue of elements to map
        self.all_elements = []  # All elements for selection
        self.mapped_elements = {}  # element -> region_ids
        self.mapping_history = []  # Stack for undo functionality
        
        # Initialize UI
        self.setWindowTitle("NaviGraph - Interactive Graph Mapping Setup")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Set application style with improved UI
        self.setStyleSheet("""
            QMainWindow {
                background-color: #fafafa;
            }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                margin: 8px;
                padding-top: 20px;
                background-color: white;
                font-size: 13px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
                color: #424242;
                background-color: white;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton {
                background-color: #ffffff;
                color: #424242;
                border: 2px solid #d0d0d0;
                padding: 8px 15px;
                border-radius: 6px;
                font-weight: 500;
                font-size: 13px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #f8f9fa;
                border-color: #999999;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
                border-color: #888888;
            }
            QPushButton:disabled {
                background-color: #fafafa;
                color: #b0b0b0;
                border-color: #e0e0e0;
            }
            QPushButton:checked {
                background-color: #e8e8e8;
                border: 2px solid #666666;
                border-style: inset;
                color: #212121;
                font-weight: 600;
                padding: 9px 15px 7px 15px;
            }
            QPushButton:checked:hover {
                background-color: #e0e0e0;
                border-color: #555555;
            }
            QLabel {
                color: #424242;
                font-size: 13px;
            }
            QStatusBar {
                background-color: #f5f5f5;
                border-top: 1px solid #e0e0e0;
                color: #616161;
                font-size: 12px;
            }
            QListWidget {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #f0f0f0;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            QListWidget::item:hover {
                background-color: #f5f5f5;
            }
            QComboBox {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 13px;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #b0b0b0;
            }
            QComboBox::drop-down {
                border: none;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 4px;
                font-size: 13px;
                min-height: 24px;
            }
            QCheckBox {
                font-size: 13px;
                color: #424242;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QProgressBar {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: #f5f5f5;
                text-align: center;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: #2196f3;
                border-radius: 3px;
            }
        """)
        
        self._init_ui()
        self._connect_signals()
        
    def _init_ui(self):
        """Initialize the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left panel - Controls (fixed width)
        self.control_panel = self._create_control_panel()
        self.control_panel.setFixedWidth(420)
        self.control_panel.setStyleSheet(
            "QWidget { background-color: #f8f9fa; border-right: 2px solid #dee2e6; }"
        )
        main_layout.addWidget(self.control_panel)
        
        # Right panel - Views (expandable)
        views_widget = QWidget()
        views_layout = QVBoxLayout(views_widget)
        views_layout.setContentsMargins(5, 5, 5, 5)
        
        # Graph view (top, smaller)
        graph_group = QGroupBox("📊 Graph Structure")
        graph_group.setStyleSheet("QGroupBox { font-weight: 600; font-size: 14px; color: #424242; }")
        graph_layout = QVBoxLayout(graph_group)
        graph_layout.setContentsMargins(8, 20, 8, 8)
        
        self.graph_widget = GraphWidget(self.graph)
        self.graph_widget.setMinimumHeight(300)
        self.graph_widget.setMaximumHeight(500)
        graph_layout.addWidget(self.graph_widget)
        
        views_layout.addWidget(graph_group)
        
        # Map view (bottom, larger)
        map_group = QGroupBox("🗺️ Interactive Mapping Area")
        map_group.setStyleSheet("QGroupBox { font-weight: 600; font-size: 14px; color: #424242; }")
        map_layout = QVBoxLayout(map_group)
        map_layout.setContentsMargins(8, 20, 8, 8)
        
        self.map_widget = MapWidget(self.map_image)
        self.map_widget.setMinimumHeight(350)
        map_layout.addWidget(self.map_widget)
        
        views_layout.addWidget(map_group)
        
        # Set stretch factors (graph:map = 2:3 to give more space to graph)
        views_layout.setStretchFactor(graph_group, 2)
        views_layout.setStretchFactor(map_group, 3)
        
        main_layout.addWidget(views_widget)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Select a setup mode to begin")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(len(self.graph.nodes) + len(self.graph.edges))
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def _create_control_panel(self) -> QWidget:
        """Create the simplified control panel widget."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Mode Selection (Always Visible)
        mode_group = QGroupBox("Setup Mode")
        mode_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: 600; }")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(10, 20, 10, 10)
        
        # Create button layout with proper spacing
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)
        
        self.grid_mode_button = QPushButton("Grid Setup")
        self.grid_mode_button.setCheckable(True)
        self.grid_mode_button.clicked.connect(self._on_grid_mode)
        self.grid_mode_button.setFixedHeight(40)
        self.grid_mode_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button_layout.addWidget(self.grid_mode_button)
        
        self.manual_mode_button = QPushButton("Manual Drawing")
        self.manual_mode_button.setCheckable(True)
        self.manual_mode_button.clicked.connect(self._on_manual_mode)
        self.manual_mode_button.setFixedHeight(40)
        self.manual_mode_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button_layout.addWidget(self.manual_mode_button)
        
        self.test_mode_button = QPushButton("Test Mode")
        self.test_mode_button.setCheckable(True)
        self.test_mode_button.clicked.connect(self._on_test_mode)
        self.test_mode_button.setFixedHeight(40)
        self.test_mode_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button_layout.addWidget(self.test_mode_button)
        
        mode_layout.addLayout(button_layout)
        
        layout.addWidget(mode_group)
        
        # Mode-specific controls (using QStackedWidget)
        self.mode_stack = QStackedWidget()
        
        # Default/empty page
        empty_widget = QWidget()
        empty_layout = QVBoxLayout(empty_widget)
        empty_layout.addWidget(QLabel("Select a mode to begin mapping"))
        self.mode_stack.addWidget(empty_widget)
        
        # Grid mode page
        self.grid_controls = self._create_grid_controls()
        self.mode_stack.addWidget(self.grid_controls)
        
        # Manual mode page
        self.manual_controls = self._create_manual_controls()
        self.mode_stack.addWidget(self.manual_controls)
        
        # Test mode page
        self.test_controls = self._create_test_controls()
        self.mode_stack.addWidget(self.test_controls)
        
        layout.addWidget(self.mode_stack)
        
        # Visualization options
        viz_group = QGroupBox("👁️ Display Options")
        viz_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: 600; }")
        viz_layout = QVBoxLayout(viz_group)
        viz_layout.setContentsMargins(10, 15, 10, 10)
        
        self.show_mappings_checkbox = QCheckBox("Show All Mappings")
        self.show_mappings_checkbox.setChecked(True)
        self.show_mappings_checkbox.stateChanged.connect(self._on_toggle_mappings)
        viz_layout.addWidget(self.show_mappings_checkbox)
        
        self.show_labels_checkbox = QCheckBox("Show Cell Labels")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.stateChanged.connect(self._on_toggle_labels)
        viz_layout.addWidget(self.show_labels_checkbox)
        
        self.adaptive_font_checkbox = QCheckBox("Adaptive Font Size")
        self.adaptive_font_checkbox.setChecked(True)
        self.adaptive_font_checkbox.stateChanged.connect(self._on_toggle_adaptive_font)
        viz_layout.addWidget(self.adaptive_font_checkbox)
        
        # Test Mode Layout Options (only visible in test mode)
        self.test_layout_spacer = QLabel("")  # Spacer
        viz_layout.addWidget(self.test_layout_spacer)
        
        self.test_layout_label = QLabel("Test Mode Layout:")
        self.test_layout_label.setStyleSheet("font-size: 12px; color: #616161; margin-top: 8px;")
        viz_layout.addWidget(self.test_layout_label)
        
        self.test_layout_vertical = QRadioButton("Vertical (Graph Top, Map Bottom)")
        self.test_layout_vertical.setChecked(True)  # Default
        self.test_layout_vertical.toggled.connect(self._on_test_layout_changed)
        viz_layout.addWidget(self.test_layout_vertical)
        
        self.test_layout_horizontal = QRadioButton("Horizontal (Map Left, Graph Right)")
        self.test_layout_horizontal.toggled.connect(self._on_test_layout_changed)
        viz_layout.addWidget(self.test_layout_horizontal)
        
        # Store test mode widgets for show/hide
        self.test_mode_widgets = [
            self.test_layout_spacer,
            self.test_layout_label,
            self.test_layout_vertical,
            self.test_layout_horizontal
        ]
        
        # Initially hide test mode widgets (will be shown when test mode is activated)
        for widget in self.test_mode_widgets:
            widget.hide()
        
        layout.addWidget(viz_group)
        
        return panel
        
    def _create_grid_controls(self) -> QWidget:
        """Create simplified grid setup controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Remove redundant current element display - using status bar instead
        
        # Element selection
        element_layout = QHBoxLayout()
        element_layout.addWidget(QLabel("Element:"))
        self.grid_element_combo = QComboBox()
        element_layout.addWidget(self.grid_element_combo)
        self.grid_jump_button = QPushButton("Jump")
        self.grid_jump_button.clicked.connect(self._on_jump_to_element)
        element_layout.addWidget(self.grid_jump_button)
        layout.addLayout(element_layout)
        
        # Grid configuration
        config_group = QGroupBox("Grid Configuration")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(10)
        
        # Grid dimensions in a properly aligned grid
        dims_layout = QGridLayout()
        dims_layout.setColumnStretch(0, 1)
        dims_layout.setColumnStretch(1, 2)
        dims_layout.setColumnStretch(2, 1)
        dims_layout.setColumnStretch(3, 2)
        
        # Rows
        dims_layout.addWidget(QLabel("Rows:"), 0, 0, Qt.AlignRight)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 20)
        self.rows_spin.setValue(8)
        self.rows_spin.setButtonSymbols(QSpinBox.NoButtons)  # Hide default buttons
        
        # Create custom buttons for rows
        rows_button_layout = QHBoxLayout()
        rows_button_layout.setSpacing(0)
        rows_button_layout.addWidget(self.rows_spin)
        
        rows_minus_btn = QPushButton("-")
        rows_minus_btn.setFixedSize(20, 24)
        rows_minus_btn.setStyleSheet("QPushButton { padding: 0px; margin: 0px; border: 1px solid #d0d0d0; font-weight: bold; font-size: 14px; }")
        rows_minus_btn.clicked.connect(lambda: self.rows_spin.setValue(self.rows_spin.value() - 1))
        rows_button_layout.addWidget(rows_minus_btn)
        
        rows_plus_btn = QPushButton("+")
        rows_plus_btn.setFixedSize(20, 24)
        rows_plus_btn.setStyleSheet("QPushButton { padding: 0px; margin: 0px; border: 1px solid #d0d0d0; font-weight: bold; font-size: 14px; }")
        rows_plus_btn.clicked.connect(lambda: self.rows_spin.setValue(self.rows_spin.value() + 1))
        rows_button_layout.addWidget(rows_plus_btn)
        
        dims_layout.addLayout(rows_button_layout, 0, 1)
        
        # Columns
        dims_layout.addWidget(QLabel("Cols:"), 0, 2, Qt.AlignRight)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 20)
        self.cols_spin.setValue(8)
        self.cols_spin.setButtonSymbols(QSpinBox.NoButtons)  # Hide default buttons
        
        # Create custom buttons for columns
        cols_button_layout = QHBoxLayout()
        cols_button_layout.setSpacing(0)
        cols_button_layout.addWidget(self.cols_spin)
        
        cols_minus_btn = QPushButton("-")
        cols_minus_btn.setFixedSize(20, 24)
        cols_minus_btn.setStyleSheet("QPushButton { padding: 0px; margin: 0px; border: 1px solid #d0d0d0; font-weight: bold; font-size: 14px; }")
        cols_minus_btn.clicked.connect(lambda: self.cols_spin.setValue(self.cols_spin.value() - 1))
        cols_button_layout.addWidget(cols_minus_btn)
        
        cols_plus_btn = QPushButton("+")
        cols_plus_btn.setFixedSize(20, 24)
        cols_plus_btn.setStyleSheet("QPushButton { padding: 0px; margin: 0px; border: 1px solid #d0d0d0; font-weight: bold; font-size: 14px; }")
        cols_plus_btn.clicked.connect(lambda: self.cols_spin.setValue(self.cols_spin.value() + 1))
        cols_button_layout.addWidget(cols_plus_btn)
        
        dims_layout.addLayout(cols_button_layout, 0, 3)
        
        # Cell Size
        dims_layout.addWidget(QLabel("Cell Size:"), 1, 0, Qt.AlignRight)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(10, 200)
        self.width_spin.setValue(50)
        self.width_spin.setSuffix(" px")
        self.width_spin.setDecimals(1)
        self.width_spin.setButtonSymbols(QDoubleSpinBox.NoButtons)  # Hide default buttons
        
        # Create custom buttons for cell size
        size_button_layout = QHBoxLayout()
        size_button_layout.setSpacing(0)
        size_button_layout.addWidget(self.width_spin)
        
        size_minus_btn = QPushButton("-")
        size_minus_btn.setFixedSize(20, 24)
        size_minus_btn.setStyleSheet("QPushButton { padding: 0px; margin: 0px; border: 1px solid #d0d0d0; font-weight: bold; font-size: 14px; }")
        size_minus_btn.clicked.connect(lambda: self.width_spin.setValue(self.width_spin.value() - 5))
        size_button_layout.addWidget(size_minus_btn)
        
        size_plus_btn = QPushButton("+")
        size_plus_btn.setFixedSize(20, 24)
        size_plus_btn.setStyleSheet("QPushButton { padding: 0px; margin: 0px; border: 1px solid #d0d0d0; font-weight: bold; font-size: 14px; }")
        size_plus_btn.clicked.connect(lambda: self.width_spin.setValue(self.width_spin.value() + 5))
        size_button_layout.addWidget(size_plus_btn)
        
        dims_layout.addLayout(size_button_layout, 1, 1)
        
        config_layout.addLayout(dims_layout)
        
        # Apply and place buttons
        config_buttons = QHBoxLayout()
        config_buttons.setSpacing(8)
        self.apply_grid_button = QPushButton("Apply Config")
        self.apply_grid_button.clicked.connect(self._on_apply_grid_config)
        config_buttons.addWidget(self.apply_grid_button)
        
        self.place_grid_button = QPushButton("Place Grid")
        self.place_grid_button.setCheckable(True)  # Make it toggleable
        self.place_grid_button.clicked.connect(self._on_place_grid)
        config_buttons.addWidget(self.place_grid_button)
        
        self.clear_grid_button = QPushButton("Clear Grid")
        self.clear_grid_button.clicked.connect(self._on_clear_grid)
        self.clear_grid_button.setEnabled(False)  # Disabled until grid is placed
        config_buttons.addWidget(self.clear_grid_button)
        config_layout.addLayout(config_buttons)
        
        # Compact status label for grid origin only
        self.grid_status_label = QLabel("")
        self.grid_status_label.setStyleSheet("QLabel { color: #616161; font-size: 11px; padding: 2px; }")
        self.grid_status_label.setMaximumHeight(20)
        config_layout.addWidget(self.grid_status_label)
        layout.addWidget(config_group)
        
        # Assignment controls
        assign_group = QGroupBox("Cell Assignment")
        assign_layout = QVBoxLayout(assign_group)
        
        self.assign_button = QPushButton("Assign Selected Cells")
        self.assign_button.clicked.connect(self._on_assign_cells)
        self.assign_button.setEnabled(False)
        assign_layout.addWidget(self.assign_button)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.next_element_button = QPushButton("Next")
        self.next_element_button.clicked.connect(self._on_next_element)
        self.next_element_button.setEnabled(False)
        control_layout.addWidget(self.next_element_button)
        
        self.undo_grid_button = QPushButton("Undo")
        self.undo_grid_button.clicked.connect(self._on_undo_last)
        self.undo_grid_button.setEnabled(False)
        control_layout.addWidget(self.undo_grid_button)
        
        self.clear_all_grid_button = QPushButton("Clear All")
        self.clear_all_grid_button.clicked.connect(self._on_clear_all)
        self.clear_all_grid_button.setEnabled(True)  # Always enabled
        control_layout.addWidget(self.clear_all_grid_button)
        
        assign_layout.addLayout(control_layout)
        layout.addWidget(assign_group)
        
        # Save/Load Mapping buttons
        mapping_group = QGroupBox("File Operations")
        mapping_layout = QHBoxLayout(mapping_group)
        
        self.save_intermediate_button = QPushButton("Save Mapping")
        self.save_intermediate_button.clicked.connect(self._on_save_intermediate_mapping)
        mapping_layout.addWidget(self.save_intermediate_button)
        
        self.load_intermediate_button = QPushButton("Load Mapping")
        self.load_intermediate_button.clicked.connect(self._on_load_intermediate_mapping)
        mapping_layout.addWidget(self.load_intermediate_button)
        
        layout.addWidget(mapping_group)
        
        return widget
        
    def _create_manual_controls(self) -> QWidget:
        """Create simplified manual drawing controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Remove redundant current element display - using status bar instead
        
        # Element selection
        element_layout = QHBoxLayout()
        element_layout.addWidget(QLabel("Element:"))
        self.manual_element_combo = QComboBox()
        element_layout.addWidget(self.manual_element_combo)
        self.manual_jump_button = QPushButton("Jump")
        self.manual_jump_button.clicked.connect(self._on_jump_to_element)
        element_layout.addWidget(self.manual_jump_button)
        layout.addLayout(element_layout)
        
        # Drawing instructions
        layout.addWidget(QLabel("Left click on map to draw points"))
        
        # Simplified drawing controls (only 2 buttons)
        drawing_group = QGroupBox("Drawing Controls")
        drawing_layout = QVBoxLayout(drawing_group)
        
        control_layout = QHBoxLayout()
        self.clear_contour_button = QPushButton("Clear Contour")
        self.clear_contour_button.clicked.connect(self._on_clear_contour)
        control_layout.addWidget(self.clear_contour_button)
        
        self.commit_contour_button = QPushButton("Commit Contour")
        self.commit_contour_button.clicked.connect(self._on_commit_contour)
        control_layout.addWidget(self.commit_contour_button)
        
        drawing_layout.addLayout(control_layout)
        layout.addWidget(drawing_group)
        
        # Contour list
        mgmt_group = QGroupBox("Existing Contours")
        mgmt_layout = QVBoxLayout(mgmt_group)
        
        self.contour_list = QListWidget()
        self.contour_list.itemClicked.connect(self._on_contour_selected)
        self.contour_list.setStyleSheet(
            "QListWidget { background-color: white; border: 1px solid #ccc; border-radius: 4px; }"
            "QListWidget::item { padding: 4px; border-bottom: 1px solid #eee; }"
            "QListWidget::item:selected { background-color: #ffeb3b; color: black; }"
            "QListWidget::item:hover { background-color: #f5f5f5; }"
        )
        mgmt_layout.addWidget(self.contour_list)
        
        contour_buttons = QHBoxLayout()
        self.delete_contour_button = QPushButton("Delete")
        self.delete_contour_button.clicked.connect(self._on_delete_contour)
        self.delete_contour_button.setEnabled(False)
        contour_buttons.addWidget(self.delete_contour_button)
        
        self.clear_all_contours_button = QPushButton("Clear All")
        self.clear_all_contours_button.clicked.connect(self._on_clear_all_contours)
        contour_buttons.addWidget(self.clear_all_contours_button)
        
        mgmt_layout.addLayout(contour_buttons)
        layout.addWidget(mgmt_group)
        
        # File operations for manual mode
        control_group = QGroupBox("File Operations")
        control_group_layout = QHBoxLayout(control_group)
        
        self.manual_save_button = QPushButton("Save Mapping")
        self.manual_save_button.clicked.connect(self._on_save_intermediate_mapping)
        control_group_layout.addWidget(self.manual_save_button)
        
        self.manual_load_button = QPushButton("Load Mapping")
        self.manual_load_button.clicked.connect(self._on_load_intermediate_mapping)
        control_group_layout.addWidget(self.manual_load_button)
        
        layout.addWidget(control_group)
        
        return widget
        
    def _create_test_controls(self) -> QWidget:
        """Create simplified test mode controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Simple Instructions
        instructions = QLabel(
            "Test Mode:\n"
            "• Click map to highlight graph element\n"
            "• Select element below to highlight map regions"
        )
        instructions.setStyleSheet(
            "background-color: #f8f9fa; border: 1px solid #dee2e6; "
            "border-radius: 6px; padding: 8px; font-size: 11px;"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Element selection
        element_group = QGroupBox("Graph Element Selection")
        element_layout = QVBoxLayout(element_group)
        
        # Dropdown and select button
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Element:"))
        
        self.test_element_combo = QComboBox()
        self.test_element_combo.setMinimumWidth(200)
        selection_layout.addWidget(self.test_element_combo)
        
        self.test_select_button = QPushButton("Select")
        self.test_select_button.clicked.connect(self._on_test_select_element)
        selection_layout.addWidget(self.test_select_button)
        
        element_layout.addLayout(selection_layout)
        layout.addWidget(element_group)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.test_load_button = QPushButton("Load Mapping")
        self.test_load_button.clicked.connect(self._on_load_intermediate_mapping)
        control_layout.addWidget(self.test_load_button)
        
        self.test_clear_button = QPushButton("Clear All")
        self.test_clear_button.clicked.connect(self._on_clear_test_markings)
        control_layout.addWidget(self.test_clear_button)
        
        layout.addLayout(control_layout)
        
        # Current selection info
        self.test_info_label = QLabel("No selection")
        self.test_info_label.setStyleSheet(
            "background-color: white; border: 1px solid #ccc; "
            "border-radius: 4px; padding: 6px; font-size: 11px;"
        )
        layout.addWidget(self.test_info_label)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return widget
        
    def _on_grid_mode(self, checked: bool):
        """Handle grid mode selection."""
        try:
            if checked:
                # Check for unsaved progress before switching
                if not self._confirm_mode_switch("Grid Setup"):
                    # User cancelled, revert button state
                    self.grid_mode_button.setChecked(False)
                    return
                self.manual_mode_button.setChecked(False)
                self.test_mode_button.setChecked(False)
                self.setup_mode = 'grid'
                self.mode_stack.setCurrentIndex(1)  # Grid controls
                
                # Always reset to fresh state when switching modes
                try:
                    self._reset_mapping_state()
                    self._init_element_queue()
                    self._populate_element_combos()
                    self._update_progress_display()
                    
                    # Set mode-specific visualization defaults
                    self._set_mode_specific_defaults('grid')
                except Exception as init_error:
                    self.status_bar.showMessage(f"Error initializing grid mode: {str(init_error)}")
                    print(f"Grid mode initialization error: {init_error}")
                    return
                
                # Ensure we're back to normal layout after test mode
                if hasattr(self, 'test_splitter'):
                    self._restore_normal_layout()
                
                # Enable drawing mode
                if hasattr(self.map_widget, 'set_interaction_mode'):
                    self.map_widget.set_interaction_mode('none')
                elif self.map_widget:
                    pass  # Grid mode doesn't need special interaction mode
                else:
                    print("Warning: map_widget not available after layout restoration")
                    
                self.status_bar.showMessage("Grid Setup Mode - Configure and place grid")
            else:
                self.mode_stack.setCurrentIndex(0)  # Empty
        except Exception as e:
            self.status_bar.showMessage(f"Error switching to grid mode: {str(e)}")
            print(f"Grid mode error: {e}")
            import traceback
            traceback.print_exc()
            
    def _on_manual_mode(self, checked: bool):
        """Handle manual drawing mode selection."""
        try:
            if checked:
                # Check for unsaved progress before switching
                if not self._confirm_mode_switch("Manual Drawing"):
                    # User cancelled, revert button state
                    self.manual_mode_button.setChecked(False)
                    return
                self.grid_mode_button.setChecked(False)
                self.test_mode_button.setChecked(False)
                self.setup_mode = 'manual'
                self.mode_stack.setCurrentIndex(2)  # Manual controls
                
                # Always reset to fresh state when switching modes
                try:
                    self._reset_mapping_state()
                    self._init_element_queue()
                    self._populate_element_combos()
                    self._update_progress_display()
                    
                    # Set mode-specific visualization defaults
                    self._set_mode_specific_defaults('manual')
                except Exception as init_error:
                    self.status_bar.showMessage(f"Error initializing manual mode: {str(init_error)}")
                    print(f"Manual mode initialization error: {init_error}")
                    return
                
                # Ensure we're back to normal layout after test mode
                if hasattr(self, 'test_splitter'):
                    self._restore_normal_layout()
                
                # Enable drawing mode
                if hasattr(self.map_widget, 'set_interaction_mode'):
                    try:
                        self.map_widget.set_interaction_mode('draw_contour')
                        self.status_bar.showMessage("Manual Drawing Mode - Left click to draw points")
                        
                        # Connect map widget mouse events to update button states
                        if not hasattr(self, '_original_map_mouse_press'):
                            self._original_map_mouse_press = self.map_widget.mousePressEvent
                        self.map_widget.mousePressEvent = self._wrap_map_mouse_press(self._original_map_mouse_press)
                        
                    except RuntimeError as e:
                        print(f"Error setting map widget interaction mode: {e}")
                        self.status_bar.showMessage("Manual mode activated - but drawing may not work properly")
                elif self.map_widget:
                    # If set_interaction_mode doesn't exist, just show the status
                    self.status_bar.showMessage("Manual Drawing Mode - Left click to draw points")
                else:
                    print("Warning: map_widget not available after layout restoration")
            else:
                self.mode_stack.setCurrentIndex(0)  # Empty
        except Exception as e:
            self.status_bar.showMessage(f"Error switching to manual mode: {str(e)}")
            print(f"Manual mode error: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_test_mode(self, checked: bool):
        """Handle test mode selection."""
        try:
            if checked:
                # Check for unsaved progress before switching
                if not self._confirm_mode_switch("Test"):
                    # User cancelled, revert button state
                    self.test_mode_button.setChecked(False)
                    return
                self.grid_mode_button.setChecked(False)
                self.manual_mode_button.setChecked(False)
                self.setup_mode = 'test'
                self.mode_stack.setCurrentIndex(3)  # Test controls
                
                # Always reset to fresh state when switching modes
                try:
                    self._reset_mapping_state()
                except Exception as reset_error:
                    self.status_bar.showMessage(f"Error resetting state for test mode: {str(reset_error)}")
                    print(f"Test mode reset error: {reset_error}")
                
                # Switch to test mode layout
                self._setup_test_layout()
                
                # Enable test interactions
                if hasattr(self.test_map_widget, 'set_interaction_mode'):
                    self.test_map_widget.set_interaction_mode('test')
                    
                # Initialize test mode state
                self.test_selected_point = None  # Selected point on map
                self.test_selected_element = None  # Selected graph element (node or edge)
                
                # Set mode-specific visualization defaults
                self._set_mode_specific_defaults('test')
                
                # Populate element selection dropdown
                self._populate_test_element_combo()
                    
                self._update_progress_display()
                self.status_bar.showMessage("Test Mode - Click map or select element to test mapping")
            else:
                # Restore normal layout
                self._restore_normal_layout()
                self.mode_stack.setCurrentIndex(0)  # Empty
        except Exception as e:
            self.status_bar.showMessage(f"Error switching to test mode: {str(e)}")
            print(f"Test mode error: {e}")
            import traceback
            traceback.print_exc()
    
    def _populate_element_combos(self):
        """Populate element selection combo boxes for both modes."""
        # Clear both combo boxes
        self.grid_element_combo.clear()
        self.manual_element_combo.clear()
        
        # Get mapped elements to show status
        mapped_elements = set()
        if hasattr(self, 'mapping') and self.mapping:
            # Check node mappings
            for node_id, regions in getattr(self.mapping, '_node_to_regions', {}).items():
                if regions:  # Has mapped regions
                    mapped_elements.add(('node', node_id))
            
            # Check edge mappings  
            for edge_id, regions in getattr(self.mapping, '_edge_to_regions', {}).items():
                if regions:  # Has mapped regions
                    mapped_elements.add(('edge', edge_id))
        
        # Populate with all elements, marking mapped ones
        for elem_type, elem_id in self.all_elements:
            if elem_type == 'node':
                base_label = f"Node: {elem_id}"
            else:
                base_label = f"Edge: {elem_id}"
            
            # Add status indicator for mapped elements
            if (elem_type, elem_id) in mapped_elements:
                label = f"✓ {base_label} (mapped)"
            else:
                label = base_label
                
            self.grid_element_combo.addItem(label)
            self.manual_element_combo.addItem(label)
    
    def _update_progress_display(self):
        """Update the progress information display."""
        try:
            stats = self.mapping.validate_mapping()
            
            # Update progress bar safely
            progress_value = stats.mapped_nodes + stats.mapped_edges
            if hasattr(self, 'progress_bar') and self.progress_bar:
                self.progress_bar.setValue(progress_value)
            
            # Update progress info text safely
            total_elements = len(self.graph.nodes) + len(self.graph.edges)
            if total_elements > 0:
                completion = (progress_value / total_elements) * 100
                progress_text = f"{stats.mapped_nodes}/{stats.total_nodes} nodes, {stats.mapped_edges}/{stats.total_edges} edges ({completion:.0f}% complete)"
            else:
                progress_text = "No elements to map"
                
            # Update progress info label if it exists
            if hasattr(self, 'progress_info') and self.progress_info:
                self.progress_info.setText(progress_text)
            
            # Enable test button if there are mappings
            has_mappings = stats.mapped_nodes > 0 or stats.mapped_edges > 0
            # Test mapping button was moved to test mode controls
            if hasattr(self, 'test_mode_button'):
                self.test_mode_button.setEnabled(True)  # Always allow test mode access
            
            # Current element is now displayed in the status bar only
            
        except Exception as e:
            print(f"Error updating progress display: {e}")
            if hasattr(self, 'status_bar') and self.status_bar:
                self.status_bar.showMessage("Error updating progress display")
        
    def _connect_signals(self):
        """Connect widget signals."""
        self.map_widget.gridPlaced.connect(self._on_grid_placed)
        self.map_widget.cellClicked.connect(self._on_cell_clicked)
        self.map_widget.contourDrawn.connect(self._on_contour_drawn)
        
        self.graph_widget.nodeClicked.connect(self._on_node_clicked)
        self.graph_widget.edgeClicked.connect(self._on_edge_clicked)
        
    # Old _on_start_mapping method removed - replaced with _on_grid_mode and _on_manual_mode
        
    def _init_element_queue(self):
        """Initialize the queue of elements to map."""
        self.element_queue = []
        self.all_elements = []
        
        # Add all nodes
        for node in self.graph.nodes:
            self.element_queue.append(('node', node))
            self.all_elements.append(('node', node))
            
        # Add all edges
        for edge in self.graph.edges:
            self.element_queue.append(('edge', edge))
            self.all_elements.append(('edge', edge))
            
        # Get first element
        if self.element_queue:
            self._select_next_element()
    
    
    def _reset_mapping_state(self):
        """Reset all mapping state when switching modes to fresh initial state."""
        # Clear all visual elements from map widget (this handles grid, contours, mappings)
        self.map_widget.reset_all()
        
        # Reset core mapping data
        self.mapping = SpatialMapping(self.graph)
        self.mapped_elements.clear()
        self.mapping_history.clear()
        self.element_queue.clear()
        self.current_element = None
        
        # Clear graph highlights
        self.graph_widget.clear_highlights()
        
        # Reset grid-specific state
        if hasattr(self, 'grid_placed'):
            self.grid_placed = False
        if hasattr(self, 'selected_cells'):
            self.selected_cells.clear()
        
        # Reset manual mode specific state
        if hasattr(self, 'contour_list'):
            self.contour_list.clear()
        if hasattr(self, 'highlighted_contour_id'):
            self.highlighted_contour_id = None
        
        # Reset button states
        if hasattr(self, 'undo_grid_button'):
            self.undo_grid_button.setEnabled(False)
        if hasattr(self, 'assign_button'):
            self.assign_button.setEnabled(False)
        if hasattr(self, 'next_element_button'):
            self.next_element_button.setEnabled(False)
        if hasattr(self, 'place_grid_button'):
            self.place_grid_button.setEnabled(True)
        if hasattr(self, 'clear_contour_button'):
            self.clear_contour_button.setEnabled(False)
        if hasattr(self, 'commit_contour_button'):
            self.commit_contour_button.setEnabled(False)
            
        # Update progress display
        self._update_progress()
        
        # Clear status bar
        self.status_bar.clearMessage()
    
    def _create_standardized_contour_mapping(self):
        """Create standardized contour representation for test mode compatibility.
        
        Returns:
            dict: {
                'nodes': {node_id: [list_of_contours]},
                'edges': {edge_tuple_str: [list_of_contours]}
            }
            where each contour is a list of (x,y) points
        """
        from navigraph.core.graph.regions import GridCell, RectangleRegion, ContourRegion
        
        standardized = {
            'nodes': {},
            'edges': {}
        }
        
        if not hasattr(self, 'mapping') or not self.mapping:
            return standardized
            
        # Process node mappings
        for node_id in self.mapping.get_mapped_nodes():
            regions = self.mapping.get_node_regions(node_id)
            contour_list = []
            
            for region in regions:
                if isinstance(region, ContourRegion):
                    # Already a contour - use as-is
                    contour_list.append(region.contour_points)
                elif isinstance(region, (GridCell, RectangleRegion)):
                    # Convert rectangle/grid cell to 4-point contour
                    x, y = region.x, region.y
                    w, h = region.width, region.height
                    rect_contour = [
                        (x, y),        # top-left
                        (x + w, y),    # top-right  
                        (x + w, y + h), # bottom-right
                        (x, y + h)     # bottom-left
                    ]
                    contour_list.append(rect_contour)
                # Add other region types as needed
                    
            if contour_list:
                standardized['nodes'][str(node_id)] = contour_list
        
        # Process edge mappings  
        for edge_tuple in self.mapping.get_mapped_edges():
            regions = self.mapping.get_edge_regions(edge_tuple)
            contour_list = []
            
            for region in regions:
                if isinstance(region, ContourRegion):
                    # Already a contour - use as-is
                    contour_list.append(region.contour_points)
                elif isinstance(region, (GridCell, RectangleRegion)):
                    # Convert rectangle/grid cell to 4-point contour
                    x, y = region.x, region.y
                    w, h = region.width, region.height
                    rect_contour = [
                        (x, y),        # top-left
                        (x + w, y),    # top-right  
                        (x + w, y + h), # bottom-right
                        (x, y + h)     # bottom-left
                    ]
                    contour_list.append(rect_contour)
                # Add other region types as needed
                    
            if contour_list:
                # Convert edge tuple to string for JSON serialization
                edge_str = f"{edge_tuple[0]}_{edge_tuple[1]}"
                standardized['edges'][edge_str] = contour_list
                
        return standardized
    
    def _set_mode_specific_defaults(self, mode: str):
        """Set mode-specific default visualization options."""
        if mode == 'grid':
            # Grid mode: Show mappings and labels, adaptive font enabled
            self.show_mappings_checkbox.setChecked(True)
            self.show_labels_checkbox.setChecked(True) 
            self.adaptive_font_checkbox.setChecked(True)
            # Hide test mode widgets
            for widget in self.test_mode_widgets:
                widget.hide()
                
        elif mode == 'manual':
            # Manual mode: Show mappings and labels, adaptive font enabled
            self.show_mappings_checkbox.setChecked(True)
            self.show_labels_checkbox.setChecked(True)
            self.adaptive_font_checkbox.setChecked(True)
            # Hide test mode widgets
            for widget in self.test_mode_widgets:
                widget.hide()
                
        elif mode == 'test':
            # Test mode: Start clean (unchecked) for testing clarity
            self.show_mappings_checkbox.setChecked(False)
            self.show_labels_checkbox.setChecked(False)
            self.adaptive_font_checkbox.setChecked(False)
            # Show test mode widgets
            for widget in self.test_mode_widgets:
                widget.show()
    
    def _wrap_map_mouse_press(self, original_mouse_press):
        """Wrap the map widget's mouse press event to update button states."""
        def wrapper(event):
            # Call original mouse press handler
            result = original_mouse_press(event)
            
            # Update button states after mouse press in manual mode
            if self.setup_mode == 'manual':
                self._update_manual_mode_buttons()
                
            return result
        return wrapper
    
    def _update_manual_mode_buttons(self):
        """Update manual mode button states based on current contour."""
        if hasattr(self, 'map_widget') and hasattr(self.map_widget, 'current_contour'):
            has_points = len(self.map_widget.current_contour) > 0
            
            # Enable clear/commit buttons if there are points
            if hasattr(self, 'clear_contour_button'):
                self.clear_contour_button.setEnabled(has_points)
            if hasattr(self, 'commit_contour_button'):
                self.commit_contour_button.setEnabled(has_points and len(self.map_widget.current_contour) > 2)
            
    def _select_next_element(self):
        """Select the next element to map."""
        if not self.element_queue:
            self.current_element = None
            self.status_bar.showMessage("All elements mapped!")
            QMessageBox.information(self, "Complete", "All elements have been mapped!")
            return
            
        self.current_element = self.element_queue.pop(0)
        elem_type, elem_id = self.current_element
        
        # Update map widget with current element
        self.map_widget.set_current_element(elem_type, elem_id)
        
        # Update UI based on mode
        if self.setup_mode == 'grid':
            self.assign_button.setEnabled(True)
            self.next_element_button.setEnabled(True)
            
        elif self.setup_mode == 'manual':
            # Manual mode navigation handled via jump functionality
            pass
            
        # Highlight in graph
        self.graph_widget.clear_highlights()
        if elem_type == 'node':
            self.graph_widget.highlight_nodes({elem_id})
        elif elem_type == 'edge':
            self.graph_widget.highlight_edges({elem_id})
            
        # Update combo box selection
        for i, element in enumerate(self.all_elements):
            if element == self.current_element:
                if self.setup_mode == 'grid':
                    self.grid_element_combo.setCurrentIndex(i)
                elif self.setup_mode == 'manual':
                    self.manual_element_combo.setCurrentIndex(i)
                break
                
        self.status_bar.showMessage(f"Mapping {elem_type} {elem_id}")
        
    def _on_apply_grid_config(self):
        """Apply grid configuration changes."""
        config = GridConfig(
            structure_type='rectangle',  # Fixed to rectangle for now
            rows=self.rows_spin.value(),
            cols=self.cols_spin.value(),
            cell_width=self.width_spin.value(),
            cell_height=self.width_spin.value()  # Use same value for square cells
        )
        self.map_widget.set_grid_config(config)
        self.grid_status_label.setText("")  # Clear status until grid is placed
        self.status_bar.showMessage("Grid configuration updated - click 'Place Grid' to position it")
        
        # Enable Clear Grid button if grid is already placed
        if hasattr(self.map_widget, 'grid_enabled') and self.map_widget.grid_enabled:
            self.clear_grid_button.setEnabled(True)
        
    def _on_place_grid(self):
        """Start or cancel grid placement mode."""
        if self.place_grid_button.isChecked():
            # Button was pressed - start placement mode
            self.map_widget.set_interaction_mode('place_grid')
            self.status_bar.showMessage("Click on the map to place grid origin")
        else:
            # Button was unpressed - cancel placement mode
            self.map_widget.set_interaction_mode('none')
            self.status_bar.showMessage("Grid placement cancelled")
        
    def _on_grid_placed(self, x: float, y: float):
        """Handle grid placement."""
        self.grid_status_label.setText(f"Origin: ({x:.0f}, {y:.0f})")
        self.status_bar.showMessage("Grid placed - Click cells to select")
        # Uncheck the Place Grid button after placement
        self.place_grid_button.setChecked(False)
        # Enable Clear Grid button
        self.clear_grid_button.setEnabled(True)
        
    def _on_clear_grid(self):
        """Clear the grid and reset all mappings."""
        from PyQt5.QtWidgets import QMessageBox
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self, "Clear Grid", 
            "This will remove the grid and clear all mappings. Are you sure?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reset the map widget
            self.map_widget.reset_all()
            
            # Reset mapping state
            self.mapping = SpatialMapping(self.graph)
            self.mapped_elements.clear()
            self.mapping_history.clear()
            
            # Reset element queue to start fresh
            self._init_element_queue()
            self._populate_element_combos()
            
            # Clear graph highlights
            self.graph_widget.clear_highlights()
            
            # Update UI
            self.grid_status_label.setText("")
            self.clear_grid_button.setEnabled(False)
            # Keep clear_all_grid_button always enabled
            self.assign_button.setEnabled(False)
            self.next_element_button.setEnabled(False)
            self.undo_grid_button.setEnabled(False)
            
            # Update progress
            self._update_progress_display()
            
            self.status_bar.showMessage("Grid cleared - all mappings reset")
        
    def _on_cell_clicked(self, cell_id: str):
        """Handle cell click."""
        if cell_id in self.map_widget.selected_cells:
            self.status_bar.showMessage(f"Selected cell {cell_id}")
        else:
            self.status_bar.showMessage(f"Deselected cell {cell_id}")
            
    def _on_assign_cells(self):
        """Assign selected cells to current element."""
        if not self.current_element or not self.map_widget.selected_cells:
            return
            
        elem_type, elem_id = self.current_element
        
        # Track for undo
        assigned_cells = list(self.map_widget.selected_cells)
        regions_added = []
        
        # Create regions for selected cells
        for cell_id in self.map_widget.selected_cells:
            if cell_id in self.map_widget.grid_cells:
                rect = self.map_widget.grid_cells[cell_id]
                region = RectangleRegion(
                    region_id=f"{elem_type}_{elem_id}_{cell_id}",
                    x=rect.x(),
                    y=rect.y(),
                    width=rect.width(),
                    height=rect.height()
                )
                
                if elem_type == 'node':
                    self.mapping.add_node_region(region, elem_id)
                    color = QColor(150, 255, 150, 100)  # Light green for nodes
                elif elem_type == 'edge':
                    self.mapping.add_edge_region(region, elem_id)
                    color = QColor(255, 200, 120, 100)  # Light orange for edges
                    
                # Update map widget with mapping
                self.map_widget.add_cell_mapping(cell_id, elem_type, elem_id, color)
                regions_added.append(region)
                    
        # Update mapped elements tracking
        self.mapped_elements[self.current_element] = assigned_cells
        
        # Add to history for undo
        self.mapping_history.append({
            'action': 'assign_cells',
            'element': self.current_element,
            'cells': assigned_cells,
            'regions': regions_added
        })
        self.undo_grid_button.setEnabled(True)
        
        # Clear selection and move to next
        self.map_widget.clear_selection()
        self._on_next_element()
        self._update_progress()
        
    def _on_next_element(self):
        """Skip to next element."""
        self._select_next_element()
        self._update_progress_display()
        
    def _on_start_drawing(self, checked: bool):
        """Toggle contour drawing mode."""
        if checked:
            self.map_widget.set_interaction_mode('draw_contour')
            self.status_bar.showMessage("Drawing mode - Left click to add points")
            # Enable contour control buttons
            self.remove_point_button.setEnabled(True)
            self.finish_contour_button.setEnabled(True)
            self.cancel_contour_button.setEnabled(True)
        else:
            self.map_widget.set_interaction_mode('none')
            self.status_bar.showMessage("Drawing mode disabled")
            # Disable contour control buttons
            self.remove_point_button.setEnabled(False)
            self.finish_contour_button.setEnabled(False)
            self.cancel_contour_button.setEnabled(False)
    
    def _on_remove_last_point(self):
        """Remove last point from current contour."""
        self.map_widget.remove_last_contour_point()
        
    def _on_finish_contour(self):
        """Finish the current contour."""
        self.map_widget.finish_current_contour()
        
    def _on_cancel_contour(self):
        """Cancel current contour drawing."""
        self.map_widget.cancel_current_contour()
        
    def _on_contour_selected(self, item):
        """Handle contour selection from list with toggle support."""
        region_id = item.data(Qt.UserRole)
        if region_id:
            # Toggle behavior: if clicking the same contour, deselect it
            if self.map_widget.highlighted_contour_id == region_id:
                # Deselect: unhighlight and clear list selection
                self.map_widget.unhighlight_current_contour()
                self.contour_list.clearSelection()
                self.delete_contour_button.setEnabled(False)
            else:
                # Select new contour: highlight it
                self.map_widget.highlight_contour(region_id)
                self.delete_contour_button.setEnabled(True)
        
    def _on_delete_contour(self):
        """Delete selected contour."""
        current_item = self.contour_list.currentItem()
        if current_item:
            region_id = current_item.data(Qt.UserRole)
            
            # Clean up highlighting if this contour is currently highlighted
            if self.map_widget.highlighted_contour_id == region_id:
                self.map_widget.unhighlight_current_contour()
            
            # Remove contour from map
            self.map_widget.remove_contour(region_id)
            
            # Remove from mapping
            # TODO: Remove from actual mapping object
            
            # Remove from list
            self.contour_list.takeItem(self.contour_list.row(current_item))
            self.delete_contour_button.setEnabled(False)
            
            self._update_progress_display()
        
    def _on_contour_drawn(self, points: List[Tuple[float, float]]):
        """Handle completed contour."""
        if self.current_element:
            elem_type, elem_id = self.current_element
            region_id = f"{elem_type}_{elem_id}_contour_{len(self.map_widget.completed_contours)}"
            
            # Add to map display
            color = QColor(150, 255, 150, 100) if elem_type == 'node' else QColor(255, 200, 120, 100)
            self.map_widget.add_contour(points, region_id, elem_type, elem_id, color)
            
            # Add to contour list
            list_label = f"{elem_type.title()} {elem_id}: {region_id}"
            item = QListWidgetItem(list_label)
            item.setData(Qt.UserRole, region_id)  # Store region_id for reference
            self.contour_list.addItem(item)
            
            # Create region and add to mapping
            region = ContourRegion(region_id=region_id, contour_points=points)
            
            if elem_type == 'node':
                self.mapping.add_node_region(region, elem_id)
            elif elem_type == 'edge':
                self.mapping.add_edge_region(region, elem_id)
                
            # Add to history for undo
            self.mapping_history.append({
                'action': 'add_contour',
                'element': self.current_element,
                'region': region,
                'region_id': region_id
            })
            # Manual mode undo functionality removed
            
            # Reset drawing state but keep drawing mode active for next element
            # Drawing mode stays active for continuous element mapping
                
            self.status_bar.showMessage(f"Contour assigned to {elem_type} {elem_id}")
            
            # Update progress and automatically move to next element
            self._update_progress_display()
            self._on_next_element()  # Auto-advance to next element
            
    def _on_assign_contour(self):
        """Assign drawn contour to current element."""
        # This is handled in _on_contour_drawn
        pass
        
    def _on_node_clicked(self, node_id):
        """Handle node click in graph widget."""
        self.status_bar.showMessage(f"Clicked node {node_id}")
        
    def _on_edge_clicked(self, edge):
        """Handle edge click in graph widget."""
        self.status_bar.showMessage(f"Clicked edge {edge}")
        
    def _update_progress(self):
        """Update progress indicators (backward compatibility)."""
        self._update_progress_display()
        
    def _update_unmapped_list(self):
        """Update the list of unmapped elements (no longer needed in simplified UI)."""
        # This method is kept for backward compatibility but no longer updates a list
        pass
            
    def _on_save_mapping(self):
        """Save the current mapping to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mapping", "", "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if file_path:
            if MappingStorage.save_mapping(self.mapping, file_path):
                QMessageBox.information(self, "Success", f"Mapping saved to {file_path}")
                self.status_bar.showMessage(f"Mapping saved to {file_path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save mapping")
                
    def _on_load_mapping(self):
        """Load a mapping from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Mapping", "", "Pickle Files (*.pkl);;All Files (*)"
        )
        
        if file_path:
            loaded_mapping = MappingStorage.load_mapping(file_path)
            if loaded_mapping:
                self.mapping = loaded_mapping
                self._visualize_loaded_mapping()
                QMessageBox.information(self, "Success", f"Mapping loaded from {file_path}")
                self.status_bar.showMessage(f"Mapping loaded from {file_path}")
                self._update_progress()
            else:
                QMessageBox.critical(self, "Error", "Failed to load mapping")
                
    def _visualize_loaded_mapping(self):
        """Visualize a loaded mapping on the map."""
        # Clear existing visualizations
        self.map_widget.clear_selection()
        self.map_widget.clear_contours()
        
        # Visualize node regions
        for node_id, regions in getattr(self.mapping, '_node_to_regions', {}).items():
            color = QColor(150, 255, 150, 100)
            for region in regions:
                if isinstance(region, RectangleRegion):
                    # For grid cells
                    # TODO: Properly map back to grid cells
                    pass
                elif isinstance(region, ContourRegion):
                    # For contours
                    self.map_widget.add_contour(region.points, region.region_id, 'node', node_id, color)
                    
        # Visualize edge regions
        for edge, regions in getattr(self.mapping, '_edge_to_regions', {}).items():
            color = QColor(255, 200, 120, 100)
            for region in regions:
                if isinstance(region, ContourRegion):
                    self.map_widget.add_contour(region.points, region.region_id, 'edge', edge, color)
    
    def _on_save_intermediate_mapping(self):
        """Save current mapping with complete state to file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Intermediate Mapping", "", "Mapping Files (*.pkl);;All Files (*)")
        
        if file_path:
            try:
                # Create standardized contour representation for test mode compatibility
                standardized_contours = self._create_standardized_contour_mapping()
                
                # Create complete state dict
                mapping_state = {
                    'format_version': '2.1',  # Increment version for new standardized format
                    'mapping': self.mapping,  # Original mapping (preserved for mode compatibility)
                    'standardized_contours': standardized_contours,  # Universal test mode format
                    'grid_config': {
                        'structure_type': self.map_widget.grid_config.structure_type,
                        'rows': self.map_widget.grid_config.rows,
                        'cols': self.map_widget.grid_config.cols,
                        'cell_width': self.map_widget.grid_config.cell_width,
                        'cell_height': self.map_widget.grid_config.cell_height,
                        'origin_x': self.map_widget.grid_config.origin_x,
                        'origin_y': self.map_widget.grid_config.origin_y,
                    },
                    'grid_enabled': self.map_widget.grid_enabled,
                    'current_mode': getattr(self, 'setup_mode', 'grid'),
                    'element_queue_index': getattr(self, 'element_queue_index', 0),
                    'mapping_history': len(self.mapping_history) if hasattr(self, 'mapping_history') else 0
                }
                
                # Save to pickle file
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(mapping_state, f)
                
                self.status_bar.showMessage(f"Intermediate mapping saved to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save mapping: {str(e)}")
    
    def _on_load_intermediate_mapping(self):
        """Load mapping with complete state from file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Load Intermediate Mapping", "", "Mapping Files (*.pkl);;All Files (*)")
        
        if file_path:
            try:
                import pickle
                with open(file_path, 'rb') as f:
                    mapping_state = pickle.load(f)
                
                # Validate format
                if not isinstance(mapping_state, dict) or 'mapping' not in mapping_state:
                    QMessageBox.warning(self, "Error", "Invalid mapping file format")
                    return
                
                # Handle format versions and backward compatibility
                format_version = mapping_state.get('format_version', '1.0')
                
                # Load the mapping
                self.mapping = mapping_state['mapping']
                
                # For test mode, always ensure standardized contours are available
                if hasattr(self, 'setup_mode') and self.setup_mode == 'test':
                    if 'standardized_contours' in mapping_state:
                        # Use pre-computed standardized contours (v2.1+)
                        self._test_mode_standardized_contours = mapping_state['standardized_contours']
                    else:
                        # Create standardized contours from loaded mapping (backward compatibility)
                        self._test_mode_standardized_contours = self._create_standardized_contour_mapping()
                
                # Restore grid configuration if present
                if 'grid_config' in mapping_state:
                    grid_config = mapping_state['grid_config']
                    self.map_widget.grid_config.structure_type = grid_config.get('structure_type', 'rectangle')
                    self.map_widget.grid_config.rows = grid_config.get('rows', 8)
                    self.map_widget.grid_config.cols = grid_config.get('cols', 8)
                    self.map_widget.grid_config.cell_width = grid_config.get('cell_width', 50.0)
                    self.map_widget.grid_config.cell_height = grid_config.get('cell_height', 50.0)
                    self.map_widget.grid_config.origin_x = grid_config.get('origin_x', 0.0)
                    self.map_widget.grid_config.origin_y = grid_config.get('origin_y', 0.0)
                    
                    # Update UI controls with loaded grid config
                    self._update_grid_ui_from_config()
                
                # Restore grid if it was enabled
                if mapping_state.get('grid_enabled', False):
                    self.map_widget.enable_grid(
                        self.map_widget.grid_config.origin_x,
                        self.map_widget.grid_config.origin_y
                    )
                    # Update grid UI elements
                    self._update_grid_status_after_load()
                    # Set interaction mode to allow cell selection
                    self.map_widget.set_interaction_mode('select_cells')
                
                # Validate mode compatibility
                loaded_mode = mapping_state.get('current_mode', 'grid')
                
                # Check if mapping contains incompatible region types
                has_grid_regions = False
                has_contour_regions = False
                for region in self.mapping._regions.values():
                    from .regions import RectangleRegion, ContourRegion
                    if isinstance(region, RectangleRegion):
                        has_grid_regions = True
                    elif isinstance(region, ContourRegion):
                        has_contour_regions = True
                
                if has_grid_regions and loaded_mode != 'grid':
                    QMessageBox.warning(self, "Mode Mismatch", 
                                      "This mapping was created in Grid mode and cannot be loaded in Manual mode.")
                    return
                elif has_contour_regions and loaded_mode != 'manual':
                    QMessageBox.warning(self, "Mode Mismatch", 
                                      "This mapping was created in Manual mode and cannot be loaded in Grid mode.")
                    return
                
                # Switch to appropriate mode without resetting mapping (unless in test mode)
                current_mode = None
                if self.grid_mode_button.isChecked():
                    current_mode = 'grid'
                elif self.manual_mode_button.isChecked():
                    current_mode = 'manual'  
                elif self.test_mode_button.isChecked():
                    current_mode = 'test'
                
                if current_mode != 'test':  # Only switch modes if not in test mode
                    self._loading_intermediate = True  # Flag to prevent reset during mode switch
                    if loaded_mode == 'grid':
                        self._on_grid_mode(True)
                        # Ensure interaction mode is correct for grid
                        if self.map_widget.grid_enabled:
                            self.map_widget.set_interaction_mode('select_cells')
                    else:
                        self._on_manual_mode(True)
                        # Ensure interaction mode is correct for manual drawing
                        self.map_widget.set_interaction_mode('draw_contour')
                    self._loading_intermediate = False
                else:
                    # In test mode, just load the mapping without switching modes
                    # Don't auto-visualize - let user control via Display Options
                    current_mode = 'test'
                
                # Visualize loaded mapping only if not in test mode
                if current_mode != 'test':
                    self._visualize_loaded_intermediate_mapping()
                # In test mode, don't auto-display anything - let user control via Show All Mappings toggle
                
                # Update progress
                self._update_progress_display()
                
                # Restore element queue position if available
                if 'element_queue_index' in mapping_state:
                    self.element_queue_index = mapping_state['element_queue_index']
                    
                # Always repopulate element combos to ensure proper state
                self._populate_element_combos()
                
                # Set current element to continue mapping from where we left off
                if hasattr(self, 'element_queue') and self.element_queue and len(self.element_queue) > 0:
                    self.current_element = self.element_queue[0]
                    elem_type, elem_id = self.current_element
                    
                    # Update map widget with current element
                    self.map_widget.set_current_element(elem_type, elem_id)
                    
                    # Update combo box selection based on mode
                    if loaded_mode == 'grid' and hasattr(self, 'grid_element_combo'):
                        # Find the element in the grid combo box
                        for i in range(self.grid_element_combo.count()):
                            combo_text = self.grid_element_combo.itemText(i)
                            if f"{elem_type.title()}: {elem_id}" in combo_text:
                                self.grid_element_combo.setCurrentIndex(i)
                                break
                    elif loaded_mode == 'manual' and hasattr(self, 'manual_element_combo'):
                        # Find the element in the manual combo box
                        for i in range(self.manual_element_combo.count()):
                            combo_text = self.manual_element_combo.itemText(i)
                            if f"{elem_type.title()}: {elem_id}" in combo_text:
                                self.manual_element_combo.setCurrentIndex(i)
                                break
                
                
                self.status_bar.showMessage(f"Intermediate mapping loaded from {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load mapping: {str(e)}")
    
    def _update_grid_ui_from_config(self):
        """Update UI controls to reflect loaded grid configuration."""
        if hasattr(self, 'rows_spinbox'):
            self.rows_spinbox.setValue(self.map_widget.grid_config.rows)
        if hasattr(self, 'cols_spinbox'):
            self.cols_spinbox.setValue(self.map_widget.grid_config.cols)
        if hasattr(self, 'cell_size_spinbox'):
            self.cell_size_spinbox.setValue(self.map_widget.grid_config.cell_width)
    
    def _visualize_loaded_intermediate_mapping(self):
        """Visualize loaded intermediate mapping on the map."""
        # Determine which map widget to use
        current_mode = None
        if self.test_mode_button.isChecked():
            current_mode = 'test'
            map_widget = self.test_map_widget
        else:
            map_widget = self.map_widget
            
        # Clear only selections, not mappings
        map_widget.selected_cells.clear()
        map_widget.completed_contours.clear()
        
        # Clear contour list if it exists (manual mode)
        if hasattr(self, 'contour_list') and self.contour_list is not None:
            self.contour_list.clear()
        
        # Restore all mapped regions
        for region_id, region in self.mapping._regions.items():
            element_info = self.mapping._region_to_element.get(region_id)
            if element_info:
                element_type, element_id = element_info
                
                # Determine color based on element type
                if element_type == 'node':
                    color = QColor(0, 255, 0, 100)  # Green for nodes
                else:
                    color = QColor(255, 165, 0, 100)  # Orange for edges
                
                # Handle different region types
                from .regions import RectangleRegion, ContourRegion
                
                if isinstance(region, RectangleRegion):  # Grid-based region
                    # Find which cell this rectangle corresponds to
                    for cell_id, cell_rect in map_widget.grid_cells.items():
                        # Check if this region matches this cell
                        if (abs(region.x - cell_rect.x()) < 1 and 
                            abs(region.y - cell_rect.y()) < 1 and
                            abs(region.width - cell_rect.width()) < 1 and
                            abs(region.height - cell_rect.height()) < 1):
                            map_widget.add_cell_mapping(cell_id, element_type, element_id, color)
                            break
                
                elif isinstance(region, ContourRegion):  # Contour-based region
                    # Restore contour mappings
                    contour_color = QColor(color.red(), color.green(), color.blue(), 150)
                    map_widget.completed_contours.append(
                        (region.contour, region_id, contour_color, element_type, element_id)
                    )
                    map_widget.contour_mappings[region_id] = (element_type, element_id)
                    
                    # Add to contour list if we're in manual mode
                    if hasattr(self, 'contour_list') and self.contour_list is not None:
                        list_label = f"{element_type.title()} {element_id}: {region_id}"
                        item = QListWidgetItem(list_label)
                        item.setData(Qt.UserRole, region_id)  # Store region_id for reference
                        self.contour_list.addItem(item)
        
        # Force widget update
        map_widget.update()
        
    def _update_grid_status_after_load(self):
        """Update grid-related UI elements after loading."""
        if self.map_widget.grid_enabled:
            # Update grid status
            origin_text = f"({self.map_widget.grid_config.origin_x:.1f}, {self.map_widget.grid_config.origin_y:.1f})"
            if hasattr(self, 'grid_status_label'):
                self.grid_status_label.setText(f"Origin: {origin_text}")
            
            # Enable relevant buttons
            if hasattr(self, 'clear_grid_button'):
                self.clear_grid_button.setEnabled(True)
            if hasattr(self, 'place_grid_button'):
                self.place_grid_button.setEnabled(False)  # Grid already placed

    def _on_jump_to_element(self):
        """Jump to selected element from combo box."""
        # Determine which combo box to use based on current mode
        if self.setup_mode == 'grid':
            index = self.grid_element_combo.currentIndex()
        elif self.setup_mode == 'manual':
            index = self.manual_element_combo.currentIndex()
        else:
            return
            
        if index >= 0 and index < len(self.all_elements):
            selected_element = self.all_elements[index]
            
            # Remove from queue if present
            if selected_element in self.element_queue:
                self.element_queue.remove(selected_element)
                
            # Set as current element
            self.current_element = selected_element
            elem_type, elem_id = self.current_element
            
            # Update map widget
            self.map_widget.set_current_element(elem_type, elem_id)
            
            # Update UI based on mode
            if self.setup_mode == 'grid':
                self.assign_button.setEnabled(True)
                self.next_element_button.setEnabled(True)
            elif self.setup_mode == 'manual':
                # Manual mode navigation handled via jump functionality
                pass
                
            # Highlight in graph
            self.graph_widget.clear_highlights()
            if elem_type == 'node':
                self.graph_widget.highlight_nodes({elem_id})
            elif elem_type == 'edge':
                self.graph_widget.highlight_edges({elem_id})
                
            self.status_bar.showMessage(f"Jumped to {elem_type} {elem_id}")
    
    def _on_toggle_mappings(self, state):
        """Toggle visibility of all mappings."""
        try:
            show_mappings = (state == Qt.Checked)
            
            # In test mode, handle Show All Mappings specially
            if hasattr(self, 'setup_mode') and self.setup_mode == 'test' and hasattr(self, 'test_map_widget'):
                if show_mappings:
                    # Clear test selections first
                    self._clear_test_selections()
                    
                    # Display all saved mappings using standardized contours
                    if hasattr(self, '_test_mode_standardized_contours'):
                        self._display_all_standardized_mappings()
                else:
                    # Clear all displayed mappings
                    if hasattr(self.test_map_widget, 'completed_contours'):
                        self.test_map_widget.completed_contours.clear()
                    if hasattr(self.test_map_widget, 'grid_cells'):
                        self.test_map_widget.grid_cells.clear()
                    self.test_map_widget.update()
            else:
                # Normal mode - just toggle visibility flag
                self.map_widget.show_all_mappings = show_mappings
                self.map_widget.update()
                
        except Exception as e:
            print(f"Error toggling mappings: {e}")
            # Don't crash the app, just log the error
        
    def _on_toggle_labels(self, state):
        """Toggle visibility of cell labels."""
        try:
            show_labels = (state == Qt.Checked)
            self.map_widget.show_cell_labels = show_labels
            self.map_widget.update()
            
            # Also update test mode widget if it exists
            if hasattr(self, 'test_map_widget'):
                self.test_map_widget.show_cell_labels = show_labels
                self.test_map_widget.update()
        except Exception as e:
            print(f"Error toggling labels: {e}")
        
    def _on_toggle_adaptive_font(self, state):
        """Toggle adaptive font sizing."""
        try:
            adaptive_font = (state == Qt.Checked)
            self.map_widget.adaptive_font_size = adaptive_font
            self.map_widget.update()
            
            # Also update test mode widget if it exists  
            if hasattr(self, 'test_map_widget'):
                self.test_map_widget.adaptive_font_size = adaptive_font
                self.test_map_widget.update()
        except Exception as e:
            print(f"Error toggling adaptive font: {e}")
    
    def _on_test_layout_changed(self):
        """Handle test mode layout orientation change."""
        if self.setup_mode == 'test':
            # Re-setup the test layout with new orientation
            self._setup_test_layout()
    
    def _on_clear_all(self):
        """Clear all mappings and start fresh."""
        reply = QMessageBox.question(self, "Clear All Mappings", 
                                    "Are you sure you want to clear all mappings? This will reset all progress and start fresh.",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._reset_mapping_state()
            self._init_element_queue()
            self._populate_element_combos()
            self._update_progress_display()
            self.status_bar.showMessage("All mappings cleared")
    
    def _on_undo_last(self):
        """Undo the last mapping action."""
        try:
            if not self.mapping_history:
                self.status_bar.showMessage("No actions to undo")
                return
                
            last_action = self.mapping_history.pop()
            
            if last_action['action'] == 'assign_cells':
                # Remove cell mappings
                elem_type, elem_id = last_action['element']
                for cell_id in last_action.get('cells', []):
                    if hasattr(self.map_widget, 'cell_mappings') and cell_id in self.map_widget.cell_mappings:
                        del self.map_widget.cell_mappings[cell_id]
                    if hasattr(self.map_widget, 'cell_colors') and cell_id in self.map_widget.cell_colors:
                        del self.map_widget.cell_colors[cell_id]
                        
                # Remove from mapping safely
                if elem_type == 'node':
                    node_to_regions = getattr(self.mapping, '_node_to_regions', {})
                    if elem_id in node_to_regions:
                        for region in last_action.get('regions', []):
                            if region in node_to_regions[elem_id]:
                                node_to_regions[elem_id].remove(region)
                elif elem_type == 'edge':
                    edge_to_regions = getattr(self.mapping, '_edge_to_regions', {})
                    if elem_id in edge_to_regions:
                        for region in last_action.get('regions', []):
                            if region in edge_to_regions[elem_id]:
                                edge_to_regions[elem_id].remove(region)
                                
                # Add element back to queue safely
                if hasattr(self, 'element_queue') and self.element_queue is not None:
                    self.element_queue.insert(0, last_action['element'])
                    
            elif last_action['action'] == 'add_contour':
                # Remove contour from display
                region_id = last_action.get('region_id')
                if region_id and hasattr(self.map_widget, 'completed_contours'):
                    self.map_widget.completed_contours = [
                        c for c in self.map_widget.completed_contours 
                        if len(c) > 1 and c[1] != region_id
                    ]
                if region_id and hasattr(self.map_widget, 'contour_mappings') and region_id in self.map_widget.contour_mappings:
                    del self.map_widget.contour_mappings[region_id]
            
            # Update displays
            self._update_progress_display()
            if hasattr(self.map_widget, 'update'):
                self.map_widget.update()
            self.status_bar.showMessage(f"Undid last {last_action.get('action', 'action')}")
            
        except Exception as e:
            self.status_bar.showMessage(f"Undo failed: {str(e)}")
            # Log the error but don't crash
            print(f"Undo error: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_clear_contour(self):
        """Clear the current contour being drawn."""
        self.map_widget.cancel_current_contour()
        
    def _on_commit_contour(self):
        """Commit the current contour."""
        self.map_widget.finish_current_contour()
        
    def _on_clear_all_contours(self):
        """Clear all contours."""
        reply = QMessageBox.question(self, "Clear All Contours", 
                                    "Are you sure you want to delete all contours?",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Clean up any highlighting state
            self.map_widget.unhighlight_current_contour()
            
            # Clear from map
            self.map_widget.clear_contours()
            # Clear from list
            self.contour_list.clear()
            # Clear from mapping
            # TODO: Remove from actual mapping object
            self._update_progress_display()
    
    def _on_test_mapping(self):
        """Launch test mode for current mapping."""
        try:
            # Save current mapping to temp file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
            temp_path = temp_file.name
            temp_file.close()
            
            from .storage import MappingStorage
            success = MappingStorage.save_mapping(self.mapping, temp_path)
            
            if success:
                # TODO: Launch test GUI with temp mapping
                QMessageBox.information(self, "Test Mode", 
                                      f"Test mode coming soon!\nMapping temporarily saved to: {temp_path}")
                self.status_bar.showMessage("Test mode launched")
            else:
                QMessageBox.critical(self, "Error", "Failed to save mapping for testing")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch test mode: {e}")
                    
    def _setup_test_layout(self):
        """Setup test mode layout based on orientation preference."""
        if not hasattr(self, 'original_layout'):
            # Store the original layout for restoration
            self.original_control_panel = self.control_panel
            self.original_views_widget = self.centralWidget().layout().itemAt(1).widget()
            
        # Create test layout widgets if they don't exist
        if not hasattr(self, 'test_map_widget'):
            self.test_map_widget = MapWidget(self.map_image)
            self.test_map_widget.setMinimumSize(400, 300)
        
        if not hasattr(self, 'test_graph_widget'):
            self.test_graph_widget = GraphWidget(self.graph)
            self.test_graph_widget.setMinimumSize(400, 300)
        
        # Clear the main layout
        main_layout = self.centralWidget().layout()
        while main_layout.count():
            child = main_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
        
        # Determine orientation
        is_vertical = self.test_layout_vertical.isChecked()
        
        # Create the test splitter
        if not hasattr(self, 'test_splitter'):
            self.test_splitter = QSplitter()
        
        # Set orientation
        if is_vertical:
            self.test_splitter.setOrientation(Qt.Vertical)
            # Graph on top, map on bottom
            self.test_splitter.addWidget(self.test_graph_widget)
            self.test_splitter.addWidget(self.test_map_widget)
            self.test_splitter.setSizes([1, 1])  # Equal sizes
        else:
            self.test_splitter.setOrientation(Qt.Horizontal)
            # Map on left, graph on right
            self.test_splitter.addWidget(self.test_map_widget)
            self.test_splitter.addWidget(self.test_graph_widget)
            self.test_splitter.setSizes([1, 1])  # Equal sizes
        
        # Add control panel (minimal) and test splitter to main layout
        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.test_splitter)
        
        # Connect test mode events
        # Store original mousePressEvent for restoration
        self.original_mouse_press = self.test_map_widget.mousePressEvent
        self.test_map_widget.mousePressEvent = self._on_test_map_click
        # Note: Graph click events removed - using dropdown selection instead
        
    def _restore_normal_layout(self):
        """Restore the normal layout when leaving test mode."""
        if hasattr(self, 'original_control_panel') and hasattr(self, 'original_views_widget'):
            # Clear current layout more carefully - only remove what we added
            main_layout = self.centralWidget().layout()
            
            # Store widgets we want to preserve
            widgets_to_preserve = [self.original_control_panel, self.original_views_widget]
            
            # Remove only test mode widgets, not the originals
            widgets_to_remove = []
            for i in range(main_layout.count()):
                item = main_layout.itemAt(i)
                if item and item.widget():
                    widget = item.widget()
                    # Only remove if it's not one of our preserved widgets
                    if widget not in widgets_to_preserve:
                        widgets_to_remove.append(widget)
            
            # Remove the test mode widgets
            for widget in widgets_to_remove:
                main_layout.removeWidget(widget)
                # Don't delete - just remove from layout
            
            # Make sure our original widgets are in the layout
            if main_layout.indexOf(self.original_control_panel) == -1:
                main_layout.addWidget(self.original_control_panel)
            if main_layout.indexOf(self.original_views_widget) == -1:
                main_layout.addWidget(self.original_views_widget)
            
            # Restore original map widget reference to self.map_widget
            if hasattr(self, 'original_views_widget'):
                # The map widget should be the second child (index 1) of the splitter
                views_splitter = self.original_views_widget
                if hasattr(views_splitter, 'widget') and views_splitter.widget(1):
                    self.map_widget = views_splitter.widget(1)
                else:
                    # Try to find the map widget by name/type
                    for child in views_splitter.findChildren(MapWidget):
                        self.map_widget = child
                        break
            
            # Restore original mousePressEvent if it was stored
            if hasattr(self, 'test_map_widget') and hasattr(self, 'original_mouse_press'):
                self.test_map_widget.mousePressEvent = self.original_mouse_press
            
            # Clean up test mode references
            if hasattr(self, 'test_splitter'):
                del self.test_splitter
            if hasattr(self, 'test_map_widget'):
                del self.test_map_widget
            if hasattr(self, 'test_graph_widget'):
                del self.test_graph_widget
            
    def _on_test_map_click(self, event):
        """Handle map click in test mode."""
        # Handle non-left clicks with original functionality (panning, etc.)
        if event.button() != Qt.LeftButton:
            if hasattr(self, 'original_mouse_press'):
                self.original_mouse_press(event)
            return
        
        # Get click position in image coordinates
        x = (event.x() - self.test_map_widget.offset_x) / self.test_map_widget.scale_factor
        y = (event.y() - self.test_map_widget.offset_y) / self.test_map_widget.scale_factor
        
        # Clear previous selection
        self._clear_test_selections()
        
        # Store click position for visual indicator
        self.test_selected_point = (x, y)
        
        # Store click position in test map widget for drawing
        if hasattr(self, 'test_map_widget'):
            self.test_map_widget.test_click_position = (x, y)
            self.test_map_widget.update()
        
        # Find mapped element at this position
        element_found = self._find_element_at_position(x, y)
        
        if element_found:
            elem_type, elem_id = element_found
            self.test_selected_element = (elem_type, elem_id)
            
            # Highlight graph element with highlight color (yellow)
            self.test_graph_widget.clear_highlights()
            if elem_type == 'node':
                self.test_graph_widget.highlight_nodes({elem_id}, color='highlight')
                self.test_info_label.setText(f"Found: Node {elem_id}")
            else:
                self.test_graph_widget.highlight_edges({elem_id}, color='highlight')
                self.test_info_label.setText(f"Found: Edge {elem_id}")
        else:
            self.test_info_label.setText(f"No mapping at ({x:.0f}, {y:.0f})")
            
    def _on_test_node_click(self, node_id):
        """Handle node click in test mode."""
        # Clear previous selection
        self._clear_test_selections()
        
        self.test_selected_element = ('node', node_id)
        
        # Highlight node with selection color
        self.test_graph_widget.clear_highlights()
        self.test_graph_widget.highlight_nodes({node_id}, color='selected')
        
        # Highlight mapped regions on map
        self._highlight_element_regions('node', node_id)
        
        self.test_info_label.setText(f"Selected: Node {node_id}")
        
    def _on_test_edge_click(self, edge):
        """Handle edge click in test mode."""
        # Clear previous selection
        self._clear_test_selections()
        
        self.test_selected_element = ('edge', edge)
        
        # Highlight edge with selection color
        self.test_graph_widget.clear_highlights()
        self.test_graph_widget.highlight_edges({edge}, color='selected')
        
        # Highlight mapped regions on map
        self._highlight_element_regions('edge', edge)
        
        self.test_info_label.setText(f"Selected: Edge {edge}")
        
    def _on_clear_test_markings(self):
        """Clear all test mode markings."""
        self._clear_test_selections()
        self.test_info_label.setText("No selection")
        
    def _on_test_select_element(self):
        """Handle element selection from dropdown in test mode."""
        if not hasattr(self, 'test_element_combo'):
            return
            
        selected_text = self.test_element_combo.currentText()
        if not selected_text or selected_text == "-- Select Element --":
            return
            
        # Parse the selected text to get element type and id
        # Format: "Node: 1" or "Edge: (1, 2)"
        if selected_text.startswith("Node: "):
            elem_type = 'node'
            elem_id = int(selected_text.replace("Node: ", ""))
            self._on_test_node_click(elem_id)
        elif selected_text.startswith("Edge: "):
            elem_type = 'edge'
            edge_str = selected_text.replace("Edge: ", "")
            # Parse tuple format "(1, 2)" 
            edge_str = edge_str.strip("()")
            parts = edge_str.split(", ")
            elem_id = (int(parts[0]), int(parts[1]))
            self._on_test_edge_click(elem_id)
    
    def _populate_test_element_combo(self):
        """Populate the test mode element dropdown with all graph elements."""
        if not hasattr(self, 'test_element_combo') or not hasattr(self, 'graph'):
            return
            
        self.test_element_combo.clear()
        self.test_element_combo.addItem("-- Select Element --")
        
        # Add all nodes
        for node in sorted(self.graph.nodes):
            self.test_element_combo.addItem(f"Node: {node}")
            
        # Add all edges  
        for edge in sorted(self.graph.edges):
            self.test_element_combo.addItem(f"Edge: {edge}")
        
    def _clear_test_selections(self):
        """Clear current test selections and highlights."""
        try:
            if hasattr(self, 'test_graph_widget'):
                self.test_graph_widget.clear_highlights()
            if hasattr(self, 'test_map_widget'):
                if hasattr(self.test_map_widget, 'clear_highlights'):
                    self.test_map_widget.clear_highlights()
                # Clear test highlight contours
                if hasattr(self.test_map_widget, 'test_highlight_contours'):
                    self.test_map_widget.test_highlight_contours = []
                # Clear click position indicator
                if hasattr(self.test_map_widget, 'test_click_position'):
                    self.test_map_widget.test_click_position = None
                self.test_map_widget.update()
            self.test_selected_point = None
            self.test_selected_element = None
        except Exception as e:
            print(f"Error clearing test selections: {e}")
            # Don't crash, reset state anyway
            self.test_selected_point = None
            self.test_selected_element = None
        
    def _find_element_at_position(self, x, y):
        """Find which graph element is mapped at the given position."""
        # Use standardized contours if available (more reliable for test mode)
        if hasattr(self, '_test_mode_standardized_contours'):
            return self._find_element_using_standardized_contours(x, y)
        
        # Fallback to original mapping method
        # Check node regions first
        for node_id, regions in getattr(self.mapping, '_node_to_regions', {}).items():
            for region_id in regions:
                region = self.mapping._regions.get(region_id)
                if region and region.contains_point(x, y):
                    return ('node', node_id)
        
        # Check edge regions
        for edge_id, regions in getattr(self.mapping, '_edge_to_regions', {}).items():
            for region_id in regions:
                region = self.mapping._regions.get(region_id)
                if region and region.contains_point(x, y):
                    return ('edge', edge_id)
        
        return None
    
    def _find_element_using_standardized_contours(self, x, y):
        """Find element at position using standardized contour representation."""
        import cv2
        import numpy as np
        
        if not hasattr(self, '_test_mode_standardized_contours'):
            return None
            
        standardized = self._test_mode_standardized_contours
        
        # Check nodes first
        for node_id_str, contour_list in standardized.get('nodes', {}).items():
            node_id = int(node_id_str) if node_id_str.isdigit() else node_id_str
            
            for contour_points in contour_list:
                # Convert to numpy array for OpenCV
                contour_array = np.array(contour_points, dtype=np.float32)
                
                # Use OpenCV point in polygon test
                result = cv2.pointPolygonTest(contour_array, (x, y), False)
                if result >= 0:  # Point is inside or on the boundary
                    return ('node', node_id)
        
        # Check edges
        for edge_str, contour_list in standardized.get('edges', {}).items():
            # Parse edge string back to tuple
            parts = edge_str.split('_')
            if len(parts) >= 2:
                try:
                    edge_id = (int(parts[0]) if parts[0].isdigit() else parts[0],
                              int(parts[1]) if parts[1].isdigit() else parts[1])
                except ValueError:
                    edge_id = (parts[0], parts[1])
            else:
                continue
                
            for contour_points in contour_list:
                # Convert to numpy array for OpenCV
                contour_array = np.array(contour_points, dtype=np.float32)
                
                # Use OpenCV point in polygon test
                result = cv2.pointPolygonTest(contour_array, (x, y), False)
                if result >= 0:  # Point is inside or on the boundary
                    return ('edge', edge_id)
        
        return None
        
    def _highlight_element_regions(self, elem_type, elem_id):
        """Highlight all regions mapped to the given element."""
        if not hasattr(self, 'mapping') or self.mapping is None:
            return
            
        # Use standardized contours if available (for better test mode compatibility)
        if hasattr(self, '_test_mode_standardized_contours'):
            self._highlight_using_standardized_contours(elem_type, elem_id)
            return
            
        # Fallback to original method
        try:
            if elem_type == 'node':
                regions_dict = getattr(self.mapping, '_node_to_regions', {})
                regions = regions_dict.get(elem_id, [])
            else:
                regions_dict = getattr(self.mapping, '_edge_to_regions', {})
                regions = regions_dict.get(elem_id, [])
                
            for region_id in regions:
                if hasattr(self.mapping, '_regions'):
                    region = self.mapping._regions.get(region_id)
                    if region:
                        # Highlight region on map with corresponding color (yellow)
                        self.test_map_widget.highlight_region(region_id, color_type='highlight')
                
        except AttributeError as e:
            # Don't fail the whole operation
            pass
    
    def _highlight_using_standardized_contours(self, elem_type, elem_id):
        """Highlight element regions using standardized contour representation."""
        if not hasattr(self, '_test_mode_standardized_contours'):
            return
            
        standardized = self._test_mode_standardized_contours
        
        # Clear existing test highlights first (but not the Show All Mappings display)
        if hasattr(self.test_map_widget, 'test_highlights'):
            self.test_map_widget.test_highlights = []
        if hasattr(self.test_map_widget, 'test_highlight_contours'):
            self.test_map_widget.test_highlight_contours = []
        
        if elem_type == 'node':
            node_id_str = str(elem_id)
            contour_list = standardized.get('nodes', {}).get(node_id_str, [])
        else:  # edge
            # Convert edge tuple to string format
            edge_str = f"{elem_id[0]}_{elem_id[1]}"
            contour_list = standardized.get('edges', {}).get(edge_str, [])
        
        # Add each contour as a temporary highlight on the map
        from PyQt5.QtGui import QColor
        
        # Use selection colors - purple for highlight
        highlight_color = QColor(160, 100, 255, 140)
        
        for i, contour_points in enumerate(contour_list):
            # Create a unique region ID for this highlight
            region_id = f"test_highlight_{elem_type}_{elem_id}_{i}"
            
            # Add the contour temporarily for display (not to the permanent mapping)
            if not hasattr(self.test_map_widget, 'test_highlight_contours'):
                self.test_map_widget.test_highlight_contours = []
            
            # Store as a test highlight that will be drawn on top
            self.test_map_widget.test_highlight_contours.append({
                'points': contour_points,
                'region_id': region_id,
                'elem_type': elem_type,
                'elem_id': elem_id,
                'color': highlight_color
            })
        
        # Trigger repaint
        self.test_map_widget.update()
                
    def get_mapping(self) -> SpatialMapping:
        """Get the current mapping."""
        return self.mapping
    
    def _has_unsaved_progress(self) -> bool:
        """Check if there's any unsaved progress that would be lost."""
        # Check if there are any mapped regions
        if hasattr(self, 'mapping') and self.mapping:
            has_regions = bool(getattr(self.mapping, '_regions', {}))
            if has_regions:
                return True
        
        # Check if there are selected cells in grid mode
        if hasattr(self, 'map_widget') and hasattr(self.map_widget, 'selected_cells'):
            if self.map_widget.selected_cells:
                return True
                
        # Check if there's a current contour being drawn
        if hasattr(self, 'map_widget') and hasattr(self.map_widget, 'current_contour'):
            if self.map_widget.current_contour:
                return True
                
        # Check if there are completed contours
        if hasattr(self, 'map_widget') and hasattr(self.map_widget, 'completed_contours'):
            if self.map_widget.completed_contours:
                return True
                
        return False
    
    def _confirm_mode_switch(self, new_mode: str) -> bool:
        """Ask user to confirm mode switch since we always reset to fresh state."""
        if not self._has_unsaved_progress():
            return True
            
        reply = QMessageBox.question(
            self, 
            "Switch Mode", 
            f"Switching to {new_mode} mode will clear all current progress and start fresh.\n\nDo you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        return reply == QMessageBox.Yes
    
    def _display_all_standardized_mappings(self):
        """Display all standardized mappings on test map widget with original colors."""
        if not hasattr(self, '_test_mode_standardized_contours') or not hasattr(self, 'test_map_widget'):
            return
            
        from PyQt5.QtGui import QColor
        
        # Clear existing displays
        self.test_map_widget.completed_contours.clear()
        
        # Display all node mappings in green
        for node_id_str, contour_list in self._test_mode_standardized_contours.get('nodes', {}).items():
            node_id = int(node_id_str) if node_id_str.isdigit() else node_id_str
            color = QColor(0, 255, 0, 100)  # Green for nodes
            
            for i, contour_points in enumerate(contour_list):
                region_id = f"node_{node_id}_region_{i}"
                # Add to completed contours for display
                self.test_map_widget.completed_contours.append(
                    (contour_points, region_id, color, 'node', node_id)
                )
        
        # Display all edge mappings in orange
        for edge_str, contour_list in self._test_mode_standardized_contours.get('edges', {}).items():
            # Parse edge string back to tuple
            parts = edge_str.split('_')
            if len(parts) >= 2:
                edge_id = (int(parts[0]) if parts[0].isdigit() else parts[0],
                          int(parts[1]) if parts[1].isdigit() else parts[1])
            else:
                continue
                
            color = QColor(255, 165, 0, 100)  # Orange for edges
            
            for i, contour_points in enumerate(contour_list):
                region_id = f"edge_{edge_str}_region_{i}"
                # Add to completed contours for display
                self.test_map_widget.completed_contours.append(
                    (contour_points, region_id, color, 'edge', edge_id)
                )
        
        # Trigger repaint
        self.test_map_widget.update()
    
    def _copy_mapping_to_test_widget(self):
        """Copy mapping data to test_map_widget for display via Show All Mappings toggle."""
        if not hasattr(self, 'test_map_widget') or not hasattr(self, 'mapping'):
            return
            
        # Clear existing data
        self.test_map_widget.completed_contours.clear()
        self.test_map_widget.grid_cells.clear()
        
        # Copy all mapped regions to test_map_widget
        for region_id, region in self.mapping._regions.items():
            element_info = self.mapping._region_to_element.get(region_id)
            if element_info:
                element_type, element_id = element_info
                
                # Determine color based on element type
                if element_type == 'node':
                    color = QColor(0, 255, 0, 100)  # Green for nodes
                else:
                    color = QColor(255, 165, 0, 100)  # Orange for edges
                
                # Handle different region types
                from .regions import RectangleRegion, ContourRegion
                
                if isinstance(region, RectangleRegion):  # Grid-based region
                    # Convert to grid cell format
                    cell_id = f"cell_{region_id}"
                    cell_rect = QRectF(region.x, region.y, region.width, region.height)
                    self.test_map_widget.grid_cells[cell_id] = cell_rect
                    self.test_map_widget.add_cell_mapping(cell_id, element_type, element_id, color)
                
                elif isinstance(region, ContourRegion):  # Contour-based region
                    # Add to completed contours
                    contour_color = QColor(color.red(), color.green(), color.blue(), 150)
                    self.test_map_widget.completed_contours.append(
                        (region.contour, region_id, contour_color, element_type, element_id)
                    )
                    self.test_map_widget.contour_mappings[region_id] = (element_type, element_id)


def launch_setup_gui(graph: GraphStructure, map_image: np.ndarray) -> Optional[SpatialMapping]:
    """Launch the setup GUI and return the created mapping.
    
    Args:
        graph: Graph structure to map
        map_image: Map image as numpy array
        
    Returns:
        Created spatial mapping or None if cancelled
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        
    window = GraphSetupWindow(graph, map_image)
    window.show()
    
    app.exec_()
    
    return window.get_mapping()