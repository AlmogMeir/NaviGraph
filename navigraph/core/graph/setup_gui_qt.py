"""PyQt5-based interactive dual-view GUI for graph-space mapping setup.

This module provides a professional Qt-based interface for creating spatial mappings
between graph nodes/edges and regions on a map, with both grid-based and manual
contour drawing modes.
"""

from dataclasses import dataclass
from typing import Optional, Set, List, Tuple, Dict, Any

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .mapping import SpatialMapping
from .regions import (
    ContourRegion,
    RectangleRegion,
)
from .structures import GraphStructure


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
        self.grid_cells: Dict[str, QRectF] = {}  # cell_id -> QRectF
        self.selected_cells: Set[str] = set()
        self.highlighted_cells: Set[str] = set()  # For highlighting specific assignments
        self.cell_colors: Dict[str, QColor] = {}  # cell_id -> QColor
        self.cell_mappings: Dict[str, Tuple[str, Any]] = {}  # cell_id -> (elem_type, elem_id) for mapped cells
        
        # Drawing state
        self.drawing_mode = False
        self.current_contour: List[Tuple[float, float]] = []
        self.completed_contours: List[Tuple[Any, str, QColor, str, Any]] = []  # List of (contour_points, region_id, color, elem_info)
        self.contour_mappings: Dict[str, Tuple[str, Any]] = {}  # region_id -> (elem_type, elem_id)
        
        # Interaction state
        self.interaction_mode = 'none'  # 'place_grid', 'select_cells', 'draw_contour'
        self.show_all_mappings = True  # Toggle for showing all mapped regions
        self.show_cell_labels = True  # Toggle for showing cell labels
        self.adaptive_font_size = True  # Toggle for adaptive font sizing
        self.unified_font_size = False  # Toggle for unified font size
        self.current_element_type = None
        self.current_element_id = None
        
        # Contour highlighting state
        self.highlighted_contour_id = None
        self.original_contour_colors: Dict[str, QColor] = {}  # Store original colors for unhighlighting
        
        self.setMouseTracking(True)
        # Remove minimum size to prevent window resizing
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # Enable keyboard focus
        
        # Enable tooltips with faster response
        self.setToolTip("")
        self.current_tooltip = ""  # Track tooltip state for immediate updates
        
    def set_interaction_mode(self, mode: str):
        """Set the current interaction mode."""
        self.interaction_mode = mode
        self.current_contour = []
        try:
            self.update()
        except RuntimeError:
            # Widget deleted, skip update
            pass
        
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
        
    def remove_cell_mapping(self, cell_id: str):
        """Remove mapping for a specific cell."""
        if cell_id in self.cell_mappings:
            del self.cell_mappings[cell_id]
        if cell_id in self.cell_colors:
            del self.cell_colors[cell_id]
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
                
                # Check if cell is mapped, selected, or highlighted
                is_mapped = cell_id in self.cell_mappings
                is_selected = cell_id in self.selected_cells
                is_highlighted = cell_id in self.highlighted_cells
                
                # Draw cell based on state (priority: selected > highlighted > mapped > empty)
                if is_selected:
                    # Currently selected for mapping - bright green
                    painter.fillRect(scaled_rect, QColor(0, 255, 0, 100))
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                    painter.drawRect(scaled_rect)
                elif is_highlighted:
                    # Highlighted from assignment list - bright yellow
                    painter.fillRect(scaled_rect, QColor(255, 235, 59, 120))  # Yellow highlight
                    painter.setPen(QPen(QColor(255, 193, 7), 2))  # Darker yellow border
                    painter.drawRect(scaled_rect)
                elif is_mapped and self.show_all_mappings:
                    # Mapped cell - show with persistent color
                    color = self.cell_colors.get(cell_id, QColor(200, 200, 200, 100))
                    painter.fillRect(scaled_rect, color)
                    painter.setPen(QPen(color.darker(), 2))
                    painter.drawRect(scaled_rect)
                    
                    # Text will be drawn from mapping section (unified approach)
                else:
                    # Empty cell
                    painter.setPen(QPen(QColor(100, 100, 100), 1))
                    painter.drawRect(scaled_rect)
                
        # Draw regions from mapping (single source of truth for all committed contours)
        # Only draw if show_all_mappings is enabled
        if (hasattr(self, 'gui_parent') and hasattr(self.gui_parent, 'mapping') and 
            self.gui_parent.mapping and hasattr(self, 'show_all_mappings') and self.show_all_mappings):
            try:
                from .regions import RectangleRegion, ContourRegion
                
                # Draw node regions in green
                for node_id in self.gui_parent.mapping.get_mapped_nodes():
                    regions = self.gui_parent.mapping.get_node_regions(node_id)
                    color = QColor(150, 255, 150, 100)  # Light green
                    
                    for region in regions:
                        # Extract points based on region type
                        if isinstance(region, RectangleRegion):
                            points = [
                                (region.x, region.y),
                                (region.x + region.width, region.y),
                                (region.x + region.width, region.y + region.height),
                                (region.x, region.y + region.height)
                            ]
                        elif isinstance(region, ContourRegion):
                            points = [(float(p[0]), float(p[1])) for p in region.contour_points]
                        else:
                            continue
                            
                        if len(points) > 2:
                            poly_points = [QPointF(self.offset_x + p[0] * self.scale_factor, 
                                                 self.offset_y + p[1] * self.scale_factor) 
                                         for p in points]
                            polygon = QPolygonF(poly_points)
                            
                            painter.setPen(QPen(color.darker(), 2))
                            painter.setBrush(QBrush(color))
                            painter.drawPolygon(polygon)
                            
                            # Draw node label based on adaptive font mode
                            if self.show_cell_labels:
                                bounding_rect = polygon.boundingRect()
                                centroid = bounding_rect.center()
                                
                                label_text = f"N{node_id}"
                                
                                if self.unified_font_size:
                                    # Unified font size mode - fixed size for all labels
                                    font_size = 8
                                    font = painter.font()
                                    font.setFamily('Arial')  # More compact font
                                    font.setPointSize(font_size)
                                    painter.setFont(font)
                                    painter.setPen(QPen(QColor(50, 50, 50), 1))  # Darker grey for better contrast
                                    
                                    # Draw text truly centered
                                    from PyQt5.QtGui import QFontMetrics
                                    metrics = QFontMetrics(font)
                                    text_rect = metrics.boundingRect(label_text)
                                    text_x = centroid.x() - text_rect.width() / 2
                                    text_y = centroid.y() + text_rect.height() / 4  # Adjust for baseline
                                    painter.drawText(QPointF(text_x, text_y), label_text)
                                elif self.adaptive_font_size:
                                    # Adaptive mode: Size text to fit inside contour
                                    # Start with a reasonable font size and reduce until text fits
                                    max_font_size = 20
                                    min_font_size = 5
                                    font_size = max_font_size
                                    
                                    # Get available space (use 80% of contour dimensions for padding)
                                    available_width = bounding_rect.width() * 0.8
                                    available_height = bounding_rect.height() * 0.8
                                    
                                    # Find the right font size that fits
                                    font = painter.font()
                                    font.setFamily('Arial')  # More compact font
                                    for size in range(max_font_size, min_font_size - 1, -1):
                                        font.setPointSize(size)
                                        painter.setFont(font)
                                        
                                        # Measure text with this font size
                                        from PyQt5.QtGui import QFontMetrics
                                        metrics = QFontMetrics(font)
                                        text_rect = metrics.boundingRect(label_text)
                                        
                                        # Check if text fits within available space
                                        if text_rect.width() <= available_width and text_rect.height() <= available_height:
                                            font_size = size
                                            break
                                    else:
                                        font_size = min_font_size
                                    
                                    # Set final font
                                    font.setPointSize(font_size)
                                    painter.setFont(font)
                                    
                                    # Draw grey text centered in contour
                                    painter.setPen(QPen(QColor(100, 100, 100), 1))  # Grey text
                                    
                                    # Draw text truly centered (accounting for text dimensions)
                                    metrics = QFontMetrics(font)
                                    text_rect = metrics.boundingRect(label_text)
                                    text_x = centroid.x() - text_rect.width() / 2
                                    text_y = centroid.y() + text_rect.height() / 4  # Adjust for baseline
                                    painter.drawText(QPointF(text_x, text_y), label_text)
                                else:
                                    # Fixed mode: Draw green text at centroid with fixed size
                                    font_size = 9
                                    font = painter.font()
                                    font.setFamily('Arial')  # More compact font
                                    font.setPointSize(font_size)
                                    painter.setFont(font)
                                    painter.setPen(QPen(QColor(0, 80, 0), 1))  # Darker green text
                                    painter.drawText(centroid, label_text)
                
                # Draw edge regions in orange
                for edge in self.gui_parent.mapping.get_mapped_edges():
                    regions = self.gui_parent.mapping.get_edge_regions(edge)
                    color = QColor(255, 165, 0, 100)  # Orange
                    
                    for region in regions:
                        # Extract points based on region type
                        if isinstance(region, RectangleRegion):
                            points = [
                                (region.x, region.y),
                                (region.x + region.width, region.y),
                                (region.x + region.width, region.y + region.height),
                                (region.x, region.y + region.height)
                            ]
                        elif isinstance(region, ContourRegion):
                            points = [(float(p[0]), float(p[1])) for p in region.contour_points]
                        else:
                            continue
                            
                        if len(points) > 2:
                            poly_points = [QPointF(self.offset_x + p[0] * self.scale_factor, 
                                                 self.offset_y + p[1] * self.scale_factor) 
                                         for p in points]
                            polygon = QPolygonF(poly_points)
                            
                            painter.setPen(QPen(color.darker(), 2))
                            painter.setBrush(QBrush(color))
                            painter.drawPolygon(polygon)
                            
                            # Draw edge label based on adaptive font mode
                            if self.show_cell_labels:
                                bounding_rect = polygon.boundingRect()
                                centroid = bounding_rect.center()
                                
                                # Prepare edge label text
                                if isinstance(edge, tuple):
                                    label_text = f"E{edge[0]},{edge[1]}"
                                else:
                                    label_text = f"E{edge}"
                                
                                if self.unified_font_size:
                                    # Unified font size mode - fixed size for edge labels (smaller than nodes)
                                    font_size = 7
                                    font = painter.font()
                                    font.setFamily('Arial')  # More compact font
                                    font.setPointSize(font_size)
                                    painter.setFont(font)
                                    painter.setPen(QPen(QColor(120, 60, 0), 1))  # Darker brown for better contrast
                                    
                                    # Draw text truly centered
                                    from PyQt5.QtGui import QFontMetrics
                                    metrics = QFontMetrics(font)
                                    text_rect = metrics.boundingRect(label_text)
                                    text_x = centroid.x() - text_rect.width() / 2
                                    text_y = centroid.y() + text_rect.height() / 4  # Adjust for baseline
                                    painter.drawText(QPointF(text_x, text_y), label_text)
                                elif self.adaptive_font_size:
                                    # Adaptive mode: Size text to fit inside contour
                                    # Start with a reasonable font size and reduce until text fits
                                    max_font_size = 20
                                    min_font_size = 5
                                    font_size = max_font_size
                                    
                                    # Get available space (use 80% of contour dimensions for padding)
                                    available_width = bounding_rect.width() * 0.8
                                    available_height = bounding_rect.height() * 0.8
                                    
                                    # Find the right font size that fits
                                    font = painter.font()
                                    font.setFamily('Arial')  # More compact font
                                    for size in range(max_font_size, min_font_size - 1, -1):
                                        font.setPointSize(size)
                                        painter.setFont(font)
                                        
                                        # Measure text with this font size
                                        from PyQt5.QtGui import QFontMetrics
                                        metrics = QFontMetrics(font)
                                        text_rect = metrics.boundingRect(label_text)
                                        
                                        # Check if text fits within available space
                                        if text_rect.width() <= available_width and text_rect.height() <= available_height:
                                            font_size = size
                                            break
                                    else:
                                        font_size = min_font_size
                                    
                                    # Set final font
                                    font.setPointSize(font_size)
                                    painter.setFont(font)
                                    
                                    # Draw grey text centered in contour
                                    painter.setPen(QPen(QColor(100, 100, 100), 1))  # Grey text
                                    
                                    # Draw text truly centered (accounting for text dimensions)
                                    metrics = QFontMetrics(font)
                                    text_rect = metrics.boundingRect(label_text)
                                    text_x = centroid.x() - text_rect.width() / 2
                                    text_y = centroid.y() + text_rect.height() / 4  # Adjust for baseline
                                    painter.drawText(QPointF(text_x, text_y), label_text)
                                else:
                                    # Fixed mode: Draw orange text at centroid with fixed size
                                    font_size = 9
                                    font = painter.font()
                                    font.setFamily('Arial')  # More compact font
                                    font.setPointSize(font_size)
                                    painter.setFont(font)
                                    painter.setPen(QPen(QColor(120, 60, 0), 1))  # Darker orange text
                                    painter.drawText(centroid, label_text)
                
            except Exception as e:
                print(f"Error drawing regions from mapping: {e}")
        
        # Draw selected region highlighting (yellow)
        if hasattr(self, 'gui_parent') and hasattr(self.gui_parent, 'mapping') and self.gui_parent.mapping and hasattr(self, 'highlighted_region_id'):
            try:
                from .regions import RectangleRegion, ContourRegion
                if self.highlighted_region_id:
                    region = self.gui_parent.mapping.get_region_by_id(self.highlighted_region_id)
                    if region:
                        # Extract points based on region type
                        if isinstance(region, RectangleRegion):
                            points = [
                                (region.x, region.y),
                                (region.x + region.width, region.y),
                                (region.x + region.width, region.y + region.height),
                                (region.x, region.y + region.height)
                            ]
                        elif isinstance(region, ContourRegion):
                            points = [(float(p[0]), float(p[1])) for p in region.contour_points]
                        else:
                            points = None
                            
                        if points and len(points) > 2:
                            poly_points = [QPointF(self.offset_x + p[0] * self.scale_factor, 
                                                 self.offset_y + p[1] * self.scale_factor) 
                                         for p in points]
                            polygon = QPolygonF(poly_points)
                            
                            # Draw yellow highlight
                            highlight_color = QColor(255, 255, 0, 150)  # Yellow highlight
                            painter.setPen(QPen(highlight_color.darker(), 3))
                            painter.setBrush(QBrush(highlight_color))
                            painter.drawPolygon(polygon)
                            
            except Exception as e:
                print(f"Error drawing region highlight: {e}")
                
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
        
        # Draw element highlights (purple) for selected nodes/edges
        if hasattr(self, 'highlighted_elements') and hasattr(self, 'gui_parent') and hasattr(self.gui_parent, 'mapping') and self.gui_parent.mapping:
            try:
                from .regions import RectangleRegion, ContourRegion
                highlight_color = QColor(160, 100, 255, 140)  # Purple highlight
                
                for elem_type, elem_id in self.highlighted_elements:
                    if elem_type == 'node':
                        regions = self.gui_parent.mapping.get_node_regions(elem_id)
                    else:  # edge
                        regions = self.gui_parent.mapping.get_edge_regions(elem_id)
                    
                    for region in regions:
                        # Extract points based on region type
                        if isinstance(region, RectangleRegion):
                            points = [
                                (region.x, region.y),
                                (region.x + region.width, region.y),
                                (region.x + region.width, region.y + region.height),
                                (region.x, region.y + region.height)
                            ]
                        elif isinstance(region, ContourRegion):
                            points = [(float(p[0]), float(p[1])) for p in region.contour_points]
                        else:
                            continue
                            
                        if len(points) > 2:
                            poly_points = [QPointF(self.offset_x + p[0] * self.scale_factor, 
                                                 self.offset_y + p[1] * self.scale_factor) 
                                         for p in points]
                            polygon = QPolygonF(poly_points)
                            
                            # Draw highlight with purple color
                            painter.setPen(QPen(highlight_color, 3))
                            painter.setBrush(QBrush(highlight_color))
                            painter.drawPolygon(polygon)
                            
            except Exception as e:
                print(f"Error drawing element highlights: {e}")
        
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
        
        # Draw labels independently only when contours are not shown
        if self.show_cell_labels and not self.show_all_mappings:
            self._draw_node_labels(painter)
            self._draw_edge_labels(painter)
                
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
                
            elif self.interaction_mode == 'test':
                # Handle test mode clicks - delegate to parent GUI
                if hasattr(self, 'gui_parent') and hasattr(self.gui_parent, '_on_test_map_click'):
                    self.gui_parent._on_test_map_click(event)
    
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
    
    def _draw_node_labels(self, painter):
        """Draw node labels independently of contours."""
        if not (hasattr(self, 'gui_parent') and hasattr(self.gui_parent, 'mapping') and self.gui_parent.mapping):
            return
            
        try:
            from .regions import RectangleRegion, ContourRegion
            
            for node_id in self.gui_parent.mapping.get_mapped_nodes():
                regions = self.gui_parent.mapping.get_node_regions(node_id)
                
                for region in regions:
                    # Calculate centroid based on region type
                    centroid = self._get_region_centroid(region)
                    if centroid is None:
                        continue
                        
                    label_text = f"N{node_id}"
                    self._draw_text_label(painter, centroid, label_text, QColor(100, 100, 100))
                    
        except Exception as e:
            print(f"Error drawing node labels: {e}")
    
    def _draw_edge_labels(self, painter):
        """Draw edge labels independently of contours.""" 
        if not (hasattr(self, 'gui_parent') and hasattr(self.gui_parent, 'mapping') and self.gui_parent.mapping):
            return
            
        try:
            from .regions import RectangleRegion, ContourRegion
            
            for edge in self.gui_parent.mapping.get_mapped_edges():
                regions = self.gui_parent.mapping.get_edge_regions(edge)
                
                for region in regions:
                    # Calculate centroid based on region type  
                    centroid = self._get_region_centroid(region)
                    if centroid is None:
                        continue
                        
                    # Prepare edge label text
                    if isinstance(edge, tuple):
                        label_text = f"E{edge[0]},{edge[1]}"
                    else:
                        label_text = f"E{edge}"
                    self._draw_text_label(painter, centroid, label_text, QColor(150, 80, 0))
                    
        except Exception as e:
            print(f"Error drawing edge labels: {e}")
    
    def _get_region_centroid(self, region):
        """Calculate the centroid of a region in screen coordinates."""
        from .regions import RectangleRegion, ContourRegion
        
        if isinstance(region, RectangleRegion):
            # Rectangle centroid
            center_x = region.x + region.width / 2
            center_y = region.y + region.height / 2
            
        elif isinstance(region, ContourRegion):
            # Contour centroid (average of points)
            if region.contour_points is None or len(region.contour_points) == 0:
                return None
            try:
                # Handle both numpy arrays and regular lists/tuples
                points = []
                for p in region.contour_points:
                    if hasattr(p, '__len__') and len(p) >= 2:
                        points.append((float(p[0]), float(p[1])))
                
                if not points:
                    return None
                    
                sum_x = sum(p[0] for p in points)
                sum_y = sum(p[1] for p in points) 
                center_x = sum_x / len(points)
                center_y = sum_y / len(points)
            except (ValueError, TypeError, IndexError) as e:
                print(f"Error processing contour points: {e}")
                return None
            
        else:
            return None
            
        # Convert to screen coordinates
        screen_x = self.offset_x + center_x * self.scale_factor
        screen_y = self.offset_y + center_y * self.scale_factor
        
        return QPointF(screen_x, screen_y)
    
    def _draw_text_label(self, painter, centroid, label_text, color):
        """Draw a text label at the given centroid."""
        if self.unified_font_size:
            # Unified font size mode - smaller size for edge labels
            if label_text.startswith('E'):
                font_size = 7  # Smaller for edges
            else:
                font_size = 8  # Nodes
        elif self.adaptive_font_size:
            # Fixed font size for labels without contours (no bounding box to fit)
            font_size = 8
        else:
            # Fixed mode font size
            font_size = 7
            
        font = painter.font()
        font.setFamily('Arial')  # More compact font
        font.setPointSize(font_size)
        painter.setFont(font)
        painter.setPen(QPen(color, 1))
        
        # Center the text
        from PyQt5.QtGui import QFontMetrics
        metrics = QFontMetrics(font)
        text_rect = metrics.boundingRect(label_text)
        text_x = centroid.x() - text_rect.width() / 2
        text_y = centroid.y() + text_rect.height() / 4  # Adjust for baseline
        
        painter.drawText(int(text_x), int(text_y), label_text)
    
    def resizeEvent(self, event):
        """Handle widget resize by recalculating scale factors."""
        super().resizeEvent(event)
        # Recalculate base scale factor to fit the new widget size
        self.update()  # This will recalculate base_scale_factor in paintEvent
    
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
    
    def highlight_element(self, elem_type: str, elem_id):
        """Highlight a specific element (node or edge)."""
        try:
            if not hasattr(self, 'highlighted_elements'):
                self.highlighted_elements = []
            
            # Remove existing highlight for this element
            self.highlighted_elements = [h for h in self.highlighted_elements if h != (elem_type, elem_id)]
            
            # Add new highlight
            self.highlighted_elements.append((elem_type, elem_id))
            self.update()
        except Exception as e:
            print(f"Error highlighting {elem_type} {elem_id}: {e}")
    
    def clear_highlights(self):
        """Clear all element highlights."""
        try:
            if hasattr(self, 'highlighted_elements'):
                self.highlighted_elements = []
            self.update()
        except Exception as e:
            print(f"Error clearing highlights: {e}")


class GraphWidget(QWidget):
    """Widget for displaying the graph structure using the builder's visualization."""
    
    nodeClicked = pyqtSignal(object)  # Emitted when a node is clicked
    edgeClicked = pyqtSignal(tuple)  # Emitted when an edge is clicked
    
    def __init__(self, graph: GraphStructure, parent=None):
        super().__init__(parent)
        
        self.graph = graph
        self.node_colors = {}
        self.edge_colors = {}
        self.edge_widths = {}
        self.highlighted_nodes = set()
        self.highlighted_edges = set()
        
        # Default colors and widths
        self.default_node_color = 'lightblue'
        self.default_edge_color = 'gray'
        self.default_edge_width = 2.0  # Thicker than NetworkX default 1.0
        
        # Prevent recursive resize events
        self._drawing = False
        
        # Setup UI
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QLabel, QVBoxLayout
        
        layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ddd; background-color: white;")
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        
        # Set reasonable minimum size for stability
        self.setMinimumSize(300, 200)
        self.draw_graph()
        
        # Note: Graph click detection removed - using dropdown selection in test mode instead

    def draw_graph(self):
        """Draw or redraw the graph using the builder's visualization."""
        if self._drawing:
            return  # Prevent recursive calls
            
        self._drawing = True
        try:
            # Prepare node colors
            node_colors = []
            for node in self.graph.nodes:
                if node in self.node_colors:
                    node_colors.append(self.node_colors[node])
                else:
                    node_colors.append(self.default_node_color)
            
            # Prepare edge colors
            edge_colors = []
            for edge in self.graph.edges:
                if edge in self.edge_colors:
                    edge_colors.append(self.edge_colors[edge])
                elif (edge[1], edge[0]) in self.edge_colors:  # Check reverse edge
                    edge_colors.append(self.edge_colors[(edge[1], edge[0])])
                else:
                    edge_colors.append(self.default_edge_color)
            
            # Prepare edge widths
            edge_widths = []
            for edge in self.graph.edges:
                if edge in self.edge_widths:
                    edge_widths.append(self.edge_widths[edge])
                elif (edge[1], edge[0]) in self.edge_widths:  # Check reverse edge
                    edge_widths.append(self.edge_widths[(edge[1], edge[0])])
                else:
                    edge_widths.append(self.default_edge_width)
            
            # Determine if we need custom colors/widths
            has_custom_node_colors = len(set(node_colors)) > 1
            has_custom_edge_colors = len(set(edge_colors)) > 1
            has_custom_edge_widths = len(set(edge_widths)) > 1
            
            # Get appropriate sizing based on graph size - increased for better visibility
            node_count = len(self.graph.nodes)
            if node_count > 100:
                node_size, font_size = 400, 9  # Increased from 250, 7
            elif node_count > 50:
                node_size, font_size = 600, 10  # Increased from 400, 8
            elif node_count > 20:
                node_size, font_size = 800, 12  # Increased from 600, 10
            else:
                node_size, font_size = 1000, 14  # Increased from 800, 12
            
            # Calculate figsize based on graph type
            # For binary trees, width should scale with tree height (number of leaf nodes)
            if hasattr(self.graph.builder, 'height'):
                # Binary tree: width scales with 2^height leaf nodes
                tree_height = self.graph.builder.height
                width = max(8, min(25, 2.5 * tree_height))  # Scale: height 7 ≈ 17.5 width
                figsize = (width, 5)
            else:
                # Default for other graph types
                figsize = (12, 8)
            
            visualization_params = {
                'figsize': figsize,
                'node_size': node_size,
                'font_size': font_size,
                'with_labels': True,
                'font_weight': 'normal',
                'font_color': 'black',
                'font_family': 'sans-serif'
            }
            
            # Add colors if we have custom ones
            if has_custom_node_colors:
                visualization_params['node_color'] = node_colors
            else:
                visualization_params['node_color'] = self.default_node_color
                
            if has_custom_edge_colors:
                visualization_params['edge_color'] = edge_colors  
            else:
                visualization_params['edge_color'] = self.default_edge_color
            
            # Add widths if we have custom ones
            if has_custom_edge_widths:
                visualization_params['width'] = edge_widths
            else:
                visualization_params['width'] = self.default_edge_width
            
            image_array = self.graph.get_visualization(**visualization_params)
            
            # Convert numpy array to QPixmap
            from PyQt5.QtGui import QImage, QPixmap
            
            height, width, channel = image_array.shape
            bytes_per_line = 3 * width
            # Ensure array is contiguous and convert to bytes for QImage
            image_array = np.ascontiguousarray(image_array)
            q_image = QImage(image_array.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale to fit widget while maintaining aspect ratio
            try:
                # Check if image_label still exists and is valid
                if hasattr(self, 'image_label') and self.image_label is not None:
                    self.image_label.isVisible()  # This will raise if deleted
                    if not self.image_label.size().isEmpty():
                        scaled_pixmap = pixmap.scaled(
                            self.image_label.size(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                        self.image_label.setPixmap(scaled_pixmap)
                    else:
                        self.image_label.setPixmap(pixmap)
            except RuntimeError:
                # Widget deleted, skip scaling
                return
            
        except Exception as e:
            print(f"Error updating graph visualization: {e}")
            import traceback
            traceback.print_exc()
            # Show error message
            self.image_label.setText(f"Error displaying graph:\n{str(e)}")
        finally:
            self._drawing = False

    def resizeEvent(self, event):
        """Handle widget resize."""
        super().resizeEvent(event)
        # Update visualization when widget is resized to maintain proper scaling
        # But prevent recursive calls during drawing
        if hasattr(self, 'image_label') and self.image_label.pixmap() and not self._drawing:
            self.draw_graph()

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
        
        # Store colors and thicker widths for highlighted edges
        highlight_width = 4.0  # Thicker than default 2.0
        for edge in edges:
            self.edge_colors[edge] = actual_color
            self.edge_widths[edge] = highlight_width
        
        self.draw_graph()
        
    def clear_highlights(self):
        """Clear all highlights."""
        self.highlighted_nodes.clear()
        self.highlighted_edges.clear()
        self.node_colors.clear()
        self.edge_colors.clear()
        self.edge_widths.clear()
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
        self.highlighted_assignment_cells = None  # Track currently highlighted grid assignment
        
        # Initialize UI
        self.setWindowTitle("NaviGraph - Interactive Graph Mapping Setup")
        self.setGeometry(100, 100, 1600, 1000)
        # Remove minimum size to prevent unwanted window resizing
        
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
        
        # Create toolbar with window controls
        self._create_toolbar()
        
        self._init_ui()
        self._connect_signals()
        
        # Add keyboard shortcuts
        self._setup_shortcuts()
        
        # Initialize scaled fonts
        self.update_font_sizes()
        
        # Let splitters handle their own initial sizing
        
    def _init_ui(self):
        """Initialize the user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create main horizontal splitter for resizable panels
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # Left panel - Controls (resizable)
        self.control_panel = self._create_control_panel()
        # Set minimum width for stability but remove maximum to allow splitter expansion
        self.control_panel.setMinimumWidth(350)
        self.control_panel.setStyleSheet(
            "QWidget { background-color: #f8f9fa; border-right: 2px solid #dee2e6; }"
        )
        self.main_splitter.addWidget(self.control_panel)
        
        # Right panel - Views with vertical splitter between graph and map
        self.views_splitter = QSplitter(Qt.Vertical)
        
        # Graph view (top)
        graph_group = QGroupBox("📊 Graph Structure")
        graph_group.setStyleSheet("QGroupBox { font-weight: 600; font-size: 14px; color: #424242; }")
        graph_layout = QVBoxLayout(graph_group)
        graph_layout.setContentsMargins(8, 20, 8, 8)
        
        self.graph_widget = GraphWidget(self.graph)
        # Remove size constraints to allow flexible splitter resizing
        graph_layout.addWidget(self.graph_widget)
        
        self.views_splitter.addWidget(graph_group)
        
        # Map view (bottom)
        map_group = QGroupBox("🗺️ Interactive Mapping Area")
        map_group.setStyleSheet("QGroupBox { font-weight: 600; font-size: 14px; color: #424242; }")
        map_layout = QVBoxLayout(map_group)
        map_layout.setContentsMargins(8, 20, 8, 8)
        
        self.map_widget = MapWidget(self.map_image)
        self.map_widget.gui_parent = self  # Allow map widget to access mapping
        # Remove minimum height to allow flexible splitter resizing
        map_layout.addWidget(self.map_widget)
        
        self.views_splitter.addWidget(map_group)
        
        # Set initial sizes for graph:map (1:2 ratio - map larger by default)
        self.views_splitter.setSizes([300, 600])
        self.views_splitter.setStretchFactor(0, 1)  # Graph
        self.views_splitter.setStretchFactor(1, 2)  # Map
        
        # Add views splitter to main splitter
        self.main_splitter.addWidget(self.views_splitter)
        
        # Set initial sizes based on proportions rather than fixed values
        initial_width = 1600  # Default window width
        control_width = 400   # Control panel width
        views_width = initial_width - control_width
        self.main_splitter.setSizes([control_width, views_width])
        self.main_splitter.setStretchFactor(0, 0)  # Control panel doesn't stretch much
        self.main_splitter.setStretchFactor(1, 1)  # Views stretch
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Select a setup mode to begin")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(len(self.graph.nodes) + len(self.graph.edges))
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def _create_control_panel(self) -> QWidget:
        """Create the simplified control panel widget with scroll support."""
        # Create scroll area container
        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create the actual panel widget
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
        
        self.show_labels_checkbox = QCheckBox("Show Contour Labels")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.stateChanged.connect(self._on_toggle_labels)
        viz_layout.addWidget(self.show_labels_checkbox)
        
        self.adaptive_font_checkbox = QCheckBox("Adaptive Font Size")
        self.adaptive_font_checkbox.setChecked(True)
        self.adaptive_font_checkbox.stateChanged.connect(self._on_toggle_adaptive_font)
        viz_layout.addWidget(self.adaptive_font_checkbox)
        
        self.unified_font_checkbox = QCheckBox("Unified Font Size")
        self.unified_font_checkbox.setChecked(False)
        self.unified_font_checkbox.stateChanged.connect(self._on_toggle_unified_font)
        viz_layout.addWidget(self.unified_font_checkbox)
        
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
        
        # Set the panel as the scroll area's widget and return scroll area
        scroll_area.setWidget(panel)
        return scroll_area
        
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
        self.rows_spin.setRange(1, 999999)
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
        self.cols_spin.setRange(1, 999999)
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
        self.width_spin.setRange(1, 9999999)
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
        
        # Place grid button
        config_buttons = QHBoxLayout()
        config_buttons.setSpacing(8)
        
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
        
        # Assignment management section
        mgmt_group = QGroupBox("Assignment Management")
        mgmt_layout = QVBoxLayout(mgmt_group)
        
        self.grid_assignment_list = QListWidget()
        self.grid_assignment_list.itemClicked.connect(self._on_grid_assignment_selected)
        self.grid_assignment_list.setStyleSheet(
            "QListWidget { background-color: white; border: 1px solid #ccc; border-radius: 4px; }"
            "QListWidget::item { padding: 4px; border-bottom: 1px solid #eee; }"
            "QListWidget::item:selected { background-color: #ffeb3b; color: black; }"
            "QListWidget::item:hover { background-color: #f5f5f5; }"
        )
        mgmt_layout.addWidget(self.grid_assignment_list)
        
        assignment_buttons = QHBoxLayout()
        self.delete_grid_assignment_button = QPushButton("Delete")
        self.delete_grid_assignment_button.clicked.connect(self._on_delete_grid_assignment)
        self.delete_grid_assignment_button.setEnabled(False)
        assignment_buttons.addWidget(self.delete_grid_assignment_button)
        
        self.clear_all_grid_button = QPushButton("Clear All")
        self.clear_all_grid_button.clicked.connect(self._on_clear_all)
        self.clear_all_grid_button.setEnabled(True)
        assignment_buttons.addWidget(self.clear_all_grid_button)
        
        mgmt_layout.addLayout(assignment_buttons)
        assign_layout.addWidget(mgmt_group)
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
                
                # Recreate UI fresh when entering grid mode
                self._recreate_fresh_ui()
                
                # Then reset to fresh state with new widgets
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
                
                # Enable drawing mode
                if hasattr(self.map_widget, 'set_interaction_mode'):
                    self.map_widget.set_interaction_mode('none')
                else:
                    print("Warning: map_widget not available for grid mode")
                    
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
                
                # Recreate UI fresh when entering manual mode
                self._recreate_fresh_ui()
                
                # Then reset to fresh state with new widgets
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
                else:
                    print("Warning: map_widget not available for manual mode")
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
                
                # Recreate UI fresh when entering test mode
                self._recreate_fresh_ui()
                
                # Then setup test-specific state with new widgets
                try:
                    self._reset_mapping_state()
                    # Test mode initialization - populate element combos
                    self._populate_element_combos()
                    self.status_bar.showMessage("Test mode ready - use same layout as other modes")
                except Exception as reset_error:
                    self.status_bar.showMessage(f"Error setting up test mode: {str(reset_error)}")
                    print(f"Test mode setup error: {reset_error}")
                
                # Enable test interactions
                if hasattr(self.map_widget, 'set_interaction_mode'):
                    self.map_widget.set_interaction_mode('test')
                    
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
                # Recreate fresh UI when leaving test mode
                self._recreate_fresh_ui()
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
            # Check node mappings using new API
            for node_id in self.mapping.get_mapped_nodes():
                mapped_elements.add(('node', node_id))
            
            # Check edge mappings using new API
            for edge_id in self.mapping.get_mapped_edges():
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
        
        # Sort nodes and edges numerically 
        def numeric_sort_key(x):
            """Sort key that handles numeric node IDs properly."""
            try:
                return int(x)
            except (ValueError, TypeError):
                return float('inf')
        
        def edge_sort_key(edge):
            """Sort key for edges based on numeric node IDs."""
            try:
                return (int(edge[0]), int(edge[1]))
            except (ValueError, TypeError):
                return (float('inf'), float('inf'))
        
        # Add all nodes (sorted)
        for node in sorted(self.graph.nodes, key=numeric_sort_key):
            self.element_queue.append(('node', node))
            self.all_elements.append(('node', node))
            
        # Add all edges (sorted)
        for edge in sorted(self.graph.edges, key=edge_sort_key):
            self.element_queue.append(('edge', edge))
            self.all_elements.append(('edge', edge))
            
        # Get first element
        if self.element_queue:
            self._select_next_element()
    
    
    def _reset_mapping_state(self):
        """Reset all mapping state when switching modes to fresh initial state."""
        try:
            # Ensure we have a valid map_widget
            if not hasattr(self, 'map_widget') or not self.map_widget:
                print("Warning: map_widget missing during reset, attempting to find it")
                for widget in self.findChildren(MapWidget):
                    self.map_widget = widget
                    break
                else:
                    print("Error: Could not find MapWidget during reset")
                    return
            
            # Clear all visual elements from map widget (this handles grid, contours, mappings)
            if hasattr(self, 'map_widget') and self.map_widget is not None:
                try:
                    # Check if widget still exists
                    self.map_widget.isVisible()
                    self.map_widget.reset_all()
                except RuntimeError:
                    # Widget deleted, skip this operation
                    pass
            
            # Reset core mapping data
            self.mapping = SpatialMapping(self.graph)
            self.mapped_elements.clear()
            self.mapping_history.clear()
            self.element_queue.clear()
            self.current_element = None
            
            # Clear graph highlights
            if hasattr(self, 'graph_widget') and self.graph_widget:
                self.graph_widget.clear_highlights()
            
            # Reset grid-specific state
            if hasattr(self, 'grid_placed'):
                self.grid_placed = False
            if hasattr(self, 'selected_cells'):
                self.selected_cells.clear()
            
            # Reset manual mode specific state
            if hasattr(self, 'contour_list') and self.contour_list:
                self.contour_list.clear()
            if hasattr(self, 'highlighted_contour_id'):
                self.highlighted_contour_id = None
                
            # Reset grid mode specific state
            if hasattr(self, 'grid_assignment_list') and self.grid_assignment_list:
                self.grid_assignment_list.clear()
            if hasattr(self, 'map_widget') and self.map_widget:
                self.map_widget.highlighted_cells.clear()
            # Clear highlighted assignment tracking
            self.highlighted_assignment_cells = None
            
            # Reset button states
            if hasattr(self, 'assign_button'):
                self.assign_button.setEnabled(False)
            if hasattr(self, 'delete_grid_assignment_button'):
                self.delete_grid_assignment_button.setEnabled(False)
            if hasattr(self, 'place_grid_button'):
                self.place_grid_button.setEnabled(True)
            if hasattr(self, 'clear_contour_button'):
                self.clear_contour_button.setEnabled(False)
            if hasattr(self, 'commit_contour_button'):
                self.commit_contour_button.setEnabled(False)
                
            # Update progress display
            self._update_progress()
            
            # Clear status bar
            if hasattr(self, 'status_bar'):
                self.status_bar.clearMessage()
                
        except Exception as e:
            print(f"Error in _reset_mapping_state: {e}")
            import traceback
            traceback.print_exc()
    
    
    def _set_mode_specific_defaults(self, mode: str):
        """Set mode-specific default visualization options."""
        if mode == 'grid':
            # Grid mode: Show mappings and labels, adaptive font enabled
            self.show_mappings_checkbox.setChecked(True)
            self.show_labels_checkbox.setChecked(True) 
            self.adaptive_font_checkbox.setChecked(True)
            # Sync map widget flag with checkbox
            self.map_widget.show_all_mappings = True
            # Hide test mode widgets
            for widget in self.test_mode_widgets:
                widget.hide()
                
        elif mode == 'manual':
            # Manual mode: Show mappings and labels, adaptive font enabled
            self.show_mappings_checkbox.setChecked(True)
            self.show_labels_checkbox.setChecked(True)
            self.adaptive_font_checkbox.setChecked(True)
            # Sync map widget flag with checkbox
            self.map_widget.show_all_mappings = True
            # Hide test mode widgets
            for widget in self.test_mode_widgets:
                widget.hide()
                
        elif mode == 'test':
            # Test mode: Start clean (unchecked) for testing clarity
            self.show_mappings_checkbox.setChecked(False)
            self.show_labels_checkbox.setChecked(False)
            self.adaptive_font_checkbox.setChecked(False)
            # Sync map widget flag with checkbox
            self.map_widget.show_all_mappings = False
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
            # Try to get next from all_elements
            if hasattr(self, 'all_elements') and self.current_element:
                try:
                    current_idx = self.all_elements.index(self.current_element)
                    if current_idx + 1 < len(self.all_elements):
                        self.current_element = self.all_elements[current_idx + 1]
                        self._set_current_element_and_highlight()
                        return
                except ValueError:
                    pass
            self.status_bar.showMessage("No more elements")
            return
            
        self.current_element = self.element_queue.pop(0)
        self._set_current_element_and_highlight()
        
    def _select_previous_element(self):
        """Select the previous element to map."""
        if not hasattr(self, 'all_elements') or not self.all_elements:
            self.status_bar.showMessage("No elements available")
            return
            
        if self.current_element is None:
            self.current_element = self.all_elements[-1]
        else:
            try:
                current_idx = self.all_elements.index(self.current_element)
                if current_idx > 0:
                    self.current_element = self.all_elements[current_idx - 1]
                    # Add back to queue if it was removed
                    if self.current_element not in self.element_queue:
                        self.element_queue.insert(0, self.current_element)
                else:
                    self.status_bar.showMessage("Already at first element")
                    return
            except ValueError:
                self.current_element = self.all_elements[-1]
                
        self._set_current_element_and_highlight()
        
    def _set_current_element_and_highlight(self):
        """Set current element and update UI highlighting."""
        if not self.current_element:
            return
            
        elem_type, elem_id = self.current_element
        
        # Update map widget with current element
        self.map_widget.set_current_element(elem_type, elem_id)
        
        # Update UI based on mode
        if self.setup_mode == 'grid':
            self.assign_button.setEnabled(True)
            
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
        
    def _on_place_grid(self):
        """Start or cancel grid placement mode."""
        if self.place_grid_button.isChecked():
            # Button was pressed - first apply current config, then start placement mode
            config = GridConfig(
                structure_type='rectangle',  # Fixed to rectangle for now
                rows=self.rows_spin.value(),
                cols=self.cols_spin.value(),
                cell_width=self.width_spin.value(),
                cell_height=self.width_spin.value()  # Use same value for square cells
            )
            self.map_widget.set_grid_config(config)
            
            # Now start placement mode
            self.map_widget.set_interaction_mode('place_grid')
            self.status_bar.showMessage("Click on the map to place grid origin")
        else:
            # Button was unpressed - cancel placement mode
            self.map_widget.set_interaction_mode('none')
            self.status_bar.showMessage("Grid placement cancelled")
        
    def _on_grid_placed(self, x: float, y: float):
        """Handle grid placement."""
        self.grid_status_label.setText(f"Origin: ({x:.0f}, {y:.0f})")
        self.status_bar.showMessage("Grid placed - Ready to map elements")
        # Uncheck the Place Grid button after placement
        self.place_grid_button.setChecked(False)
        # Enable Clear Grid button
        self.clear_grid_button.setEnabled(True)
        
        # Automatically start with the first element
        if self.all_elements and len(self.all_elements) > 0:
            # Set the first element as current
            first_element = self.all_elements[0]
            elem_type, elem_id = first_element
            
            # Set current element
            self.current_element = first_element
            
            # Update combo box selection
            self.grid_element_combo.setCurrentIndex(0)
            
            # Set interaction mode for cell selection
            self.map_widget.set_interaction_mode('select_cells')
            
            # Update map widget
            self.map_widget.set_current_element(elem_type, elem_id)
            
            # Highlight in graph 
            self.graph_widget.clear_highlights()
            if elem_type == 'node':
                self.graph_widget.highlight_nodes({elem_id})
            elif elem_type == 'edge':
                self.graph_widget.highlight_edges({elem_id})
            
            # Update status
            self.status_bar.showMessage(f"Ready to map {elem_type} {elem_id} - Click cells to select")
        
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
            
            # Clear grid assignment list
            if hasattr(self, 'grid_assignment_list') and self.grid_assignment_list:
                self.grid_assignment_list.clear()
            
            # Clear highlighted assignment tracking
            self.highlighted_assignment_cells = None
            
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
            if hasattr(self, 'delete_grid_assignment_button'):
                self.delete_grid_assignment_button.setEnabled(False)
            
            # Update progress
            self._update_progress_display()
            
            self.status_bar.showMessage("Grid cleared - all mappings reset")
        
    def _on_cell_clicked(self, cell_id: str):
        """Handle cell click."""
        if cell_id in self.map_widget.selected_cells:
            self.status_bar.showMessage(f"Selected cell {cell_id}")
        else:
            self.status_bar.showMessage(f"Deselected cell {cell_id}")
            
        # Update assign button state based on whether we have selected cells and current element
        if hasattr(self, 'assign_button'):
            has_selected_cells = bool(self.map_widget.selected_cells)
            has_current_element = bool(self.current_element)
            self.assign_button.setEnabled(has_selected_cells and has_current_element)
            
    def _on_assign_cells(self):
        """Assign selected cells to current element."""
        if not self.current_element or not self.map_widget.selected_cells:
            return
            
        elem_type, elem_id = self.current_element
        
        # Track assigned cells for the assignment list
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
        
        # Add to grid assignment list
        cell_names = ", ".join(sorted(assigned_cells))
        list_label = f"{elem_type.title()} {elem_id}: {cell_names}"
        item = QListWidgetItem(list_label)
        # Store assignment data for later reference
        item.setData(Qt.UserRole, {
            'element': self.current_element,
            'cells': assigned_cells,
            'regions': regions_added
        })
        self.grid_assignment_list.addItem(item)
        
        # Clear selection and automatically advance to next element
        self.map_widget.clear_selection()
        self._select_next_element()
        self._update_progress_display()
        
    def _on_next_element(self):
        """Skip to next element."""
        self._select_next_element()
        self._update_progress_display()
    
    def _on_previous_element(self):
        """Go back to previous element."""
        self._select_previous_element()
        self._update_progress_display()
    
    def _on_enter_key(self):
        """Handle Enter key for assign/commit in both modes."""
        if self.setup_mode == 'grid':
            # In grid mode: assign selected cells to current element
            if (self.current_element and 
                hasattr(self.map_widget, 'selected_cells') and 
                self.map_widget.selected_cells):
                self._on_assign_cells()
        elif self.setup_mode == 'manual':
            # In manual mode: commit current contour if drawing
            if (hasattr(self.map_widget, 'current_contour') and 
                self.map_widget.current_contour and 
                len(self.map_widget.current_contour) > 2):
                # Complete contour using the existing method
                self.map_widget.finish_current_contour()
        
    def _on_grid_assignment_selected(self, item):
        """Handle grid assignment selection with toggle support."""
        if item:
            data = item.data(Qt.UserRole)
            if data and 'cells' in data:
                # Toggle behavior: if clicking the same assignment, deselect it
                current_highlight = getattr(self, 'highlighted_assignment_cells', None)
                if current_highlight == data['cells']:
                    # Deselect: clear highlight and list selection
                    self.highlighted_assignment_cells = None
                    self.grid_assignment_list.clearSelection()
                    self.delete_grid_assignment_button.setEnabled(False)
                    # Clear highlights
                    if hasattr(self.map_widget, 'highlighted_cells'):
                        self.map_widget.highlighted_cells.clear()
                        self.map_widget.update()
                else:
                    # Select: highlight this assignment in yellow
                    self.highlighted_assignment_cells = data['cells']
                    self.delete_grid_assignment_button.setEnabled(True)
                    # Clear previous highlights and set new ones
                    if hasattr(self.map_widget, 'highlighted_cells'):
                        self.map_widget.highlighted_cells.clear()
                    else:
                        self.map_widget.highlighted_cells = set()
                    
                    # Highlight the cells for this assignment
                    self.map_widget.highlighted_cells.update(data['cells'])
                    self.map_widget.update()
        else:
            self.delete_grid_assignment_button.setEnabled(False)
            
    def _on_delete_grid_assignment(self):
        """Delete selected assignment."""
        current_item = self.grid_assignment_list.currentItem()
        if not current_item:
            return
            
        # Get assignment data
        data = current_item.data(Qt.UserRole)
        if not data:
            return
            
        element = data['element']
        cells = data['cells']
        regions = data['regions']
        
        # Remove regions from mapping
        for region in regions:
            try:
                self.mapping.remove_region(region.region_id)
            except Exception as e:
                print(f"Warning: Could not remove region {region.region_id}: {e}")
        
        # Remove cell mappings from map widget
        for cell_id in cells:
            self.map_widget.remove_cell_mapping(cell_id)
        
        # Remove from mapped elements tracking
        if element in self.mapped_elements:
            del self.mapped_elements[element]
        
        # Remove from list
        row = self.grid_assignment_list.row(current_item)
        self.grid_assignment_list.takeItem(row)
        
        # Clear highlights
        if hasattr(self.map_widget, 'highlighted_cells'):
            self.map_widget.highlighted_cells.clear()
            self.map_widget.update()
        
        # Clear highlighted assignment tracking
        self.highlighted_assignment_cells = None
        
        # Update button state
        self.delete_grid_assignment_button.setEnabled(False)
        
        # Update progress
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
            # Toggle behavior: if clicking the same region, deselect it
            current_highlight = getattr(self.map_widget, 'highlighted_region_id', None)
            if current_highlight == region_id:
                # Deselect: clear highlight and list selection
                self.map_widget.highlighted_region_id = None
                self.contour_list.clearSelection()
                self.delete_contour_button.setEnabled(False)
                self.map_widget.update()
            else:
                # Select: highlight this region in yellow
                self.map_widget.highlighted_region_id = region_id
                self.delete_contour_button.setEnabled(True)
                self.map_widget.update()
        
    def _on_delete_contour(self):
        """Delete selected contour."""
        current_item = self.contour_list.currentItem()
        if current_item:
            region_id = current_item.data(Qt.UserRole)
            
            # Remove from mapping
            self.mapping.remove_region(region_id)
            
            # Clear highlighting
            self.map_widget.highlighted_region_id = None
            
            # Remove from list
            self.contour_list.takeItem(self.contour_list.row(current_item))
            self.delete_contour_button.setEnabled(False)
            
            # Update map to reflect removal
            self.map_widget.update()
            self.status_bar.showMessage(f"Deleted region {region_id}")
            
            self._update_progress_display()
        
    def _on_contour_drawn(self, points: List[Tuple[float, float]]):
        """Handle completed contour - add directly to mapping."""
        if self.current_element:
            # We have an element selected - add directly to mapping only
            elem_type, elem_id = self.current_element
            
            # Generate unique region ID based on existing regions for this element
            if elem_type == 'node':
                existing_regions = len(self.mapping.get_node_regions(elem_id))
            else:
                existing_regions = len(self.mapping.get_edge_regions(elem_id))
            region_id = f"{elem_type}_{elem_id}_contour_{existing_regions}"
            
            # Create region and add to mapping (single source of truth)
            from .regions import ContourRegion
            import numpy as np
            # Ensure points is a numpy array
            points_array = np.array(points) if not isinstance(points, np.ndarray) else points
            region = ContourRegion(region_id=region_id, contour_points=points_array)
            
            if elem_type == 'node':
                self.mapping.add_node_region(region, elem_id)
            elif elem_type == 'edge':
                self.mapping.add_edge_region(region, elem_id)
            
            # Add to contour list widget for UI reference
            list_label = f"{elem_type.title()} {elem_id}: {region_id}"
            item = QListWidgetItem(list_label)
            item.setData(Qt.UserRole, region_id)  # Store region_id for reference
            self.contour_list.addItem(item)
            
            # Force map update to show the new region from mapping
            self.map_widget.update()
            self.status_bar.showMessage(f"Contour assigned to {elem_type} {elem_id}")
            
        else:
            # No element selected - just notify user to select element first
            self.status_bar.showMessage("Please select an element first to assign the contour")
                
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
        
    def _validate_builder_compatibility(self, loaded_mapping):
        """Validate that loaded mapping is compatible with current graph structure.
        
        Args:
            loaded_mapping: The mapping loaded from file
            
        Raises:
            ValueError: If builders are incompatible
        """
        if not self.graph or not loaded_mapping.graph:
            return
            
        current_metadata = self.graph.metadata
        loaded_metadata = loaded_mapping.graph.metadata
        
        current_builder = current_metadata.get('builder_type', 'unknown')
        loaded_builder = loaded_metadata.get('builder_type', 'unknown')
        
        # Check if builder types match
        if current_builder != loaded_builder:
            raise ValueError(
                f"Graph builder mismatch!\n\n"
                f"Current config uses: '{current_builder}'\n"
                f"Loaded mapping uses: '{loaded_builder}'\n\n"
                f"The mapping was created with a different graph structure.\n"
                f"Please use a config with the same builder type, or create a new mapping."
            )
        
        # For same builder types, check if key parameters match
        current_params = current_metadata.get('parameters', {})
        loaded_params = loaded_metadata.get('parameters', {})
        
        # Check critical parameters that affect graph structure
        critical_params = ['height', 'n_nodes', 'filepath', 'graph_type']  # Add more as needed
        mismatched_params = []
        
        for param in critical_params:
            if param in current_params and param in loaded_params:
                if current_params[param] != loaded_params[param]:
                    mismatched_params.append(
                        f"  {param}: current='{current_params[param]}', mapping='{loaded_params[param]}'"
                    )
        
        if mismatched_params:
            param_details = '\n'.join(mismatched_params)
            raise ValueError(
                f"Graph parameters mismatch!\n\n"
                f"Builder type: '{current_builder}'\n"
                f"Parameter differences:\n{param_details}\n\n"
                f"The mapping was created with different graph parameters.\n"
                f"Please use a config with matching parameters, or create a new mapping."
            )
        
        # Final check: compare actual node sets
        current_nodes = set(self.graph.graph.nodes())
        loaded_nodes = set(loaded_mapping.get_mapped_nodes())
        missing_nodes = loaded_nodes - current_nodes
        
        if missing_nodes:
            raise ValueError(
                f"Graph structure mismatch!\n\n"
                f"Mapping contains nodes that don't exist in current graph:\n"
                f"Missing nodes: {sorted(list(missing_nodes))}\n\n"
                f"Current graph nodes: {sorted(list(current_nodes))}\n"
                f"Mapped nodes: {sorted(list(loaded_nodes))}\n\n"
                f"This usually means the graphs have different sizes or structures.\n"
                f"Please verify your graph configuration matches the saved mapping."
            )
        
    def _update_unmapped_list(self):
        """Update the list of unmapped elements (no longer needed in simplified UI)."""
        # This method is kept for backward compatibility but no longer updates a list
        pass
            
    def _on_save_mapping(self):
        """Save the current mapping using new self-contained format."""
        if not self.mapping:
            QMessageBox.warning(self, "Warning", "No mapping to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mapping", "", 
            "Pickle files (*.pkl);;JSON files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                from pathlib import Path
                self.mapping.save_with_builder_info(Path(file_path))
                QMessageBox.information(self, "Success", f"Mapping saved to {file_path}")
                self.status_bar.showMessage(f"Mapping saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save mapping: {str(e)}")
                
    def _on_load_mapping(self):
        """Load a mapping using new self-contained format."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Mapping", "",
            "Mapping files (*.pkl *.json);;Pickle Files (*.pkl);;JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                from pathlib import Path
                # Load mapping data but preserve original graph structure
                loaded_mapping = SpatialMapping.load_with_builder_reconstruction(Path(file_path))
                
                # Validate builder compatibility before attempting to copy mappings
                self._validate_builder_compatibility(loaded_mapping)
                
                # IMPORTANT: Preserve the original graph from config, only load contour mappings
                # This ensures the graph structure (e.g., height=7) stays as configured
                # regardless of which nodes were actually mapped in the file
                
                # Clear current mapping but keep original graph
                self.mapping.clear_all_mappings()
                
                # Copy only the contour mappings to preserve graph structure
                for node_id in loaded_mapping.get_mapped_nodes():
                    regions = loaded_mapping.get_node_regions(node_id)
                    for region in regions:
                        self.mapping.add_node_region(region, node_id)
                
                for edge in loaded_mapping.get_mapped_edges():
                    regions = loaded_mapping.get_edge_regions(edge)
                    for region in regions:
                        self.mapping.add_edge_region(region, edge)
                
                # Update UI to reflect loaded mapping
                self._update_ui_from_mapping()
                
                QMessageBox.information(self, "Success", f"Mapping loaded from {file_path}")
                self.status_bar.showMessage(f"Mapping loaded from {file_path}")
                self._update_progress()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load mapping: {str(e)}")
                
    def _update_ui_from_mapping(self):
        """Update UI to reflect loaded mapping with new self-contained format."""
        # Clear existing visualizations
        self.map_widget.clear_selection()
        self.map_widget.clear_contours()
        
        # Update builder selection if available (skip in test mode)
        if (self.graph and hasattr(self.graph, 'metadata') and 
            hasattr(self, 'setup_mode') and self.setup_mode != 'test'):
            builder_type = self.graph.metadata.get('builder_type', '')
            if builder_type and hasattr(self, 'builder_combo'):
                # Find and select the appropriate builder in the dropdown
                for i in range(self.builder_combo.count()):
                    if self.builder_combo.itemText(i) == builder_type:
                        self.builder_combo.setCurrentIndex(i)
                        self._update_builder_config()
                        break
        
        # In test mode, update the graph widget with the new graph
        if (hasattr(self, 'setup_mode') and self.setup_mode == 'test' and 
            hasattr(self, 'test_graph_widget')):
            # Update the test graph widget with the new graph
            self.test_graph_widget.graph = self.graph
            self.test_graph_widget.draw_graph()
        
        # Set flag for paintEvent to draw mappings if checkbox is checked
        if hasattr(self, 'show_all_mappings_checkbox'):
            self.map_widget.show_all_mappings = self.show_all_mappings_checkbox.isChecked()
            self.map_widget.update()
    
    def _visualize_loaded_mapping(self):
        """Visualize a loaded mapping on the map widget - same widget for all modes."""
        if not self.mapping:
            return
            
        # Always use the same map_widget
        target_widget = self.map_widget
            
        # Clear existing visualizations first
        target_widget.clear_contours()
        target_widget.clear_selection()
        
        # ALL mappings are treated the same - they're just contour points
        # Whether from grid or manual mode, they're all stored as contour points
        
        # Visualize node regions
        for node_id in self.mapping.get_mapped_nodes():
            regions = self.mapping.get_node_regions(node_id)
            color = QColor(150, 255, 150, 100)  # Light green
            
            for region in regions:
                # Extract contour points - works for both RectangleRegion and ContourRegion
                if isinstance(region, RectangleRegion):
                    # Convert rectangle to contour points (4 corners)
                    points = [
                        (region.x, region.y),
                        (region.x + region.width, region.y),
                        (region.x + region.width, region.y + region.height),
                        (region.x, region.y + region.height)
                    ]
                elif isinstance(region, ContourRegion):
                    # Already contour points
                    points = [(float(p[0]), float(p[1])) for p in region.contour_points]
                else:
                    continue
                
                # Add as contour - same method for all
                # Use the actual region ID for highlighting to work
                target_widget.add_contour(points, region.region_id, 'node', node_id, color)
        
        # Visualize edge regions - exact same approach
        for edge in self.mapping.get_mapped_edges():
            regions = self.mapping.get_edge_regions(edge)
            color = QColor(255, 165, 0, 100)  # Orange
            
            for region in regions:
                # Extract contour points - works for both RectangleRegion and ContourRegion
                if isinstance(region, RectangleRegion):
                    # Convert rectangle to contour points (4 corners)
                    points = [
                        (region.x, region.y),
                        (region.x + region.width, region.y),
                        (region.x + region.width, region.y + region.height),
                        (region.x, region.y + region.height)
                    ]
                elif isinstance(region, ContourRegion):
                    # Already contour points
                    points = [(float(p[0]), float(p[1])) for p in region.contour_points]
                else:
                    continue
                
                # Add as contour - same method for all
                # Use the actual region ID for highlighting to work
                target_widget.add_contour(points, region.region_id, 'edge', edge, color)
        
        # Force update
        target_widget.update()
    
    def _on_save_intermediate_mapping(self):
        """Save current mapping with complete state using new self-contained format."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save Intermediate Mapping", "", "Mapping Files (*.pkl);;JSON Files (*.json);;All Files (*)")
        
        if file_path:
            try:
                # Create setup mode state for GUI continuation
                setup_mode_state = {
                    'current_mode': getattr(self, 'setup_mode', 'grid'),
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
                    'element_queue_index': getattr(self, 'element_queue_index', 0),
                    'mapping_history': len(self.mapping_history) if hasattr(self, 'mapping_history') else 0
                }
                
                # Use the new self-contained save method
                from pathlib import Path
                self.mapping.save_with_builder_info(Path(file_path), setup_mode_state)
                
                self.status_bar.showMessage(f"Mapping saved to {file_path}")
                QMessageBox.information(self, "Success", f"Mapping saved successfully to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save mapping: {str(e)}")
    
    def _on_load_intermediate_mapping(self):
        """Load mapping with complete state from file using new self-contained format."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Load Intermediate Mapping", "", 
            "Mapping Files (*.pkl);;JSON Files (*.json);;All Files (*)")
        
        if file_path:
            try:
                from pathlib import Path
                # Load mapping data but preserve original graph structure
                loaded_mapping = SpatialMapping.load_with_builder_reconstruction(Path(file_path))
                
                # Validate builder compatibility before attempting to copy mappings
                self._validate_builder_compatibility(loaded_mapping)
                
                # IMPORTANT: Preserve the original graph from config, only load contour mappings
                # This ensures the graph structure (e.g., height=7) stays as configured
                
                # Clear current mapping but keep original graph
                self.mapping.clear_all_mappings()
                
                # Copy only the contour mappings to preserve graph structure
                for node_id in loaded_mapping.get_mapped_nodes():
                    regions = loaded_mapping.get_node_regions(node_id)
                    for region in regions:
                        self.mapping.add_node_region(region, node_id)
                
                for edge in loaded_mapping.get_mapped_edges():
                    regions = loaded_mapping.get_edge_regions(edge)
                    for region in regions:
                        self.mapping.add_edge_region(region, edge)
                
                # Get setup mode state for GUI continuation
                setup_mode_state = loaded_mapping.get_setup_mode_state()
                
                # Restore setup mode state if available
                if setup_mode_state:
                    self._restore_setup_mode_state(setup_mode_state)
                
                # Update UI to reflect loaded mapping
                self._update_ui_from_mapping()
                
                self.status_bar.showMessage(f"Mapping loaded from {file_path}")
                QMessageBox.information(self, "Success", f"Mapping loaded successfully from {file_path}")
                self._update_progress()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load mapping: {str(e)}")
    
    def _restore_setup_mode_state(self, setup_mode_state: dict):
        """Restore GUI state from loaded setup mode configuration."""
        # Restore grid configuration if present
        if 'grid_config' in setup_mode_state:
            grid_config = setup_mode_state['grid_config']
            self.map_widget.grid_config.structure_type = grid_config.get('structure_type', 'rectangle')
            self.map_widget.grid_config.rows = grid_config.get('rows', 8)
            self.map_widget.grid_config.cols = grid_config.get('cols', 8)
            self.map_widget.grid_config.cell_width = grid_config.get('cell_width', 50.0)
            self.map_widget.grid_config.cell_height = grid_config.get('cell_height', 50.0)
            self.map_widget.grid_config.origin_x = grid_config.get('origin_x', 0.0)
            self.map_widget.grid_config.origin_y = grid_config.get('origin_y', 0.0)
            
            # Update UI controls with loaded grid config
            if hasattr(self, '_update_grid_ui_from_config'):
                self._update_grid_ui_from_config()
        
        # Restore grid if it was enabled (but not in test mode)
        if setup_mode_state.get('grid_enabled', False) and getattr(self, 'setup_mode', None) != 'test':
            self.map_widget.enable_grid(
                self.map_widget.grid_config.origin_x,
                self.map_widget.grid_config.origin_y
            )
            # Update grid UI elements
            if hasattr(self, '_update_grid_status_after_load'):
                self._update_grid_status_after_load()
            # Set interaction mode to allow cell selection
            self.map_widget.set_interaction_mode('select_cells')
        
        # Restore element queue index for grid mode continuation
        if 'element_queue_index' in setup_mode_state:
            self.element_queue_index = setup_mode_state['element_queue_index']
    
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
            map_widget = self.map_widget
        else:
            map_widget = self.map_widget
            
        # Clear only selections, not mappings
        map_widget.selected_cells.clear()
        map_widget.completed_contours.clear()
        
        # Clear contour list if it exists (manual mode)
        if hasattr(self, 'contour_list') and self.contour_list is not None:
            self.contour_list.clear()
        
        # Restore all mapped regions using new API
        from .regions import ContourRegion, RectangleRegion
        
        # First restore node regions
        for node_id in self.mapping.get_mapped_nodes():
            regions = self.mapping.get_node_regions(node_id)
            color = QColor(0, 255, 0, 100)  # Green for nodes
            
            for region in regions:
                self._restore_region_visualization(region, 'node', node_id, color, map_widget)
        
        # Then restore edge regions
        for edge_id in self.mapping.get_mapped_edges():
            regions = self.mapping.get_edge_regions(edge_id)
            color = QColor(255, 165, 0, 100)  # Orange for edges
            
            for region in regions:
                self._restore_region_visualization(region, 'edge', edge_id, color, map_widget)
    
    def _restore_region_visualization(self, region, element_type, element_id, color, map_widget):
        """Helper method to restore visualization of a single region."""
        from .regions import ContourRegion, RectangleRegion
        
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
            region_id = region.region_id
            map_widget.completed_contours.append(
                (region.contour_points, region_id, contour_color, element_type, element_id)
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
                # Enable assign button only if we have selected cells
                has_selected_cells = bool(self.map_widget.selected_cells)
                self.assign_button.setEnabled(has_selected_cells)
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
            # Same logic for all modes since we use the same map_widget
            if show_mappings:
                # Clear test selections first (if in test mode)
                if hasattr(self, 'setup_mode') and self.setup_mode == 'test':
                    self._clear_test_selections()
                
                # Enable showing all mappings (paintEvent will draw from mapping directly)
                self.map_widget.show_all_mappings = True
                self.map_widget.update()
            else:
                # Hide all mappings by setting flag to False
                self.map_widget.show_all_mappings = False
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
    
    def _on_toggle_unified_font(self, state):
        """Toggle unified font sizing."""
        try:
            unified_font = (state == Qt.Checked)
            self.map_widget.unified_font_size = unified_font
            self.map_widget.update()
            
            # Also update test mode widget if it exists  
            if hasattr(self, 'test_map_widget'):
                self.test_map_widget.unified_font_size = unified_font
                self.test_map_widget.update()
        except Exception as e:
            print(f"Error toggling unified font: {e}")
    
    def _on_test_layout_changed(self):
        """Handle test mode layout orientation change."""
        if self.setup_mode == 'test':
            # Re-setup the test layout with new orientation
            self._setup_test_layout()
    
    def _clear_mappings_only(self):
        """Clear only mappings, preserve grid placement."""
        # Clear mapping state but keep grid
        self.mapping = SpatialMapping(self.graph)
        self.mapped_elements.clear()
        self.mapping_history.clear()
        
        # Clear grid assignment list
        if hasattr(self, 'grid_assignment_list') and self.grid_assignment_list:
            self.grid_assignment_list.clear()
        
        # Clear highlighted assignment tracking  
        self.highlighted_assignment_cells = None
        
        # Clear cell mappings but preserve grid structure
        if hasattr(self.map_widget, 'cell_mappings'):
            self.map_widget.cell_mappings.clear()
        if hasattr(self.map_widget, 'cell_colors'):
            self.map_widget.cell_colors.clear()
        if hasattr(self.map_widget, 'selected_cells'):
            self.map_widget.selected_cells.clear()
        if hasattr(self.map_widget, 'highlighted_cells'):
            self.map_widget.highlighted_cells.clear()
            
        # Reset element queue to start fresh
        self._init_element_queue()
        self._populate_element_combos()
        
        # Clear graph highlights
        self.graph_widget.clear_highlights()
        
        # Update UI buttons
        self.assign_button.setEnabled(False)
        if hasattr(self, 'delete_grid_assignment_button'):
            self.delete_grid_assignment_button.setEnabled(False)
        
        # Update the map display
        self.map_widget.update()
        self._update_progress_display()

    def _on_clear_all(self):
        """Clear all mappings but preserve grid placement."""
        reply = QMessageBox.question(self, "Clear All Mappings", 
                                    "Are you sure you want to clear all mappings? The grid placement will be preserved.",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._clear_mappings_only()
            self.status_bar.showMessage("All mappings cleared - grid preserved")
    
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
                        
                # Remove from mapping using new API
                for region_id in last_action.get('regions', []):
                    self.mapping.remove_region(region_id)
                                
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
            # Clear all regions from mapping
            # Option 1: Clear all regions completely (create new clean mapping)
            self.mapping.clear_all_mappings()
            
            # Clear from list widget
            self.contour_list.clear()
            
            # Clear any highlights
            self.map_widget.highlighted_region_id = None
            
            # Update display
            self.map_widget.update()
            self.status_bar.showMessage("Cleared all contours")
            self._update_progress_display()
    
    def _on_test_mapping(self):
        """Launch test mode for current mapping."""
        try:
            # Save current mapping to temp file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
            temp_path = temp_file.name
            temp_file.close()
            
            from pathlib import Path
            try:
                self.mapping.save_with_builder_info(Path(temp_path))
                success = True
            except Exception:
                success = False
            
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
        """Setup test mode layout by rearranging existing fresh widgets."""
        try:
            # Find the views widget created by _recreate_fresh_ui()
            main_layout = self.centralWidget().layout()
            views_widget = None
            for i in range(main_layout.count()):
                item = main_layout.itemAt(i)
                if item and item.widget() and item.widget() != self.control_panel:
                    views_widget = item.widget()
                    break
            
            if views_widget:
                # Remove the views widget from main layout (but don't delete it)
                main_layout.removeWidget(views_widget)
                views_widget.hide()
            
            # Create fresh test splitter (the old one was deleted in _recreate_fresh_ui)
            self.test_splitter = QSplitter()
            self.test_splitter.setParent(self.centralWidget())
            
            # Move the existing fresh widgets into test splitter
            # Remove them from their current parents first
            if hasattr(self, 'graph_widget') and self.graph_widget is not None:
                self.graph_widget.setParent(None)
            if hasattr(self, 'map_widget') and self.map_widget is not None:
                self.map_widget.setParent(None)
            
            # Determine orientation
            is_vertical = self.test_layout_vertical.isChecked()
            
            # Set orientation and add the existing fresh widgets with safety checks
            try:
                if is_vertical:
                    self.test_splitter.setOrientation(Qt.Vertical)
                    # Graph on top, map on bottom
                    if hasattr(self, 'graph_widget') and self.graph_widget is not None:
                        self.test_splitter.addWidget(self.graph_widget)
                    if hasattr(self, 'map_widget') and self.map_widget is not None:
                        self.map_widget.isVisible()  # Check if widget exists
                        self.test_splitter.addWidget(self.map_widget)
                else:
                    self.test_splitter.setOrientation(Qt.Horizontal)
                    # Map on left, graph on right
                    if hasattr(self, 'map_widget') and self.map_widget is not None:
                        self.map_widget.isVisible()  # Check if widget exists
                        self.test_splitter.addWidget(self.map_widget)
                    if hasattr(self, 'graph_widget') and self.graph_widget is not None:
                        self.test_splitter.addWidget(self.graph_widget)
            except RuntimeError:
                # Widget deleted during setup, skip layout
                return
            
            self.test_splitter.setSizes([1, 1])  # Equal sizes
            
            # Add test splitter to main layout
            main_layout = self.centralWidget().layout()
            main_layout.addWidget(self.test_splitter)
            self.test_splitter.show()
            
            # Connect test mode events
            if hasattr(self.map_widget, 'mousePressEvent') and not hasattr(self, 'original_mouse_press'):
                self.original_mouse_press = self.map_widget.mousePressEvent
            if hasattr(self.map_widget, 'mousePressEvent'):
                self.map_widget.mousePressEvent = self._on_test_map_click
                
        except Exception as e:
            print(f"Error in _setup_test_layout: {e}")
            import traceback
            traceback.print_exc()
        
    def _restore_normal_layout(self):
        """Restore the normal layout when leaving test mode."""
        try:
            if hasattr(self, 'original_views_widget'):
                # Hide test splitter and remove map_widget from it
                if hasattr(self, 'test_splitter'):
                    self.test_splitter.hide()
                    # Remove map_widget from test splitter safely
                    if self.map_widget.parent() == self.test_splitter:
                        self.map_widget.setParent(None)
                
                # Show original views widget
                self.original_views_widget.show()
                
                # Find the Interactive Mapping Area group box and restore map_widget to it
                map_group = None
                for child in self.original_views_widget.findChildren(QGroupBox):
                    if "Interactive Mapping Area" in child.title():
                        map_group = child
                        break
                
                if map_group:
                    map_layout = map_group.layout()
                    if not map_layout:
                        # Create the layout if it doesn't exist
                        print("Creating new layout for map group")
                        map_layout = QVBoxLayout()
                        map_layout.setContentsMargins(8, 20, 8, 8)
                        map_group.setLayout(map_layout)
                    
                    # Clear the layout first
                    while map_layout.count():
                        child = map_layout.takeAt(0)
                        if child.widget() and child.widget() != self.map_widget:
                            child.widget().setParent(None)
                    
                    # Check if map_widget is already in layout
                    widget_in_layout = False
                    for i in range(map_layout.count()):
                        if map_layout.itemAt(i).widget() == self.map_widget:
                            widget_in_layout = True
                            break
                    
                    # Add map_widget back if not already there
                    if not widget_in_layout:
                        map_layout.addWidget(self.map_widget)
                    print("Successfully restored map_widget to Interactive Mapping Area")
                else:
                    print("Error: Could not find Interactive Mapping Area group box")
                
                # Restore original mousePressEvent if it was stored
                if hasattr(self, 'original_mouse_press'):
                    self.map_widget.mousePressEvent = self.original_mouse_press
                    
        except Exception as e:
            print(f"Error in _restore_normal_layout: {e}")
            import traceback
            traceback.print_exc()
    
    def _ensure_normal_layout(self):
        """Ensure we're in normal layout (not test mode) and map_widget is properly positioned."""
        try:
            # If we have a test splitter that's visible, restore normal layout
            if hasattr(self, 'test_splitter') and self.test_splitter.isVisible():
                self._restore_normal_layout()
                return
            
            # Double check that map_widget is accessible and in the right place
            if not hasattr(self, 'map_widget') or not self.map_widget:
                print("Critical: map_widget missing, attempting to find it")
                # Try to find the map widget in the widget tree
                for widget in self.findChildren(MapWidget):
                    self.map_widget = widget
                    print(f"Found map_widget: {widget}")
                    break
                else:
                    print("Error: Could not find MapWidget in widget tree")
                    return
            
            # Find the Interactive Mapping Area and ensure map_widget is properly placed
            map_group = None
            for child in self.findChildren(QGroupBox):
                if "Interactive Mapping Area" in child.title():
                    map_group = child
                    break
            
            if map_group:
                map_layout = map_group.layout()
                if not map_layout:
                    # Create the layout if it doesn't exist
                    print("Creating new layout for map group in ensure_normal_layout")
                    map_layout = QVBoxLayout()
                    map_layout.setContentsMargins(8, 20, 8, 8)
                    map_group.setLayout(map_layout)
                
                # Check if map_widget is already in this layout
                widget_found = False
                for i in range(map_layout.count()):
                    if map_layout.itemAt(i).widget() == self.map_widget:
                        widget_found = True
                        break
                
                # If not found, add it
                if not widget_found:
                    # Remove from current parent if any
                    if self.map_widget.parent() and self.map_widget.parent() != map_group:
                        self.map_widget.setParent(None)
                    # Add to correct layout
                    map_layout.addWidget(self.map_widget)
                    print("Ensured map_widget is in Interactive Mapping Area")
            else:
                print("Error: Could not find Interactive Mapping Area in ensure_normal_layout")
                        
        except Exception as e:
            print(f"Error in _ensure_normal_layout: {e}")
            import traceback
            traceback.print_exc()
    
    def _recreate_fresh_ui(self):
        """Recreate the entire UI from scratch, like initial startup."""
        try:
            
            # Store current mapping data if any
            current_mapping = None
            if hasattr(self, 'mapping'):
                current_mapping = self.mapping
            
            # FIRST: Completely disconnect all signals from old widgets
            if hasattr(self, 'map_widget') and self.map_widget is not None:
                try:
                    # Disconnect all signals
                    self.map_widget.disconnect()
                    # Reset mouse event to original (remove wrapper)
                    if hasattr(self.map_widget, 'original_mouse_press'):
                        self.map_widget.mousePressEvent = self.map_widget.original_mouse_press
                except:
                    pass
            
            # Clear all widget references BEFORE layout operations  
            old_graph_widget = getattr(self, 'graph_widget', None)
            old_map_widget = getattr(self, 'map_widget', None)
            old_test_graph_widget = getattr(self, 'test_graph_widget', None)
            old_test_splitter = getattr(self, 'test_splitter', None)
            
            # Clear references immediately
            if hasattr(self, 'graph_widget'):
                delattr(self, 'graph_widget')
            if hasattr(self, 'map_widget'):
                delattr(self, 'map_widget')
            if hasattr(self, 'test_graph_widget'):
                delattr(self, 'test_graph_widget')
            if hasattr(self, 'test_splitter'):
                delattr(self, 'test_splitter')
            
            # Clear ALL mouse event references to prevent old event handlers
            if hasattr(self, '_original_map_mouse_press'):
                delattr(self, '_original_map_mouse_press')
            if hasattr(self, 'original_mouse_press'):
                delattr(self, 'original_mouse_press')
            
            # Clear the main splitter (not main_layout which should only contain main_splitter)
            control_panel_ref = self.control_panel
            
            # Remove old views_splitter from main_splitter (keep control_panel)
            old_views_splitter = getattr(self, 'views_splitter', None)
            if old_views_splitter and hasattr(self, 'main_splitter'):
                # Proper way to remove from splitter - just set parent to None
                if old_views_splitter.parent() == self.main_splitter:
                    old_views_splitter.setParent(None)
            
            # Safely delete old widgets
            def safe_delete_widget(widget):
                if widget is not None:
                    try:
                        widget.isVisible()  # Test if widget is valid
                        widget.deleteLater()
                    except RuntimeError:
                        pass  # Already deleted
            
            safe_delete_widget(old_graph_widget)
            safe_delete_widget(old_map_widget)
            safe_delete_widget(old_test_graph_widget)
            safe_delete_widget(old_test_splitter)
            safe_delete_widget(old_views_splitter)
            
            # Recreate the right panel (views) with vertical splitter
            self.views_splitter = QSplitter(Qt.Vertical)
            
            # Graph view (top)
            graph_group = QGroupBox("📊 Graph Structure")
            graph_group.setStyleSheet("QGroupBox { font-weight: 600; font-size: 14px; color: #424242; }")
            graph_layout = QVBoxLayout(graph_group)
            graph_layout.setContentsMargins(8, 20, 8, 8)
            
            # Recreate graph widget
            self.graph_widget = GraphWidget(self.graph)
            # Remove size constraints to allow flexible splitter resizing
            graph_layout.addWidget(self.graph_widget)
            
            self.views_splitter.addWidget(graph_group)
            
            # Map view (bottom) - recreate from scratch
            map_group = QGroupBox("🗺️ Interactive Mapping Area")
            map_group.setStyleSheet("QGroupBox { font-weight: 600; font-size: 14px; color: #424242; }")
            map_layout = QVBoxLayout(map_group)
            map_layout.setContentsMargins(8, 20, 8, 8)
            
            # Recreate map widget completely fresh
            self.map_widget = MapWidget(self.map_image)
            self.map_widget.gui_parent = self  # Allow map widget to access mapping
            # Remove minimum height to allow flexible splitter resizing
            map_layout.addWidget(self.map_widget)
            
            self.views_splitter.addWidget(map_group)
            
            # Set initial sizes for graph:map (1:2 ratio - map larger by default)
            self.views_splitter.setSizes([300, 600])
            self.views_splitter.setStretchFactor(0, 1)  # Graph
            self.views_splitter.setStretchFactor(1, 2)  # Map
            
            # Add views_splitter back to main_splitter (control_panel should still be there)
            self.main_splitter.addWidget(self.views_splitter)
            
            # Set proper default splitter sizes to restore good ratio
            self.main_splitter.setSizes([400, 1200])  # 1:3 ratio like original
            
            # Reconnect signals since we have new map_widget
            self._connect_signals()
            
            # Restore mapping if we had one
            if current_mapping:
                self.mapping = current_mapping
                # Ensure map widget gets the mapping reference
                self.map_widget.gui_parent = self
            
        except Exception as e:
            print(f"Error in _recreate_fresh_ui: {e}")
            import traceback
            traceback.print_exc()
    
            
    def _on_test_map_click(self, event):
        """Handle map click in test mode."""
        # Handle non-left clicks with original functionality (panning, etc.)
        if event.button() != Qt.LeftButton:
            if hasattr(self, 'original_mouse_press'):
                self.original_mouse_press(event)
            return
        
        # Get click position in image coordinates
        x = (event.x() - self.map_widget.offset_x) / self.map_widget.scale_factor
        y = (event.y() - self.map_widget.offset_y) / self.map_widget.scale_factor
        
        # Clear previous selection
        self._clear_test_selections()
        
        # Store click position for visual indicator
        self.test_selected_point = (x, y)
        
        # Store click position in map widget for drawing
        if hasattr(self, 'map_widget'):
            self.map_widget.test_click_position = (x, y)
            self.map_widget.update()
        
        # Find mapped element at this position
        element_found = self._find_element_at_position(x, y)
        
        if element_found:
            elem_type, elem_id = element_found
            self.test_selected_element = (elem_type, elem_id)
            
            # Highlight graph element with highlight color (yellow)
            self.graph_widget.clear_highlights()
            if elem_type == 'node':
                self.graph_widget.highlight_nodes({elem_id}, color='highlight')
                self.test_info_label.setText(f"Found: Node {elem_id}")
            else:
                self.graph_widget.highlight_edges({elem_id}, color='highlight')
                self.test_info_label.setText(f"Found: Edge {elem_id}")
        else:
            self.test_info_label.setText(f"No mapping at ({x:.0f}, {y:.0f})")
            
    def _on_test_node_click(self, node_id):
        """Handle node click in test mode."""
        # Clear previous selection
        self._clear_test_selections()
        
        self.test_selected_element = ('node', node_id)
        
        # Highlight node with selection color
        self.graph_widget.clear_highlights()
        self.graph_widget.highlight_nodes({node_id}, color='selected')
        
        # Highlight mapped regions on map
        self._highlight_element_regions('node', node_id)
        
        self.test_info_label.setText(f"Selected: Node {node_id}")
        
    def _on_test_edge_click(self, edge):
        """Handle edge click in test mode."""
        # Clear previous selection
        self._clear_test_selections()
        
        self.test_selected_element = ('edge', edge)
        
        # Highlight edge with selection color
        self.graph_widget.clear_highlights()
        self.graph_widget.highlight_edges({edge}, color='selected')
        
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
        
        # Add all nodes (sort numerically)
        def numeric_sort_key(x):
            """Sort key that handles numeric node IDs properly."""
            try:
                return int(x)
            except (ValueError, TypeError):
                return float('inf')  # Put non-numeric items at the end
        
        sorted_nodes = sorted(self.graph.nodes, key=numeric_sort_key)
        
        for node in sorted_nodes:
            self.test_element_combo.addItem(f"Node: {node}")
            
        # Add all edges (sort by both nodes numerically)
        def edge_sort_key(edge):
            """Sort key for edges based on numeric node IDs."""
            try:
                return (int(edge[0]), int(edge[1]))
            except (ValueError, TypeError):
                return (float('inf'), float('inf'))
        
        sorted_edges = sorted(self.graph.edges, key=edge_sort_key)
            
        for edge in sorted_edges:
            self.test_element_combo.addItem(f"Edge: {edge}")
        
    def _clear_test_selections(self):
        """Clear current test selections and highlights."""
        try:
            if hasattr(self, 'graph_widget'):
                self.graph_widget.clear_highlights()
            if hasattr(self, 'map_widget'):
                if hasattr(self.map_widget, 'clear_highlights'):
                    self.map_widget.clear_highlights()
                # Clear test highlight contours
                if hasattr(self.map_widget, 'test_highlight_contours'):
                    self.map_widget.test_highlight_contours = []
                # Clear click position indicator
                if hasattr(self.map_widget, 'test_click_position'):
                    self.map_widget.test_click_position = None
                self.map_widget.update()
            self.test_selected_point = None
            self.test_selected_element = None
        except Exception as e:
            print(f"Error clearing test selections: {e}")
            # Don't crash, reset state anyway
            self.test_selected_point = None
            self.test_selected_element = None
        
    def _find_element_at_position(self, x, y):
        """Find which graph element is mapped at the given position using mapping API."""
        if not self.mapping:
            return None
        
        # Use the SpatialMapping query_point method - clean and simple
        node_id, edge_id = self.mapping.query_point(x, y)
        if node_id is not None:
            return ('node', node_id)
        elif edge_id is not None:
            return ('edge', edge_id)
        else:
            return None
    
        
    def _highlight_element_regions(self, elem_type, elem_id):
        """Highlight all regions mapped to the given element and show status message."""
        if not hasattr(self, 'mapping') or self.mapping is None:
            self.status_bar.showMessage(f"No mapping loaded - cannot highlight {elem_type} {elem_id}")
            return
            
        try:
            if elem_type == 'node':
                region_objects = self.mapping.get_node_regions(elem_id)
            else:
                region_objects = self.mapping.get_edge_regions(elem_id)
            
            if not region_objects:
                # No regions found for this element
                self.status_bar.showMessage(f"Selected {elem_type} {elem_id} - no corresponding map region found")
                self.map_widget.clear_highlights()
            else:
                # Regions found - highlight them
                self.status_bar.showMessage(f"Selected {elem_type} {elem_id} - highlighting {len(region_objects)} region(s)")
                self.map_widget.highlight_element(elem_type, elem_id)
                
        except Exception as e:
            self.status_bar.showMessage(f"Error highlighting {elem_type} {elem_id}: {e}")
            print(f"Error highlighting {elem_type} {elem_id}: {e}")
    
    def calculate_font_scale(self):
        """Calculate font scale factor based on window size."""
        base_width = 1600  # Original design width
        current_width = self.width()
        
        # More conservative scaling: 0.85x to 1.15x (instead of 0.7x to 1.3x)
        # This prevents text from getting too large on big screens
        scale = max(0.85, min(1.15, current_width / base_width))
        
        # Further reduce scale if window is very wide (maximized)
        if current_width > 2000:
            scale = min(scale, 1.0)  # Don't go above normal size on very wide screens
        
        return scale
        
    def _set_initial_splitter_sizes(self):
        """Set initial splitter sizes based on current window size."""
        if hasattr(self, 'main_splitter'):
            total_width = self.width()
            # Use 25% for control panel, but cap at 450px and minimum 350px
            control_width = max(350, min(450, int(total_width * 0.25)))
            views_width = total_width - control_width - 20  # Account for splitter handle
            self.main_splitter.setSizes([control_width, views_width])

    def generate_scaled_stylesheet(self, scaled_sizes):
        """Generate stylesheet with scaled font sizes."""
        return f"""
            QMainWindow {{
                background-color: #fafafa;
            }}
            QGroupBox {{
                font-weight: 600;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                margin: 8px;
                padding-top: 20px;
                background-color: white;
                font-size: {scaled_sizes['title']}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 10px;
                color: #424242;
                background-color: white;
                font-size: {scaled_sizes['title']}px;
                font-weight: 600;
            }}
            QPushButton {{
                background-color: #ffffff;
                color: #424242;
                border: 2px solid #d0d0d0;
                padding: 8px 15px;
                border-radius: 6px;
                font-weight: 500;
                font-size: {scaled_sizes['normal']}px;
                text-align: center;
            }}
            QPushButton:hover {{
                background-color: #f5f5f5;
                border-color: #b0b0b0;
            }}
            QPushButton:pressed {{
                background-color: #e0e0e0;
            }}
            QPushButton:checked {{
                background-color: #e3f2fd;
                border-color: #2196f3;
                color: #1976d2;
                font-weight: 600;
            }}
            QLabel {{
                color: #424242;
                font-size: {scaled_sizes['normal']}px;
            }}
            QStatusBar {{
                background-color: #f5f5f5;
                border-top: 1px solid #e0e0e0;
                color: #616161;
                font-size: {scaled_sizes['small']}px;
            }}
            QListWidget {{
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                font-size: {scaled_sizes['small']}px;
            }}
            QListWidget::item {{
                padding: 6px;
                border-bottom: 1px solid #f0f0f0;
            }}
            QListWidget::item:selected {{
                background-color: #e3f2fd;
                color: #1976d2;
            }}
            QListWidget::item:hover {{
                background-color: #f5f5f5;
            }}
            QComboBox {{
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: {scaled_sizes['normal']}px;
                min-width: 120px;
            }}
            QComboBox:hover {{
                border-color: #b0b0b0;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: white;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 4px;
                font-size: {scaled_sizes['normal']}px;
                min-height: 24px;
            }}
            QCheckBox {{
                font-size: {scaled_sizes['normal']}px;
                color: #424242;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
            QProgressBar {{
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: #f5f5f5;
                text-align: center;
                font-size: {scaled_sizes['tiny']}px;
            }}
            QProgressBar::chunk {{
                background-color: #2196f3;
                border-radius: 3px;
            }}
        """

    def update_font_sizes(self):
        """Update all font sizes based on window size."""
        scale = self.calculate_font_scale()
        
        # Base font sizes
        base_sizes = {
            'title': 14,
            'normal': 13,
            'small': 12,
            'tiny': 11
        }
        
        # Scale all sizes
        scaled_sizes = {k: int(v * scale) for k, v in base_sizes.items()}
        
        # Update stylesheet dynamically
        self.setStyleSheet(self.generate_scaled_stylesheet(scaled_sizes))
        
        # Update button heights
        if hasattr(self, 'grid_mode_button'):
            button_height = max(30, int(40 * scale))
            self.grid_mode_button.setFixedHeight(button_height)
            self.manual_mode_button.setFixedHeight(button_height)
            self.test_mode_button.setFixedHeight(button_height)
        
        # Update +/- button sizes
        button_size = max(18, int(24 * scale))
        for widget_name in ['rows_minus_btn', 'rows_plus_btn', 'cols_minus_btn', 'cols_plus_btn', 'size_minus_btn', 'size_plus_btn']:
            if hasattr(self, widget_name):
                button = getattr(self, widget_name)
                button.setFixedSize(button_size, button_size)

    def resizeEvent(self, event):
        """Handle window resize by updating font sizes."""
        super().resizeEvent(event)
        self.update_font_sizes()
        
        # Remove automatic splitter adjustment to allow manual control

    def _create_toolbar(self):
        """Create toolbar with window controls."""
        self.toolbar = QToolBar("Window Controls")
        self.toolbar.setMovable(False)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: #f8f9fa;
                border-bottom: 1px solid #dee2e6;
                spacing: 5px;
                padding: 5px;
            }
            QAction {
                padding: 5px 10px;
                margin: 2px;
            }
            QToolBar QAction:hover {
                background-color: #e9ecef;
                border-radius: 4px;
            }
        """)
        
        # Add spacer to push controls to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)
        
        # Note: Maximize button removed - using native window controls instead
        # Keep keyboard shortcuts (F11, Ctrl+M) available
        
        self.addToolBar(self.toolbar)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts for window controls and navigation."""
        # F11 for fullscreen toggle
        fullscreen_shortcut = QShortcut("F11", self)
        fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
        
        # Ctrl+M for maximize
        maximize_shortcut = QShortcut("Ctrl+M", self)
        maximize_shortcut.activated.connect(self.toggle_maximize)
        
        # Enter key for assign/commit in both modes
        enter_shortcut = QShortcut("Return", self)
        enter_shortcut.activated.connect(self._on_enter_key)
        
        # Left/Right arrow keys for graph traversal
        left_shortcut = QShortcut("Left", self)
        left_shortcut.activated.connect(self._on_previous_element)
        
        right_shortcut = QShortcut("Right", self)
        right_shortcut.activated.connect(self._on_next_element)
    
    def toggle_maximize(self):
        """Toggle between maximized and normal window state."""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and normal window state."""
        if self.isFullScreen():
            self.showNormal()
            self.toolbar.setVisible(True)
        else:
            self.showFullScreen()
            # Hide toolbar in fullscreen for maximum space
            self.toolbar.setVisible(False)
                
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