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
    QCheckBox, QFileDialog, QSizePolicy
)
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QPen, QBrush, QColor, QPolygonF, QPainter

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
    """Widget for displaying and interacting with the map image."""
    
    gridPlaced = pyqtSignal(float, float)  # Emitted when grid origin is placed
    cellClicked = pyqtSignal(str)  # Emitted when a grid cell is clicked
    contourDrawn = pyqtSignal(list)  # Emitted when a contour is completed
    
    def __init__(self, map_image: np.ndarray, parent=None):
        super().__init__(parent)
        self.map_image = map_image
        self.original_image = map_image.copy()
        
        # Display state
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
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
        self.current_element_type = None
        self.current_element_id = None
        
        self.setMouseTracking(True)
        self.setMinimumSize(800, 600)
        
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
            
            # Calculate scale factor to fit both dimensions
            scale_x = available_width / width if width > 0 else 1.0
            scale_y = available_height / height if height > 0 else 1.0
            scale_factor = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            # Calculate final dimensions
            final_width = int(width * scale_factor)
            final_height = int(height * scale_factor)
            
            # Scale the pixmap
            scaled_pixmap = pixmap.scaled(final_width, final_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Center the image
            self.offset_x = (widget_size.width() - final_width) // 2
            self.offset_y = (widget_size.height() - final_height) // 2
            painter.drawPixmap(self.offset_x, self.offset_y, scaled_pixmap)
            
            # Store scale factor for coordinate conversion
            self.scale_factor = scale_factor
            
        # Draw grid if enabled
        if self.grid_enabled:
            font = painter.font()
            font.setPointSize(8)
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
                    
                    # Draw element ID
                    elem_type, elem_id = self.cell_mappings[cell_id]
                    if elem_type == 'node':
                        label = f"N:{elem_id}"
                        painter.setPen(QPen(QColor(0, 100, 0), 1))
                    else:
                        label = f"E:{elem_id[0]},{elem_id[1]}" if isinstance(elem_id, tuple) else f"E:{elem_id}"
                        painter.setPen(QPen(QColor(0, 0, 100), 1))
                    painter.drawText(scaled_rect.center(), label)
                    
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
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            
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
                    
                    # Draw element ID at centroid
                    if elem_type and elem_id:
                        centroid = polygon.boundingRect().center()
                        if elem_type == 'node':
                            label = f"N:{elem_id}"
                        else:
                            label = f"E:{elem_id[0]},{elem_id[1]}" if isinstance(elem_id, tuple) else f"E:{elem_id}"
                        painter.setPen(QPen(color.darker().darker(), 1))
                        painter.drawText(centroid, label)
                
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
                
    def mousePressEvent(self, event):
        """Handle mouse press events."""
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
    
    def highlight_contour(self, region_id: str):
        """Highlight a specific contour on the map."""
        # Find and highlight the contour
        for i, contour_data in enumerate(self.completed_contours):
            if contour_data[1] == region_id:  # region_id is at index 1
                # Temporarily change the color to highlight
                original_color = contour_data[2]
                highlight_color = QColor(255, 255, 0, 150)  # Yellow highlight
                # Update the contour data
                self.completed_contours[i] = (contour_data[0], contour_data[1], highlight_color) + contour_data[3:]
                self.update()
                
                # Reset color after a delay
                QTimer.singleShot(2000, lambda: self._reset_contour_color(region_id, original_color))
                break
    
    def _reset_contour_color(self, region_id: str, original_color: QColor):
        """Reset contour color after highlighting."""
        for i, contour_data in enumerate(self.completed_contours):
            if contour_data[1] == region_id:
                self.completed_contours[i] = (contour_data[0], contour_data[1], original_color) + contour_data[3:]
                self.update()
                break
    
    def remove_contour(self, region_id: str):
        """Remove a contour from the map."""
        self.completed_contours = [
            contour for contour in self.completed_contours 
            if contour[1] != region_id
        ]
        if region_id in self.contour_mappings:
            del self.contour_mappings[region_id]
        self.update()


class GraphWidget(FigureCanvas):
    """Widget for displaying the graph structure using matplotlib."""
    
    nodeClicked = pyqtSignal(object)  # Emitted when a node is clicked
    edgeClicked = pyqtSignal(tuple)  # Emitted when an edge is clicked
    
    def __init__(self, graph: GraphStructure, parent=None):
        self.graph = graph
        self.figure = Figure(figsize=(8, 6))
        super().__init__(self.figure)
        self.setParent(parent)
        
        self.ax = self.figure.add_subplot(111)
        self.node_colors = {}
        self.edge_colors = {}
        self.highlighted_nodes = set()
        self.highlighted_edges = set()
        
        self._compute_layout()
        self.draw_graph()
        
    def _compute_layout(self):
        """Compute graph layout for visualization."""
        # Use stored positions if available, otherwise compute
        if self.graph.node_positions:
            self.pos = self.graph.node_positions
        else:
            # For binary trees, create a hierarchical layout manually
            if hasattr(self.graph, 'graph') and nx.is_tree(self.graph.graph):
                try:
                    # Try graphviz first
                    self.pos = nx.nx_agraph.graphviz_layout(self.graph.graph, prog='dot')
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
            y = -level * 4  # Increased vertical spacing
            for i, node in enumerate(level_nodes):
                # Spread nodes horizontally with better spacing
                if len(level_nodes) == 1:
                    x = 0
                else:
                    # Increased horizontal spacing multiplier
                    spacing_factor = max(3, max_width / len(level_nodes)) * 3
                    x = (i - (len(level_nodes) - 1) / 2) * spacing_factor
                pos[node] = (x, y)
        
        return pos
                
    def draw_graph(self):
        """Draw or redraw the graph."""
        self.ax.clear()
        
        # Prepare node colors
        node_colors = []
        for node in self.graph.nodes:
            if node in self.highlighted_nodes:
                node_colors.append('lightgreen')
            elif node in self.node_colors:
                node_colors.append(self.node_colors[node])
            else:
                node_colors.append('lightblue')
                
        # Draw graph with appropriate sizing for large trees
        node_count = len(self.graph.nodes)
        if node_count > 50:
            # Small nodes for large graphs
            node_size = 300
            font_size = 8
        elif node_count > 20:
            # Medium nodes for medium graphs
            node_size = 500
            font_size = 9
        else:
            # Large nodes for small graphs
            node_size = 800
            font_size = 11
            
        nx.draw(self.graph.graph, pos=self.pos, ax=self.ax,
               node_color=node_colors, node_size=node_size,
               with_labels=True, font_size=font_size, font_weight='bold',
               edge_color='gray', linewidths=1, font_color='black',
               node_shape='o', alpha=0.9)
               
        # Highlight specific edges if needed
        if self.highlighted_edges:
            edge_list = list(self.highlighted_edges)
            nx.draw_networkx_edges(self.graph.graph, pos=self.pos, ax=self.ax,
                                  edgelist=edge_list, edge_color='green', width=2)
                                  
        self.ax.set_title(f"Graph Structure ({len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges)")
        self.draw()
        
    def highlight_nodes(self, nodes: Set, color: str = 'lightgreen'):
        """Highlight specific nodes."""
        self.highlighted_nodes = nodes
        self.draw_graph()
        
    def highlight_edges(self, edges: Set[Tuple], color: str = 'green'):
        """Highlight specific edges."""
        self.highlighted_edges = edges
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
                padding: 6px;
                font-size: 13px;
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
        self.graph_widget.setMinimumHeight(200)
        self.graph_widget.setMaximumHeight(300)
        graph_layout.addWidget(self.graph_widget)
        
        views_layout.addWidget(graph_group)
        
        # Map view (bottom, larger)
        map_group = QGroupBox("🗺️ Interactive Mapping Area")
        map_group.setStyleSheet("QGroupBox { font-weight: 600; font-size: 14px; color: #424242; }")
        map_layout = QVBoxLayout(map_group)
        map_layout.setContentsMargins(8, 20, 8, 8)
        
        self.map_widget = MapWidget(self.map_image)
        self.map_widget.setMinimumHeight(400)
        map_layout.addWidget(self.map_widget)
        
        views_layout.addWidget(map_group)
        
        # Set stretch factors (graph:map = 1:3)
        views_layout.setStretchFactor(graph_group, 1)
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
        
        # Grid dimensions and size in compact layout
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 20)
        self.rows_spin.setValue(8)
        grid_layout.addWidget(self.rows_spin)
        
        grid_layout.addWidget(QLabel("Cols:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 20)
        self.cols_spin.setValue(8)
        grid_layout.addWidget(self.cols_spin)
        config_layout.addLayout(grid_layout)
        
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Cell Size:"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(10, 200)
        self.width_spin.setValue(50)
        size_layout.addWidget(self.width_spin)
        config_layout.addLayout(size_layout)
        
        # Apply and place buttons
        config_buttons = QHBoxLayout()
        self.apply_grid_button = QPushButton("Apply Config")
        self.apply_grid_button.clicked.connect(self._on_apply_grid_config)
        config_buttons.addWidget(self.apply_grid_button)
        
        self.place_grid_button = QPushButton("Place Grid")
        self.place_grid_button.clicked.connect(self._on_place_grid)
        config_buttons.addWidget(self.place_grid_button)
        config_layout.addLayout(config_buttons)
        
        self.grid_status_label = QLabel("Configure and place grid")
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
        
        self.clear_grid_button = QPushButton("Clear All")
        self.clear_grid_button.clicked.connect(self._on_clear_all)
        self.clear_grid_button.setEnabled(False)
        control_layout.addWidget(self.clear_grid_button)
        
        assign_layout.addLayout(control_layout)
        layout.addWidget(assign_group)
        
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
        
        # Control buttons
        control_group = QGroupBox("Element Controls")
        control_group_layout = QHBoxLayout(control_group)
        
        self.manual_next_button = QPushButton("Next")
        self.manual_next_button.clicked.connect(self._on_next_element)
        self.manual_next_button.setEnabled(False)
        control_group_layout.addWidget(self.manual_next_button)
        
        self.manual_undo_button = QPushButton("Undo")
        self.manual_undo_button.clicked.connect(self._on_undo_last)
        self.manual_undo_button.setEnabled(False)
        control_group_layout.addWidget(self.manual_undo_button)
        
        layout.addWidget(control_group)
        
        return widget
        
    def _create_test_controls(self) -> QWidget:
        """Create test mode controls."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Instructions
        instructions = QLabel(
            "Test Mode Instructions:\n\n"
            "• Click on the map to see which graph element is highlighted\n"
            "• Click on graph nodes/edges to see map regions\n"
            "• Use the load button to test with saved mappings\n"
            "• Current mapping is automatically available for testing"
        )
        instructions.setStyleSheet(
            "background-color: #f8f9fa; border: 1px solid #dee2e6; "
            "border-radius: 6px; padding: 12px; font-size: 12px; color: #495057;"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Progress info
        self.progress_info = QLabel("No mappings yet")
        self.progress_info.setStyleSheet("font-weight: 600; color: #616161; padding: 8px;")
        layout.addWidget(self.progress_info)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        file_buttons = QHBoxLayout()
        self.save_button = QPushButton("Save Mapping")
        self.save_button.clicked.connect(self._on_save_mapping)
        file_buttons.addWidget(self.save_button)
        
        self.load_button = QPushButton("Load Mapping")
        self.load_button.clicked.connect(self._on_load_mapping)
        file_buttons.addWidget(self.load_button)
        
        file_layout.addLayout(file_buttons)
        layout.addWidget(file_group)
        
        # Test results display
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout(results_group)
        
        self.test_result_label = QLabel("Click on map or graph to test...")
        self.test_result_label.setStyleSheet(
            "background-color: white; border: 1px solid #ccc; "
            "border-radius: 4px; padding: 8px; min-height: 60px;"
        )
        self.test_result_label.setWordWrap(True)
        results_layout.addWidget(self.test_result_label)
        
        layout.addWidget(results_group)
        
        return widget
        
    def _on_grid_mode(self, checked: bool):
        """Handle grid mode selection."""
        try:
            if checked:
                self.manual_mode_button.setChecked(False)
                self.test_mode_button.setChecked(False)
                self.setup_mode = 'grid'
                self.mode_stack.setCurrentIndex(1)  # Grid controls
                
                # Reset and initialize for grid mode with better error handling
                try:
                    self._reset_mapping_state()
                    self._init_element_queue()
                    self._populate_element_combos()
                    self._update_progress_display()
                except Exception as init_error:
                    self.status_bar.showMessage(f"Error initializing grid mode: {str(init_error)}")
                    print(f"Grid mode initialization error: {init_error}")
                    return
                
                # Enable drawing mode
                if hasattr(self.map_widget, 'set_interaction_mode'):
                    self.map_widget.set_interaction_mode('none')
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
                self.grid_mode_button.setChecked(False)
                self.test_mode_button.setChecked(False)
                self.setup_mode = 'manual'
                self.mode_stack.setCurrentIndex(2)  # Manual controls
                
                # Reset and initialize for manual mode with better error handling
                try:
                    self._reset_mapping_state()
                    self._init_element_queue()
                    self._populate_element_combos()
                    self._update_progress_display()
                except Exception as init_error:
                    self.status_bar.showMessage(f"Error initializing manual mode: {str(init_error)}")
                    print(f"Manual mode initialization error: {init_error}")
                    return
                
                # Enable drawing mode
                if hasattr(self.map_widget, 'set_interaction_mode'):
                    self.map_widget.set_interaction_mode('draw_contour')
                self.status_bar.showMessage("Manual Drawing Mode - Left click to draw points")
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
                self.grid_mode_button.setChecked(False)
                self.manual_mode_button.setChecked(False)
                self.setup_mode = 'test'
                self.mode_stack.setCurrentIndex(3)  # Test controls
                
                # Enable test interactions
                if hasattr(self.map_widget, 'set_interaction_mode'):
                    self.map_widget.set_interaction_mode('test')
                self._update_progress_display()
                self.status_bar.showMessage("Test Mode - Click map or graph to test mapping")
            else:
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
        
        # Populate with all elements
        for elem_type, elem_id in self.all_elements:
            if elem_type == 'node':
                label = f"Node: {elem_id}"
            else:
                label = f"Edge: {elem_id}"
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
        """Reset all mapping state when switching modes."""
        self.map_widget.reset_all()
        self.mapping = SpatialMapping(self.graph)
        self.mapped_elements.clear()
        self.mapping_history.clear()
        self.element_queue.clear()
        self.current_element = None
        self.graph_widget.clear_highlights()
        self._update_progress()
        
        # Reset button states
        if hasattr(self, 'undo_grid_button'):
            self.undo_grid_button.setEnabled(False)
        if hasattr(self, 'manual_undo_button'):
            self.manual_undo_button.setEnabled(False)
            
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
            self.manual_next_button.setEnabled(True)
            
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
        self.grid_status_label.setText(f"Config: {config.rows}x{config.cols}, size {config.cell_width}")
        self.status_bar.showMessage("Grid configuration updated - click 'Place Grid' to position it")
        
    def _on_place_grid(self):
        """Start grid placement mode."""
        self.map_widget.set_interaction_mode('place_grid')
        self.status_bar.showMessage("Click on the map to place grid origin")
        
    def _on_grid_placed(self, x: float, y: float):
        """Handle grid placement."""
        self.grid_status_label.setText(f"Grid placed at ({x:.0f}, {y:.0f})")
        self.status_bar.showMessage("Grid placed - Click cells to select")
        
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
                    color = QColor(150, 150, 255, 100)  # Light blue for edges
                    
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
        """Handle contour selection from list."""
        self.delete_contour_button.setEnabled(True)
        region_id = item.data(Qt.UserRole)
        if region_id:
            # Highlight the selected contour on the map
            self.map_widget.highlight_contour(region_id)
        
    def _on_delete_contour(self):
        """Delete selected contour."""
        current_item = self.contour_list.currentItem()
        if current_item:
            region_id = current_item.data(Qt.UserRole)
            
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
            color = QColor(150, 255, 150, 100) if elem_type == 'node' else QColor(150, 150, 255, 100)
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
            self.manual_undo_button.setEnabled(True)
            
            # Reset drawing state but keep drawing mode active for next element
            # Drawing mode stays active for continuous element mapping
                
            self.status_bar.showMessage(f"Contour assigned to {elem_type} {elem_id}")
            
            # Update progress and don't automatically move to next element
            # Let user control when to move to next element with Next button
            self._update_progress_display()
            
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
        for node_id, regions in self.mapping.node_to_regions.items():
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
        for edge, regions in self.mapping.edge_to_regions.items():
            color = QColor(150, 150, 255, 100)
            for region in regions:
                if isinstance(region, ContourRegion):
                    self.map_widget.add_contour(region.points, region.region_id, 'edge', edge, color)
    
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
                self.manual_next_button.setEnabled(True)
                
            # Highlight in graph
            self.graph_widget.clear_highlights()
            if elem_type == 'node':
                self.graph_widget.highlight_nodes({elem_id})
            elif elem_type == 'edge':
                self.graph_widget.highlight_edges({elem_id})
                
            self.status_bar.showMessage(f"Jumped to {elem_type} {elem_id}")
    
    def _on_toggle_mappings(self, state):
        """Toggle visibility of all mappings."""
        self.map_widget.show_all_mappings = (state == Qt.Checked)
        self.map_widget.update()
    
    def _on_clear_all(self):
        """Clear all mappings and start fresh."""
        reply = QMessageBox.question(self, "Clear All", 
                                    "Are you sure you want to clear all mappings?",
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
                    if hasattr(self.mapping, 'node_to_regions') and elem_id in self.mapping.node_to_regions:
                        for region in last_action.get('regions', []):
                            if region in self.mapping.node_to_regions[elem_id]:
                                self.mapping.node_to_regions[elem_id].remove(region)
                elif elem_type == 'edge':
                    if hasattr(self.mapping, 'edge_to_regions') and elem_id in self.mapping.edge_to_regions:
                        for region in last_action.get('regions', []):
                            if region in self.mapping.edge_to_regions[elem_id]:
                                self.mapping.edge_to_regions[elem_id].remove(region)
                                
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
                    
    def get_mapping(self) -> SpatialMapping:
        """Get the current mapping."""
        return self.mapping


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