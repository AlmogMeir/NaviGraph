"""
SVG Export Utility for NaviGraph MapWidget

Usage in debugger:
1. Set breakpoint in MapWidget (e.g., in _on_toggle_mappings or paintEvent)
2. In debug console, run:

   import sys
   sys.path.append('/home/elior/PycharmProjects/NaviGraph')
   from svg_export_utility import export_map_widget_to_svg
   export_map_widget_to_svg(self, "/home/elior/my_export.svg")

"""

import cv2
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QImage, QPainter, QPixmap, QBrush, QColor, QPen, QPolygonF, QFontMetrics
from PyQt5.QtSvg import QSvgGenerator


def export_map_widget_to_svg(map_widget, svg_path: str):
    """Export MapWidget's current view to SVG file.

    Args:
        map_widget: The MapWidget instance (usually 'self' in debugger)
        svg_path: Full path where to save the SVG file
    """
    # Create SVG generator
    generator = QSvgGenerator()
    generator.setFileName(svg_path)
    generator.setSize(map_widget.size())
    generator.setViewBox(map_widget.rect())
    generator.setTitle("NaviGraph Map Export")

    # Create painter
    painter = QPainter()
    painter.begin(generator)
    painter.setRenderHint(QPainter.Antialiasing)

    # === Draw map image (copied from paintEvent) ===
    if map_widget.map_image is not None:
        height, width = map_widget.map_image.shape[:2]
        bytes_per_line = 3 * width

        if len(map_widget.map_image.shape) == 2:
            # Grayscale
            image = cv2.cvtColor(map_widget.map_image, cv2.COLOR_GRAY2RGB)
        else:
            # Color (BGR to RGB)
            image = cv2.cvtColor(map_widget.map_image, cv2.COLOR_BGR2RGB)

        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Calculate optimal scaling
        widget_size = map_widget.size()
        available_width = widget_size.width() - 40
        available_height = widget_size.height() - 40

        scale_x = available_width / width if width > 0 else 1.0
        scale_y = available_height / height if height > 0 else 1.0
        map_widget.base_scale_factor = min(scale_x, scale_y, 1.0)

        # Calculate combined scale factor
        map_widget.scale_factor = map_widget.base_scale_factor * map_widget.user_scale_factor

        # Calculate final dimensions
        final_width = int(width * map_widget.scale_factor)
        final_height = int(height * map_widget.scale_factor)

        # Scale the pixmap
        scaled_pixmap = pixmap.scaled(final_width, final_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Center the image if not zoomed
        if map_widget.user_scale_factor == 1.0:
            map_widget.offset_x = (widget_size.width() - final_width) // 2
            map_widget.offset_y = (widget_size.height() - final_height) // 2

        painter.drawPixmap(int(map_widget.offset_x), int(map_widget.offset_y), scaled_pixmap)

    # === Draw grid if enabled ===
    if map_widget.grid_enabled:
        # Calculate adaptive font size
        cell_size = map_widget.grid_config.cell_width * map_widget.scale_factor
        if map_widget.adaptive_font_size:
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

        for cell_id, rect in map_widget.grid_cells.items():
            scaled_rect = QRectF(
                map_widget.offset_x + rect.x() * map_widget.scale_factor,
                map_widget.offset_y + rect.y() * map_widget.scale_factor,
                rect.width() * map_widget.scale_factor,
                rect.height() * map_widget.scale_factor
            )

            # Check cell state
            is_mapped = cell_id in map_widget.cell_mappings
            is_selected = cell_id in map_widget.selected_cells
            is_highlighted = cell_id in map_widget.highlighted_cells

            # Draw cell based on state
            if is_selected:
                painter.fillRect(scaled_rect, QColor(0, 255, 0, 100))
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.drawRect(scaled_rect)
            elif is_highlighted:
                painter.fillRect(scaled_rect, QColor(255, 235, 59, 120))
                painter.setPen(QPen(QColor(255, 193, 7), 2))
                painter.drawRect(scaled_rect)
            elif is_mapped and map_widget.show_all_mappings:
                color = map_widget.cell_colors.get(cell_id, QColor(200, 200, 200, 100))
                painter.fillRect(scaled_rect, color)
                painter.setPen(QPen(color.darker(), 2))
                painter.drawRect(scaled_rect)
            else:
                painter.setPen(QPen(QColor(100, 100, 100), 1))
                painter.drawRect(scaled_rect)

    # === Draw regions from mapping ===
    if (hasattr(map_widget, 'gui_parent') and hasattr(map_widget.gui_parent, 'mapping') and
        map_widget.gui_parent.mapping and hasattr(map_widget, 'show_all_mappings') and map_widget.show_all_mappings):
        try:
            from navigraph.core.graph.regions import RectangleRegion, ContourRegion

            # Draw node regions in green
            for node_id in map_widget.gui_parent.mapping.get_mapped_nodes():
                regions = map_widget.gui_parent.mapping.get_node_regions(node_id)
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
                        poly_points = [QPointF(map_widget.offset_x + p[0] * map_widget.scale_factor,
                                             map_widget.offset_y + p[1] * map_widget.scale_factor)
                                     for p in points]
                        polygon = QPolygonF(poly_points)

                        painter.setPen(QPen(color.darker(), 2))
                        painter.setBrush(QBrush(color))
                        painter.drawPolygon(polygon)

                        # Draw node label
                        if map_widget.show_cell_labels:
                            bounding_rect = polygon.boundingRect()
                            centroid = bounding_rect.center()
                            label_text = f"N{node_id}"

                            font = painter.font()
                            font.setFamily('Arial')
                            font.setPointSize(8)
                            painter.setFont(font)
                            painter.setPen(QPen(QColor(50, 50, 50), 1))

                            metrics = QFontMetrics(font)
                            text_rect = metrics.boundingRect(label_text)
                            text_x = centroid.x() - text_rect.width() / 2
                            text_y = centroid.y() + text_rect.height() / 4
                            painter.drawText(QPointF(text_x, text_y), label_text)

            # Draw edge regions in orange
            for edge in map_widget.gui_parent.mapping.get_mapped_edges():
                regions = map_widget.gui_parent.mapping.get_edge_regions(edge)
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
                        poly_points = [QPointF(map_widget.offset_x + p[0] * map_widget.scale_factor,
                                             map_widget.offset_y + p[1] * map_widget.scale_factor)
                                     for p in points]
                        polygon = QPolygonF(poly_points)

                        painter.setPen(QPen(color.darker(), 2))
                        painter.setBrush(QBrush(color))
                        painter.drawPolygon(polygon)

                        # Draw edge label
                        if map_widget.show_cell_labels:
                            bounding_rect = polygon.boundingRect()
                            centroid = bounding_rect.center()

                            if isinstance(edge, tuple):
                                label_text = f"E{edge[0]},{edge[1]}"
                            else:
                                label_text = f"E{edge}"

                            font = painter.font()
                            font.setFamily('Arial')
                            font.setPointSize(7)
                            painter.setFont(font)
                            painter.setPen(QPen(QColor(120, 60, 0), 1))

                            metrics = QFontMetrics(font)
                            text_rect = metrics.boundingRect(label_text)
                            text_x = centroid.x() - text_rect.width() / 2
                            text_y = centroid.y() + text_rect.height() / 4
                            painter.drawText(QPointF(text_x, text_y), label_text)

        except Exception as e:
            print(f"Error drawing regions: {e}")

    # === Draw current contour being drawn ===
    if map_widget.current_contour:
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        scaled_points = [(map_widget.offset_x + p[0] * map_widget.scale_factor,
                        map_widget.offset_y + p[1] * map_widget.scale_factor)
                       for p in map_widget.current_contour]
        for i in range(len(scaled_points) - 1):
            painter.drawLine(QPointF(*scaled_points[i]), QPointF(*scaled_points[i+1]))

        # Draw points
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        for point in scaled_points:
            painter.drawEllipse(QPointF(*point), 3, 3)

    painter.end()
    print(f"SVG successfully exported to: {svg_path}")