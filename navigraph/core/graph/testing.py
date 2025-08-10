"""Interactive testing tool for validating graph-space mappings.

This module provides an interactive interface for testing and validating
spatial mappings, allowing users to click on maps to see corresponding
nodes and vice versa.
"""

from typing import Dict, Any, List, Tuple, Optional, Set, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, CheckButtons
from matplotlib.backend_bases import MouseEvent, KeyEvent

from .gui_utils import (
    is_gui_available, create_figure, safe_show, 
    handle_gui_error, OpenCVFallback, print_backend_info
)
import networkx as nx

from .structures import GraphStructure
from .mapping import SpatialMapping
from .regions import SpatialRegion, ContourRegion, RectangleRegion, CircleRegion, GridCell
from .storage import MappingStorage


class MappingTester:
    """Interactive tool for testing graph-space mappings."""
    
    def __init__(self, graph: GraphStructure, mapping: SpatialMapping, 
                 map_image: np.ndarray):
        """Initialize mapping tester.
        
        Args:
            graph: Graph structure
            mapping: Spatial mapping to test
            map_image: Map image as numpy array
        """
        self.graph = graph
        self.mapping = mapping
        self.map_image = map_image
        
        # UI state
        self.fig = None
        self.ax_map = None
        self.ax_graph = None
        self.ax_info = None
        
        # Selection state
        self.selected_nodes = set()
        self.selected_regions = set()
        self.highlighted_path = []
        
        # Display options
        self.show_all_regions = True
        self.show_node_labels = True
        self.show_overlaps = False
        self.show_unmapped = False
        
        # Path testing mode
        self.path_mode = False
        self.path_start = None
        self.path_end = None
        
        # Graph layout
        self.graph_pos = None
        self._update_graph_layout()
        
        # Statistics
        self.stats = mapping.validate_mapping()
        
        # UI elements
        self.buttons = {}
        self.checkboxes = None
    
    def start_interactive_test(self) -> None:
        """Launch dual-view interactive testing interface."""
        # Check GUI availability and provide feedback
        if not is_gui_available():
            print("⚠️  GUI not available. Interactive testing is not possible.")
            print("💡 Alternative: Use programmatic validation methods:")
            print("   - mapping.validate_mapping()")
            print("   - mapping.find_overlaps()")
            print("   - Export reports with --report option")
            return
        
        try:
            self._create_dual_view()
            self._setup_ui_controls()
            self._connect_event_handlers()
            self._initial_draw()
            safe_show(self.fig)
        except Exception as e:
            handle_gui_error("interactive_test", e)
            print("💡 Consider using the command line report generation:")
            print("   navigraph test graph -m mapping.pkl -i image.png --report report.html")
    
    def _create_dual_view(self):
        """Create side-by-side map and graph visualization."""
        self.fig = create_figure(figsize=(18, 10))
        
        # Create subplots with custom layout
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Map view (left, takes up 2x2)
        self.ax_map = self.fig.add_subplot(gs[:2, :2])
        self.ax_map.set_title("Map View - Click to test mapping", fontsize=14, fontweight='bold')
        
        # Graph view (top right)
        self.ax_graph = self.fig.add_subplot(gs[0, 2])
        self.ax_graph.set_title("Graph View - Click nodes", fontsize=12, fontweight='bold')
        
        # Info panel (bottom right)
        self.ax_info = self.fig.add_subplot(gs[1:, 2])
        self.ax_info.set_title("Information Panel", fontsize=12, fontweight='bold')
        self.ax_info.axis('off')
    
    def _setup_ui_controls(self):
        """Setup UI control buttons and checkboxes."""
        # Button positions (bottom of figure)
        button_height = 0.04
        button_width = 0.08
        y_pos = 0.02
        x_start = 0.02
        x_spacing = button_width + 0.01
        
        # Toggle buttons
        buttons_config = [
            ('Regions', self._toggle_regions),
            ('Labels', self._toggle_labels),
            ('Overlaps', self._toggle_overlaps),
            ('Unmapped', self._toggle_unmapped),
            ('Path Mode', self._toggle_path_mode),
            ('Clear', self._clear_selections),
            ('Stats', self._show_statistics),
            ('Export', self._export_report)
        ]
        
        for i, (label, callback) in enumerate(buttons_config):
            x_pos = x_start + i * x_spacing
            button_ax = plt.axes([x_pos, y_pos, button_width, button_height])
            button = Button(button_ax, label, color='lightgray')
            button.on_clicked(callback)
            self.buttons[label] = button
    
    def _connect_event_handlers(self):
        """Connect mouse and keyboard event handlers."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
    
    def _initial_draw(self):
        """Draw initial state of both views."""
        self._draw_map_view()
        self._draw_graph_view()
        self._update_info_panel()
        self.fig.canvas.draw()
    
    def _draw_map_view(self):
        """Draw map with regions and overlays."""
        self.ax_map.clear()
        
        # Display map image
        if len(self.map_image.shape) == 3:
            self.ax_map.imshow(self.map_image)
        else:
            self.ax_map.imshow(self.map_image, cmap='gray')
        
        # Draw regions if enabled
        if self.show_all_regions:
            self._draw_all_regions()
        
        # Draw overlaps if enabled
        if self.show_overlaps:
            self._draw_overlaps()
        
        # Draw unmapped areas if enabled
        if self.show_unmapped:
            self._draw_unmapped_areas()
        
        # Highlight selected regions
        self._highlight_selected_regions()
        
        # Draw path on map if available
        if self.highlighted_path:
            self._draw_path_on_map()
        
        self.ax_map.set_title("Map View - Click to test mapping")
    
    def _draw_graph_view(self):
        """Draw graph with highlighted nodes."""
        self.ax_graph.clear()
        
        if not self.graph_pos:
            self._update_graph_layout()
        
        # Draw all nodes
        node_colors = []
        for node in self.graph.nodes:
            if node in self.selected_nodes:
                node_colors.append('red')
            elif node in self.mapping.get_mapped_nodes():
                node_colors.append('lightblue')
            else:
                node_colors.append('lightgray')
        
        # Draw graph
        nx.draw(self.graph.graph, pos=self.graph_pos, ax=self.ax_graph,
                node_color=node_colors, node_size=300, 
                with_labels=self.show_node_labels, font_size=8,
                edge_color='gray', font_weight='bold')
        
        # Highlight path if available
        if self.highlighted_path and len(self.highlighted_path) > 1:
            path_edges = [(self.highlighted_path[i], self.highlighted_path[i+1]) 
                         for i in range(len(self.highlighted_path)-1)]
            nx.draw_networkx_edges(self.graph.graph, pos=self.graph_pos, 
                                 edgelist=path_edges, ax=self.ax_graph,
                                 edge_color='red', width=3)
        
        self.ax_graph.set_title("Graph View - Click nodes")
    
    def _draw_all_regions(self):
        """Draw all regions on the map."""
        for region_id, region in self.mapping.get_all_regions().items():
            self._draw_region(region, alpha=0.3, color='blue')
    
    def _draw_region(self, region: SpatialRegion, alpha: float = 0.3, 
                    color: str = 'blue', highlight: bool = False):
        """Draw a single region on the map."""
        linewidth = 3 if highlight else 1
        
        if isinstance(region, ContourRegion):
            contour = np.array(region.contour_points)
            if highlight:
                self.ax_map.plot(contour[:, 0], contour[:, 1], 
                               color='red', linewidth=linewidth)
            self.ax_map.fill(contour[:, 0], contour[:, 1], 
                           alpha=alpha, color=color)
        
        elif isinstance(region, (RectangleRegion, GridCell)):
            rect = patches.Rectangle((region.x, region.y), 
                                   region.width, region.height,
                                   linewidth=linewidth, 
                                   edgecolor='red' if highlight else color,
                                   facecolor=color, alpha=alpha)
            self.ax_map.add_patch(rect)
        
        elif isinstance(region, CircleRegion):
            circle = patches.Circle((region.center_x, region.center_y),
                                  region.radius, linewidth=linewidth,
                                  edgecolor='red' if highlight else color,
                                  facecolor=color, alpha=alpha)
            self.ax_map.add_patch(circle)
        
        # Add label if enabled
        if self.show_node_labels:
            element = self.mapping.get_element_for_region(region.region_id)
            if element:
                elem_type, elem_id = element
                center = region.get_center()
                label = f"{elem_type[0].upper()}: {elem_id}"  # N: node_id or E: edge_id
                self.ax_map.text(center.x, center.y, label,
                               ha='center', va='center', fontsize=8,
                               color='white', weight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', 
                                       facecolor='black', alpha=0.7))
    
    def _draw_overlaps(self):
        """Highlight overlapping regions."""
        overlaps = self.mapping.find_overlaps()
        conflicts = self.mapping.find_node_conflicts()
        
        # Draw overlapping regions
        for overlap in overlaps:
            region1 = self.mapping.get_region_by_id(overlap.region1_id)
            region2 = self.mapping.get_region_by_id(overlap.region2_id)
            
            if region1:
                center1 = region1.get_center()
                self.ax_map.plot(center1.x, center1.y, 'rx', 
                               markersize=15, markeredgewidth=4)
            if region2:
                center2 = region2.get_center()
                self.ax_map.plot(center2.x, center2.y, 'rx', 
                               markersize=15, markeredgewidth=4)
        
        # Draw node conflicts with different marker
        for conflict in conflicts:
            region = self.mapping.get_region_by_id(conflict.region_id)
            if region:
                center = region.get_center()
                self.ax_map.plot(center.x, center.y, 'go', 
                               markersize=20, markeredgewidth=4, fillstyle='none')
    
    def _draw_unmapped_areas(self):
        """Highlight unmapped nodes and edges."""
        unmapped_nodes = self.mapping.get_unmapped_nodes()
        unmapped_edges = self.mapping.get_unmapped_edges()
        
        y_offset = 0.98
        
        if unmapped_nodes:
            # Show text listing unmapped nodes
            unmapped_text = f"Unmapped nodes: {', '.join(map(str, list(unmapped_nodes)[:5]))}"
            if len(unmapped_nodes) > 5:
                unmapped_text += f" (+{len(unmapped_nodes) - 5} more)"
            self.ax_map.text(0.02, y_offset, unmapped_text, 
                           transform=self.ax_map.transAxes,
                           verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            y_offset -= 0.08
        
        if unmapped_edges:
            # Show text listing unmapped edges
            edge_strs = [f"{e[0]}-{e[1]}" for e in list(unmapped_edges)[:3]]
            unmapped_text = f"Unmapped edges: {', '.join(edge_strs)}"
            if len(unmapped_edges) > 3:
                unmapped_text += f" (+{len(unmapped_edges) - 3} more)"
            self.ax_map.text(0.02, y_offset, unmapped_text, 
                           transform=self.ax_map.transAxes,
                           verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
    
    def _highlight_selected_regions(self):
        """Highlight currently selected regions."""
        for region_id in self.selected_regions:
            region = self.mapping.get_region_by_id(region_id)
            if region:
                self._draw_region(region, alpha=0.5, color='red', highlight=True)
    
    def _draw_path_on_map(self):
        """Draw highlighted path on map."""
        if len(self.highlighted_path) < 2:
            return
        
        # Get centers of regions for path nodes
        path_points = []
        for node in self.highlighted_path:
            regions = self.mapping.get_node_regions(node)
            if regions:
                center = regions[0].get_center()
                path_points.append((center.x, center.y))
        
        if len(path_points) >= 2:
            path_array = np.array(path_points)
            self.ax_map.plot(path_array[:, 0], path_array[:, 1], 
                           'r-', linewidth=3, alpha=0.8)
            self.ax_map.plot(path_array[:, 0], path_array[:, 1], 
                           'ro', markersize=8)
    
    def _update_graph_layout(self):
        """Update graph layout for visualization."""
        if self.graph.node_positions:
            self.graph_pos = self.graph.node_positions
        else:
            # Use spring layout if no positions available
            self.graph_pos = nx.spring_layout(self.graph.graph, seed=42)
    
    def _update_info_panel(self):
        """Update information panel with current state."""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = []
        
        # Basic statistics
        info_text.append("=== MAPPING STATISTICS ===")
        info_text.append(f"Total nodes: {self.stats.total_nodes}")
        info_text.append(f"Mapped nodes: {self.stats.mapped_nodes}")
        info_text.append(f"Unmapped nodes: {self.stats.unmapped_nodes}")
        info_text.append(f"Node completeness: {self.stats.node_mapping_completeness:.1f}%")
        info_text.append(f"Mapped edges: {self.stats.mapped_edges}")
        info_text.append(f"Unmapped edges: {self.stats.unmapped_edges}")
        info_text.append(f"Edge completeness: {self.stats.edge_mapping_completeness:.1f}%")
        info_text.append(f"Total regions: {self.stats.total_regions}")
        info_text.append("")
        
        # Current selections
        if self.selected_nodes:
            info_text.append("=== SELECTED NODES ===")
            info_text.append(", ".join(map(str, self.selected_nodes)))
            info_text.append("")
        
        if self.selected_regions:
            info_text.append("=== SELECTED REGIONS ===")
            info_text.append(", ".join(self.selected_regions))
            info_text.append("")
        
        # Path information
        if self.highlighted_path:
            info_text.append("=== CURRENT PATH ===")
            info_text.append(f"Length: {len(self.highlighted_path)} nodes")
            info_text.append(f"Path: {' → '.join(map(str, self.highlighted_path))}")
            info_text.append("")
        
        # Overlaps and conflicts
        if self.stats.overlaps:
            info_text.append("=== OVERLAPS DETECTED ===")
            for overlap in self.stats.overlaps[:3]:  # Show first 3
                elem1_type, elem1_id = overlap.element1
                elem2_type, elem2_id = overlap.element2
                info_text.append(f"{elem1_type} {elem1_id} ↔ {elem2_type} {elem2_id}")
            if len(self.stats.overlaps) > 3:
                info_text.append(f"... and {len(self.stats.overlaps) - 3} more")
            info_text.append("")
        
        if self.stats.node_conflicts:
            info_text.append("=== NODE CONFLICTS ===")
            for conflict in self.stats.node_conflicts[:3]:  # Show first 3
                info_text.append(f"Nodes {conflict.node1} & {conflict.node2} → {conflict.region_id}")
            if len(self.stats.node_conflicts) > 3:
                info_text.append(f"... and {len(self.stats.node_conflicts) - 3} more")
            info_text.append("")
        
        # Controls help
        info_text.append("=== CONTROLS ===")
        info_text.append("• Click map → identify node/edge")
        info_text.append("• Click graph → highlight regions")
        info_text.append("• P: Path mode")
        info_text.append("• R: Toggle regions")
        info_text.append("• L: Toggle labels")
        info_text.append("• C: Clear selections")
        info_text.append("• S: Show statistics")
        
        # Display text
        text_str = "\n".join(info_text)
        self.ax_info.text(0.05, 0.95, text_str, transform=self.ax_info.transAxes,
                         verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    def _on_click(self, event: MouseEvent):
        """Handle mouse click events."""
        if event.inaxes == self.ax_map:
            self._handle_map_click(event.xdata, event.ydata)
        elif event.inaxes == self.ax_graph:
            self._handle_graph_click(event.xdata, event.ydata)
    
    def _handle_map_click(self, x: float, y: float):
        """Handle click on map view."""
        if x is None or y is None:
            return
        
        # Find which elements contain the point
        node, edge = self.mapping.map_point_to_elements(x, y)
        
        if self.path_mode:
            self._handle_path_selection(node)
        else:
            # Regular selection mode
            if node is not None or edge is not None:
                if node is not None:
                    self._select_node(node)
                    feedback = f"Point ({x:.1f}, {y:.1f}) → Node {node}"
                if edge is not None:
                    feedback += f" & Edge {edge}" if node else f"Point ({x:.1f}, {y:.1f}) → Edge {edge}"
                self._show_feedback(feedback)
            else:
                self._show_feedback(f"Point ({x:.1f}, {y:.1f}) → Unmapped")
        
        self._refresh_display()
    
    def _handle_graph_click(self, x: float, y: float):
        """Handle click on graph view."""
        if x is None or y is None or not self.graph_pos:
            return
        
        # Find closest node to click position
        min_distance = float('inf')
        closest_node = None
        
        for node, (nx, ny) in self.graph_pos.items():
            distance = np.sqrt((x - nx) ** 2 + (y - ny) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        
        # Select if click is close enough
        if min_distance < 0.05 and closest_node is not None:  # Threshold for selection
            if self.path_mode:
                self._handle_path_selection(closest_node)
            else:
                self._select_node(closest_node)
                regions = self.mapping.get_node_regions(closest_node)
                region_names = [r.region_id for r in regions]
                self._show_feedback(f"Node {closest_node} → Regions: {region_names}")
            
            self._refresh_display()
    
    def _handle_path_selection(self, node):
        """Handle node selection in path mode."""
        if node is None:
            return
        
        if self.path_start is None:
            self.path_start = node
            self._show_feedback(f"Path start: {node}. Click another node for end.")
        elif node != self.path_start:
            self.path_end = node
            self._calculate_and_show_path()
            # Reset path mode
            self.path_start = None
            self.path_end = None
        else:
            self._show_feedback("Select a different node for path end.")
    
    def _calculate_and_show_path(self):
        """Calculate and display path between selected nodes."""
        if self.path_start is None or self.path_end is None:
            return
        
        path = self.graph.get_shortest_path(self.path_start, self.path_end)
        
        if path:
            self.highlighted_path = path
            self._show_feedback(f"Path from {self.path_start} to {self.path_end}: "
                              f"{len(path)} nodes, distance: "
                              f"{self.graph.get_shortest_path_length(self.path_start, self.path_end)}")
        else:
            self._show_feedback(f"No path found between {self.path_start} and {self.path_end}")
            self.highlighted_path = []
    
    def _select_node(self, node):
        """Select/deselect a node."""
        if node in self.selected_nodes:
            self.selected_nodes.remove(node)
            # Remove associated regions from selection
            regions = self.mapping.get_node_regions(node)
            for region in regions:
                self.selected_regions.discard(region.region_id)
        else:
            self.selected_nodes.add(node)
            # Add associated regions to selection
            regions = self.mapping.get_node_regions(node)
            for region in regions:
                self.selected_regions.add(region.region_id)
    
    def _show_feedback(self, message: str):
        """Show feedback message."""
        print(f"[MappingTester] {message}")
    
    def _refresh_display(self):
        """Refresh both map and graph displays."""
        self._draw_map_view()
        self._draw_graph_view()
        self._update_info_panel()
        self.fig.canvas.draw()
    
    def _on_key_press(self, event: KeyEvent):
        """Handle keyboard events."""
        if event.key == 'r':
            self._toggle_regions(None)
        elif event.key == 'l':
            self._toggle_labels(None)
        elif event.key == 'o':
            self._toggle_overlaps(None)
        elif event.key == 'u':
            self._toggle_unmapped(None)
        elif event.key == 'p':
            self._toggle_path_mode(None)
        elif event.key == 'c':
            self._clear_selections(None)
        elif event.key == 's':
            self._show_statistics(None)
        elif event.key == 'h':
            self._show_help()
    
    def _on_mouse_move(self, event):
        """Handle mouse movement for hover effects."""
        # Could implement hover effects here
        pass
    
    # Button callbacks
    def _toggle_regions(self, event):
        """Toggle region display."""
        self.show_all_regions = not self.show_all_regions
        self.buttons['Regions'].color = 'lightgreen' if self.show_all_regions else 'lightgray'
        self._refresh_display()
    
    def _toggle_labels(self, event):
        """Toggle label display."""
        self.show_node_labels = not self.show_node_labels
        self.buttons['Labels'].color = 'lightgreen' if self.show_node_labels else 'lightgray'
        self._refresh_display()
    
    def _toggle_overlaps(self, event):
        """Toggle overlap highlighting."""
        self.show_overlaps = not self.show_overlaps
        self.buttons['Overlaps'].color = 'lightgreen' if self.show_overlaps else 'lightgray'
        self._refresh_display()
    
    def _toggle_unmapped(self, event):
        """Toggle unmapped areas display."""
        self.show_unmapped = not self.show_unmapped
        self.buttons['Unmapped'].color = 'lightgreen' if self.show_unmapped else 'lightgray'
        self._refresh_display()
    
    def _toggle_path_mode(self, event):
        """Toggle path selection mode."""
        self.path_mode = not self.path_mode
        self.buttons['Path Mode'].color = 'lightgreen' if self.path_mode else 'lightgray'
        if self.path_mode:
            self.path_start = None
            self.path_end = None
            self._show_feedback("Path mode ON: Click two nodes to show path")
        else:
            self._show_feedback("Path mode OFF")
    
    def _clear_selections(self, event):
        """Clear all selections."""
        self.selected_nodes.clear()
        self.selected_regions.clear()
        self.highlighted_path.clear()
        self.path_start = None
        self.path_end = None
        self._show_feedback("Selections cleared")
        self._refresh_display()
    
    def _show_statistics(self, event):
        """Show detailed statistics."""
        print("\n" + "="*50)
        print("DETAILED MAPPING STATISTICS")
        print("="*50)
        print(self.stats)
        
        if self.stats.overlaps:
            print(f"\nOVERLAPS ({len(self.stats.overlaps)}):")
            for i, overlap in enumerate(self.stats.overlaps, 1):
                print(f"  {i}. {overlap}")
    
    def _show_help(self):
        """Show help information."""
        help_text = """
MAPPING TESTER HELP
==================

Mouse Controls:
• Left click on map → Show corresponding node
• Left click on graph → Show corresponding region

Keyboard Shortcuts:
• R: Toggle region display
• L: Toggle node labels
• O: Toggle overlap highlighting  
• U: Toggle unmapped areas
• P: Toggle path mode
• C: Clear all selections
• S: Show detailed statistics
• H: Show this help

Path Mode:
• Click first node → Set as path start
• Click second node → Calculate and show shortest path
• Path is displayed on both map and graph

Button Controls:
• All keyboard shortcuts also available as buttons
• Export: Save validation report
"""
        print(help_text)
    
    def _export_report(self, event):
        """Export validation report."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"mapping_validation_{timestamp}.txt"
        
        success = MappingStorage.export_mapping_report(
            self.mapping, report_path, format='txt'
        )
        
        if success:
            self._show_feedback(f"Report exported to: {report_path}")
        else:
            self._show_feedback("Failed to export report")
    
    def export_test_session(self, filepath: Union[str, Path]) -> bool:
        """Export current test session state.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if export successful
        """
        session_data = {
            'mapping_statistics': self.stats,
            'selected_nodes': list(self.selected_nodes),
            'selected_regions': list(self.selected_regions),
            'highlighted_path': self.highlighted_path,
            'display_options': {
                'show_all_regions': self.show_all_regions,
                'show_node_labels': self.show_node_labels,
                'show_overlaps': self.show_overlaps,
                'show_unmapped': self.show_unmapped
            }
        }
        
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Failed to export test session: {e}")
            return False


class InteractiveValidator:
    """Additional validation tools for testing mode."""
    
    def __init__(self, mapping: SpatialMapping, map_image: np.ndarray):
        """Initialize validator.
        
        Args:
            mapping: Spatial mapping to validate
            map_image: Map image for visualization
        """
        self.mapping = mapping
        self.map_image = map_image
    
    def find_overlaps(self, visualize: bool = True) -> List:
        """Find and optionally visualize overlapping regions."""
        overlaps = self.mapping.find_overlaps()
        
        if visualize and overlaps:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Display map
            if len(self.map_image.shape) == 3:
                ax.imshow(self.map_image)
            else:
                ax.imshow(self.map_image, cmap='gray')
            
            # Highlight overlapping regions
            for overlap in overlaps:
                region1 = self.mapping.get_region_by_id(overlap.region1_id)
                region2 = self.mapping.get_region_by_id(overlap.region2_id)
                
                if region1:
                    center1 = region1.get_center()
                    ax.plot(center1.x, center1.y, 'rx', markersize=15, markeredgewidth=4)
                    ax.text(center1.x + 10, center1.y + 10, f"Node {overlap.node1}",
                           fontsize=10, color='red', weight='bold')
                
                if region2:
                    center2 = region2.get_center()
                    ax.plot(center2.x, center2.y, 'rx', markersize=15, markeredgewidth=4)
                    ax.text(center2.x + 10, center2.y + 10, f"Node {overlap.node2}",
                           fontsize=10, color='red', weight='bold')
            
            ax.set_title(f"Overlapping Regions ({len(overlaps)} found)")
            plt.show()
        
        return overlaps
    
    def create_coverage_heatmap(self) -> plt.Figure:
        """Create coverage heatmap showing mapping density."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a grid for density calculation
        h, w = self.map_image.shape[:2]
        density_grid = np.zeros((h // 10, w // 10))  # Downsample for performance
        
        # Sample points and check coverage
        for i in range(0, h, 10):
            for j in range(0, w, 10):
                node, edge = self.mapping.map_point_to_elements(j, i)
                # Consider mapped if either node or edge is found
                mapped = node is not None or edge is not None
                if mapped:
                    density_grid[i // 10, j // 10] = 1
        
        # Display heatmap
        ax.imshow(density_grid, extent=[0, w, h, 0], alpha=0.7, cmap='RdYlGn')
        
        # Overlay original map
        if len(self.map_image.shape) == 3:
            ax.imshow(self.map_image, alpha=0.5)
        else:
            ax.imshow(self.map_image, alpha=0.5, cmap='gray')
        
        ax.set_title("Coverage Heatmap (Green = Mapped, Red = Unmapped)")
        return fig