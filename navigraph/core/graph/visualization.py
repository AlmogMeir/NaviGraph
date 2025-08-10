"""Visualization utilities for graph structures and mappings.

This module provides utilities for visualizing graph structures, spatial mappings,
and their relationships in various formats.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

from .structures import GraphStructure
from .mapping import SpatialMapping
from .regions import (SpatialRegion, ContourRegion, RectangleRegion, 
                     CircleRegion, GridCell, HexagonalCell, EllipseRegion)


class GraphVisualizer:
    """Utilities for visualizing graph structures."""
    
    @staticmethod
    def plot_graph(graph: GraphStructure, layout: str = 'spring', 
                   node_size: int = 300, node_color: str = 'lightblue',
                   with_labels: bool = True, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot graph structure.
        
        Args:
            graph: Graph structure to plot
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'planar')
            node_size: Size of nodes
            node_color: Color of nodes
            with_labels: Whether to show node labels
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(graph.graph, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(graph.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph.graph)
        elif layout == 'planar' and nx.is_planar(graph.graph):
            pos = nx.planar_layout(graph.graph)
        else:
            pos = nx.spring_layout(graph.graph, seed=42)
        
        # Use stored positions if available
        if graph.node_positions:
            pos = graph.node_positions
        
        # Draw graph
        nx.draw(graph.graph, pos=pos, ax=ax,
                node_size=node_size, node_color=node_color,
                with_labels=with_labels, font_size=10, font_weight='bold',
                edge_color='gray', linewidths=1)
        
        ax.set_title(f"Graph Structure ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")
        
        return fig
    
    @staticmethod
    def plot_graph_with_metrics(graph: GraphStructure, node_metrics: Dict[Any, float],
                               metric_name: str = 'Value', colormap: str = 'viridis',
                               layout: str = 'spring', figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot graph with node colors based on metrics.
        
        Args:
            graph: Graph structure to plot
            node_metrics: Dictionary mapping node IDs to metric values
            metric_name: Name of the metric for display
            colormap: Matplotlib colormap name
            layout: Layout algorithm
            figsize: Figure size
            
        Returns:
            Matplotlib figure with colorbar
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(graph.graph, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(graph.graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph.graph)
        else:
            pos = nx.spring_layout(graph.graph, seed=42)
        
        # Use stored positions if available
        if graph.node_positions:
            pos = graph.node_positions
        
        # Create color array based on metrics
        node_colors = []
        for node in graph.nodes:
            value = node_metrics.get(node, 0.0)
            node_colors.append(value)
        
        # Draw graph with colored nodes
        nodes = nx.draw_networkx_nodes(graph.graph, pos=pos, ax=ax,
                                     node_color=node_colors, node_size=300,
                                     cmap=plt.cm.get_cmap(colormap))
        
        nx.draw_networkx_edges(graph.graph, pos=pos, ax=ax, edge_color='gray')
        nx.draw_networkx_labels(graph.graph, pos=pos, ax=ax, font_size=8, font_weight='bold')
        
        # Add colorbar
        plt.colorbar(nodes, ax=ax, label=metric_name)
        
        ax.set_title(f"Graph with {metric_name} ({len(graph.nodes)} nodes)")
        
        return fig


class MappingVisualizer:
    """Utilities for visualizing spatial mappings."""
    
    @staticmethod
    def plot_mapping_overview(mapping: SpatialMapping, map_image: np.ndarray,
                             show_labels: bool = True, show_overlaps: bool = True,
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Plot complete mapping overview.
        
        Args:
            mapping: Spatial mapping to visualize
            map_image: Map image as numpy array
            show_labels: Whether to show node labels
            show_overlaps: Whether to highlight overlaps
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, (ax_map, ax_graph) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot map with regions
        MappingVisualizer._plot_map_with_regions(ax_map, mapping, map_image, 
                                               show_labels, show_overlaps)
        
        # Plot graph structure
        if mapping.graph:
            MappingVisualizer._plot_mapped_graph(ax_graph, mapping)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _plot_map_with_regions(ax: plt.Axes, mapping: SpatialMapping, 
                              map_image: np.ndarray, show_labels: bool,
                              show_overlaps: bool):
        """Plot map with spatial regions."""
        # Display map image
        if len(map_image.shape) == 3:
            ax.imshow(map_image)
        else:
            ax.imshow(map_image, cmap='gray')
        
        # Draw all regions
        colors = plt.cm.Set3(np.linspace(0, 1, len(mapping.get_all_regions())))
        
        for i, (region_id, region) in enumerate(mapping.get_all_regions().items()):
            color = colors[i % len(colors)]
            node_id = mapping.get_node_for_region(region_id)
            
            MappingVisualizer._draw_region_on_axes(ax, region, color, alpha=0.3)
            
            if show_labels and node_id is not None:
                center = region.get_center()
                ax.text(center.x, center.y, str(node_id),
                       ha='center', va='center', fontsize=10,
                       color='white', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', 
                               facecolor='black', alpha=0.7))
        
        # Highlight overlaps
        if show_overlaps:
            overlaps = mapping.find_overlaps()
            for overlap in overlaps:
                region1 = mapping.get_region_by_id(overlap.region1_id)
                region2 = mapping.get_region_by_id(overlap.region2_id)
                
                if region1:
                    center1 = region1.get_center()
                    ax.plot(center1.x, center1.y, 'rx', markersize=15, markeredgewidth=4)
                if region2:
                    center2 = region2.get_center()
                    ax.plot(center2.x, center2.y, 'rx', markersize=15, markeredgewidth=4)
        
        ax.set_title("Spatial Regions on Map")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
    
    @staticmethod
    def _plot_mapped_graph(ax: plt.Axes, mapping: SpatialMapping):
        """Plot graph with mapped/unmapped node visualization."""
        graph = mapping.graph
        if not graph:
            ax.text(0.5, 0.5, "No graph available", ha='center', va='center', 
                   transform=ax.transAxes)
            return
        
        # Determine layout
        if graph.node_positions:
            pos = graph.node_positions
        else:
            pos = nx.spring_layout(graph.graph, seed=42)
        
        # Color nodes based on mapping status
        node_colors = []
        for node in graph.nodes:
            if node in mapping.get_mapped_nodes():
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightcoral')
        
        # Draw graph
        nx.draw(graph.graph, pos=pos, ax=ax,
                node_color=node_colors, node_size=300,
                with_labels=True, font_size=8, font_weight='bold',
                edge_color='gray')
        
        # Add legend
        mapped_patch = patches.Patch(color='lightgreen', label='Mapped')
        unmapped_patch = patches.Patch(color='lightcoral', label='Unmapped')
        ax.legend(handles=[mapped_patch, unmapped_patch], loc='upper right')
        
        ax.set_title("Graph Structure (Mapping Status)")
    
    @staticmethod
    def _draw_region_on_axes(ax: plt.Axes, region: SpatialRegion, 
                            color, alpha: float = 0.3):
        """Draw a spatial region on matplotlib axes."""
        if isinstance(region, ContourRegion):
            contour = np.array(region.contour_points)
            ax.fill(contour[:, 0], contour[:, 1], alpha=alpha, color=color)
        
        elif isinstance(region, (RectangleRegion, GridCell)):
            rect = patches.Rectangle((region.x, region.y), 
                                   region.width, region.height,
                                   linewidth=1, edgecolor=color,
                                   facecolor=color, alpha=alpha)
            ax.add_patch(rect)
        
        elif isinstance(region, CircleRegion):
            circle = patches.Circle((region.center_x, region.center_y),
                                  region.radius, linewidth=1,
                                  edgecolor=color, facecolor=color, alpha=alpha)
            ax.add_patch(circle)
        
        elif isinstance(region, HexagonalCell):
            # Draw hexagon using vertices
            vertices = np.array(region.vertices)
            polygon = patches.Polygon(vertices, linewidth=1,
                                    edgecolor=color, facecolor=color, alpha=alpha)
            ax.add_patch(polygon)
        
        elif isinstance(region, EllipseRegion):
            ellipse = patches.Ellipse((region.center_x, region.center_y),
                                    2 * region.radius_x, 2 * region.radius_y,
                                    angle=np.degrees(region.angle),
                                    linewidth=1, edgecolor=color,
                                    facecolor=color, alpha=alpha)
            ax.add_patch(ellipse)
    
    @staticmethod
    def plot_path_on_mapping(mapping: SpatialMapping, map_image: np.ndarray,
                            path: List[Any], figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot a path through the mapped regions.
        
        Args:
            mapping: Spatial mapping
            map_image: Map image as numpy array
            path: List of node IDs representing the path
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Display map
        if len(map_image.shape) == 3:
            ax.imshow(map_image)
        else:
            ax.imshow(map_image, cmap='gray')
        
        # Draw all regions (faded)
        for region_id, region in mapping.get_all_regions().items():
            MappingVisualizer._draw_region_on_axes(ax, region, 'lightblue', alpha=0.2)
        
        # Draw path
        if len(path) >= 2:
            path_points = []
            for node in path:
                regions = mapping.get_node_regions(node)
                if regions:
                    center = regions[0].get_center()
                    path_points.append((center.x, center.y))
            
            if len(path_points) >= 2:
                path_array = np.array(path_points)
                ax.plot(path_array[:, 0], path_array[:, 1], 
                       'r-', linewidth=3, alpha=0.8, label='Path')
                ax.plot(path_array[:, 0], path_array[:, 1], 
                       'ro', markersize=8)
                
                # Label start and end
                ax.text(path_points[0][0], path_points[0][1], 'START',
                       ha='center', va='center', fontsize=8, weight='bold',
                       bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))
                ax.text(path_points[-1][0], path_points[-1][1], 'END',
                       ha='center', va='center', fontsize=8, weight='bold',
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
        
        ax.set_title(f"Path Visualization ({len(path)} nodes)")
        ax.legend()
        
        return fig
    
    @staticmethod
    def create_mapping_report_figure(mapping: SpatialMapping, 
                                    map_image: np.ndarray) -> plt.Figure:
        """Create a comprehensive mapping report figure.
        
        Args:
            mapping: Spatial mapping to report
            map_image: Map image as numpy array
            
        Returns:
            Matplotlib figure with multiple subplots
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main mapping overview (top row, spans 2 columns)
        ax_main = fig.add_subplot(gs[0, :2])
        MappingVisualizer._plot_map_with_regions(ax_main, mapping, map_image, 
                                               show_labels=True, show_overlaps=True)
        
        # Graph structure (top right)
        ax_graph = fig.add_subplot(gs[0, 2])
        if mapping.graph:
            MappingVisualizer._plot_mapped_graph(ax_graph, mapping)
        
        # Statistics panel (middle left)
        ax_stats = fig.add_subplot(gs[1, 0])
        ax_stats.axis('off')
        stats = mapping.validate_mapping()
        stats_text = f"""Mapping Statistics:
        
Total Nodes: {stats.total_nodes}
Mapped Nodes: {stats.mapped_nodes}
Unmapped Nodes: {stats.unmapped_nodes}
Mapping Completeness: {stats.mapping_completeness:.1f}%

Total Regions: {stats.total_regions}
Overlapping Regions: {stats.overlapping_regions}
Coverage: {stats.coverage_percentage:.1f}%

Overlaps Detected: {len(stats.overlaps)}"""
        
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax_stats.set_title("Statistics")
        
        # Region size distribution (middle center)
        ax_sizes = fig.add_subplot(gs[1, 1])
        region_sizes = [region.get_area() for region in mapping.get_all_regions().values()]
        if region_sizes:
            ax_sizes.hist(region_sizes, bins=20, alpha=0.7, color='skyblue')
            ax_sizes.set_xlabel("Region Area")
            ax_sizes.set_ylabel("Count")
            ax_sizes.set_title("Region Size Distribution")
        
        # Node degree distribution (middle right)
        ax_degrees = fig.add_subplot(gs[1, 2])
        if mapping.graph:
            degrees = [mapping.graph.graph.degree(node) for node in mapping.graph.nodes]
            ax_degrees.hist(degrees, bins=max(1, len(set(degrees))), 
                           alpha=0.7, color='lightcoral')
            ax_degrees.set_xlabel("Node Degree")
            ax_degrees.set_ylabel("Count")
            ax_degrees.set_title("Node Degree Distribution")
        
        # Overlap details (bottom row)
        if stats.overlaps:
            ax_overlaps = fig.add_subplot(gs[2, :])
            ax_overlaps.axis('off')
            overlap_text = "Detected Overlaps:\n\n"
            for i, overlap in enumerate(stats.overlaps[:10], 1):  # Show first 10
                overlap_text += f"{i}. {overlap}\n"
            if len(stats.overlaps) > 10:
                overlap_text += f"... and {len(stats.overlaps) - 10} more"
            
            ax_overlaps.text(0.05, 0.95, overlap_text, transform=ax_overlaps.transAxes,
                           fontsize=9, verticalalignment='top', fontfamily='monospace')
            ax_overlaps.set_title("Overlap Details")
        
        plt.suptitle("NaviGraph Spatial Mapping Report", fontsize=16, fontweight='bold')
        
        return fig


def save_visualization(fig: plt.Figure, filepath: Union[str, Path], 
                      dpi: int = 300, format: str = 'png') -> bool:
    """Save visualization figure to file.
    
    Args:
        fig: Matplotlib figure to save
        filepath: Output file path
        dpi: Resolution for raster formats
        format: File format ('png', 'pdf', 'svg', 'eps')
        
    Returns:
        True if save successful
    """
    try:
        fig.savefig(filepath, dpi=dpi, format=format, bbox_inches='tight')
        return True
    except Exception as e:
        print(f"Failed to save visualization: {e}")
        return False