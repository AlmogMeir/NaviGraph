"""Time series visualizer for NaviGraph.

Displays real-time time series data with ECG-style plotting.
Supports multiple columns as subplots with configurable history window.
"""

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Dict, Any, List, Tuple, Optional, Union
from loguru import logger
from collections import deque

from ..core.registry import register_visualizer

# Module-level cache for time series data (per column)
_time_series_cache = {}  # Key: column_name, Value: {'data': deque or list, 'mode': str}
_frame_counter = 0


@register_visualizer("time_series")
def visualize_time_series(frame: np.ndarray, frame_data: pd.Series, shared_resources: Dict[str, Any], **config) -> np.ndarray:
    """Display real-time time series data with ECG-style plotting.
    
    Args:
        frame: Input video frame (H, W, 3)
        frame_data: DataFrame row for current frame
        shared_resources: Session shared resources (unused)
        **config: Visualization configuration
        
    Config:
        # Data selection
        columns: List of column names to plot or single column name (required)
        
        # Display settings
        mode: 'overlay' or 'side_by_side' (default: 'side_by_side')
        position: Position when mode='overlay' - 'top_left', 'top_right', 'bottom_left', 'bottom_right' (default: 'bottom_right')
        size: Size factor for overlay mode (0.1 to 1.0, default: 0.4)
        opacity: Graph transparency for overlay mode (0.0 to 1.0, default: 0.8)
        
        # Time series settings
        window_size: Number of frames to keep in history (default: 150)
        
        # Plot styling (ECG-style)
        figure_size: [width, height] in inches (default: [8, 6])
        dpi: Figure DPI for resolution (default: 80)
        background_color: Plot background color (default: 'black')
        grid_color: Grid line color (default: '#004400')
        grid_alpha: Grid transparency (default: 0.3)
        
        # Line styling
        colors: List of colors for each subplot, or single color (default: ['#00ff00', '#ff6600', '#0099ff'])
        line_width: Line thickness (default: 2)
        
        # Subplot settings
        subplot_spacing: Vertical spacing between subplots (default: 0.1)
        
        # Axes and labels
        show_axes: Whether to show x/y axes and ticks (default: True)
        show_labels: Whether to show column names as y-labels (default: True)
        show_values: Whether to show current values as text (default: True)
        font_size: Font size for labels and values (default: 10)
        
        # Y-axis range settings
        y_range_mode: 'dynamic' or 'fixed' (default: 'dynamic')
        y_range: Fixed range as [min, max] when y_range_mode='fixed' (default: None)
        y_margin: Margin factor for dynamic range (0.1 = 10% margin, default: 0.1)
        
        # Plotting behavior
        plot_mode: 'continuous' or 'fixed_window' (default: 'continuous')
        # continuous: always plot from frame 0 to current frame (growing line)
        # fixed_window: always plot last N frames (sliding window, current default)
        
        # Performance
        update_interval: Update plot every N frames (default: 1)
        
    Returns:
        Frame with time series overlay applied
    """
    global _time_series_cache, _frame_counter
    
    if frame_data.empty:
        return frame
        
    # Configuration
    columns = config.get('columns', [])
    if isinstance(columns, str):
        columns = [columns]
    if not columns:
        return frame
    
    mode = config.get('mode', 'side_by_side')
    position = config.get('position', 'bottom_right')
    size = config.get('size', 0.4)
    opacity = config.get('opacity', 0.8)
    
    window_size = config.get('window_size', 150)
    update_interval = config.get('update_interval', 1)
    plot_mode = config.get('plot_mode', 'continuous')
    
    # Only update every N frames for performance
    _frame_counter += 1
    if _frame_counter % update_interval != 0:
        # Still need to update cache, but don't redraw
        _update_time_series_cache(columns, frame_data, window_size, plot_mode)
        return frame
    
    # Update time series cache
    _update_time_series_cache(columns, frame_data, window_size, plot_mode)
    
    # Generate plot
    plot_image = _generate_time_series_plot(columns, config)
    
    if plot_image is None:
        return frame
    
    # Apply overlay based on mode
    if mode == 'side_by_side':
        return _apply_side_by_side_mode(frame, plot_image)
    else:  # overlay mode
        return _apply_overlay_mode(frame, plot_image, size, opacity, position)


def _update_time_series_cache(columns: List[str], frame_data: pd.Series, window_size: int, plot_mode: str = 'continuous') -> None:
    """Update the time series cache with current frame data."""
    global _time_series_cache
    
    for column in columns:
        if column not in _time_series_cache:
            if plot_mode == 'fixed_window':
                _time_series_cache[column] = {'data': deque(maxlen=window_size), 'mode': 'fixed_window'}
            else:  # continuous
                _time_series_cache[column] = {'data': [], 'mode': 'continuous'}
        
        # Get current value
        value = frame_data.get(column, np.nan)
        
        if _time_series_cache[column]['mode'] == 'fixed_window':
            _time_series_cache[column]['data'].append(value)
        else:  # continuous
            _time_series_cache[column]['data'].append(value)
            # For continuous mode, keep all data but limit to reasonable size to prevent memory issues
            if len(_time_series_cache[column]['data']) > 10000:  # Keep last 10k points max
                _time_series_cache[column]['data'] = _time_series_cache[column]['data'][-5000:]


def _generate_time_series_plot(columns: List[str], config: Dict[str, Any]) -> Optional[np.ndarray]:
    """Generate ECG-style time series plot."""
    global _time_series_cache
    
    # Plot configuration
    figure_size = config.get('figure_size', [8, 6])
    dpi = config.get('dpi', 80)
    background_color = config.get('background_color', 'black')
    grid_color = config.get('grid_color', '#004400')
    grid_alpha = config.get('grid_alpha', 0.3)
    
    colors = config.get('colors', ['#00ff00', '#ff6600', '#0099ff', '#ff0099', '#ffff00'])
    line_width = config.get('line_width', 2)
    subplot_spacing = config.get('subplot_spacing', 0.1)
    
    show_axes = config.get('show_axes', True)
    show_labels = config.get('show_labels', True)
    show_values = config.get('show_values', True)
    font_size = config.get('font_size', 10)
    
    # Y-axis range settings
    y_range_mode = config.get('y_range_mode', 'dynamic')
    y_range = config.get('y_range', None)
    y_margin = config.get('y_margin', 0.1)
    
    # Plotting behavior
    plot_mode = config.get('plot_mode', 'continuous')
    window_size = config.get('window_size', 150)
    
    # Ensure we have enough colors
    if len(colors) < len(columns):
        colors = colors * ((len(columns) // len(colors)) + 1)
    
    try:
        # Set matplotlib to avoid GUI and manage memory
        plt.ioff()  # Turn off interactive mode
        
        # Create figure with black background
        fig, axes = plt.subplots(len(columns), 1, figsize=figure_size, dpi=dpi, 
                                facecolor=background_color)
        if len(columns) == 1:
            axes = [axes]
        
        fig.patch.set_facecolor(background_color)
        plt.subplots_adjust(hspace=subplot_spacing, left=0.1, right=0.95, top=0.95, bottom=0.05)
        
        for i, column in enumerate(columns):
            ax = axes[i]
            
            # Get data for this column
            if column in _time_series_cache and len(_time_series_cache[column]['data']) > 0:
                data = list(_time_series_cache[column]['data'])
                
                # For fixed_window mode, show all data in window
                # For continuous mode, show last window_size points for performance but keep growing effect
                if plot_mode == 'continuous' and len(data) > window_size:
                    display_data = data[-window_size:]
                    x_offset = len(data) - window_size  # Offset to maintain continuous X axis
                else:
                    display_data = data
                    x_offset = 0
                
                x_values = np.arange(len(display_data)) + x_offset
                
                # Remove NaN values for plotting
                valid_mask = ~np.isnan(display_data)
                if np.any(valid_mask):
                    x_valid = x_values[valid_mask]
                    y_valid = np.array(display_data)[valid_mask]
                    
                    # Plot the line
                    ax.plot(x_valid, y_valid, color=colors[i], linewidth=line_width, 
                           antialiased=True)
                    
                    # Highlight current point
                    if len(y_valid) > 0:
                        ax.plot(x_valid[-1], y_valid[-1], 'o', color=colors[i], 
                               markersize=6, markeredgecolor='white', markeredgewidth=1)
                    
                    # Show current value as text
                    if show_values and len(data) > 0:
                        current_value = data[-1]
                        if not np.isnan(current_value):
                            ax.text(0.02, 0.98, f'{current_value:.3f}', 
                                   transform=ax.transAxes, color=colors[i],
                                   fontsize=font_size + 2, fontweight='bold',
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
                    
                    # Set Y-axis range
                    if y_range_mode == 'fixed' and y_range and len(y_range) == 2:
                        ax.set_ylim(y_range[0], y_range[1])
                    elif y_range_mode == 'dynamic':
                        if len(y_valid) > 0:
                            y_min, y_max = np.min(y_valid), np.max(y_valid)
                            if y_min != y_max:  # Avoid division by zero
                                y_range_size = y_max - y_min
                                margin = y_range_size * y_margin
                                ax.set_ylim(y_min - margin, y_max + margin)
                            else:
                                ax.set_ylim(y_min - 1, y_max + 1)
            
            # Style the subplot (ECG-style)
            ax.set_facecolor(background_color)
            ax.grid(True, color=grid_color, alpha=grid_alpha, linewidth=0.5)
            
            # Handle axes display
            if show_axes:
                ax.tick_params(colors='white', labelsize=font_size-2)
                
                # Remove x-axis for all but bottom subplot
                if i < len(columns) - 1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel('Time (frames)', color='white', fontsize=font_size)
                
                # Y-label
                if show_labels:
                    ax.set_ylabel(column, color=colors[i], fontsize=font_size, fontweight='bold')
                
                # Style spines
                for spine in ax.spines.values():
                    spine.set_color('#333333')
                    spine.set_linewidth(0.5)
            else:
                # Hide all axes elements for cleaner ECG-like look
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Minimal or no labels
                if show_labels:
                    ax.set_ylabel(column, color=colors[i], fontsize=font_size, fontweight='bold')
                
                # Hide spines
                for spine in ax.spines.values():
                    spine.set_visible(False)
        
        # Convert plot to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # Get raw buffer from canvas - try different methods for different matplotlib versions
        try:
            # Modern matplotlib (3.3+)
            buf = canvas.buffer_rgba()
            width, height = canvas.get_width_height()
            plot_image = np.asarray(buf).reshape(height, width, 4)
            # Convert RGBA to RGB
            plot_image = plot_image[:, :, :3]
        except AttributeError:
            try:
                # Older matplotlib
                buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                width, height = canvas.get_width_height()
                plot_image = buf.reshape(height, width, 3)
            except AttributeError:
                # Very old matplotlib or different method needed
                buf = canvas.print_to_buffer()
                width, height = canvas.get_width_height()
                plot_image = np.frombuffer(buf[0], dtype=np.uint8).reshape(height, width, 4)
                # Convert RGBA to RGB
                plot_image = plot_image[:, :, :3]
        
        plt.close(fig)
        plt.clf()  # Clear current figure
        
        return plot_image
        
    except Exception as e:
        logger.error(f"Failed to generate time series plot: {e}")
        # Make sure to clean up even on error
        try:
            plt.close(fig)
        except:
            pass
        plt.clf()
        return None


def _apply_side_by_side_mode(frame: np.ndarray, plot_image: np.ndarray) -> np.ndarray:
    """Apply side-by-side display of video and time series plot."""
    frame_h, frame_w = frame.shape[:2]
    plot_h, plot_w = plot_image.shape[:2]
    
    # Resize plot to match frame height
    aspect_ratio = plot_w / plot_h
    new_height = frame_h
    new_width = int(frame_h * aspect_ratio)
    
    resized_plot = cv2.resize(plot_image, (new_width, new_height))
    
    # Create combined frame
    combined_frame = np.zeros((frame_h, frame_w + new_width, 3), dtype=np.uint8)
    combined_frame[:, :frame_w] = frame
    combined_frame[:, frame_w:frame_w + new_width] = resized_plot
    
    return combined_frame


def _apply_overlay_mode(frame: np.ndarray, plot_image: np.ndarray, 
                       size: float, opacity: float, position: str) -> np.ndarray:
    """Apply overlay mode with plot positioned on frame."""
    frame_h, frame_w = frame.shape[:2]
    plot_h, plot_w = plot_image.shape[:2]
    
    # Resize plot based on size factor
    overlay_w = int(frame_w * size)
    overlay_h = int((overlay_w / plot_w) * plot_h)
    
    # Ensure overlay fits in frame
    if overlay_h > frame_h * 0.8:
        overlay_h = int(frame_h * 0.8)
        overlay_w = int((overlay_h / plot_h) * plot_w)
    
    resized_plot = cv2.resize(plot_image, (overlay_w, overlay_h))
    
    # Calculate position
    x, y = _calculate_overlay_position(frame_w, frame_h, overlay_w, overlay_h, position)
    
    # Apply overlay with opacity
    overlay_region = frame[y:y + overlay_h, x:x + overlay_w]
    blended = cv2.addWeighted(overlay_region, 1 - opacity, resized_plot, opacity, 0)
    frame[y:y + overlay_h, x:x + overlay_w] = blended
    
    return frame


def _calculate_overlay_position(frame_w: int, frame_h: int, overlay_w: int, overlay_h: int, position: str) -> Tuple[int, int]:
    """Calculate overlay position based on position string."""
    margin = 10
    
    if position == 'top_left':
        return margin, margin
    elif position == 'top_right':
        return frame_w - overlay_w - margin, margin
    elif position == 'bottom_left':
        return margin, frame_h - overlay_h - margin
    elif position == 'bottom_right':
        return frame_w - overlay_w - margin, frame_h - overlay_h - margin
    else:  # default to bottom_right
        return frame_w - overlay_w - margin, frame_h - overlay_h - margin