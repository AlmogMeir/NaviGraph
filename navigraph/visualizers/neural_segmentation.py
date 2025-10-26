"""Neural spatial segmentation visualizer for NaviGraph.

Displays neural activity video frames with overlaid neuron contours from spatial footprints,
synchronized with the main navigation video frame index. Supports multiple display modes
including side-by-side layout for comprehensive behavioral-neural visualization.
"""

import numpy as np
import cv2
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from ..core.registry import register_visualizer


@register_visualizer("neural_segmentation")
def visualize_neural_segmentation(
    frame: np.ndarray,
    frame_data: pd.Series,
    shared_resources: Dict[str, Any],
    **config
) -> np.ndarray:
    """Visualize neural activity video with segmentation contours.

    Args:
        frame: Input video frame (H, W, 3)
        frame_data: DataFrame row for current frame with neural contour data
        shared_resources: Session shared resources containing 'neural_video'
        **config: Visualization configuration

    Config:
        # Display modes
        mode: 'side_by_side', 'overlay', 'replace' (default: 'side_by_side')

        # Side-by-side mode settings
        panel_width: 'auto' or specific pixel width (default: 'auto')

        # Overlay mode settings
        position: 'top_right', 'top_left', 'bottom_right', 'bottom_left'
        size: Scale factor for overlay (0.1 to 1.0, default: 0.3)
        opacity: Transparency for overlay (0.0 to 1.0, default: 0.7)

        # Contour selection
        columns: List of contour column names or null for all (default: null)

        # Activity filtering and modulation
        activity_threshold: Minimum activity level to show contour (default: 0.05)
        normalize_activity: Normalize activity values to 0-1 range (default: True)
        modulate_opacity: Vary contour opacity based on activity (default: True)
        modulate_color: Vary color brightness based on activity (default: False)
        max_neurons_per_frame: Limit neurons shown for performance (default: 50)

        # Visual styling
        contour_thickness: Line thickness for contour outlines (default: 2)
        contour_fill: Whether to fill contours semi-transparently (default: True)
        fill_opacity: Opacity for filled contours (default: 0.3)

        # Label settings
        show_labels: Whether to show neuron ID labels (default: True)
        show_activity_values: Include activity value in labels (default: False)
        label_font_scale: Text size for labels (default: 0.4)
        label_position: 'centroid', 'top_left', 'bottom_right' (default: 'centroid')
        label_offset: [x, y] pixel offset from position (default: [5, -5])
        label_background: Whether to draw background behind text (default: True)

        # Color settings
        color_scheme: 'auto', 'manual' (default: 'auto')
        custom_colors: Dict mapping neuron IDs to [B,G,R] colors (for manual scheme)

    Returns:
        Frame with neural segmentation applied based on mode
    """
    # Check if neural video is available
    neural_video_info = shared_resources.get('neural_video')
    if not neural_video_info:
        return frame  # No neural video, return unchanged

    # Get synchronized neural frame
    neural_frame = _get_neural_frame(config, frame_data, neural_video_info)
    if neural_frame is None:
        return frame  # Could not read neural frame

    # PERFORMANCE OPTIMIZATION: Resize BEFORE drawing for much faster rendering
    mode = config.get('mode', 'side_by_side')

    if mode == 'side_by_side':
        # Calculate target size first, then resize before drawing
        target_size = _calculate_target_size_for_side_by_side(frame, neural_frame, config)
        neural_frame = cv2.resize(neural_frame, target_size)

        # Draw contours on smaller frame (much faster)
        neural_frame = _draw_contours_on_frame(neural_frame, frame_data, config, target_size)

        # Concatenate without additional resizing
        return _apply_side_by_side_direct(frame, neural_frame)

    elif mode == 'overlay':
        # For overlay, resize to overlay size first
        target_size = _calculate_target_size_for_overlay(frame, neural_frame, config)
        neural_frame = cv2.resize(neural_frame, target_size)

        # Draw contours on smaller frame
        neural_frame = _draw_contours_on_frame(neural_frame, frame_data, config, target_size)

        return _apply_overlay_direct(frame, neural_frame, config)

    elif mode == 'replace':
        # For replace mode, draw on original size or resize as configured
        neural_frame = _draw_contours_on_frame(neural_frame, frame_data, config)
        return neural_frame
    else:
        # Unknown mode, default to side_by_side
        target_size = _calculate_target_size_for_side_by_side(frame, neural_frame, config)
        neural_frame = cv2.resize(neural_frame, target_size)
        neural_frame = _draw_contours_on_frame(neural_frame, frame_data, config, target_size)
        return _apply_side_by_side_direct(frame, neural_frame)


def _get_neural_frame(
    config: Dict[str, Any],
    frame_data: pd.Series,
    neural_video_info: Dict[str, Any]
) -> Optional[np.ndarray]:
    """Get synchronized frame from neural video.

    Args:
        config: Visualizer config (for caching video capture)
        frame_data: Current frame data containing frame index
        neural_video_info: Neural video metadata

    Returns:
        Neural video frame or None if failed
    """
    # Initialize video capture once (cached in config)
    if '_video_capture' not in config:
        config['_video_capture'] = cv2.VideoCapture(neural_video_info['path'])
        if not config['_video_capture'].isOpened():
            return None
        config['_neural_frame_count'] = int(
            config['_video_capture'].get(cv2.CAP_PROP_FRAME_COUNT)
        )

    # Get frame index from frame_data
    if hasattr(frame_data, 'name'):
        frame_idx = frame_data.name
    elif 'frame' in frame_data:
        frame_idx = int(frame_data['frame'])
    else:
        frame_idx = 0

    # Handle frame count mismatch between videos
    if frame_idx >= config['_neural_frame_count']:
        # Use last frame if beyond neural video length
        frame_idx = config['_neural_frame_count'] - 1
    elif frame_idx < 0:
        frame_idx = 0

    # Seek to frame and read
    cap = config['_video_capture']
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, neural_frame = cap.read()

    return neural_frame if ret else None


def _draw_contours_on_frame(
    neural_frame: np.ndarray,
    frame_data: pd.Series,
    config: Dict[str, Any],
    target_size: Tuple[int, int] = None
) -> np.ndarray:
    """Draw neuron contours on neural activity frame with activity-based filtering.

    Args:
        neural_frame: Neural video frame to draw on (may be resized)
        frame_data: Current frame data containing activity-modulated contour columns
        config: Visualizer configuration
        target_size: (width, height) if frame has been resized, for contour scaling

    Returns:
        Neural frame with contours drawn
    """
    # Find available contour columns
    all_contour_columns = [col for col in frame_data.index if col.endswith('_A_contour')]

    # Filter based on config
    columns_config = config.get('columns')
    if columns_config is None:
        # null means all columns
        contour_columns = all_contour_columns
    elif isinstance(columns_config, list):
        # Specific columns requested
        contour_columns = [col for col in columns_config if col in all_contour_columns]
    else:
        # Invalid config, use all
        contour_columns = all_contour_columns

    if not contour_columns:
        return neural_frame  # No contours to draw

    # Configuration
    activity_threshold = config.get('activity_threshold', 0.05)
    normalize_activity = config.get('normalize_activity', True)
    modulate_opacity = config.get('modulate_opacity', True)
    modulate_color = config.get('modulate_color', False)
    max_neurons_per_frame = config.get('max_neurons_per_frame', 50)
    contour_thickness = config.get('contour_thickness', 2)
    contour_fill = config.get('contour_fill', True)  # NEW: Can be disabled for performance
    fill_opacity = config.get('fill_opacity', 0.3)

    # Performance optimizations
    downsample_contours = config.get('downsample_contours', False)
    contour_simplify_tolerance = config.get('contour_simplify_tolerance', 2.0)
    label_every_n_neurons = config.get('label_every_n_neurons', 1)
    label_min_activity = config.get('label_min_activity', 0.0)

    # Calculate scaling factors if frame was resized
    original_size = config.get('_original_neural_size')
    scale_x = scale_y = 1.0
    if target_size and original_size:
        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]

    # Extract and filter contour data based on activity
    active_contours = []
    for col in contour_columns:
        try:
            # Extract neuron ID from column name (unit_id_X_A_contour)
            neuron_id = col.split('_')[2]
            contour_data = frame_data[col]

            # Handle new tuple format (contour_points, activity_value)
            if isinstance(contour_data, tuple) and len(contour_data) == 2:
                contour, activity = contour_data

                # Filter by activity threshold
                if activity > activity_threshold and contour and len(contour) > 0:
                    active_contours.append({
                        'neuron_id': neuron_id,
                        'contour': contour,
                        'activity': activity,
                        'col_index': len(active_contours)  # For color mapping
                    })
            # Fallback for old format (just contour points)
            elif isinstance(contour_data, list) and len(contour_data) > 0:
                active_contours.append({
                    'neuron_id': neuron_id,
                    'contour': contour_data,
                    'activity': 1.0,  # Default activity
                    'col_index': len(active_contours)
                })

        except (IndexError, ValueError, TypeError):
            continue

    if not active_contours:
        return neural_frame  # No active contours to draw

    # Limit number of neurons for performance
    if len(active_contours) > max_neurons_per_frame:
        # Sort by activity and take top N
        active_contours = sorted(active_contours, key=lambda x: x['activity'], reverse=True)[:max_neurons_per_frame]

    # Initialize color mapping based on total neurons seen
    total_neurons = len(contour_columns)
    if '_neuron_colors' not in config:
        config['_neuron_colors'] = _generate_distinct_colors(total_neurons)

    # Normalize activity values for visual modulation
    if normalize_activity and active_contours:
        activities = [c['activity'] for c in active_contours]
        max_activity = max(activities)
        min_activity = min(activities)
        activity_range = max_activity - min_activity

        if activity_range > 0:
            for contour_info in active_contours:
                contour_info['normalized_activity'] = (contour_info['activity'] - min_activity) / activity_range
        else:
            for contour_info in active_contours:
                contour_info['normalized_activity'] = 1.0
    else:
        for contour_info in active_contours:
            contour_info['normalized_activity'] = 1.0

    # OPTIMIZED: Single overlay for all filled contours
    overlay = None
    if contour_fill:
        overlay = neural_frame.copy()

    # Draw each active contour
    for contour_info in active_contours:
        try:
            neuron_id = contour_info['neuron_id']
            contour = contour_info['contour']
            activity = contour_info['activity']
            normalized_activity = contour_info['normalized_activity']

            # Convert contour to numpy array and scale if needed
            points = np.array(contour, dtype=np.float32)

            # Scale contour points if frame was resized
            if scale_x != 1.0 or scale_y != 1.0:
                points[:, 0] *= scale_x  # Scale x coordinates
                points[:, 1] *= scale_y  # Scale y coordinates

            # Simplify contour for performance if requested
            if downsample_contours and len(points) > 10:
                # Use Douglas-Peucker algorithm to reduce points
                epsilon = contour_simplify_tolerance
                points = cv2.approxPolyDP(points, epsilon, True)
                points = points.reshape(-1, 2)

            # Convert to integers for drawing
            points = points.astype(np.int32)

            # Get base color for this neuron
            color_idx = int(neuron_id) % len(config['_neuron_colors'])
            base_color = config['_neuron_colors'][color_idx]

            # Modulate color/opacity based on activity
            if modulate_color:
                # Scale color brightness based on activity
                color = tuple(int(c * (0.3 + 0.7 * normalized_activity)) for c in base_color)
            else:
                color = base_color

            # Calculate dynamic opacity
            if modulate_opacity:
                dynamic_fill_opacity = fill_opacity * (0.2 + 0.8 * normalized_activity)
            else:
                dynamic_fill_opacity = fill_opacity

            # Draw filled contour on overlay (single overlay for performance)
            if contour_fill and overlay is not None:
                cv2.fillPoly(overlay, [points], color)

            # Draw contour outline directly on frame
            cv2.drawContours(neural_frame, [points], -1, color, contour_thickness)

            # Draw label if requested (with filtering for performance)
            should_draw_label = (
                config.get('show_labels', True) and
                (len(active_contours) % label_every_n_neurons == 0) and
                activity >= label_min_activity
            )
            if should_draw_label:
                neural_frame = _draw_neuron_label(neural_frame, points, neuron_id, color, config, activity)

        except Exception as e:
            # Skip invalid contour data
            continue

    # Apply overlay once for all filled contours (MUCH faster than per-contour)
    if contour_fill and overlay is not None:
        neural_frame = cv2.addWeighted(neural_frame, 1 - fill_opacity, overlay, fill_opacity, 0)

    return neural_frame


def _calculate_target_size_for_side_by_side(
    main_frame: np.ndarray,
    neural_frame: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[int, int]:
    """Calculate target size for neural frame in side-by-side mode.

    Args:
        main_frame: Main video frame
        neural_frame: Neural video frame
        config: Visualizer configuration

    Returns:
        (width, height) tuple for target size
    """
    # Store original size for scaling calculations
    config['_original_neural_size'] = (neural_frame.shape[1], neural_frame.shape[0])

    frame_h = main_frame.shape[0]
    neural_h, neural_w = neural_frame.shape[:2]

    panel_width = config.get('panel_width', 'auto')

    if panel_width == 'auto':
        # Scale to match height, maintain aspect ratio
        scale = frame_h / neural_h
        new_width = int(neural_w * scale)
        new_height = frame_h
    else:
        # Use specified width
        try:
            new_width = int(panel_width)
            new_height = frame_h
        except (ValueError, TypeError):
            # Fallback to auto
            scale = frame_h / neural_h
            new_width = int(neural_w * scale)
            new_height = frame_h

    return (new_width, new_height)


def _calculate_target_size_for_overlay(
    main_frame: np.ndarray,
    neural_frame: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[int, int]:
    """Calculate target size for neural frame in overlay mode.

    Args:
        main_frame: Main video frame
        neural_frame: Neural video frame
        config: Visualizer configuration

    Returns:
        (width, height) tuple for target size
    """
    # Store original size for scaling calculations
    config['_original_neural_size'] = (neural_frame.shape[1], neural_frame.shape[0])

    frame_h, frame_w = main_frame.shape[:2]
    neural_h, neural_w = neural_frame.shape[:2]

    size = config.get('size', 0.3)

    # Calculate overlay dimensions
    overlay_w = int(frame_w * size)
    overlay_h = int((overlay_w / neural_w) * neural_h)

    # Ensure overlay fits in frame
    if overlay_h > frame_h * 0.8:
        overlay_h = int(frame_h * 0.8)
        overlay_w = int((overlay_h / neural_h) * neural_w)

    return (overlay_w, overlay_h)


def _apply_side_by_side_direct(
    main_frame: np.ndarray,
    neural_frame: np.ndarray
) -> np.ndarray:
    """Apply side-by-side concatenation without resizing (frame already sized).

    Args:
        main_frame: Main video frame
        neural_frame: Neural frame already resized to target

    Returns:
        Combined frame
    """
    return np.hstack([main_frame, neural_frame])


def _apply_overlay_direct(
    main_frame: np.ndarray,
    neural_frame: np.ndarray,
    config: Dict[str, Any]
) -> np.ndarray:
    """Apply overlay without resizing (frame already sized).

    Args:
        main_frame: Main video frame
        neural_frame: Neural frame already resized to overlay size
        config: Visualizer configuration

    Returns:
        Main frame with neural overlay
    """
    frame_h, frame_w = main_frame.shape[:2]
    overlay_h, overlay_w = neural_frame.shape[:2]

    position = config.get('position', 'top_right')
    opacity = config.get('opacity', 0.7)

    # Calculate position
    positions = {
        'top_right': (frame_w - overlay_w - 10, 10),
        'top_left': (10, 10),
        'bottom_right': (frame_w - overlay_w - 10, frame_h - overlay_h - 10),
        'bottom_left': (10, frame_h - overlay_h - 10)
    }

    x, y = positions.get(position, positions['top_right'])

    # Ensure overlay doesn't go out of bounds
    x = max(0, min(x, frame_w - overlay_w))
    y = max(0, min(y, frame_h - overlay_h))

    # Apply overlay with opacity
    overlay_region = main_frame[y:y + overlay_h, x:x + overlay_w]
    blended = cv2.addWeighted(overlay_region, 1 - opacity, neural_frame, opacity, 0)
    main_frame[y:y + overlay_h, x:x + overlay_w] = blended

    return main_frame


def _draw_neuron_label(
    frame: np.ndarray,
    contour_points: np.ndarray,
    neuron_id: str,
    color: Tuple[int, int, int],
    config: Dict[str, Any],
    activity: float = None
) -> np.ndarray:
    """Draw neuron ID label next to contour.

    Args:
        frame: Frame to draw on
        contour_points: Contour points array
        neuron_id: Neuron identifier
        color: Color for label (matching contour)
        config: Visualizer configuration
        activity: Current activity value (optional)

    Returns:
        Frame with label drawn
    """
    # Create label with activity if available
    if activity is not None and config.get('show_activity_values', False):
        label = f"ID:{neuron_id} ({activity:.2f})"
    else:
        label = f"ID:{neuron_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = config.get('label_font_scale', 0.4)
    thickness = 1

    # Calculate label position
    label_position_type = config.get('label_position', 'centroid')
    label_offset = config.get('label_offset', [5, -5])

    if label_position_type == 'centroid':
        # Use centroid of contour
        moments = cv2.moments(contour_points)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            # Fallback to bounding box center
            x, y, w, h = cv2.boundingRect(contour_points)
            cx, cy = x + w // 2, y + h // 2
        position = (cx + label_offset[0], cy + label_offset[1])
    elif label_position_type == 'top_left':
        # Use top-left of bounding box
        x, y, w, h = cv2.boundingRect(contour_points)
        position = (x + label_offset[0], y + label_offset[1])
    elif label_position_type == 'bottom_right':
        # Use bottom-right of bounding box
        x, y, w, h = cv2.boundingRect(contour_points)
        position = (x + w + label_offset[0], y + h + label_offset[1])
    else:
        # Default to centroid
        position = (int(contour_points[:, 0].mean()) + label_offset[0],
                   int(contour_points[:, 1].mean()) + label_offset[1])

    # Draw background rectangle if requested
    if config.get('label_background', True):
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Background rectangle
        bg_pt1 = (position[0] - 2, position[1] - text_height - 2)
        bg_pt2 = (position[0] + text_width + 2, position[1] + baseline + 2)
        cv2.rectangle(frame, bg_pt1, bg_pt2, (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, label, position, font, font_scale, color, thickness)

    return frame


def _generate_distinct_colors(n_neurons: int) -> List[Tuple[int, int, int]]:
    """Generate visually distinct colors using HSV space.

    Args:
        n_neurons: Number of colors to generate

    Returns:
        List of (B, G, R) color tuples
    """
    colors = []

    # Use golden ratio for better color distribution
    golden_ratio = 0.618033988749895
    hue = 0

    for i in range(n_neurons):
        hue += golden_ratio
        hue %= 1.0

        # Convert HSV to BGR (OpenCV format)
        # Use high saturation and value for vibrant colors
        hsv = np.array([[[int(hue * 179), 255, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in bgr))

    return colors


def _apply_side_by_side(
    main_frame: np.ndarray,
    neural_frame: np.ndarray,
    config: Dict[str, Any]
) -> np.ndarray:
    """Apply side-by-side display mode.

    Args:
        main_frame: Main video frame (left side)
        neural_frame: Neural video frame (to be placed on right)
        config: Visualizer configuration

    Returns:
        Combined frame with neural frame on the right
    """
    frame_h = main_frame.shape[0]
    neural_h, neural_w = neural_frame.shape[:2]

    # Handle panel width setting
    panel_width = config.get('panel_width', 'auto')

    if panel_width == 'auto':
        # Scale neural frame to match main frame height, maintain aspect ratio
        scale = frame_h / neural_h
        new_width = int(neural_w * scale)
        new_height = frame_h
    else:
        # Use specified width
        try:
            new_width = int(panel_width)
            new_height = frame_h
        except (ValueError, TypeError):
            # Fallback to auto if invalid
            scale = frame_h / neural_h
            new_width = int(neural_w * scale)
            new_height = frame_h

    # Resize neural frame
    neural_resized = cv2.resize(neural_frame, (new_width, new_height))

    # Concatenate horizontally
    combined = np.hstack([main_frame, neural_resized])

    return combined


def _apply_overlay(
    main_frame: np.ndarray,
    neural_frame: np.ndarray,
    config: Dict[str, Any]
) -> np.ndarray:
    """Apply overlay display mode.

    Args:
        main_frame: Main video frame (background)
        neural_frame: Neural video frame (overlay)
        config: Visualizer configuration

    Returns:
        Main frame with neural frame overlaid
    """
    frame_h, frame_w = main_frame.shape[:2]
    neural_h, neural_w = neural_frame.shape[:2]

    # Configuration
    position = config.get('position', 'top_right')
    size = config.get('size', 0.3)
    opacity = config.get('opacity', 0.7)

    # Calculate overlay dimensions
    overlay_w = int(frame_w * size)
    overlay_h = int((overlay_w / neural_w) * neural_h)

    # Ensure overlay fits in frame
    if overlay_h > frame_h * 0.8:
        overlay_h = int(frame_h * 0.8)
        overlay_w = int((overlay_h / neural_h) * neural_w)

    # Resize neural frame for overlay
    neural_resized = cv2.resize(neural_frame, (overlay_w, overlay_h))

    # Calculate position
    positions = {
        'top_right': (frame_w - overlay_w - 10, 10),
        'top_left': (10, 10),
        'bottom_right': (frame_w - overlay_w - 10, frame_h - overlay_h - 10),
        'bottom_left': (10, frame_h - overlay_h - 10)
    }

    x, y = positions.get(position, positions['top_right'])

    # Ensure overlay doesn't go out of bounds
    x = max(0, min(x, frame_w - overlay_w))
    y = max(0, min(y, frame_h - overlay_h))

    # Apply overlay with opacity
    overlay_region = main_frame[y:y + overlay_h, x:x + overlay_w]
    blended = cv2.addWeighted(overlay_region, 1 - opacity, neural_resized, opacity, 0)
    main_frame[y:y + overlay_h, x:x + overlay_w] = blended

    return main_frame


def cleanup_neural_segmentation(config: Dict[str, Any]) -> None:
    """Clean up neural segmentation resources.

    Args:
        config: Visualizer config containing cached resources
    """
    if '_video_capture' in config:
        config['_video_capture'].release()
        del config['_video_capture']

    if '_neural_frame_count' in config:
        del config['_neural_frame_count']

    if '_neuron_colors' in config:
        del config['_neuron_colors']