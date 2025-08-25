"""Stream info plugin for NaviGraph unified architecture.

Extracts video stream metadata (fps, frame count, duration) and provides as shared resource.
"""

import cv2
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd

from ...core.navigraph_plugin import NaviGraphPlugin
from ...core.exceptions import NavigraphError
from ...core.registry import register_data_source_plugin


@register_data_source_plugin("stream_info")
class StreamInfoPlugin(NaviGraphPlugin):
    """Provides video stream information as a shared resource."""
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.config.get('name', 'stream_info'),
            'type': 'stream_info',
            'description': 'Extracts video stream metadata (fps, frame count, duration)',
            'provides': ['stream_info'],
            'augments': []
        }
    
    def provide(self, shared_resources: Dict[str, Any]) -> None:
        """Extract video metadata and add to shared resources.
        
        Args:
            shared_resources: Dictionary to store stream info
        """
        # Validate we have discovered files
        if not self.discovered_files:
            raise NavigraphError(
                f"StreamInfoPlugin requires video file but none found. "
                f"Check file_pattern in config: {self.config.get('file_pattern')}"
            )
        
        video_file = self.discovered_files[0]  # Use first discovered video
        self.logger.info(f"Extracting stream info from: {video_file.name}")
        
        try:
            # Open video file with OpenCV
            cap = cv2.VideoCapture(str(video_file))
            
            if not cap.isOpened():
                raise NavigraphError(f"Failed to open video file: {video_file}")
            
            # Extract video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            # Release video capture
            cap.release()
            
            # Create stream info dictionary
            stream_info = {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'width': width,
                'height': height,
                'video_path': str(video_file)
            }
            
            # Add to shared resources
            shared_resources['stream_info'] = stream_info
            
            self.logger.info(
                f"✓ Stream info extracted: {frame_count} frames @ {fps:.1f} fps, "
                f"{width}x{height}, duration={duration:.1f}s"
            )
            
        except Exception as e:
            raise NavigraphError(
                f"Failed to extract stream info from {video_file}: {str(e)}"
            ) from e
    
    def augment_data(self, dataframe: pd.DataFrame, shared_resources: Dict[str, Any]) -> pd.DataFrame:
        """Stream info plugin doesn't augment data - only provides resource.
        
        Args:
            dataframe: Current DataFrame
            shared_resources: Available shared resources
            
        Returns:
            DataFrame unchanged
        """
        # Stream info doesn't add columns, just provides metadata
        return dataframe