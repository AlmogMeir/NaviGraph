"""Publishers for visualization outputs.

Publishers handle the final output of visualization pipelines, converting
processed data into files (video, images, etc.). This separates the 
visualization logic from file I/O operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, Union, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
from loguru import logger
import matplotlib.pyplot as plt


class Publisher(ABC):
    """Base class for all visualization publishers.
    
    Publishers are responsible for taking processed visualization data
    and writing it to the appropriate output format.
    """
    
    def __init__(self, output_path: str, output_name: str):
        """Initialize publisher with output location.
        
        Args:
            output_path: Directory where output will be saved
            output_name: Base name for output file(s)
        """
        self.output_path = Path(output_path)
        self.output_name = output_name
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def publish(self, data: Any, **kwargs) -> Optional[str]:
        """Publish visualization data to output format.
        
        Args:
            data: Processed visualization data
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to created output file(s), or None if failed
        """
        pass


class VideoPublisher(Publisher):
    """Publisher for video outputs.
    
    Takes an iterator of frames and writes them to a video file.
    """
    
    def publish(
        self, 
        frames: Iterator[np.ndarray],
        fps: float = 30.0,
        codec: str = 'mp4v',
        **kwargs
    ) -> Optional[str]:
        """Publish frames as a video file.
        
        Args:
            frames: Iterator yielding numpy arrays (H, W, C)
            fps: Frames per second for output video
            codec: FourCC codec string
            **kwargs: Additional parameters
            
        Returns:
            Path to created video file
        """
        output_file = self.output_path / f"{self.output_name}.mp4"
        writer = None
        
        try:
            frame_count = 0
            for i, frame in enumerate(frames):
                if writer is None:
                    # Initialize writer with first frame dimensions
                    height, width = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(
                        str(output_file),
                        fourcc,
                        fps,
                        (width, height)
                    )
                
                writer.write(frame)
                frame_count += 1
                
                # Progress logging every 100 frames
                if frame_count % 100 == 0:
                    logger.debug(f"Published {frame_count} frames")
            
            if writer:
                writer.release()
                logger.info(f"Video published: {output_file} ({frame_count} frames)")
                return str(output_file)
            else:
                logger.error("No frames to publish")
                return None
                
        except Exception as e:
            logger.error(f"Video publishing failed: {e}")
            if writer:
                writer.release()
            return None


class ImagePublisher(Publisher):
    """Publisher for single image outputs.
    
    Takes a single image (numpy array or matplotlib figure) and saves it.
    """
    
    def publish(
        self, 
        image: Union[np.ndarray, plt.Figure],
        format: str = 'png',
        dpi: int = 100,
        **kwargs
    ) -> Optional[str]:
        """Publish a single image.
        
        Args:
            image: Numpy array (H, W, C) or matplotlib Figure
            format: Output format (png, jpg, pdf, etc.)
            dpi: DPI for figure saving (matplotlib only)
            **kwargs: Additional parameters
            
        Returns:
            Path to created image file
        """
        output_file = self.output_path / f"{self.output_name}.{format}"
        
        try:
            if isinstance(image, plt.Figure):
                # Save matplotlib figure
                image.savefig(output_file, dpi=dpi, bbox_inches='tight')
                plt.close(image)
            else:
                # Save numpy array as image
                cv2.imwrite(str(output_file), image)
            
            logger.info(f"Image published: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Image publishing failed: {e}")
            return None


class ImageSequencePublisher(Publisher):
    """Publisher for image sequence outputs.
    
    Takes an iterator of images and saves them as numbered files.
    """
    
    def publish(
        self, 
        images: Iterator[Union[np.ndarray, plt.Figure]],
        format: str = 'png',
        name_pattern: str = "{name}_{index:04d}",
        **kwargs
    ) -> Optional[str]:
        """Publish images as a numbered sequence.
        
        Args:
            images: Iterator yielding images
            format: Output format for each image
            name_pattern: Pattern for naming files
            **kwargs: Additional parameters
            
        Returns:
            Path to output directory containing images
        """
        try:
            output_dir = self.output_path / self.output_name
            output_dir.mkdir(exist_ok=True)
            
            image_publisher = ImagePublisher(str(output_dir), "")
            published_files = []
            
            for i, image in enumerate(images):
                filename = name_pattern.format(name=self.output_name, index=i)
                image_publisher.output_name = filename
                
                output_file = image_publisher.publish(image, format=format, **kwargs)
                if output_file:
                    published_files.append(output_file)
            
            logger.info(f"Image sequence published: {len(published_files)} images in {output_dir}")
            return str(output_dir)
            
        except Exception as e:
            logger.error(f"Image sequence publishing failed: {e}")
            return None


def get_publisher(
    publisher_type: str,
    output_path: str,
    output_name: str
) -> Publisher:
    """Factory function to create publishers.
    
    Args:
        publisher_type: Type of publisher ('video', 'image', 'sequence')
        output_path: Output directory path
        output_name: Base name for outputs
        
    Returns:
        Publisher instance
        
    Raises:
        ValueError: If publisher type is unknown
    """
    publishers = {
        'video': VideoPublisher,
        'image': ImagePublisher,
        'sequence': ImageSequencePublisher
    }
    
    publisher_class = publishers.get(publisher_type)
    if not publisher_class:
        raise ValueError(f"Unknown publisher type: {publisher_type}")
    
    return publisher_class(output_path, output_name)