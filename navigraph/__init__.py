"""NaviGraph: Flexible graph-based analysis framework for behavioral research.

A modern, extensible framework for analyzing behavioral data in maze paradigms,
integrating multiple data sources (keypoints, neural activity, behavioral events)
into unified graph-based analysis workflows.
"""

__version__ = "0.1.0"
__author__ = "NaviGraph Team"

from loguru import logger

# Configure default logger
logger.add(
    lambda msg: print(msg, end=""),
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    colorize=True,
)

__all__ = ["__version__", "__author__", "logger"]