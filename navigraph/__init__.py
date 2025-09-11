"""NaviGraph: Flexible graph-based analysis framework for behavioral research.

A modern, extensible framework for analyzing behavioral data in maze paradigms,
integrating multiple data sources (keypoints, neural activity, behavioral events)
into unified graph-based analysis workflows.
"""

__version__ = "1.0.0"
__author__ = "NaviGraph Team"

from loguru import logger

# Configure clean logger for CLI (default level, can be overridden)  
logger.remove()  # Remove default handler

def default_format_record(record):
    level = record["level"].name
    colors = {
        "INFO": "<blue>",
        "DEBUG": "<yellow>", 
        "WARNING": "<light-red>",
        "ERROR": "<red>"
    }
    color = colors.get(level, "<white>")
    
    return (
        f"<green>{record['time']:YYYY-MM-DD HH:mm:ss}</green> | "
        f"{color}{level: <8}</> | "
        f"{record['message']}"
    )

logger.add(
    lambda msg: print(msg),
    level="INFO", 
    format=default_format_record,
    colorize=True,
)

def configure_logging(log_level: str = "info"):
    """Configure logger level based on config."""
    logger.remove()  # Remove all handlers
    
    # Custom format with specific colors for each level
    def format_record(record):
        level = record["level"].name
        colors = {
            "INFO": "<blue>",
            "DEBUG": "<yellow>", 
            "WARNING": "<light-red>",
            "ERROR": "<red>"
        }
        color = colors.get(level, "<white>")
        
        return (
            f"<green>{record['time']:YYYY-MM-DD HH:mm:ss}</green> | "
            f"{color}{level: <8}</> | "
            f"{record['message']}"
        )
    
    logger.add(
        lambda msg: print(msg),
        level=log_level.upper(),
        format=format_record,
        colorize=True,
    )

__all__ = ["__version__", "__author__", "logger", "configure_logging"]