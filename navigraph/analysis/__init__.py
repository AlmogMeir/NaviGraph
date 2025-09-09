"""Analysis functions for NaviGraph.

Contains session-level and cross-session analysis functions.
All functions are auto-discovered and registered via decorators.
"""

from . import metrics  # Import to register metrics
from .analyzer import Analyzer

__all__ = ['Analyzer']