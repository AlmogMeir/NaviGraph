"""Simple exception hierarchy for NaviGraph.

Essential exceptions for the behavioral analysis framework.
"""


class NavigraphError(Exception):
    """Base exception for all NaviGraph errors."""
    pass


class DataSourceError(NavigraphError):
    """Raised when data source processing fails."""
    pass


class AnalysisError(NavigraphError):
    """Raised when analysis computation fails."""
    pass


class ConfigurationError(NavigraphError):
    """Raised when configuration is invalid or missing."""
    pass


