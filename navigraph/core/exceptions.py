"""Exception hierarchy for NaviGraph.

This module provides a comprehensive, structured exception hierarchy for
all NaviGraph components. Exceptions are organized by functional area
with clear inheritance relationships and helpful error messages.
"""

from typing import List, Optional, Any, Dict


class NavigraphError(Exception):
    """Base exception for all NaviGraph errors.
    
    All NaviGraph-specific exceptions inherit from this class.
    Provides common functionality for error handling and reporting.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize NavigraphError with message and optional details.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return formatted error message with details if available."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# =============================================================================
# PLUGIN SYSTEM EXCEPTIONS
# =============================================================================

class PluginError(NavigraphError):
    """Base exception for all plugin-related errors."""
    pass


class PluginRegistrationError(PluginError):
    """Raised when plugin registration fails.
    
    This includes errors during plugin discovery, validation,
    or registration in the plugin registry.
    """
    pass


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin cannot be found."""
    
    def __init__(self, plugin_name: str, plugin_type: str, available_plugins: List[str]):
        """Initialize with plugin information and suggestions.
        
        Args:
            plugin_name: Name of the plugin that was not found
            plugin_type: Type of plugin (data_source, analyzer, etc.)
            available_plugins: List of available plugins of this type
        """
        message = f"{plugin_type.title()} plugin '{plugin_name}' not found"
        details = {
            "plugin_name": plugin_name,
            "plugin_type": plugin_type,
            "available_plugins": available_plugins
        }
        super().__init__(message, details)


class PluginValidationError(PluginError):
    """Raised when plugin validation fails.
    
    This includes missing required methods, incorrect inheritance,
    or invalid plugin configuration.
    """
    pass


class PluginInitializationError(PluginError):
    """Raised when plugin initialization fails after registration."""
    pass


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================

class ConfigurationError(NavigraphError):
    """Base exception for configuration-related errors."""
    pass


class ConfigurationFileError(ConfigurationError):
    """Raised when configuration file cannot be found or parsed."""
    pass


class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    
    def __init__(self, key: str, value: Any, expected: str, details: Optional[str] = None):
        """Initialize with validation details.
        
        Args:
            key: Configuration key that failed validation
            value: Invalid value that was provided
            expected: Description of expected value format
            details: Optional additional validation details
        """
        message = f"Invalid configuration for '{key}': expected {expected}, got {value}"
        if details:
            message += f" - {details}"
        
        super().__init__(message, {
            "key": key,
            "value": value,
            "expected": expected
        })


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, missing_keys: List[str], section: Optional[str] = None):
        """Initialize with missing key information.
        
        Args:
            missing_keys: List of missing configuration keys
            section: Optional configuration section name
        """
        if section:
            message = f"Missing required configuration keys in section '{section}': {missing_keys}"
        else:
            message = f"Missing required configuration keys: {missing_keys}"
        
        super().__init__(message, {
            "missing_keys": missing_keys,
            "section": section
        })


# =============================================================================
# DATA SOURCE EXCEPTIONS
# =============================================================================

class DataSourceError(NavigraphError):
    """Base exception for data source-related errors."""
    pass


class DataSourceIntegrationError(DataSourceError):
    """Raised when data source integration into session fails."""
    pass


class DataSourcePrerequisiteError(DataSourceError):
    """Raised when data source prerequisites are not met."""
    
    def __init__(self, data_source: str, missing_requirements: List[str]):
        """Initialize with prerequisite information.
        
        Args:
            data_source: Name of the data source
            missing_requirements: List of missing requirements
        """
        message = f"Prerequisites not met for data source '{data_source}'"
        super().__init__(message, {
            "data_source": data_source,
            "missing_requirements": missing_requirements
        })


class DataSourceFileError(DataSourceError):
    """Raised when data source file operations fail."""
    pass


class DataFormatError(DataSourceError):
    """Raised when data format is invalid or corrupted."""
    
    def __init__(self, file_path: str, expected_format: str, details: Optional[str] = None):
        """Initialize with file format information.
        
        Args:
            file_path: Path to the problematic file
            expected_format: Expected file format
            details: Optional additional format details
        """
        message = f"Invalid data format in '{file_path}': expected {expected_format}"
        if details:
            message += f" - {details}"
        
        super().__init__(message, {
            "file_path": file_path,
            "expected_format": expected_format
        })


# =============================================================================
# SESSION EXCEPTIONS
# =============================================================================

class SessionError(NavigraphError):
    """Base exception for session-related errors."""
    pass


class SessionInitializationError(SessionError):
    """Raised when session initialization fails."""
    pass


class SessionDataError(SessionError):
    """Raised when session data operations fail."""
    pass


class SessionNotFoundError(SessionError):
    """Raised when a requested session cannot be found."""
    
    def __init__(self, session_id: str, available_sessions: List[str]):
        """Initialize with session information.
        
        Args:
            session_id: ID of the session that was not found
            available_sessions: List of available session IDs
        """
        message = f"Session '{session_id}' not found"
        super().__init__(message, {
            "session_id": session_id,
            "available_sessions": available_sessions
        })


# =============================================================================
# SHARED RESOURCE EXCEPTIONS
# =============================================================================

class SharedResourceError(NavigraphError):
    """Base exception for shared resource-related errors."""
    pass


class ResourceInitializationError(SharedResourceError):
    """Raised when shared resource initialization fails."""
    pass


class ResourceNotFoundError(SharedResourceError):
    """Raised when a required shared resource is not available."""
    
    def __init__(self, resource_name: str, resource_type: str):
        """Initialize with resource information.
        
        Args:
            resource_name: Name of the missing resource
            resource_type: Type of the missing resource
        """
        message = f"Required {resource_type} resource '{resource_name}' not found"
        super().__init__(message, {
            "resource_name": resource_name,
            "resource_type": resource_type
        })


class ResourceValidationError(SharedResourceError):
    """Raised when shared resource validation fails."""
    pass


# =============================================================================
# ANALYSIS EXCEPTIONS
# =============================================================================

class AnalysisError(NavigraphError):
    """Base exception for analysis-related errors."""
    pass


class AnalysisPrerequisiteError(AnalysisError):
    """Raised when analysis prerequisites are not met."""
    
    def __init__(self, analyzer: str, missing_columns: List[str], missing_resources: List[str]):
        """Initialize with analysis prerequisite information.
        
        Args:
            analyzer: Name of the analyzer
            missing_columns: List of missing DataFrame columns
            missing_resources: List of missing shared resources
        """
        message = f"Prerequisites not met for analyzer '{analyzer}'"
        super().__init__(message, {
            "analyzer": analyzer,
            "missing_columns": missing_columns,
            "missing_resources": missing_resources
        })


class AnalysisComputationError(AnalysisError):
    """Raised when analysis computation fails."""
    pass


class InsufficientDataError(AnalysisError):
    """Raised when there is insufficient data for analysis."""
    
    def __init__(self, required_samples: int, available_samples: int, details: Optional[str] = None):
        """Initialize with data availability information.
        
        Args:
            required_samples: Minimum number of samples required
            available_samples: Number of samples available
            details: Optional additional details about the requirement
        """
        message = f"Insufficient data for analysis: need {required_samples}, have {available_samples}"
        if details:
            message += f" - {details}"
        
        super().__init__(message, {
            "required_samples": required_samples,
            "available_samples": available_samples
        })


# =============================================================================
# VISUALIZATION EXCEPTIONS
# =============================================================================

class VisualizationError(NavigraphError):
    """Base exception for visualization-related errors."""
    pass


class VisualizationConfigurationError(VisualizationError):
    """Raised when visualization configuration is invalid."""
    pass


class VisualizationRenderingError(VisualizationError):
    """Raised when visualization rendering fails."""
    pass


class VisualizationResourceError(VisualizationError):
    """Raised when required visualization resources are missing."""
    
    def __init__(self, visualizer: str, missing_resources: List[str]):
        """Initialize with visualization resource information.
        
        Args:
            visualizer: Name of the visualizer
            missing_resources: List of missing resources
        """
        message = f"Missing required resources for visualizer '{visualizer}': {missing_resources}"
        super().__init__(message, {
            "visualizer": visualizer,
            "missing_resources": missing_resources
        })


# =============================================================================
# FILE SYSTEM EXCEPTIONS
# =============================================================================

class FileSystemError(NavigraphError):
    """Base exception for file system-related errors."""
    pass


class FileNotFoundError(FileSystemError):
    """Raised when a required file cannot be found."""
    
    def __init__(self, file_path: str, context: Optional[str] = None):
        """Initialize with file information.
        
        Args:
            file_path: Path to the missing file
            context: Optional context about why the file was needed
        """
        message = f"File not found: {file_path}"
        if context:
            message += f" ({context})"
        
        super().__init__(message, {"file_path": file_path, "context": context})


class FileReadError(FileSystemError):
    """Raised when file reading fails."""
    pass


class FileWriteError(FileSystemError):
    """Raised when file writing fails."""
    pass


class DirectoryError(FileSystemError):
    """Raised when directory operations fail."""
    pass


# =============================================================================
# EXPERIMENT RUNNER EXCEPTIONS
# =============================================================================

class ExperimentError(NavigraphError):
    """Base exception for experiment execution errors."""
    pass


class ExperimentConfigurationError(ExperimentError):
    """Raised when experiment configuration is invalid."""
    pass


class ExperimentExecutionError(ExperimentError):
    """Raised when experiment execution fails."""
    pass


class ModeNotSupportedError(ExperimentError):
    """Raised when an unsupported system mode is requested."""
    
    def __init__(self, mode: str, supported_modes: List[str]):
        """Initialize with mode information.
        
        Args:
            mode: The unsupported mode that was requested
            supported_modes: List of supported system modes
        """
        message = f"System mode '{mode}' not supported"
        super().__init__(message, {
            "mode": mode,
            "supported_modes": supported_modes
        })


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_exception_context(exc: Exception, context: Dict[str, Any]) -> str:
    """Format exception with additional context information.
    
    Args:
        exc: The exception to format
        context: Additional context information
        
    Returns:
        Formatted error message with context
    """
    base_message = str(exc)
    
    if context:
        context_items = [f"{k}={v}" for k, v in context.items()]
        context_str = ", ".join(context_items)
        return f"{base_message} [Context: {context_str}]"
    
    return base_message


def create_chained_error(
    primary_error: Exception,
    secondary_errors: List[Exception],
    context: Optional[str] = None
) -> NavigraphError:
    """Create a chained error from multiple related exceptions.
    
    Args:
        primary_error: The main exception that occurred
        secondary_errors: List of related exceptions
        context: Optional context description
        
    Returns:
        NavigraphError with chained error information
    """
    message = str(primary_error)
    
    if context:
        message = f"{context}: {message}"
    
    if secondary_errors:
        error_messages = [str(err) for err in secondary_errors]
        message += f" (Related errors: {'; '.join(error_messages)})"
    
    details = {
        "primary_error": type(primary_error).__name__,
        "secondary_error_count": len(secondary_errors)
    }
    
    return NavigraphError(message, details)