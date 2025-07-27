"""Constants module for NaviGraph.

This module centralizes all magic strings, configuration keys, and other
constants used throughout the NaviGraph codebase to improve maintainability
and reduce errors from typos.
"""


class DataColumns:
    """Standard DataFrame column names."""
    
    # Keypoint data columns
    KEYPOINTS_X = "keypoints_x"
    KEYPOINTS_Y = "keypoints_y"
    KEYPOINTS_LIKELIHOOD = "keypoints_likelihood"
    
    # Spatial data columns
    TILE_ID = "tile_id"
    TREE_POSITION = "tree_position"
    MAP_X = "map_x"
    MAP_Y = "map_y"
    
    # Temporal data columns
    FRAME_NUMBER = "frame_number"
    TIMESTAMP = "timestamp"
    
    # Velocity and movement columns
    VELOCITY = "velocity"
    SPEED = "speed"
    DISTANCE = "distance"
    
    # Analysis result columns
    SESSION_ID = "session_id"
    METRIC_NAME = "metric_name"
    METRIC_VALUE = "metric_value"


class ConfigKeys:
    """Configuration file keys and sections."""
    
    # Main configuration sections
    DATA_SOURCES = "data_sources"
    SHARED_RESOURCES = "shared_resources"
    ANALYZERS = "analyzers"
    VISUALIZATIONS = "visualizations"
    
    # Data source configuration
    PLUGIN_NAME = "plugin_name"
    PLUGIN_TYPE = "type"
    FILE_PATTERN = "file_pattern"
    DISCOVERED_FILE_PATH = "discovered_file_path"
    REQUIRED = "required"
    ENABLED = "enabled"
    
    # Location settings
    BODYPART = "bodypart"
    LIKELIHOOD = "likelihood"
    LIKELIHOOD_THRESHOLD = "likelihood_threshold"
    
    # Map settings
    MAP_PATH = "map_path"
    MAP_SETTINGS = "map_settings"
    ORIGIN = "origin"
    GRID_SIZE = "grid_size"
    SEGMENT_LENGTH = "segment_length"
    PIXEL_TO_METER = "pixel_to_meter"
    
    # Graph settings
    GRAPH = "graph"
    HEIGHT = "height"
    DRAW = "draw"
    OPTIONS = "options"
    
    # Analysis settings
    ANALYZE = "analyze"
    METRICS = "metrics"
    FUNC_NAME = "func_name"
    ARGS = "args"
    
    # Visualization settings
    SHOW_VISUALIZATION = "show_visualization"
    RECORD_VISUALIZATION = "record_visualization"
    FPS = "fps"
    RESIZE = "resize"
    OUTPUT_PATH = "output_path"
    
    # Session settings
    SESSION_ID = "session_id"
    REWARD_TILE_ID = "reward_tile_id"
    LEARNING_START_TILE_ID = "learning_start_tile_id"


class FileExtensions:
    """Common file extensions used in NaviGraph."""
    
    # Video formats
    VIDEO_MP4 = ".mp4"
    VIDEO_AVI = ".avi"
    VIDEO_MOV = ".mov"
    
    # Data formats
    HDF5 = ".h5"
    CSV = ".csv"
    PICKLE = ".pkl"
    NUMPY = ".npy"
    
    # Image formats
    PNG = ".png"
    JPG = ".jpg"
    JPEG = ".jpeg"
    PDF = ".pdf"
    SVG = ".svg"
    
    # Configuration formats
    YAML = ".yaml"
    YML = ".yml"
    JSON = ".json"


class DefaultValues:
    """Default values for various parameters."""
    
    # Data processing defaults
    DEFAULT_LIKELIHOOD_THRESHOLD = 0.3
    DEFAULT_FPS = 30.0
    DEFAULT_BODYPART = "nose"
    
    # Visualization defaults
    DEFAULT_MARKER_RADIUS = 5
    DEFAULT_TRAIL_LENGTH = 30
    DEFAULT_OPACITY = 0.7
    DEFAULT_OVERLAY_SIZE = 0.3
    
    # Analysis defaults
    DEFAULT_MIN_NODES_ON_PATH = 0
    DEFAULT_TREE_HEIGHT = 7
    
    # File discovery defaults
    DEFAULT_TIMEOUT_MS = 120000  # 2 minutes
    
    # Plugin defaults
    DEFAULT_PLUGIN_ENABLED = True


class ErrorMessages:
    """Standardized error messages."""
    
    # Plugin errors
    PLUGIN_NOT_FOUND = "Plugin '{plugin_name}' of type '{plugin_type}' not found"
    PLUGIN_REGISTRATION_FAILED = "Failed to register plugin '{plugin_name}': {error}"
    PLUGIN_VALIDATION_FAILED = "Plugin '{plugin_name}' validation failed: {error}"
    PLUGIN_MISSING_INTERFACE = "Plugin '{plugin_name}' must inherit from {interface}"
    
    # Configuration errors
    CONFIG_KEY_MISSING = "Required configuration key '{key}' is missing"
    CONFIG_INVALID_VALUE = "Invalid value for configuration key '{key}': {value}"
    CONFIG_FILE_NOT_FOUND = "Configuration file not found: {path}"
    CONFIG_PARSE_ERROR = "Failed to parse configuration file: {error}"
    
    # Data source errors
    DATA_SOURCE_INTEGRATION_FAILED = "Data source '{name}' integration failed: {error}"
    DATA_SOURCE_PREREQUISITES_NOT_MET = "Prerequisites not met for data source '{name}'"
    DATA_SOURCE_FILE_NOT_FOUND = "Data file not found for source '{name}': {path}"
    DATA_SOURCE_INVALID_FORMAT = "Invalid data format for source '{name}': {error}"
    
    # Session errors
    SESSION_INITIALIZATION_FAILED = "Session initialization failed: {error}"
    SESSION_DATA_INTEGRATION_FAILED = "Session data integration failed: {error}"
    SESSION_NOT_FOUND = "Session '{session_id}' not found"
    
    # File errors
    FILE_NOT_FOUND = "File not found: {path}"
    FILE_READ_ERROR = "Failed to read file '{path}': {error}"
    FILE_WRITE_ERROR = "Failed to write file '{path}': {error}"
    FILE_INVALID_FORMAT = "Invalid file format for '{path}': expected {expected}"
    
    # Analysis errors
    ANALYSIS_FAILED = "Analysis failed for session '{session_id}': {error}"
    ANALYSIS_MISSING_COLUMNS = "Missing required columns for analysis: {columns}"
    ANALYSIS_INSUFFICIENT_DATA = "Insufficient data for analysis: {details}"
    
    # Visualization errors
    VISUALIZATION_FAILED = "Visualization '{name}' failed: {error}"
    VISUALIZATION_MISSING_RESOURCES = "Missing required resources for visualization: {resources}"
    VISUALIZATION_INVALID_CONFIG = "Invalid visualization configuration: {error}"


class LogMessages:
    """Standardized log messages."""
    
    # Info messages
    PLUGIN_REGISTERED = "Registered {plugin_type} plugin: {name}"
    SESSION_INITIALIZED = "Initialized session: {session_id}"
    DATA_INTEGRATION_COMPLETE = "Data integration complete: {columns} columns, {frames} frames"
    ANALYSIS_COMPLETE = "Analysis complete for session {session_id}: {metrics}"
    VISUALIZATION_CREATED = "Created {visualization_type}: {output_path}"
    
    # Debug messages
    PLUGIN_LOADING = "Loading plugin: {name} ({type})"
    FILE_DISCOVERED = "Discovered file for {pattern}: {path}"
    CONFIGURATION_LOADED = "Loaded configuration from: {path}"
    
    # Warning messages
    PLUGIN_OVERRIDE = "Overriding existing plugin: {name}"
    FILE_NOT_FOUND_WARNING = "File not found (optional): {path}"
    CONFIGURATION_DEFAULT = "Using default value for {key}: {value}"
    
    # Error messages
    PLUGIN_LOAD_FAILED = "Failed to load plugin {name}: {error}"
    RESOURCE_INITIALIZATION_FAILED = "Resource initialization failed: {error}"
    EXPERIMENT_FAILED = "Experiment failed: {error}"


class Paths:
    """Standard path patterns and structures."""
    
    # Configuration paths
    CONFIG_DIR = "configs"
    DEFAULT_CONFIG = "config.yaml"
    
    # Data directories
    DATA_DIR = "data"
    RESOURCES_DIR = "resources"
    OUTPUT_DIR = "output"
    
    # Output subdirectories
    ANALYSIS_OUTPUT = "analysis"
    VISUALIZATION_OUTPUT = "visualizations"
    RAW_DATA_OUTPUT = "raw_data"
    
    # File name patterns
    CALIBRATION_MATRIX = "transform_matrix.npy"
    ANALYSIS_RESULTS_CSV = "analysis_results.csv"
    ANALYSIS_RESULTS_PKL = "analysis_results.pkl"
    RAW_DATA_PKL = "raw_data.pkl"


class VisualizationDefaults:
    """Default values for visualization settings."""
    
    # Colors (RGB values)
    DEFAULT_TRAJECTORY_COLOR = [0, 255, 0]  # Green
    DEFAULT_CURRENT_TILE_COLOR = [255, 0, 0]  # Red
    DEFAULT_VISITED_TILE_COLOR = [0, 255, 0]  # Green
    DEFAULT_PATH_COLOR = [0, 0, 255]  # Blue
    DEFAULT_BACKGROUND_COLOR = [255, 255, 255]  # White
    
    # Positions
    OVERLAY_POSITIONS = ["top_left", "top_right", "bottom_left", "bottom_right"]
    DEFAULT_OVERLAY_POSITION = "bottom_right"
    
    # Plot types
    PLOT_TYPES = ["learning_curve", "distribution", "comparison", "correlation", "timeline", "summary"]
    DEFAULT_PLOT_TYPE = "learning_curve"
    
    # Output formats
    VIDEO_FORMATS = ["mp4", "avi", "mov"]
    IMAGE_FORMATS = ["png", "jpg", "pdf", "svg"]
    DEFAULT_VIDEO_FORMAT = "mp4"
    DEFAULT_IMAGE_FORMAT = "png"


class AnalysisDefaults:
    """Default values for analysis settings."""
    
    # Metric calculation defaults
    DEFAULT_WINDOW_SIZE = 10
    DEFAULT_SMOOTHING_FACTOR = 0.1
    DEFAULT_CONFIDENCE_INTERVAL = 0.95
    
    # Statistical defaults
    DEFAULT_SIGNIFICANCE_LEVEL = 0.05
    DEFAULT_EFFECT_SIZE_THRESHOLD = 0.2
    
    # Cross-session analysis
    DEFAULT_GROUP_COMPARISON_METHOD = "t_test"
    DEFAULT_MULTIPLE_COMPARISONS_CORRECTION = "bonferroni"


class SystemModes:
    """Valid system running modes."""
    
    CALIBRATE = "calibrate"
    TEST = "test"
    ANALYZE = "analyze"
    VISUALIZE = "visualize"
    
    # Combined modes
    CALIBRATE_AND_ANALYZE = "calibrate&analyze"
    ANALYZE_AND_VISUALIZE = "analyze&visualize"
    
    ALL_MODES = [CALIBRATE, TEST, ANALYZE, VISUALIZE]