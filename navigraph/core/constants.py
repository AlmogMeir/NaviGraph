"""Constants used throughout the NaviGraph system."""

# Configuration keys
class ConfigKeys:
    """Configuration key constants."""
    EXPERIMENT_PATH = 'experiment_path'
    OUTPUT_PATH = 'experiment_output_path'
    DATA_SOURCES = 'data_sources'
    ANALYZE = 'analyze'
    VISUALIZATIONS = 'visualizations'
    MAP_SETTINGS = 'map_settings'
    LOCATION_SETTINGS = 'location_settings'
    REWARD_TILE_ID = 'reward_tile_id'
    MAP_PATH = 'map_path'
    VERBOSE = 'verbose'
    SESSION_ID = 'session_id'
    SESSION_SETTINGS = 'session_settings'
    GRAPH = 'graph'


# Directory names
class Directories:
    """Standard directory names."""
    RESOURCES = 'resources'
    METRICS = 'metrics'
    RAW_DATA = 'raw_data'
    VISUALIZATIONS = 'visualizations'
    CROSS_SESSION = 'cross_session'
    SESSIONS = 'sessions'
    CALIBRATION = 'calibration'


# File names
class FileNames:
    """Standard file names."""
    CONFIG_YAML = 'config.yaml'
    EXPERIMENT_LOG = 'experiment.log'
    ANALYSIS_RESULTS_CSV = 'analysis_results.csv'
    ANALYSIS_RESULTS_PKL = 'analysis_results.pkl'
    SESSION_METRICS_CSV = 'session_metrics.csv'
    SESSION_METRICS_PKL = 'session_metrics.pkl'
    SESSION_RAW_DATA_PKL = 'session_raw_data.pkl'
    TRANSFORM_MATRIX = 'transform_matrix.npy'


# Column names
class ColumnNames:
    """Standard DataFrame column names."""
    FRAME_NUMBER = 'frame_number'
    TILE_ID = 'tile_id'
    MAP_X = 'map_x'
    MAP_Y = 'map_y'
    TILE_BBOX = 'tile_bbox'
    TREE_POSITION = 'tree_position'
    FPS = 'fps'
    
    # Keypoint columns
    KEYPOINTS_X = 'keypoints_x'
    KEYPOINTS_Y = 'keypoints_y'
    KEYPOINTS_LIKELIHOOD = 'keypoints_likelihood'
    
    # Neural activity
    NEURON_PREFIX = 'neuron_'
    NEURON_MEAN_ACTIVITY = 'neuron_mean_activity'
    NEURON_MAX_ACTIVITY = 'neuron_max_activity'
    NEURON_ACTIVE_COUNT = 'neuron_active_count'
    NEURON_ACTIVITY_STD = 'neuron_activity_std'
    
    # Head direction
    YAW = 'yaw'
    PITCH = 'pitch'
    ROLL = 'roll'


# Logging formats
class LogFormats:
    """Logging format constants."""
    PHASE_PREFIX = "[{phase}]"
    PROGRESS = "{current}/{total}"
    PERCENTAGE = "{value:.1f}%"
    
    # Phase names
    INIT = "INIT"
    DISCOVERY = "DISCOVERY"
    VALIDATION = "VALIDATION"
    SESSION = "SESSION"
    ANALYSIS = "ANALYSIS"
    VISUALIZATION = "VIZ"
    SAVE = "SAVE"
    ERROR = "ERROR"
    WARNING = "WARN"
    
    # Standard messages (no special characters)
    SUCCESS = "completed"
    FAILED = "failed"
    PROCESSING = "processing"
    LOADING = "loading"
    CREATING = "creating"
    FOUND = "found"
    

# File patterns
class FilePatterns:
    """File pattern constants."""
    VIDEO_PATTERNS = ['*.mp4', '*.avi']
    H5_PATTERN = '*.h5'
    CSV_PATTERN = '*.csv'
    NPY_PATTERN = '*.npy'
    PNG_PATTERN = '*.png'
    ZARR_PATTERN = '*.zarr'
    
    # DeepLabCut specific
    DLC_PATTERN = r'.*DLC.*\.h5$'
    
    # Head orientation
    HEAD_ORIENTATION_PATTERN = r'.*headOrientation\.csv$'
    
    # Map files
    MAZE_MAP_PATTERN = r'.*maze_map\.png$'


# Default values
class Defaults:
    """Default configuration values."""
    FPS = 30
    LIKELIHOOD_THRESHOLD = 0.3
    MIN_NODES_ON_PATH = 0
    ACTIVITY_THRESHOLD = 0.1
    OUTPUT_PATH = "{PROJECT_ROOT}/output"
    RUNNING_MODE = "analyze"