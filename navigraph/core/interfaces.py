"""Core interfaces for NaviGraph plugin system.

This module defines the abstract base classes that all plugins must implement.
The interfaces follow configuration-based ordering rather than priority-based,
meaning the execution order is determined by the order in configuration files.

Key Design Principles:
- Configuration-based ordering: Plugins execute in config file order
- Clear error messages: Validation failures provide helpful guidance
- Type safety: Comprehensive type hints throughout
- Documentation: Every method clearly documented with examples
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from .session import Session

# Type alias for logger
Logger = type(logger)


class NavigraphPluginError(Exception):
    """Base exception for all NaviGraph plugin errors."""
    pass


class DataSourceIntegrationError(NavigraphPluginError):
    """Raised when data source integration fails."""
    pass


class PluginPrerequisiteError(NavigraphPluginError):
    """Raised when plugin prerequisites are not met."""
    pass


class SharedResourceError(NavigraphPluginError):
    """Raised when shared resource operations fail."""
    pass


class IDataSource(ABC):
    """Interface for data sources that integrate into session DataFrame.
    
    Data sources are executed in the order specified in the configuration file.
    Each data source adds its columns to the accumulating DataFrame by implementing
    the integrate_data_into_session method.
    
    ⚠️  IMPORTANT: Order matters! Make sure dependencies are listed before dependents
    in your configuration file. For example:
    
    Good order:
    1. keypoints (establishes frame index)
    2. spatial (needs keypoints + calibration)
    3. graph (needs spatial data)
    4. neural (needs frame index)
    
    Example usage:
        @register_data_source_plugin("deeplabcut")
        class DeepLabCutSource(IDataSource):
            def integrate_data_into_session(self, current_df, config, resources, logger):
                # Load keypoint data and add columns
                return updated_dataframe
    """
    
    @abstractmethod
    def integrate_data_into_session(
        self,
        current_dataframe: pd.DataFrame,
        session_config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        logger: Logger
    ) -> pd.DataFrame:
        """Integrate this data source into the session DataFrame.
        
        This method receives the current state of the session DataFrame and must
        add its own columns, returning the updated DataFrame. The first data source
        typically establishes the frame_number index.
        
        Args:
            current_dataframe: Existing DataFrame from previous data sources.
                              Empty DataFrame for the first data source.
            session_config: Configuration specific to this data source instance,
                          including discovered file paths and user parameters.
            shared_resources: Cross-session shared resources like maps, graphs,
                            and calibration data.
            logger: Logger instance for debugging and progress reporting.
            
        Returns:
            Updated DataFrame with new columns added by this data source.
            Must preserve the frame_number index if it exists.
        
        Raises:
            DataSourceIntegrationError: If integration fails for any reason.
            FileNotFoundError: If required data files are missing.
            ValueError: If data format is invalid or corrupted.
            
        Example:
            def integrate_data_into_session(self, current_df, config, resources, logger):
                h5_path = config['discovered_file_path']
                df = pd.read_hdf(h5_path)
                
                # Add keypoint columns
                if current_df.empty:
                    # First data source - establish index
                    result_df = pd.DataFrame({
                        'keypoints_x': df['x'],
                        'keypoints_y': df['y']
                    })
                    result_df.index.name = 'frame_number'
                else:
                    # Add to existing DataFrame
                    current_df['keypoints_x'] = df['x']
                    current_df['keypoints_y'] = df['y']
                    result_df = current_df
                
                return result_df
        """
        pass
    
    @abstractmethod
    def validate_session_prerequisites(
        self,
        current_dataframe: pd.DataFrame,
        shared_resources: Dict[str, Any]
    ) -> bool:
        """Check if prerequisites for this data source are met.
        
        This method should validate that all required columns exist in the current
        DataFrame and that necessary shared resources are available. It should NOT
        raise exceptions but return False if prerequisites are not met.
        
        Args:
            current_dataframe: Current state of the session DataFrame.
            shared_resources: Available shared resources.
            
        Returns:
            True if all prerequisites are met, False otherwise.
            
        Note:
            For helpful error messages, implement get_required_columns() method
            so the Session can generate detailed error messages when validation fails.
        """
        pass
    
    @abstractmethod
    def get_provided_column_names(self) -> List[str]:
        """Return column names this data source provides.
        
        Returns:
            List of column names that will be added to the DataFrame.
            
        Example:
            return ['keypoints_x', 'keypoints_y', 'keypoints_likelihood']
        """
        pass
    
    def get_required_columns(self) -> List[str]:
        """Return column names required by this data source.
        
        This is optional but recommended for generating helpful error messages
        when prerequisites are not met.
        
        Returns:
            List of column names that must exist before this data source runs.
            
        Example:
            return ['keypoints_x', 'keypoints_y']  # For map integration
        """
        return []
    
    def get_required_shared_resources(self) -> List[str]:
        """Return shared resource names required by this data source.
        
        Returns:
            List of shared resource names required for integration.
            
        Example:
            return ['maze_map', 'calibration']
        """
        return []


class ISharedResource(ABC):
    """Interface for cross-session shared resources.
    
    Shared resources are objects that are used across multiple sessions,
    such as maze maps, graph structures, and calibration data.
    
    Example usage:
        @register_shared_resource_plugin("map_provider")
        class MapProvider(ISharedResource):
            def initialize(self, config):
                self.map_image = cv2.imread(config['map_path'])
                
            def get_resource(self):
                return self.map_labeler
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any], logger: Logger) -> None:
        """Initialize the shared resource from configuration.
        
        Args:
            config: Configuration dictionary with resource parameters.
            logger: Logger instance for debugging.
            
        Raises:
            ValueError: If configuration is invalid.
            FileNotFoundError: If required files are missing.
        """
        pass
    
    @abstractmethod
    def get_resource(self) -> Any:
        """Get the initialized resource object.
        
        Returns:
            The resource object that will be used by sessions.
        """
        pass
    
    @property
    @abstractmethod
    def resource_type(self) -> str:
        """Type identifier for this resource.
        
        Returns:
            String identifier for the resource type.
            
        Example:
            return "maze_map"
        """
        pass


class IAnalyzer(ABC):
    """Interface for analysis metrics with full session access.
    
    Analyzers can access both tabular data and graph structures from sessions,
    enabling sophisticated behavioral and neural analysis.
    
    Example usage:
        @register_analyzer_plugin("spatial_metrics")
        class SpatialAnalyzer(IAnalyzer):
            def analyze_session(self, session):
                df = session.get_integrated_dataframe()
                graph = session.get_graph_structure()
                return {'time_to_reward': calculate_time(df)}
    """
    
    @abstractmethod
    def analyze_session(self, session: "Session") -> Dict[str, Any]:
        """Analyze a single session with full access to data and graph.
        
        Args:
            session: Session object providing access to:
                - session.get_integrated_dataframe(): Complete multi-source DataFrame
                - session.get_graph_structure(): Graph structure for topological analysis
                - session.generate_node_level_features(): Node-aggregated features
                - session.shared_resources: Maps, calibration, etc.
                - session.logger: Logging capabilities
                - session.session_id: Session identifier
        
        Returns:
            Dictionary of computed metrics and analysis results.
            Keys should be descriptive metric names, values can be numbers,
            lists, or nested dictionaries.
            
        Example:
            {
                'time_to_reward': 45.2,  # seconds
                'average_velocity': 12.5,  # cm/s
                'exploration_percentage': 0.73,
                'path_efficiency': 0.85,
                'neural_correlations': {
                    'cell_001': 0.62,
                    'cell_002': 0.41
                }
            }
        """
        pass
    
    @abstractmethod
    def analyze_cross_session(
        self,
        sessions: List["Session"],
        session_metrics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform cross-session analysis and statistics.
        
        Args:
            sessions: List of Session objects for cross-session access.
            session_metrics: Results from analyze_session for each session.
                           Format: {session_id: {metric_name: value}}
        
        Returns:
            Dictionary of cross-session analysis results.
            
        Example:
            {
                'mean_time_to_reward': 42.1,
                'learning_curve_slope': -0.25,
                'group_differences': {
                    'control_vs_treatment': {
                        'p_value': 0.032,
                        'effect_size': 0.67
                    }
                }
            }
        """
        pass
    
    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """Required DataFrame columns for this analyzer.
        
        Returns:
            List of column names that must be present in session data.
        """
        pass
    
    @property
    @abstractmethod
    def analyzer_type(self) -> str:
        """Type of analyzer: 'session', 'cross_session', or 'both'.
        
        Returns:
            String indicating what types of analysis this analyzer supports.
        """
        pass


class IVisualizer(ABC):
    """Interface for visualization plugins.
    
    Visualizers create plots, animations, and other visual outputs from
    session data and analysis results.
    """
    
    @abstractmethod
    def visualize(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> Optional[str]:
        """Create visualization from data.
        
        Args:
            data: DataFrame with session data or analysis results.
            config: Visualization-specific configuration.
            shared_resources: Shared resources (maps, graphs, etc.).
            output_path: Directory to save visualization outputs.
            **kwargs: Additional parameters for visualization.
            
        Returns:
            Path to created visualization file, or None if no file created.
        """
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """List of supported output formats.
        
        Returns:
            List of file extensions this visualizer can produce.
            
        Example:
            return ['png', 'pdf', 'mp4']
        """
        pass


class IGraphProvider(ABC):
    """Interface for graph structure providers.
    
    Graph providers create and manage the graph representations used
    for topological analysis of behavioral data.
    """
    
    @abstractmethod
    def build_graph(self, config: Dict[str, Any], logger: Logger) -> Any:
        """Build and return graph structure.
        
        Args:
            config: Graph configuration parameters.
            logger: Logger instance.
            
        Returns:
            Graph object (typically NetworkX Graph).
        """
        pass
    
    @abstractmethod
    def get_node_mapping(self, spatial_id: int) -> Optional[Any]:
        """Map spatial ID (e.g., tile_id) to graph node.
        
        Args:
            spatial_id: Spatial identifier to map.
            
        Returns:
            Graph node identifier, or None if not found.
        """
        pass
    
    @property
    @abstractmethod
    def graph_type(self) -> str:
        """Type identifier for this graph provider.
        
        Returns:
            String identifier for the graph type.
            
        Example:
            return "binary_tree"
        """
        pass