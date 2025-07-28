"""Core interfaces for NaviGraph plugin system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from .types import AnalysisResult
    from .session import Session

# Type alias for logger
Logger = type(logger)


class IDataSource(ABC):
    """Interface for data sources that integrate data into session DataFrame."""
    
    @abstractmethod
    def integrate_data_into_session(
        self,
        current_dataframe: pd.DataFrame,
        session_config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        logger: Logger
    ) -> pd.DataFrame:
        """Integrate data into the session DataFrame."""
        pass


class ISharedResource(ABC):
    """Interface for shared resources used across sessions."""
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any], logger: Optional[Logger] = None):
        """Create resource from configuration."""
        pass
    
    @abstractmethod
    def initialize_resource(self) -> None:
        """Initialize the resource."""
        pass


class IAnalyzer(ABC):
    """Interface for analyzers that compute metrics from session data."""
    
    @abstractmethod
    def analyze_session(self, session: "Session") -> "AnalysisResult":
        """Analyze session and return metrics."""
        pass


class IVisualizer(ABC):
    """Interface for visualizers that create plots and figures."""
    
    @abstractmethod
    def generate_visualization(
        self,
        session_data: pd.DataFrame,
        config: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> Optional[str]:
        """Generate visualization and return output path.
        
        Args:
            session_data: DataFrame with integrated session data
            config: Visualization-specific configuration
            output_path: Directory to save visualization outputs
            **kwargs: Additional parameters that may include:
                - video_path: Path to source video for video-based visualizations
                - session_id: Session identifier for output naming
                - shared_resources: Dict of shared resources (map, graph, etc.)
                - Any other visualizer-specific parameters
                
        Returns:
            Path to created visualization file, or None if failed
        """
        pass


# Legacy aliases
IGraphProvider = ISharedResource