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
    """Interface for visualizers that process and transform data.
    
    Visualizers are pure data transformation functions that process input
    data and return the result. They do not handle file I/O - that is the
    responsibility of the pipeline and publishers.
    """
    
    @abstractmethod
    def process(
        self,
        session_data: pd.DataFrame,
        config: Dict[str, Any],
        input_data: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """Process data and return visualization result.
        
        Args:
            session_data: DataFrame with integrated session data
            config: Visualization-specific configuration
            input_data: Optional input from previous pipeline stage
                - For video processors: Iterator[np.ndarray] or video path
                - For plot processors: matplotlib Figure or data
            **kwargs: Additional parameters including:
                - session_path: Path to session directory for file discovery
                - session_id: Session identifier
                - shared_resources: Dict of shared resources (map, graph, etc.)
                
        Returns:
            Processed visualization data:
            - Video processors: Iterator[np.ndarray] (frame generator)
            - Plot processors: matplotlib.Figure or np.ndarray
            - Data processors: Transformed data
        """
        pass


# Legacy aliases
IGraphProvider = ISharedResource