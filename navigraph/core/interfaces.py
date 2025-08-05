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
    """Interface for visualizers that process frames.
    
    Visualizers process individual frames with session context.
    The pipeline handles all file I/O and frame iteration.
    """
    
    @abstractmethod
    def process_frame(self, frame, frame_index: int, session) -> Any:
        """Process a single frame with session context.
        
        Args:
            frame: Input frame (numpy array for video)
            frame_index: Current frame index
            session: Session object with full data access
                
        Returns:
            Processed frame (same type as input)
        """
        pass


# Legacy aliases
IGraphProvider = ISharedResource