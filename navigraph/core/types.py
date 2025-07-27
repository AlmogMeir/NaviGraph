"""Essential type definitions for NaviGraph."""

from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AnalysisMetadata:
    """Simple analysis metadata."""
    analyzer_name: str
    version: str = "1.0.0"
    timestamp: datetime = None
    computation_time: float = 0.0
    config_hash: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AnalysisResult:
    """Analysis result with metrics and metadata."""
    session_id: str
    analyzer_name: str
    metrics: Dict[str, Any]
    metadata: AnalysisMetadata
    
    def get_metric(self, name: str, default: Any = None) -> Any:
        """Get metric value by name."""
        return self.metrics.get(name, default)
    
    def has_metric(self, name: str) -> bool:
        """Check if metric exists."""
        return name in self.metrics