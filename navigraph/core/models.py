"""Data models for NaviGraph core functionality."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class SessionInfo:
    """Basic information about a discovered session."""
    name: str
    path: Path
    
    def __str__(self) -> str:
        return f"Session: {self.name} at {self.path}"


@dataclass
class PluginValidationResult:
    """Validation result from a single plugin."""
    plugin_name: str
    plugin_type: str
    is_valid: bool
    found_count: int
    message: str
    found_files: List[Path] = field(default_factory=list)


@dataclass
class SessionValidation:
    """Validation results for one session."""
    session_id: str
    results: List[PluginValidationResult] = field(default_factory=list)
    
    def add_result(self, result: PluginValidationResult) -> None:
        """Add a plugin validation result."""
        self.results.append(result)
    
    @property
    def is_valid(self) -> bool:
        """Check if all plugins validated successfully."""
        return all(r.is_valid for r in self.results)
    
    @property
    def invalid_plugins(self) -> List[str]:
        """Get list of plugins that failed validation."""
        return [r.plugin_name for r in self.results if not r.is_valid]


@dataclass
class ValidationReport:
    """Complete validation report for all sessions."""
    session_validations: List[SessionValidation]
    
    @property
    def total_sessions(self) -> int:
        return len(self.session_validations)
    
    @property
    def valid_sessions(self) -> int:
        return sum(1 for sv in self.session_validations if sv.is_valid)
    
    @property
    def invalid_sessions(self) -> int:
        return self.total_sessions - self.valid_sessions
    
    @property
    def validation_rate(self) -> float:
        if self.total_sessions == 0:
            return 0.0
        return (self.valid_sessions / self.total_sessions) * 100
    
    def format_report(self) -> str:
        """Format a human-readable validation report."""
        lines = [
            "Session Validation Report",
            "=" * 50,
            f"Total sessions: {self.total_sessions}",
            f"Valid sessions: {self.valid_sessions} ({self.validation_rate:.1f}%)",
            ""
        ]
        
        for sv in self.session_validations:
            status_icon = "✓" if sv.is_valid else "✗"
            lines.append(f"\n{status_icon} Session: {sv.session_id}")
            
            for result in sv.results:
                status = "✓" if result.is_valid else "✗"
                lines.append(f"  {status} {result.plugin_name}: {result.message}")
        
        return "\n".join(lines)


@dataclass 
class CrossSessionResults:
    """Container for cross-session analysis results."""
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)
    session_ids: List[str] = field(default_factory=list)
    analyzer_name: str = ""
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a cross-session metric."""
        self.metrics[name] = value
    
    def get_metric(self, name: str, default: float = 0.0) -> float:
        """Get a metric value by name."""
        return self.metrics.get(name, default)