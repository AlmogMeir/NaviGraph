"""Session Analyzer orchestrator for NaviGraph.

Manages and orchestrates analysis functions for a session.
Gets configuration from session, validates analysis functions,
runs them with dataframe and shared resources, and organizes results.
"""

from typing import Dict, Any
import pandas as pd
from loguru import logger

from .registry import registry
from .exceptions import NavigraphError


class SessionAnalyzer:
    """Orchestrates analysis functions for a session."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with analysis configuration.
        
        Args:
            config: Dict with structure:
                {
                    'analyses': {
                        'spatial_metrics': {
                            'enabled': True,
                            'target_tile': 5,
                            ...
                        },
                        'movement_analysis': {
                            'enabled': True,
                            ...
                        }
                    }
                }
        """
        self.config = config.get('analyses', {})
        self.results = {}
        self.logger = logger
    
    def run_analyses(self, dataframe: pd.DataFrame, shared_resources: Dict[str, Any]) -> Dict[str, Any]:
        """Run all configured analyses.
        
        Args:
            dataframe: Session dataframe with all plugin data
            shared_resources: Shared resources (graph, mapping, etc.)
            
        Returns:
            Dict of analysis results by name
        """
        if not self.config:
            self.logger.info("No analyses configured")
            return {}
        
        results = {}
        
        for analysis_name, analysis_config in self.config.items():
            # Skip disabled analyses
            if not analysis_config.get('enabled', True):
                self.logger.debug(f"Skipping disabled analysis: {analysis_name}")
                continue
            
            try:
                # Get analysis function from registry
                analysis_func = registry.get_analysis(analysis_name)
                
                # Run analysis with config
                self.logger.info(f"Running analysis: {analysis_name}")
                result = analysis_func(
                    dataframe=dataframe,
                    shared_resources=shared_resources,
                    **analysis_config
                )
                
                if not isinstance(result, dict):
                    raise TypeError(f"Analysis '{analysis_name}' must return a dict, got {type(result)}")
                
                results[analysis_name] = result
                self.logger.info(f"✓ {analysis_name}: {len(result)} metrics")
                
            except NavigraphError as e:
                # Analysis not found in registry
                self.logger.error(f"Analysis '{analysis_name}' not found: {e}")
                if analysis_config.get('required', False):
                    raise
            except TypeError as e:
                # Invalid return type
                self.logger.error(f"Analysis '{analysis_name}' returned invalid type: {e}")
                if analysis_config.get('required', False):
                    raise
            except Exception as e:
                # Analysis execution failed
                self.logger.error(f"Analysis '{analysis_name}' failed: {e}")
                if analysis_config.get('required', False):
                    raise
                results[analysis_name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all analysis results.
        
        Returns:
            Dict with summary statistics
        """
        summary = {
            'total_analyses': len(self.results),
            'successful': sum(1 for r in self.results.values() if isinstance(r, dict) and 'error' not in r),
            'failed': sum(1 for r in self.results.values() if isinstance(r, dict) and 'error' in r),
            'metrics': {}
        }
        
        # Flatten all metrics
        for analysis_name, result in self.results.items():
            if isinstance(result, dict) and 'error' not in result:
                for metric_name, value in result.items():
                    summary['metrics'][f"{analysis_name}.{metric_name}"] = value
        
        return summary
    
    def get_results(self) -> Dict[str, Any]:
        """Get all analysis results.
        
        Returns:
            Dict of all analysis results
        """
        return self.results.copy()
    
    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results = {}