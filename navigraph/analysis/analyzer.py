"""Analyzer system for NaviGraph using registry-based metrics.

This module provides an analyzer that loads metric functions from the registry
based on configuration and applies them to session data.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import pickle
from datetime import datetime
from loguru import logger

from ..core.registry import registry


class Analyzer:
    """Analyzer that uses registry to load and execute metrics on session data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize analyzer with experiment configuration.
        
        Args:
            config: Complete experiment configuration containing analysis settings
        """
        self.config = config
        self.analysis_config = config.get('analyze', {})
        self.metrics_config = self.analysis_config.get('metrics', {})
        self.cross_metrics_config = self.analysis_config.get('cross_session_metrics', {})
        self._load_metrics()
    
    def run(self, sessions: List[Any], output_dir: Path) -> Dict[str, Any]:
        """Run complete analysis pipeline on all sessions.
        
        Args:
            sessions: List of session objects with get_integrated_dataframe() method
            output_dir: Directory where results will be saved
            
        Returns:
            Complete analysis results including session and cross-session metrics
        """
        logger.info(f"Starting analysis of {len(sessions)} sessions")
        
        # Process each session
        session_results = []
        raw_data_collection = {}
        
        for i, session in enumerate(sessions, 1):
            logger.info(f"Processing session {i}/{len(sessions)}: {getattr(session, 'session_path', 'unknown')}")
            result = self.analyze_session(session)
            session_results.append(result)
            
            # Collect raw dataframe if export is configured
            if self.analysis_config.get('save_raw_data_as_pkl', False):
                session_id = getattr(session, 'session_id', f'session_{i}')
                raw_dataframe = session.get_integrated_dataframe()
                raw_data_collection[session_id] = {
                    'dataframe': raw_dataframe,
                    'session_path': str(getattr(session, 'session_path', 'unknown'))
                }
        
        # Run cross-session analysis
        cross_results = self.analyze_cross_session(session_results)
        
        # Export results
        self.export_results(session_results, cross_results, output_dir, raw_data_collection)
        
        logger.info("Analysis pipeline completed")
        
        return {
            'session_results': session_results,
            'cross_session_results': cross_results
        }
    
    def _load_metrics(self):
        """Load metric functions from registry based on configuration."""
        # Load session-level metrics
        self.session_metrics = {}
        for metric_name, metric_config in self.metrics_config.items():
            func_name = metric_config.get('func_name')
            if func_name:
                try:
                    metric_func = registry.get_session_metric(func_name)
                    args = metric_config.get('args', {})
                    self.session_metrics[metric_name] = (metric_func, args)
                    logger.debug(f"Loaded session metric: {metric_name} -> {func_name}")
                except Exception as e:
                    logger.warning(f"Failed to load session metric {func_name}: {e}")
        
        # Load cross-session metrics
        self.cross_session_metrics = {}
        for metric_name, metric_config in self.cross_metrics_config.items():
            func_name = metric_config.get('func_name')
            if func_name:
                try:
                    metric_func = registry.get_cross_session_metric(func_name)
                    args = metric_config.get('args', {})
                    self.cross_session_metrics[metric_name] = (metric_func, args)
                    logger.debug(f"Loaded cross-session metric: {metric_name} -> {func_name}")
                except Exception as e:
                    logger.warning(f"Failed to load cross-session metric {func_name}: {e}")
    
    def analyze_session(self, session: Any) -> Dict[str, Any]:
        """Apply all configured metrics to a single session.
        
        Args:
            session: Session object with get_integrated_dataframe() method
            
        Returns:
            Dictionary containing session path, frame count, and computed metric values
        """
        results = {
            'session_path': str(getattr(session, 'session_path', 'unknown')),
            'metrics': {}
        }
        
        # Get basic session info
        try:
            dataframe = session.get_integrated_dataframe()
            results['num_frames'] = len(dataframe)
        except Exception as e:
            logger.error(f"Failed to get dataframe from session: {e}")
            results['num_frames'] = 0
        
        # Apply each configured session metric
        for metric_name, (metric_func, args) in self.session_metrics.items():
            try:
                resolved_args = self._resolve_args(args)
                result_value = metric_func(session, **resolved_args)
                results['metrics'][metric_name] = result_value
                logger.debug(f"Computed {metric_name}: {result_value}")
            except Exception as e:
                logger.error(f"Failed to compute {metric_name}: {e}")
                results['metrics'][metric_name] = None
        
        return results
    
    def analyze_cross_session(self, session_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply cross-session metrics to aggregate data from multiple sessions.
        
        Args:
            session_results: List of session analysis results from analyze_session()
            
        Returns:
            Dictionary containing cross-session metric results
        """
        results = {}
        
        if len(session_results) < 2:
            logger.info("Skipping cross-session analysis: less than 2 sessions")
            return results
        
        # Apply each configured cross-session metric
        for metric_name, (metric_func, args) in self.cross_session_metrics.items():
            try:
                resolved_args = self._resolve_args(args)
                result_value = metric_func(session_results, **resolved_args)
                results[metric_name] = result_value
                logger.debug(f"Computed cross-session {metric_name}: {result_value}")
            except Exception as e:
                logger.error(f"Failed to compute cross-session {metric_name}: {e}")
                results[metric_name] = None
        
        return results
    
    def _resolve_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Replace @graph.metadata references with actual values from config.
        
        Args:
            args: Metric arguments that may contain @graph.metadata.field_name references
            
        Returns:
            Arguments with metadata references replaced by actual values
        """
        resolved = {}
        graph_metadata = self.config.get('graph', {}).get('metadata', {})
        
        for key, value in args.items():
            if isinstance(value, str) and value.startswith('@graph.metadata.'):
                metadata_key = value.split('.')[-1]
                resolved_value = graph_metadata.get(metadata_key, value)
                resolved[key] = resolved_value
                logger.debug(f"Resolved {value} -> {resolved_value}")
            else:
                resolved[key] = value
        
        return resolved
    
    def export_results(self, session_results: List[Dict[str, Any]], 
                      cross_results: Dict[str, Any], output_dir: Path,
                      raw_data_collection: Dict[str, Dict] = None) -> None:
        """Export analysis results in configured formats (CSV and/or PKL).
        
        Args:
            session_results: Session analysis results from analyze_session()
            cross_results: Cross-session analysis results from analyze_cross_session()
            output_dir: Directory where files will be saved
            raw_data_collection: Raw session dataframes for export
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export to CSV if configured
        if self.analysis_config.get('save_as_csv', True):
            csv_path = output_dir / f'session_metrics_{timestamp}.csv'
            self._export_csv(session_results, csv_path)
            
        # Export to PKL if configured  
        if self.analysis_config.get('save_as_pkl', True):
            pkl_path = output_dir / f'session_metrics_{timestamp}.pkl'
            all_results = {
                'session_results': session_results,
                'cross_session_results': cross_results,
                'config': self.config,
                'timestamp': timestamp
            }
            self._export_pickle(all_results, pkl_path)
        
        # Export raw session data if configured
        if self.analysis_config.get('save_raw_data_as_pkl', False) and raw_data_collection:
            self._export_raw_data(raw_data_collection, output_dir, timestamp)
        
        logger.info(f"Results exported to: {output_dir}")
    
    def _export_csv(self, session_results: List[Dict[str, Any]], filepath: Path) -> None:
        """Save session results as CSV with metrics as columns."""
        rows = []
        
        for result in session_results:
            row = {
                'session_path': result.get('session_path', ''),
                'num_frames': result.get('num_frames', 0)
            }
            
            # Add metric values as columns
            metrics = result.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                row[metric_name] = metric_value
            
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
            logger.info(f"CSV exported: {filepath}")
        else:
            logger.warning("No session results to export to CSV")
    
    def _export_pickle(self, results: Dict[str, Any], filepath: Path) -> None:
        """Save complete results as pickle file including config and metadata."""
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Pickle exported: {filepath}")
    
    def _export_raw_data(self, raw_data_collection: Dict[str, Dict], 
                         output_dir: Path, timestamp: str) -> None:
        """Export raw session dataframes to pickle file.
        
        Args:
            raw_data_collection: Dict mapping session_id to session data
            output_dir: Directory where file will be saved
            timestamp: Timestamp for the filename
        """
        if not raw_data_collection:
            logger.warning("No raw data to export")
            return
        
        raw_data_export = {
            **raw_data_collection,  # Session data keyed by session_id
            'timestamp': timestamp,
            'experiment_path': str(getattr(self, 'experiment_path', 'unknown'))
        }
        
        filepath = output_dir / f'raw_session_data_{timestamp}.pkl'
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(raw_data_export, f)
            
            total_sessions = len(raw_data_collection)
            logger.info(f"Raw session data exported: {filepath} ({total_sessions} sessions)")
            
        except Exception as e:
            logger.error(f"Failed to export raw session data: {e}")