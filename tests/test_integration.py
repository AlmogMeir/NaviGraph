#!/usr/bin/env python3
"""Integration tests for NaviGraph framework.

This module provides comprehensive integration tests that verify the entire
plugin architecture works correctly end-to-end.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from navigraph.core import (
    registry, Session, ExperimentRunner, VisualizationPipeline,
    VisualizationConfig, ColorTheme, OutputFormat,
    AnalysisResult, AnalysisMetadata
)
from navigraph.core.file_discovery import FileDiscoveryEngine
from navigraph.core.exceptions import NavigraphError


class TestPluginRegistration:
    """Test plugin registration and discovery."""
    
    def test_all_plugin_types_registered(self):
        """Test that all expected plugin types are registered."""
        plugins = registry.list_all_plugins()
        
        # Verify we have plugins of all types
        assert len(plugins['data_sources']) > 0, "No data source plugins registered"
        assert len(plugins['shared_resources']) > 0, "No shared resource plugins registered"
        assert len(plugins['analyzers']) > 0, "No analyzer plugins registered"
        assert len(plugins['visualizers']) > 0, "No visualizer plugins registered"
        
        # Verify specific expected plugins
        assert 'deeplabcut' in plugins['data_sources']
        assert 'spatial_metrics' in plugins['analyzers']
        assert 'trajectory_visualizer' in plugins['visualizers']
    
    def test_plugin_factory_methods(self):
        """Test that all plugins have working factory methods."""
        plugins = registry.list_all_plugins()
        
        # Test data source factory
        dlc_class = registry.get_data_source_plugin('deeplabcut')
        dlc_instance = dlc_class.from_config({'bodypart': 'nose'})
        assert dlc_instance is not None
        
        # Test analyzer factory
        spatial_class = registry.get_analyzer_plugin('spatial_metrics')
        spatial_instance = spatial_class.from_config({})
        assert spatial_instance is not None
        
        # Test visualizer factory
        traj_class = registry.get_visualizer_plugin('trajectory_visualizer')
        traj_instance = traj_class.from_config({})
        assert traj_instance is not None


class TestSessionIntegration:
    """Test session data integration."""
    
    @pytest.fixture
    def mock_session_config(self):
        """Create a mock session configuration."""
        return {
            'session_id': 'test_session_001',
            'data_sources': [
                {
                    'name': 'deeplabcut',
                    'type': 'deeplabcut',
                    'required': True,
                    'config': {
                        'bodypart': 'nose',
                        'likelihood_threshold': 0.9
                    }
                }
            ],
            'shared_resources': {},
            'session_settings': {
                'likelihood': 0.9
            }
        }
    
    @pytest.fixture
    def mock_dataframe(self):
        """Create a mock integrated DataFrame."""
        np.random.seed(42)  # For reproducible tests
        n_frames = 100
        
        data = {
            'keypoints_x': np.random.uniform(100, 500, n_frames),
            'keypoints_y': np.random.uniform(100, 400, n_frames),
            'keypoints_likelihood': np.random.uniform(0.5, 1.0, n_frames),
            'tile_id': np.random.randint(0, 16, n_frames),
            'tree_position': np.random.randint(0, 8, n_frames)
        }
        
        return pd.DataFrame(data)
    
    def test_session_creation(self, mock_session_config):
        """Test basic session creation."""
        with patch('navigraph.core.session.FileDiscoveryEngine'):
            session = Session(
                session_configuration=mock_session_config,
                logger_instance=Mock()
            )
            
            assert session.session_id == 'test_session_001'
            assert session.likelihood == 0.9
    
    def test_session_data_integration(self, mock_session_config, mock_dataframe):
        """Test session data integration flow."""
        with patch('navigraph.core.session.FileDiscoveryEngine'), \
             patch.object(Session, '_integrate_all_data_sources', return_value=mock_dataframe):
            
            session = Session(
                session_configuration=mock_session_config,
                logger_instance=Mock()
            )
            
            df = session.get_integrated_dataframe()
            assert len(df) == 100
            assert 'keypoints_x' in df.columns
            assert 'tile_id' in df.columns


class TestAnalyzerIntegration:
    """Test analyzer plugin integration."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock session with data."""
        session = Mock()
        session.session_id = 'test_session_001'
        session.logger = Mock()
        
        # Create test data
        np.random.seed(42)
        n_frames = 100
        data = {
            'tile_id': np.random.randint(0, 16, n_frames),
            'tree_position': np.random.randint(0, 8, n_frames),
            'keypoints_x': np.random.uniform(100, 500, n_frames),
            'keypoints_y': np.random.uniform(100, 400, n_frames),
            'keypoints_likelihood': np.random.uniform(0.8, 1.0, n_frames)
        }
        df = pd.DataFrame(data)
        
        session.get_integrated_dataframe.return_value = df
        session.get_session_config.return_value = {
            'analyze': {
                'metrics': {
                    'test_metric': {
                        'func_name': 'time_a_to_b',
                        'args': {'a': 0, 'b': 15}
                    }
                }
            }
        }
        session.get_session_stream_info.return_value = {'fps': 30}
        session.shared_resources = {}
        
        return session
    
    def test_spatial_analyzer_returns_analysis_result(self, mock_session):
        """Test that spatial analyzer returns AnalysisResult."""
        analyzer_class = registry.get_analyzer_plugin('spatial_metrics')
        analyzer = analyzer_class.from_config({})
        
        result = analyzer.analyze_session(mock_session)
        
        assert isinstance(result, AnalysisResult)
        assert result.session_id == 'test_session_001'
        assert result.analyzer_name == 'spatial_metrics'
        assert isinstance(result.metadata, AnalysisMetadata)
        assert result.metadata.computation_time > 0
    
    def test_analyzer_cross_session_analysis(self, mock_session):
        """Test cross-session analysis with AnalysisResult objects."""
        analyzer_class = registry.get_analyzer_plugin('spatial_metrics')
        analyzer = analyzer_class.from_config({})
        
        # Create multiple session results
        result1 = analyzer.analyze_session(mock_session)
        
        mock_session.session_id = 'test_session_002'
        result2 = analyzer.analyze_session(mock_session)
        
        session_results = {
            'test_session_001': result1,
            'test_session_002': result2
        }
        
        cross_results = analyzer.analyze_cross_session([], session_results)
        assert isinstance(cross_results, dict)


class TestVisualizationIntegration:
    """Test visualization system integration."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_visualization_config_creation(self):
        """Test visualization configuration system."""
        config = VisualizationConfig(
            theme=ColorTheme.DARK,
            output_formats=[OutputFormat.PNG, OutputFormat.PDF]
        )
        
        assert config.theme == ColorTheme.DARK
        assert len(config.output_formats) == 2
        assert config.colors.background == (30, 30, 30)  # Dark theme background
    
    def test_visualization_pipeline_initialization(self, temp_output_dir):
        """Test visualization pipeline initialization."""
        config = {
            'visualizations': {
                'test_viz': {
                    'plugin': 'trajectory_visualizer',
                    'enabled': True,
                    'config': {'trail_length': 50}
                }
            }
        }
        
        pipeline = VisualizationPipeline(config)
        assert len(pipeline.visualizers) == 1
        assert 'test_viz' in pipeline.visualizers
    
    def test_visualization_config_save_load(self, temp_output_dir):
        """Test saving and loading visualization configurations."""
        config = VisualizationConfig(
            theme=ColorTheme.PUBLICATION,
            output_formats=[OutputFormat.PDF]
        )
        
        config_path = Path(temp_output_dir) / 'test_config.json'
        config.save(config_path)
        
        loaded_config = VisualizationConfig.load(config_path)
        assert loaded_config.theme == ColorTheme.PUBLICATION
        assert loaded_config.output_formats == [OutputFormat.PDF]


class TestExperimentRunnerIntegration:
    """Test end-to-end experiment execution."""
    
    @pytest.fixture
    def mock_experiment_config(self):
        """Create a mock experiment configuration."""
        return {
            'data_sources': ['deeplabcut'],
            'shared_resources': [],
            'analyzers': ['spatial_metrics'],
            'output_path': '/tmp/test_output'
        }
    
    @pytest.fixture
    def mock_sessions(self):
        """Create mock sessions for testing."""
        sessions = []
        
        for i in range(3):
            session = Mock()
            session.session_id = f'session_{i:03d}'
            session.logger = Mock()
            
            # Mock integrated dataframe
            np.random.seed(42 + i)
            n_frames = 50
            data = {
                'tile_id': np.random.randint(0, 16, n_frames),
                'tree_position': np.random.randint(0, 8, n_frames),
                'keypoints_x': np.random.uniform(100, 500, n_frames),
                'keypoints_y': np.random.uniform(100, 400, n_frames)
            }
            df = pd.DataFrame(data)
            session.get_integrated_dataframe.return_value = df
            session.get_session_config.return_value = {
                'analyze': {'metrics': {}}
            }
            session.get_session_stream_info.return_value = {'fps': 30}
            session.shared_resources = {}
            
            sessions.append(session)
        
        return sessions
    
    def test_experiment_runner_analysis(self, mock_experiment_config, mock_sessions):
        """Test experiment runner analysis execution."""
        # Create a simple config for ExperimentRunner
        from omegaconf import DictConfig
        config = DictConfig(mock_experiment_config)
        runner = ExperimentRunner(config)
        runner.sessions = mock_sessions  # Manually set sessions for test
        
        # Mock the analyzer to avoid complex setup
        with patch.object(registry, 'get_analyzer_plugin') as mock_get_analyzer:
            mock_analyzer_class = Mock()
            mock_analyzer = Mock()
            
            # Create a proper AnalysisResult
            result = AnalysisResult(
                session_id='test_session',
                analyzer_name='spatial_metrics',
                metrics={'test_metric': [1, 2, 3]},
                metadata=AnalysisMetadata(
                    analyzer_name='spatial_metrics',
                    version='1.0.0',
                    timestamp=datetime.now(),
                    computation_time=0.1,
                    config_hash='abc123'
                )
            )
            
            mock_analyzer.analyze_session.return_value = result
            mock_analyzer.analyze_cross_session.return_value = {'mean_test_metric': 2.0}
            mock_analyzer_class.return_value = mock_analyzer
            mock_get_analyzer.return_value = mock_analyzer_class
            
            # Run analysis
            results = runner.run_analysis()
            
            assert isinstance(results, pd.DataFrame)
            assert len(results.columns) == 3  # 3 sessions


class TestErrorHandling:
    """Test error handling throughout the system."""
    
    def test_plugin_not_found_error(self):
        """Test handling of missing plugins."""
        with pytest.raises(Exception):  # Should raise PluginNotFoundError
            registry.get_data_source_plugin('nonexistent_plugin')
    
    def test_session_initialization_error(self):
        """Test session initialization error handling."""
        invalid_config = {}  # Missing required fields
        
        with pytest.raises(Exception):  # Should handle gracefully
            Session(
                session_configuration=invalid_config,
                logger_instance=Mock()
            )
    
    def test_visualization_error_handling(self):
        """Test visualization error handling."""
        # Test missing output path with configured visualizers
        config = {
            'visualizations': {
                'test_viz': {
                    'plugin': 'trajectory_visualizer',
                    'enabled': True,
                    'config': {}
                }
            }
        }
        viz_config = VisualizationConfig(output_path=None)
        pipeline = VisualizationPipeline(config)
        pipeline.viz_config = viz_config
        
        mock_session = Mock()
        mock_session.get_integrated_dataframe.return_value = pd.DataFrame()
        mock_session.shared_resources = {}
        mock_session.session_id = 'test'
        mock_session.config = {}
        
        with pytest.raises(Exception):  # Should raise VisualizationError for missing output path
            pipeline.create_session_visualizations(mock_session)


class TestBackwardCompatibility:
    """Test backward compatibility with existing functionality."""
    
    def test_session_backward_compatibility_methods(self):
        """Test that Session maintains backward compatibility methods."""
        config = {
            'session_id': 'test_session',
            'data_sources': [],
            'session_settings': {'likelihood': 0.8}
        }
        
        with patch('navigraph.core.session.FileDiscoveryEngine'), \
             patch.object(Session, '_integrate_all_data_sources', return_value=pd.DataFrame()):
            
            session = Session(config, Mock())
            
            # Test backward compatibility properties
            assert hasattr(session, 'session_name')
            assert hasattr(session, 'bodyparts')
            assert hasattr(session, 'coords')
            assert hasattr(session, 'likelihood')
            assert session.likelihood == 0.8


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])