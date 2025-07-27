#!/usr/bin/env python3
"""Complete NaviGraph pipeline example.

This example demonstrates how to use the NaviGraph framework for a complete
behavioral analysis pipeline from data loading to visualization.
"""

import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from navigraph.core import (
    ExperimentRunner, VisualizationPipeline, Session,
    VisualizationConfig, ColorTheme, OutputFormat,
    create_default_configs, registry
)
from navigraph.core.file_discovery import FileDiscoveryEngine


def create_mock_experiment_data(experiment_path: Path, num_sessions: int = 3):
    """Create mock experimental data for demonstration.
    
    Args:
        experiment_path: Path to experiment directory
        num_sessions: Number of sessions to create
    """
    print(f"Creating mock experiment data with {num_sessions} sessions...")
    
    # Create experiment directory structure
    experiment_path.mkdir(exist_ok=True)
    shared_resources_path = experiment_path / "shared_resources"
    shared_resources_path.mkdir(exist_ok=True)
    
    # Create mock sessions
    for session_idx in range(num_sessions):
        session_id = f"session_{session_idx:03d}"
        session_path = experiment_path / session_id
        session_path.mkdir(exist_ok=True)
        
        # Create mock DeepLabCut data
        np.random.seed(42 + session_idx)  # Reproducible data
        n_frames = 1000 + session_idx * 100  # Varying session lengths
        
        # Simulate animal movement through a 4x4 grid maze
        grid_size = 4
        tile_size = 100  # pixels per tile
        
        # Create realistic trajectory
        x_coords = []
        y_coords = []
        likelihoods = []
        
        # Start at random position
        current_x = np.random.uniform(50, 350)
        current_y = np.random.uniform(50, 350)
        
        for frame in range(n_frames):
            # Add some movement noise
            dx = np.random.normal(0, 2)
            dy = np.random.normal(0, 2)
            
            # Occasional larger movements (exploring)
            if np.random.random() < 0.1:
                dx += np.random.normal(0, 20)
                dy += np.random.normal(0, 20)
            
            current_x = np.clip(current_x + dx, 0, 400)
            current_y = np.clip(current_y + dy, 0, 400)
            
            x_coords.append(current_x)
            y_coords.append(current_y)
            likelihoods.append(np.random.uniform(0.7, 0.99))
        
        # Create DeepLabCut-style DataFrame
        scorer = "DLC_researcher_test_shuffle1"
        bodypart = "nose"
        
        # Multi-level column structure
        columns = pd.MultiIndex.from_tuples([
            (scorer, bodypart, 'x'),
            (scorer, bodypart, 'y'),
            (scorer, bodypart, 'likelihood')
        ])
        
        dlc_data = pd.DataFrame({
            (scorer, bodypart, 'x'): x_coords,
            (scorer, bodypart, 'y'): y_coords,
            (scorer, bodypart, 'likelihood'): likelihoods
        })
        dlc_data.columns = columns
        
        # Save as HDF5 (DeepLabCut format)
        h5_path = session_path / f"{session_id}DLC_researcher_test_shuffle1.h5"
        dlc_data.to_hdf(h5_path, key='df_with_missing', mode='w')
        
        print(f"  Created {session_id}: {n_frames} frames")
    
    # Create mock maze map image (4x4 grid, 100px per tile)
    import cv2
    map_height, map_width = 400, 400  # 4x4 grid with 100px tiles
    map_image = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255  # White background
    
    # Draw grid lines
    for i in range(5):  # 5 lines for 4x4 grid
        x = i * 100
        y = i * 100
        cv2.line(map_image, (x, 0), (x, map_height), (0, 0, 0), 2)  # Vertical lines
        cv2.line(map_image, (0, y), (map_width, y), (0, 0, 0), 2)   # Horizontal lines
    
    # Add some colored tiles for visual interest
    cv2.rectangle(map_image, (0, 0), (100, 100), (255, 0, 0), -1)      # Red bottom-left
    cv2.rectangle(map_image, (300, 300), (400, 400), (0, 255, 0), -1)  # Green top-right
    
    # Save map image
    map_image_path = shared_resources_path / "maze_map.png"
    cv2.imwrite(str(map_image_path), map_image)
    
    print("Mock experiment data created successfully!")


def create_experiment_configuration():
    """Create experiment configuration for the pipeline."""
    config = {
        # Data sources in processing order
        'data_sources': [
            {
                'name': 'deeplabcut',
                'type': 'deeplabcut',
                'required': True,
                'file_pattern': r'.*DLC.*\.h5$',
                'config': {
                    'bodypart': 'nose',
                    'likelihood_threshold': 0.9
                }
            },
            {
                'name': 'map_integration',
                'type': 'map_integration',
                'required': True,
                'config': {}
            },
            {
                'name': 'graph_integration', 
                'type': 'graph_integration',
                'required': False,
                'config': {}
            }
        ],
        
        # Shared resources
        'shared_resources': [
            {
                'name': 'maze_map',
                'type': 'map_provider',
                'config': {
                    'map_path': 'shared_resources/maze_map.png',
                    'map_settings': {
                        'segment_length': 100,
                        'origin': [0, 0],
                        'grid_size': [4, 4],
                        'pixel_to_meter': 100.0
                    }
                }
            },
            {
                'name': 'graph',
                'type': 'graph_provider', 
                'config': {
                    'height': 3
                }
            }
        ],
        
        # Analysis configuration
        'analyzers': [
            {
                'name': 'spatial_metrics',
                'type': 'spatial_metrics',
                'config': {}
            },
            {
                'name': 'exploration_metrics',
                'type': 'exploration_metrics',
                'config': {}
            }
        ],
        
        # Session-level settings
        'session_settings': {
            'likelihood': 0.9
        },
        
        # Analysis metrics to compute
        'analyze': {
            'metrics': {
                'time_to_corner': {
                    'func_name': 'time_a_to_b',
                    'args': {'a': 0, 'b': 15}  # Bottom-left to top-right
                },
                'exploration_over_time': {
                    'func_name': 'exploration_percentage', 
                    'args': {}
                }
            }
        }
    }
    
    return config


def demonstrate_basic_pipeline(experiment_path: Path):
    """Demonstrate basic analysis pipeline."""
    print("\n=== Running Basic Analysis Pipeline ===")
    
    # Load experiment configuration
    config = create_experiment_configuration()
    
    # Initialize file discovery
    from loguru import logger
    file_discovery = FileDiscoveryEngine(str(experiment_path), logger)
    
    # Discover sessions
    session_folders = file_discovery.discover_session_folders()
    print(f"Discovered {len(session_folders)} sessions: {session_folders}")
    
    # Create sessions
    sessions = []
    for session_folder in session_folders[:2]:  # Process first 2 sessions for demo
        print(f"\nProcessing {session_folder}...")
        
        # File discovery for this session
        file_patterns = {
            'deeplabcut': r'.*DLC.*\.h5$'
        }
        discovered_files = file_discovery.match_files_in_session(session_folder, file_patterns)
        print(f"  Discovered files: {discovered_files}")
        
        # Create session configuration
        session_config = config.copy()
        session_config['session_id'] = session_folder
        session_config['experiment_path'] = str(experiment_path)  # Add experiment path for file discovery
        
        # Add discovered file paths to data source configs
        for ds_config in session_config['data_sources']:
            if ds_config['name'] in discovered_files:
                ds_config['discovered_file_path'] = discovered_files[ds_config['name']]
        
        try:
            # Create session (this will integrate all data sources)
            from loguru import logger
            session = Session(
                session_configuration=session_config,
                logger_instance=logger
            )
            sessions.append(session)
            
            # Show session info
            metadata = session.get_session_metadata()
            print(f"  ✓ Session created: {metadata['total_frames']} frames, {metadata['total_columns']} columns")
            
        except Exception as e:
            print(f"  ✗ Session creation failed: {e}")
            continue
    
    return sessions, config


def demonstrate_analysis(sessions, config):
    """Demonstrate analysis execution."""
    print("\n=== Running Analysis ===")
    
    if not sessions:
        print("No sessions available for analysis")
        return None
    
    # Create experiment runner
    from omegaconf import DictConfig
    experiment_config = DictConfig(config)
    runner = ExperimentRunner(experiment_config)
    runner.sessions = sessions  # Manually set sessions for example
    
    try:
        # Run analysis
        results = runner.run_analysis()
        print(f"✓ Analysis complete!")
        print(f"  Results shape: {results.shape}")
        print(f"  Sessions analyzed: {list(results.columns)}")
        print(f"  Metrics computed: {list(results.index)}")
        
        return results
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        return None


def demonstrate_visualization(sessions, temp_dir):
    """Demonstrate visualization system."""
    print("\n=== Creating Visualizations ===")
    
    if not sessions:
        print("No sessions available for visualization")
        return
    
    # Create output directory
    viz_output_path = Path(temp_dir) / "visualizations"
    viz_output_path.mkdir(exist_ok=True)
    
    # Use publication theme for nice outputs
    viz_config = create_default_configs()["publication"]
    viz_config.output_path = str(viz_output_path)
    
    # Configure visualizations
    pipeline_config = {
        'visualizations': {
            'trajectory': {
                'plugin': 'trajectory_visualizer',
                'enabled': True,
                'config': {
                    'trail_length': 100,
                    'show_confidence': True
                }
            },
            'map_overlay': {
                'plugin': 'map_visualizer', 
                'enabled': True,
                'config': {
                    'show_tile_ids': True,
                    'overlay_alpha': 0.7
                }
            }
        }
    }
    
    # Create visualization pipeline
    pipeline = VisualizationPipeline(pipeline_config)
    pipeline.viz_config = viz_config
    
    # Generate visualizations for each session
    for session in sessions[:1]:  # Just first session for demo
        try:
            print(f"Creating visualizations for {session.session_id}...")
            
            results = pipeline.create_session_visualizations(
                session=session,
                output_path=str(viz_output_path)
            )
            
            print(f"  ✓ Visualizations created: {sum(len(files) for files in results.values())} files")
            for viz_name, files in results.items():
                if files:
                    print(f"    {viz_name}: {len(files)} files")
            
        except Exception as e:
            print(f"  ✗ Visualization failed: {e}")


def demonstrate_configuration_management():
    """Demonstrate configuration management features."""
    print("\n=== Configuration Management ===")
    
    # Show available presets
    presets = create_default_configs()
    print(f"Available visualization presets: {list(presets.keys())}")
    
    # Create custom configuration
    custom_config = VisualizationConfig(
        theme=ColorTheme.DARK,
        output_formats=[OutputFormat.PNG, OutputFormat.SVG],
        trajectory_settings={
            'trail_length': 200,
            'show_confidence': True,
            'confidence_threshold': 0.95
        }
    )
    
    print(f"Custom config theme: {custom_config.theme.value}")
    print(f"Custom config outputs: {[fmt.value for fmt in custom_config.output_formats]}")
    
    # Show plugin information
    plugins = registry.list_all_plugins()
    print(f"\nRegistered plugins:")
    for category, plugin_list in plugins.items():
        print(f"  {category}: {len(plugin_list)} plugins")


def main():
    """Run the complete pipeline demonstration."""
    print("NaviGraph Complete Pipeline Example")
    print("=" * 50)
    
    # Create temporary directory for demonstration
    temp_dir = tempfile.mkdtemp()
    experiment_path = Path(temp_dir) / "demo_experiment"
    
    try:
        # Step 1: Create mock data
        create_mock_experiment_data(experiment_path, num_sessions=3)
        
        # Step 2: Demonstrate configuration management
        demonstrate_configuration_management()
        
        # Step 3: Run basic pipeline
        sessions, config = demonstrate_basic_pipeline(experiment_path)
        
        # Step 4: Run analysis
        if sessions:
            results = demonstrate_analysis(sessions, config)
            
            # Step 5: Create visualizations
            demonstrate_visualization(sessions, temp_dir)
        
        print(f"\n=== Pipeline Complete ===")
        print(f"Demo files created in: {temp_dir}")
        print("Note: In a real workflow, you would:")
        print("1. Prepare your experimental data in the expected format")
        print("2. Configure data sources, analyzers, and visualizations")
        print("3. Run the pipeline with your configuration")
        print("4. Examine results and visualizations")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup (comment out to inspect generated files)
        # shutil.rmtree(temp_dir)
        print(f"\nTemporary files at: {temp_dir}")


if __name__ == "__main__":
    main()