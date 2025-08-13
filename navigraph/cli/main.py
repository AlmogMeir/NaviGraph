# Configure matplotlib backend before any other imports
import os
import matplotlib
import warnings

# Fix QT_API environment variable if needed
if os.environ.get('QT_API') == 'pyqt':
    os.environ['QT_API'] = 'pyqt5'

# Setup matplotlib backend with fallback options
def setup_matplotlib_backend():
    """Setup matplotlib backend for GUI functionality."""
    backends_to_try = ['Qt5Agg', 'TkAgg', 'Qt4Agg', 'GTK3Agg']
    
    for backend in backends_to_try:
        try:
            # Suppress warnings during backend testing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                matplotlib.use(backend, force=True)
                
                # Try to import and test the backend
                import matplotlib.pyplot as plt
                fig = plt.figure()
                plt.close(fig)
                return backend
        except Exception:
            continue
    
    # If no GUI backend works, use non-interactive backend
    matplotlib.use('Agg', force=True)
    return 'Agg'

# Setup backend before any matplotlib imports
backend = setup_matplotlib_backend()

"""
NaviGraph CLI - Command-line interface for spatial navigation analysis.

This module provides the command-line interface for NaviGraph, supporting
experiment running, configuration validation, and interactive setup tools.
"""

import sys
import os
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import click
from omegaconf import OmegaConf

# Import experiment runner
from ..core.experiment_runner import ExperimentRunner


def resolve_project_root(config_path: Path) -> Path:
    """Find the project root directory from a config path.
    
    Searches upward from the config file location to find the project root,
    identified by the presence of a .git directory or pyproject.toml file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Path to the project root directory
    """
    current = config_path.parent.absolute()
    
    # Search upward for project root indicators
    while current != current.parent:
        if (current / '.git').exists() or (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    
    # If no project root found, use config directory
    return config_path.parent.absolute()


def process_config_path(config_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process configuration with proper path resolution.
    
    Updates the configuration with resolved paths based on config file location,
    adding special handling for {PROJECT_ROOT} placeholder in paths.
    
    Args:
        config_path: Path to the configuration file  
        config: Loaded configuration dictionary
        
    Returns:
        Updated configuration with resolved paths
    """
    config_dir = config_path.parent.absolute()
    project_root = resolve_project_root(config_path)
    
    # Add metadata
    config['_config_path'] = str(config_path)
    config['_config_dir'] = str(config_dir)
    config['_project_root'] = str(project_root)
    
    # Process experiment_output_path with {PROJECT_ROOT} support
    if 'experiment_output_path' in config:
        output_path = config['experiment_output_path']
        
        if isinstance(output_path, str):
            # Replace {PROJECT_ROOT} placeholder
            if '{PROJECT_ROOT}' in output_path:
                output_path = output_path.replace('{PROJECT_ROOT}', str(project_root))
            
            # Resolve path
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = config_dir / output_path
            
            config['experiment_output_path'] = str(output_path.resolve())
    
    return config


@click.group()
@click.version_option(version='0.2.0', prog_name='NaviGraph')
def cli():
    """NaviGraph - A flexible framework for spatial navigation analysis.
    
    NaviGraph provides tools for analyzing animal navigation behavior through
    integration of pose estimation, spatial mapping, and graph-based analysis.
    
    \b
    Common Commands:
      navigraph run config.yaml          - Run experiment with config
      navigraph setup graph config.yaml  - Setup graph mapping
      navigraph test graph config.yaml   - Test graph mapping
      navigraph validate config.yaml     - Validate configuration
    
    Use 'navigraph COMMAND --help' for more information on each command.
    """
    pass


@cli.command()
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def run(config_path: Path, verbose: bool):
    """Run NaviGraph experiment with configuration file.
    
    CONFIG_PATH: Path to YAML configuration file
    
    \b
    Examples:
      navigraph run config.yaml
      navigraph run experiments/mouse/config.yaml --verbose
    
    The configuration file should specify all experiment parameters including
    data sources, analysis metrics, and output settings.
    """
    try:
        click.echo(f"🚀 Starting NaviGraph experiment")
        click.echo(f"📋 Configuration: {config_path}")
        
        # Load and process configuration
        config = OmegaConf.load(config_path)
        config = process_config_path(config_path, OmegaConf.to_container(config))
        
        # Override verbose if specified
        if verbose:
            config['verbose'] = True
        
        # Create and run experiment
        runner = ExperimentRunner(config)
        
        with click.progressbar(length=1, label='Running experiment') as bar:
            results = runner.run_experiment()
            bar.update(1)
        
        if results is not None:
            click.echo(f"✅ Experiment completed successfully!")
            click.echo(f"📊 Analysis results: {len(results.columns)} sessions, {len(results.index)} metrics")
            
            output_path = config.get('experiment_output_path', '.')
            click.echo(f"📁 Results saved to: {output_path}")
        else:
            click.echo("✅ Experiment completed (no analysis results)")
            
    except Exception as e:
        click.echo(f"❌ Experiment failed: {str(e)}", err=True)
        if config.get('verbose', False):
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.command()
@click.argument('data_path', type=click.Path(exists=True, path_type=Path))
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path),
              help='Configuration file to use for validation (optional)')
@click.option('--format', '-f', 
              type=click.Choice(['table', 'json', 'tree']),
              default='table',
              help='Output format for discovered sessions')
def discover(data_path: Path, config: Optional[Path], format: str):
    """Discover and list available experimental sessions with validation.
    
    Scan a directory for sessions and validate them against configured data sources.
    Without a config file, performs basic directory scanning.
    
    DATA_PATH: Path to directory containing experimental data
    
    \b
    Examples:
      navigraph discover ./data/
      navigraph discover ./data/ --config config.yaml --format json
      navigraph discover ./sessions/ --format tree
    
    \b
    With config file:
      • Validates sessions using configured data source plugins
      • Shows detailed validation results per session
      • Respects plugin file patterns and requirements
    
    \b
    Without config file:
      • Basic directory scanning for potential sessions
      • Lists directories that could contain session data
    """
    # Load plugins for session discovery
    from ..plugins import data_sources, shared_resources, analyzers  # noqa: F401
    
    try:
        click.echo(f"🔍 Discovering sessions in: {data_path}")
        
        if config:
            # Use provided config for plugin-based validation
            click.echo(f"📋 Using configuration: {config}")
            
            # Load configuration
            config_data = OmegaConf.load(config)
            config_data._config_dir = str(config.parent)
            
            # Override experiment path to discovery location
            config_data.experiment_path = str(data_path)
            
            # Create experiment runner
            runner = ExperimentRunner(config_data)
            
            # Discover sessions
            sessions = runner.discover_sessions()
            
            if not sessions:
                click.echo("⚠️  No session directories found")
                return
            
            click.echo(f"📁 Found {len(sessions)} session directories")
            
            # Validate sessions using plugins
            validation_report = runner.validate_sessions(sessions)
            
            # Display results
            if format == 'table':
                click.echo()
                click.echo("Session Validation Results:")
                click.echo("=" * 60)
                
                for session_validation in validation_report.session_validations:
                    status = "✅ VALID" if session_validation.is_valid else "❌ INVALID"
                    click.echo(f"\n{session_validation.session_id}: {status}")
                    
                    # Show data source results
                    for ds_result in session_validation.data_source_results:
                        icon = "✓" if ds_result.is_valid else "✗"
                        click.echo(f"  {icon} {ds_result.name}: ", end="")
                        
                        if ds_result.is_valid:
                            click.echo(f"{len(ds_result.files)} files")
                            if config_data.get('verbose'):
                                for file in ds_result.files[:3]:  # Show first 3
                                    click.echo(f"      • {file.name}")
                                if len(ds_result.files) > 3:
                                    click.echo(f"      ... and {len(ds_result.files)-3} more")
                        else:
                            click.echo(f"{ds_result.error or 'No matching files'}")
                
                # Summary
                click.echo()
                click.echo("Summary:")
                click.echo(f"  Valid sessions: {validation_report.valid_count}/{validation_report.total_count}")
                
                if validation_report.warnings:
                    click.echo(f"  ⚠️  Warnings: {len(validation_report.warnings)}")
                    for warning in validation_report.warnings[:3]:
                        click.echo(f"     • {warning}")
                        
            elif format == 'json':
                import json
                output = {
                    'total_sessions': validation_report.total_count,
                    'valid_sessions': validation_report.valid_count,
                    'sessions': []
                }
                
                for sv in validation_report.session_validations:
                    session_data = {
                        'id': sv.session_id,
                        'valid': sv.is_valid,
                        'data_sources': {}
                    }
                    
                    for ds in sv.data_source_results:
                        session_data['data_sources'][ds.name] = {
                            'valid': ds.is_valid,
                            'files': len(ds.files) if ds.is_valid else 0,
                            'error': ds.error
                        }
                    
                    output['sessions'].append(session_data)
                
                click.echo(json.dumps(output, indent=2))
                
            elif format == 'tree':
                # Tree-like display
                for i, sv in enumerate(validation_report.session_validations):
                    is_last = i == len(validation_report.session_validations) - 1
                    prefix = "└── " if is_last else "├── "
                    status = "✅" if sv.is_valid else "❌"
                    click.echo(f"{prefix}{sv.session_id} {status}")
                    
                    for j, ds in enumerate(sv.data_source_results):
                        is_last_ds = j == len(sv.data_source_results) - 1
                        tree_prefix = "    " if is_last else "│   "
                        ds_prefix = "└── " if is_last_ds else "├── "
                        icon = "✓" if ds.is_valid else "✗"
                        files_info = f"({len(ds.files)} files)" if ds.is_valid else "(no files)"
                        click.echo(f"{tree_prefix}{ds_prefix}{icon} {ds.name} {files_info}")
        else:
            # Basic directory scanning without config
            click.echo("📂 Basic directory scan (no config provided)")
            
            # Find all subdirectories
            subdirs = [d for d in data_path.iterdir() if d.is_dir()]
            
            if not subdirs:
                click.echo("⚠️  No subdirectories found")
                return
            
            click.echo(f"📁 Found {len(subdirs)} potential session directories:")
            
            for subdir in sorted(subdirs):
                # Count files in directory
                file_count = len(list(subdir.glob('*')))
                click.echo(f"  • {subdir.name} ({file_count} items)")
                
    except Exception as e:
        click.echo(f"❌ Discovery failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command('list-plugins')
@click.option('--category', '-c',
              type=click.Choice(['data_sources', 'shared_resources', 'analyzers', 'all']),
              default='all',
              help='Plugin category to list')
@click.option('--format', '-f',
              type=click.Choice(['table', 'json', 'simple']),
              default='table',
              help='Output format')
def list_plugins(category: str, format: str):
    """List all available NaviGraph plugins by category.
    
    Display plugins for data integration, analysis, and visualization.
    Use this to discover available functionality and verify plugin registration.
    
    \b
    Examples:
      navigraph list-plugins
      navigraph list-plugins --category analyzers
      navigraph list-plugins --format json
    
    \b
    Plugin categories:
      data_sources      - DeepLabCut, map integration, graph integration
      shared_resources  - Map provider, graph provider, calibration
      analyzers        - Spatial metrics, navigation analysis, exploration
      visualizers      - Keypoint, map, trajectory visualizations
    """
    try:
        click.echo("🔌 NaviGraph Plugin Registry")
        
        # Load plugins to populate the registry
        from ..plugins import data_sources, shared_resources, analyzers
        from ..core.registry import registry
        
        # Get all plugins
        all_plugins = registry.list_all_plugins()
        
        # Filter by category if specified
        if category != 'all':
            all_plugins = {category: all_plugins.get(category, [])}
        
        # Remove empty categories
        all_plugins = {k: v for k, v in all_plugins.items() if v}
        
        if not all_plugins:
            click.echo("⚠️  No plugins found")
            return
        
        if format == 'simple':
            # Simple list format
            for cat, plugins in all_plugins.items():
                for plugin in plugins:
                    click.echo(f"{cat}:{plugin}")
        
        elif format == 'json':
            # JSON format
            import json
            click.echo(json.dumps(all_plugins, indent=2))
        
        else:  # table format
            # Table format with descriptions
            click.echo()
            
            category_descriptions = {
                'data_sources': '📊 Data Sources - Integrate data from various sources',
                'shared_resources': '🔗 Shared Resources - Provide cross-session resources',
                'analyzers': '🧮 Analyzers - Compute behavioral and navigation metrics'
            }
            
            for cat, plugins in all_plugins.items():
                desc = category_descriptions.get(cat, f"📦 {cat.title()}")
                click.echo(f"{desc}")
                click.echo("-" * len(desc))
                
                for plugin in plugins:
                    click.echo(f"  • {plugin}")
                
                click.echo()
            
            total_plugins = sum(len(plugins) for plugins in all_plugins.values())
            click.echo(f"Total: {total_plugins} plugins across {len(all_plugins)} categories")
            
    except Exception as e:
        click.echo(f"❌ Plugin listing failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
def validate(config_path: Path):
    """Validate configuration file for errors and compatibility.
    
    Check configuration syntax, required fields, file paths, and plugin
    dependencies before running experiments to catch issues early.
    
    CONFIG_PATH: Path to YAML configuration file to validate
    
    \b
    Examples:
      navigraph validate examples/basic_maze/config.yaml
      navigraph validate my_experiment_config.yaml
    
    \b
    Validation checks:
      ✓ YAML syntax and structure
      ✓ Required fields present
      ✓ File paths exist and accessible
      ✓ Plugin dependencies satisfied
      ✓ Configuration parameter validity
    """
    try:
        click.echo(f"🔍 Validating configuration: {config_path}")
        
        # Load plugins for validation
        from ..plugins import data_sources, shared_resources, analyzers
        
        # Load configuration
        config = OmegaConf.load(config_path)
        issues = []
        warnings = []
        
        # Check required fields
        required_fields = ['stream_path', 'keypoint_detection_file_path']
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field: {field}")
        
        # Check paths exist
        path_fields = ['stream_path', 'keypoint_detection_file_path', 'map_path']
        for field in path_fields:
            if field in config:
                path = Path(config[field])
                if not path.exists():
                    issues.append(f"Path does not exist: {field} = {path}")
        
        # Check plugin configuration
        if 'analyze' in config and 'metrics' in config.analyze:
            metrics = config.analyze.metrics
            from ..core.registry import registry
            available_analyzers = registry.list_all_plugins()['analyzers']
            
            for metric_name, metric_config in metrics.items():
                func_name = metric_config.get('func_name')
                
                # Map function names to analyzers (simplified check)
                analyzer_mapping = {
                    'time_a_to_b': 'spatial_metrics',
                    'velocity_a_to_b': 'spatial_metrics', 
                    'num_nodes_in_path': 'navigation_metrics',
                    'shortest_path_from_a_to_b': 'navigation_metrics',
                    'exploration_percentage': 'exploration_metrics',
                    'avg_node_time': 'exploration_metrics'
                }
                
                required_analyzer = analyzer_mapping.get(func_name)
                if required_analyzer and required_analyzer not in available_analyzers:
                    issues.append(f"Analyzer '{required_analyzer}' required for metric '{metric_name}' not available")
        
        # Report results
        click.echo()
        if issues:
            click.echo("❌ Configuration validation failed:")
            for issue in issues:
                click.echo(f"   • {issue}")
        else:
            click.echo("✅ Configuration is valid!")
        
        if warnings:
            click.echo("⚠️  Warnings:")
            for warning in warnings:
                click.echo(f"   • {warning}")
        
        if issues:
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Validation failed: {str(e)}", err=True)
        sys.exit(1)


# Graph setup and testing commands
@cli.group()
def setup():
    """Setup and configuration tools for NaviGraph components.
    
    Interactive tools for configuring graph mappings, calibration,
    and other system components.
    
    \b
    Examples:
      navigraph setup graph config.yaml
      navigraph setup calibration config.yaml
    """
    pass


@setup.command('graph')
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
def setup_graph(config_path: Path):
    """Setup spatial mapping between graph nodes/edges and map regions.
    
    Launch interactive dual-view GUI for creating spatial mappings that link
    graph nodes and edges to regions on a map image.
    
    CONFIG_PATH: Path to configuration file containing graph and map settings
    
    \b
    The GUI provides two mapping modes:
      Grid Setup    - Place a grid and assign nodes/edges to cells
      Manual Drawing - Draw custom contours for each element
    
    \b
    Example:
      navigraph setup graph config.yaml
    
    \b
    Required config sections:
      graph_structure: Defines graph type (binary_tree or custom)
      map_path: Path to the map image
      graph_mapping.mapping_file: Where to save the mapping
    """
    try:
        click.echo(f"Loading configuration: {config_path}")
        
        # Load configuration
        config = OmegaConf.load(config_path)
        config._config_dir = str(config_path.parent)
        
        # Import graph modules
        from ..core.graph.structures import GraphStructure
        from ..core.graph.builders import build_binary_tree
        from ..core.graph.setup_gui_qt import launch_setup_gui
        from ..core.graph.storage import MappingStorage
        import numpy as np
        import cv2
        import importlib
        
        # Get map path from config
        map_path = config.get('map_path')
        if not map_path:
            click.echo("Error: map_path not found in config", err=True)
            sys.exit(1)
        
        # Resolve map path relative to config directory
        if not Path(map_path).is_absolute():
            map_path = Path(config._config_dir) / map_path
        
        # Load map image
        map_array = cv2.imread(str(map_path))
        if map_array is None:
            click.echo(f"Error: Failed to load map image: {map_path}", err=True)
            sys.exit(1)
        
        # Create graph from config
        graph_config = config.get('graph_structure', {})
        graph_type = graph_config.get('type', 'binary_tree')
        
        if graph_type == 'binary_tree':
            # Use binary_tree_height from graph_structure, fallback to graph.height
            height = graph_config.get('binary_tree_height')
            if height is None:
                height = config.get('graph', {}).get('height', 4)
            
            graph = build_binary_tree(height)
            click.echo(f"Graph: binary tree, height {height} ({len(graph.nodes)} nodes)")
        
        elif graph_type == 'custom':
            # Import and call custom function
            func_path = graph_config.get('custom_graph_function')
            if not func_path:
                click.echo("Error: custom_graph_function required for custom graph type", err=True)
                sys.exit(1)
            
            try:
                # Import the function dynamically
                module_name, func_name = func_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
                
                # Call function with config as argument
                nx_graph = func(config)
                graph = GraphStructure(nx_graph)
                click.echo(f"Graph: custom ({len(graph.nodes)} nodes)")
            except Exception as e:
                click.echo(f"Error: Failed to create custom graph: {str(e)}", err=True)
                sys.exit(1)
        
        else:
            click.echo(f"Error: Unsupported graph type '{graph_type}' (supported: binary_tree, custom)", err=True)
            sys.exit(1)
        
        # Launch PyQt5 GUI
        click.echo("Launching mapping interface...")
        
        try:
            mapping = launch_setup_gui(graph, map_array)
            
        except ImportError as e:
            click.echo(f"Error: PyQt5 is required but not installed: {e}", err=True)
            click.echo("Install with: pip install PyQt5", err=True)
            sys.exit(1)
            
        except Exception as e:
            click.echo(f"Error: Failed to launch interface: {e}", err=True)
            sys.exit(1)
        
        # GUI closed
        click.echo("Closing GUI...")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if '--verbose' in sys.argv:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@setup.command('calibration')
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
def setup_calibration(config_path: Path):
    """Setup camera calibration for spatial coordinate transformation.
    
    Launch interactive calibration tool to establish correspondence between
    camera view and map coordinates. This needs to be done whenever the
    camera position or angle changes.
    
    CONFIG_PATH: Path to configuration file
    
    \b
    Example:
      navigraph setup calibration config.yaml
    
    \b
    Required config sections:
      map_path: Path to the map image
      calibrator_parameters: Calibration settings
    """
    try:
        click.echo(f"📋 Loading configuration from: {config_path}")
        
        # Load configuration
        config = OmegaConf.load(config_path)
        config._config_dir = str(config_path.parent)
        
        # Get map path from config
        map_path = config.get('map_path')
        if not map_path:
            click.echo("Error: map_path not found in config", err=True)
            sys.exit(1)
        
        # Resolve map path relative to config directory
        if not Path(map_path).is_absolute():
            map_path = Path(config._config_dir) / map_path
        
        click.echo(f"🗺️  Map image: {map_path}")
        
        # Get calibration settings
        calib_params = config.get('calibrator_parameters', {})
        calibration_type = calib_params.get('registration_method', 'homography&ransac')
        
        click.echo(f"🎯 Method: {calibration_type}")
        click.echo(f"🎥 Setting up camera calibration")
        
        # Import calibration modules
        import cv2
        import numpy as np
        import pickle
        
        # Load map image
        map_array = cv2.imread(str(map_path))
        if map_array is None:
            click.echo(f"❌ Failed to load map image: {map_path}", err=True)
            sys.exit(1)
        
        # Camera index from config (if available)
        camera_index = calib_params.get('camera_index', 0)
        
        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            click.echo(f"❌ Failed to open camera {camera_index}", err=True)
            sys.exit(1)
        
        click.echo("📹 Camera opened successfully")
        
        # Calibration points
        num_points = calib_params.get('num_calibration_points', 4)
        
        click.echo(f"🎯 Corner-based calibration with {num_points} points")
        click.echo("📝 Instructions:")
        click.echo("   1. Click corresponding points on camera view and map")
        click.echo("   2. Select easily identifiable landmarks")
        click.echo("   3. Press 'q' when done")
        
        # Collect calibration points
        camera_points = []
        map_points = []
        
        # Simple calibration interface (placeholder)
        click.echo("⚠️  Full calibration GUI not yet fully implemented")
        click.echo("💡 Using placeholder calibration for now")
        
        # Create calibration data
        calibration_data = {
            'type': calibration_type,
            'camera_index': camera_index,
            'camera_points': camera_points,
            'map_points': map_points,
            'transform_matrix': np.eye(3),  # Identity for now
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        # Release camera
        cap.release()
        
        # Get output path from config
        output_dir = calib_params.get('path_to_save_calibration_files', './resources')
        if not Path(output_dir).is_absolute():
            output_dir = Path(config._config_dir) / output_dir
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'calibration.pkl'
        
        # Save calibration
        with open(output_path, 'wb') as f:
            pickle.dump(calibration_data, f)
        click.echo(f"💾 Calibration saved to: {output_path}")
        
        # Also save transform matrix separately if configured
        if calib_params.get('save_transform_matrix', True):
            matrix_path = output_dir / 'transform_matrix.npy'
            np.save(matrix_path, calibration_data['transform_matrix'])
            click.echo(f"💾 Transform matrix saved to: {matrix_path}")
        
        click.echo("✅ Calibration setup complete")
        
    except Exception as e:
        click.echo(f"❌ Calibration failed: {str(e)}", err=True)
        if '--verbose' in sys.argv:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@cli.group()
def test():
    """Testing and validation tools for NaviGraph components.
    
    Interactive tools for testing graph mappings, calibration accuracy,
    and validating system configurations.
    
    \b
    Examples:
      navigraph test graph config.yaml
      navigraph test calibration config.yaml
    """
    pass


@test.command('graph')
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
@click.option('--report', '-r', type=click.Path(path_type=Path),
              help='Export validation report to file')
@click.option('--format', '-f',
              type=click.Choice(['txt', 'html', 'md']),
              default='txt',
              help='Report format for export')
def test_graph(config_path: Path, report: Optional[Path], format: str):
    """Test and validate graph-space mappings interactively.
    
    Launch interactive testing interface for validating spatial mappings.
    Click on the map to see corresponding nodes, click on nodes to see regions.
    
    CONFIG_PATH: Path to configuration file
    
    \b
    Examples:
      navigraph test graph config.yaml
      navigraph test graph config.yaml --report validation.html
    
    \b
    Interactive Features:
      • Click map regions → highlight corresponding nodes
      • Click graph nodes → highlight corresponding regions  
      • Path testing between nodes
      • Overlap and conflict detection
    
    \b
    Required config sections:
      map_path: Path to the map image
      graph_mapping.mapping_file: Path to the saved mapping
    """
    try:
        click.echo(f"📋 Loading configuration from: {config_path}")
        
        # Load configuration
        config = OmegaConf.load(config_path)
        config._config_dir = str(config_path.parent)
        
        # Import modules
        from ..core.graph.storage import MappingStorage
        from ..core.graph.testing import MappingTester
        import cv2
        
        # Get map path from config
        map_path = config.get('map_path')
        if not map_path:
            click.echo("Error: map_path not found in config", err=True)
            sys.exit(1)
        
        # Resolve map path relative to config directory
        if not Path(map_path).is_absolute():
            map_path = Path(config._config_dir) / map_path
        
        # Load map image
        map_array = cv2.imread(str(map_path))
        if map_array is None:
            click.echo(f"❌ Failed to load map image: {map_path}", err=True)
            sys.exit(1)
        click.echo(f"🗺️  Loaded map from: {map_path}")
        
        # Get mapping path from config
        mapping_path = config.get('graph_mapping', {}).get('mapping_file')
        if not mapping_path:
            click.echo("❌ graph_mapping.mapping_file not found in config", err=True)
            click.echo("💡 Run 'navigraph setup graph' first to create a mapping")
            sys.exit(1)
        
        # Resolve mapping path relative to config directory
        if not Path(mapping_path).is_absolute():
            mapping_path = Path(config._config_dir) / mapping_path
        
        # Load mapping
        mapping = MappingStorage.load_mapping(mapping_path)
        if not mapping:
            click.echo(f"❌ Failed to load mapping from: {mapping_path}", err=True)
            sys.exit(1)
        click.echo(f"📂 Loaded mapping from: {mapping_path}")
        
        # Get graph from mapping
        graph = mapping.graph
        if not graph:
            click.echo("❌ No graph found in mapping", err=True)
            sys.exit(1)
        
        click.echo(f"🌳 Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Create and launch tester
        click.echo("🎮 Launching interactive testing interface...")
        click.echo("💡 Tips:")
        click.echo("   • Click on map to identify nodes/edges")
        click.echo("   • Click on graph to highlight regions")
        click.echo("   • Press 'h' for help, 's' for statistics")
        
        tester = MappingTester(graph, mapping, map_array)
        tester.start_interactive_test()
        
        # Export report if requested
        if report:
            success = MappingStorage.export_mapping_report(mapping, report, format)
            if success:
                click.echo(f"📝 Report exported to: {report}")
            else:
                click.echo(f"❌ Failed to export report", err=True)
        
        click.echo("✅ Testing session complete")
        
    except Exception as e:
        click.echo(f"❌ Test failed: {str(e)}", err=True)
        if '--verbose' in sys.argv:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


def main():
    """Main entry point for the NaviGraph CLI."""
    cli()


if __name__ == '__main__':
    main()