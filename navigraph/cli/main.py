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
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import click
from omegaconf import OmegaConf

# Import experiment runner and enums
from ..core.experiment_runner import ExperimentRunner
from ..core.enums import SystemMode


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
    
    # Process experiment_path - should be relative to config file location
    if 'experiment_path' in config:
        exp_path = config['experiment_path']
        if isinstance(exp_path, str):
            exp_path = Path(exp_path)
            if not exp_path.is_absolute():
                exp_path = config_dir / exp_path
            config['experiment_path'] = str(exp_path.resolve())
    
    return config


@click.group()
@click.version_option(version='0.2.0', prog_name='NaviGraph')
def cli():
    """NaviGraph - A flexible framework for spatial navigation analysis.
    
    NaviGraph provides tools for analyzing animal navigation behavior through
    integration of pose estimation, spatial mapping, and graph-based analysis.
    
    \b
    Common Commands:
      navigraph run config.yaml              - Run analysis and visualization
      navigraph run analyze config.yaml      - Run analysis only  
      navigraph run visualize config.yaml    - Run visualization only
      navigraph setup graph config.yaml      - Setup graph mapping
      navigraph setup calibration config.yaml - Setup camera calibration
      navigraph validate config.yaml         - Validate configuration
    
    Use 'navigraph COMMAND --help' for more information on each command.
    """
    pass


@cli.command()
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--show', is_flag=True, help='Show visualization results after creation')
@click.option('--analyze', is_flag=True, help='Run analysis only')
@click.option('--visualize', is_flag=True, help='Run visualization only')
def run(config_path: Path, verbose: bool, show: bool, analyze: bool, visualize: bool):
    """Run NaviGraph experiments with different execution modes.
    
    CONFIG_PATH: Path to YAML configuration file
    
    \b
    Examples:
      navigraph run config.yaml              - Run both analysis and visualization
      navigraph run config.yaml --show       - Run and show visualization results
      navigraph run config.yaml --analyze    - Run analysis only
      navigraph run config.yaml --visualize  - Run visualization only
      navigraph run config.yaml --analyze --visualize  - Run both explicitly
    """
    # Determine modes based on flags
    if analyze and visualize:
        modes = [SystemMode.ANALYZE, SystemMode.VISUALIZE]
    elif analyze:
        modes = [SystemMode.ANALYZE]
    elif visualize:
        modes = [SystemMode.VISUALIZE]
    else:
        # Default: run both analysis and visualization
        modes = [SystemMode.ANALYZE, SystemMode.VISUALIZE]
    
    _run_experiment_with_modes(config_path, verbose, modes, show)


def _run_experiment_with_modes(config_path: Path, verbose: bool, modes: List[SystemMode], show: bool = False):
    """Helper function to run experiment with specified modes."""
    try:
        modes_str = ', '.join(mode.value for mode in modes)
        click.echo(f"🚀 Starting NaviGraph experiment ({modes_str})")
        click.echo(f"📋 Configuration: {config_path}")
        
        # Load and process configuration
        config = OmegaConf.load(config_path)
        config = process_config_path(config_path, OmegaConf.to_container(config))
        
        # Override verbose if specified
        if verbose:
            config['verbose'] = True
        
        # Add show flag to config for visualization system
        if show:
            config['show_visualization'] = True
            
        # Create and run experiment with specified modes
        runner = ExperimentRunner(config, system_modes=modes)
        
        results = runner.run_experiment()
        
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
    from .. import plugins  # noqa: F401
    
    try:
        click.echo(f"🔍 Discovering sessions in: {data_path}")
        
        if config:
            # Use provided config for plugin-based validation
            click.echo(f"📋 Using configuration: {config}")
            
            # Load configuration
            config_data = OmegaConf.load(config)
            config_data = process_config_path(config, OmegaConf.to_container(config_data))
            
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


@cli.command('list-graph-builders')
def list_graph_builders_cmd():
    """List all available graph builders.
    
    Display registered graph builders that can be used in configuration files.
    
    \b
    Example:
      navigraph list-graph-builders
    """
    try:
        from ..core.graph.builders import list_graph_builders, get_graph_builder_info
        
        builders = list_graph_builders()
        
        if not builders:
            click.echo("No graph builders registered")
            return
        
        click.echo("📊 Available Graph Builders")
        click.echo("-" * 40)
        
        for builder_name in sorted(builders):
            info = get_graph_builder_info(builder_name)
            click.echo(f"\n• {builder_name}")
            click.echo(f"  Class: {info['class_name']}")
            
            if info['docstring']:
                # Get first line of docstring
                doc_lines = info['docstring'].strip().split('\n')
                if doc_lines:
                    click.echo(f"  Description: {doc_lines[0]}")
            
            if info['parameters']:
                click.echo("  Parameters:")
                for param_name, param_info in info['parameters'].items():
                    required = "required" if param_info['required'] else "optional"
                    default = f", default={param_info['default']}" if param_info['default'] is not None else ""
                    click.echo(f"    - {param_name} ({required}{default})")
        
        click.echo(f"\nTotal: {len(builders)} builders")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command('list-plugins')
@click.option('--category', '-c',
              type=click.Choice(['plugins', 'session_metrics', 'cross_session_metrics', 'visualizers', 'all']),
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
      plugins           - All unified NaviGraph plugins (pose tracking, calibration, etc.)
      session_metrics   - Session-level analysis functions
      cross_session_metrics - Cross-session analysis functions
      visualizers       - Visualization functions
    """
    try:
        click.echo("🔌 NaviGraph Plugin Registry")
        
        # Load plugins to populate the registry
        from .. import plugins
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
                'plugins': '📊 Plugins - Unified NaviGraph plugins for data integration',
                'session_metrics': '🧮 Session Metrics - Session-level analysis functions',
                'cross_session_metrics': '📈 Cross-Session Metrics - Cross-session analysis functions',
                'visualizers': '🎨 Visualizers - Visualization functions'
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
        from .. import plugins
        
        # Load configuration
        config = OmegaConf.load(config_path)
        config = process_config_path(config_path, OmegaConf.to_container(config))
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
      setup.map_path: Path to the map image
      graph.builder: Defines graph type (binary_tree or custom)
      graph.mapping_file: Where to save the mapping
    """
    try:
        click.echo(f"Loading configuration: {config_path}")
        
        # Load configuration
        config = OmegaConf.load(config_path)
        config = process_config_path(config_path, OmegaConf.to_container(config))
        
        # Import graph modules
        from ..core.graph.structures import GraphStructure
        from ..core.graph.builders import get_graph_builder, list_graph_builders
        from ..core.graph.setup_gui_qt import launch_setup_gui
        from ..core.graph.storage import MappingStorage
        import numpy as np
        import cv2
        
        # Get map path from setup section
        setup_config = config.get('setup', {})
        map_path = setup_config.get('map_path')
        if not map_path:
            click.echo("Error: map_path not found in setup config section", err=True)
            click.echo("Please add 'map_path: path/to/map.png' under the 'setup:' section", err=True)
            sys.exit(1)
        
        # Resolve map path relative to config directory
        if not Path(map_path).is_absolute():
            map_path = Path(config['_config_dir']) / map_path
        
        # Load map image
        map_array = cv2.imread(str(map_path))
        if map_array is None:
            click.echo(f"Error: Failed to load map image: {map_path}", err=True)
            sys.exit(1)
        
        # Create graph from config using new builder system
        # Support both old 'graph_structure' and new 'graph.builder' formats
        graph_config = config.get('graph_structure', {})
        if not graph_config:
            # Try new format: graph.builder
            graph_section = config.get('graph', {})
            builder_config = graph_section.get('builder', {})
            if builder_config:
                graph_config = {
                    'type': builder_config.get('type'),
                    'parameters': builder_config.get('config', {})
                }
        
        graph_type = graph_config.get('type')
        if not graph_type:
            click.echo("Error: No graph builder type specified in configuration", err=True)
            click.echo("Add 'graph.builder.type' to your config file", err=True)
            click.echo(f"Available builders: {', '.join(list_graph_builders())}", err=True)
            sys.exit(1)
        
        # Get parameters for the builder
        params = graph_config.get('parameters', {})
        
        try:
            # Get builder class and create instance
            builder_class = get_graph_builder(graph_type)
            builder = builder_class(**params)
            
            # Create graph structure
            graph = GraphStructure(builder)
            
            # Get builder metadata for display
            metadata = builder.get_metadata()
            param_str = ', '.join(f"{k}={v}" for k, v in metadata['parameters'].items())
            
            # Display graph information
            click.echo(f"📊 Graph Builder: {graph_type}")
            click.echo(f"   Parameters: {param_str}")
            click.echo(f"   Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
            
        except KeyError as e:
            click.echo(f"Error: Unknown graph builder type '{graph_type}'", err=True)
            click.echo(f"Available builders: {', '.join(list_graph_builders())}", err=True)
            sys.exit(1)
        except TypeError as e:
            click.echo(f"Error: Invalid parameters for {graph_type} builder: {str(e)}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error: Failed to create graph: {str(e)}", err=True)
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
@click.option("--test", is_flag=True, help="Test existing calibration instead of creating new one")
def setup_calibration(config_path: Path, test: bool):
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
        config = process_config_path(config_path, OmegaConf.to_container(config))
        
        # Get map path from config (check setup section first, then root for backward compatibility)
        setup_config = config.get('setup', {})
        map_path = setup_config.get('map_path') or config.get('map_path')
        if not map_path:
            click.echo("Error: map_path not found in config. Add it to the setup section or root level.", err=True)
            sys.exit(1)
        
        # Resolve map path relative to config directory
        if not Path(map_path).is_absolute():
            map_path = Path(config['_config_dir']) / map_path
        
        click.echo(f"🗺️  Map image: {map_path}")
        
        # Get spatial image for calibration (from setup section)
        setup_config = config.get('setup', {})
        spatial_image_path = setup_config.get('spatial_image_for_calibration')
        
        if not spatial_image_path:
            click.echo("Error: spatial_image_for_calibration not found in setup section.", err=True)
            click.echo("Add it to config: setup.spatial_image_for_calibration: /path/to/video/or/image", err=True)
            sys.exit(1)
        
        # Resolve spatial image path relative to config directory
        if not Path(spatial_image_path).is_absolute():
            spatial_image_path = Path(config['_config_dir']) / spatial_image_path
        
        click.echo(f"📹 Spatial source: {spatial_image_path}")
        
        # Get calibration settings
        calib_params = config.get('calibrator_parameters', {})
        method = calib_params.get('registration_method', 'homography_ransac')
        
        # Convert legacy method name
        if method == 'homography&ransac':
            method = 'homography_ransac'
        
        min_points = calib_params.get('num_calibration_points', 4)
        
        if test:
            # Test mode - validate existing calibration
            click.echo("🧪 Testing existing calibration matrix")
            
            # Get calibration matrix path from config
            setup_config = config.get('setup', {})
            calibration_matrix_path = setup_config.get('calibration_matrix')
            
            if not calibration_matrix_path:
                # Default to resources directory
                calibration_matrix_path = './resources/transform_matrix.npy'
            
            # Resolve relative to config directory
            if not Path(calibration_matrix_path).is_absolute():
                calibration_matrix_path = Path(config['_config_dir']) / calibration_matrix_path
            
            # Check if calibration matrix exists
            if not Path(calibration_matrix_path).exists():
                click.echo(f"❌ Calibration matrix not found: {calibration_matrix_path}", err=True)
                click.echo("💡 Run calibration without --test to create one first", err=True)
                sys.exit(1)
            
            click.echo(f"📊 Testing calibration: {calibration_matrix_path}")
            
            # Import and run calibration tester
            from ..core.calibration import CalibrationTester
            
            tester = CalibrationTester()
            tester.test_calibration(
                spatial_image_path=spatial_image_path,
                map_image_path=map_path,
                calibration_matrix_path=calibration_matrix_path
            )
            
            click.echo("✅ Calibration test completed")
            
        else:
            # Create mode - interactive calibration
            click.echo(f"🎯 Method: {method}")
            click.echo(f"🎯 Minimum points: {min_points}")
            
            # Import and run interactive calibration
            from ..core.calibration import InteractiveCalibrator
            
            calibrator = InteractiveCalibrator()
            
            # Run calibration
            calibration_result = calibrator.calibrate_camera_to_map(
                camera_source=spatial_image_path,
                map_image_path=map_path,
                method=method,
                min_points=min_points,
                show_preview=True
            )
            
            # Determine output directory (save to resources by default)
            output_dir = Path(config['_config_dir']) / 'resources'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save transformation matrix
            matrix_path = output_dir / 'transform_matrix.npy'
            calibration_result.save(matrix_path)
            
            click.echo("✅ Interactive calibration completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Calibration failed: {str(e)}", err=True)
        if '--verbose' in sys.argv:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)




@cli.command('list-conflict-resolvers')
@click.option('--format', '-f', 
              type=click.Choice(['table', 'json', 'simple']),
              default='table',
              help='Output format')
def list_conflict_resolvers(format: str):
    """List all available conflict resolution strategies.
    
    Display registered conflict resolvers that can be used in spatial mapping
    when pixels fall within multiple regions.
    
    \b
    Examples:
      navigraph list-conflict-resolvers
      navigraph list-conflict-resolvers --format json
    """
    try:
        from ..core.graph.conflict_resolvers import ConflictResolvers
        
        strategies = ConflictResolvers._strategies
        
        if not strategies:
            click.echo("⚠️  No conflict resolvers found")
            return
        
        if format == 'simple':
            for name in strategies.keys():
                click.echo(name)
        
        elif format == 'json':
            output = {
                'available_resolvers': [
                    {
                        'name': name,
                        'description': func.__doc__.strip() if func.__doc__ else ''
                    }
                    for name, func in strategies.items()
                ],
                'total_count': len(strategies),
                'default': 'node_priority'
            }
            click.echo(json.dumps(output, indent=2))
        
        else:  # table format
            click.echo()
            click.echo("🔀 Available Conflict Resolution Strategies")
            click.echo("=" * 55)
            
            for name, func in strategies.items():
                click.echo(f"\n🔹 {name}")
                if func.__doc__:
                    doc = func.__doc__.strip().split('\n')[0]
                    click.echo(f"   {doc}")
            
            click.echo()
            click.echo(f"Total: {len(strategies)} strategies available")
            click.echo(f"Default: node_priority")
            click.echo()
            click.echo("Usage in config:")
            click.echo("  shared_resources:")
            click.echo("    - name: graph_provider")
            click.echo("      config:")
            click.echo("        conflict_strategy: <resolver_name>")
            click.echo()
            click.echo("CLI override:")
            click.echo("  navigraph run config.yaml --conflict-strategy <resolver_name>")
            
    except Exception as e:
        click.echo(f"❌ Failed to list conflict resolvers: {str(e)}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the NaviGraph CLI."""
    cli()


if __name__ == '__main__':
    main()