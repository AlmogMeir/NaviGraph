"""CLI for NaviGraph behavioral analysis framework."""

import click
import sys
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf, DictConfig
from loguru import logger

from ..core.experiment_runner import ExperimentRunner


@click.group()
@click.version_option(version="0.1.0", prog_name="navigraph")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool):
    """NaviGraph: Graph-based behavioral analysis framework.
    
    Comprehensive infrastructure to empower researcher workflows in 
    multimodal scenarios - digesting inputs, analysis, visualizations 
    and more.
    
    \b
    Quick Start:
      navigraph run examples/basic_maze/config.yaml --mode visualize
      navigraph run examples/basic_maze/config.yaml --mode analyze
      navigraph discover ./data/
      navigraph list-plugins
    
    \b
    Installation:
      poetry install    # Install dependencies
      poetry shell      # Activate environment
      navigraph --help  # You're here!
    """
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")


@cli.command()
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
@click.option('--mode', '-m', 
              type=click.Choice(['analyze', 'calibrate', 'test', 'visualize']),
              multiple=True,
              help='Experiment mode(s) to run')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Override output directory from config')
def run(config_path: Path, mode: tuple, output: Optional[Path]):
    """Run NaviGraph experiment with the specified configuration.
    
    Process video data through the complete analysis pipeline including
    calibration, spatial mapping, graph analysis, and visualization.
    
    CONFIG_PATH: Path to YAML configuration file
    
    \b
    Examples:
      navigraph run examples/basic_maze/config.yaml --mode visualize
      navigraph run examples/basic_maze/config.yaml --mode analyze  
      navigraph run config.yaml --mode calibrate --mode analyze
      navigraph run config.yaml --output ./results/
    
    \b
    Available modes:
      calibrate   - Calibrate camera-to-maze coordinate transformation
      test        - Test calibration quality and accuracy
      analyze     - Analyze behavioral metrics and navigation patterns  
      visualize   - Generate video visualizations with keypoints/maps
    
    Modes can be combined: --mode calibrate --mode analyze
    """
    # Load plugins only when actually running experiments
    from ..plugins import data_sources, shared_resources, analyzers
    
    try:
        click.echo(f"🚀 Starting NaviGraph experiment with config: {config_path}")
        
        # Load configuration (preserving Hydra compatibility)
        config = OmegaConf.load(config_path)
        
        # Add config directory for relative path resolution
        config._config_dir = str(config_path.parent)
        
        # Override mode if specified
        if mode:
            mode_string = '&'.join(mode)  # Join with & as in original system
            config.system_running_mode = mode_string
            click.echo(f"📋 Mode override: {mode_string}")
        
        # Override output path if specified
        if output:
            config.experiment_output_path = str(output)
            click.echo(f"📁 Output override: {output}")
        
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
@click.option('--format', '-f', 
              type=click.Choice(['table', 'json', 'tree']),
              default='table',
              help='Output format for discovered sessions')
def discover(data_path: Path, format: str):
    """Discover and list available experimental sessions.
    
    Scan a directory for video files (.mp4, .avi) and matching DeepLabCut
    H5 files to identify sessions ready for analysis.
    
    DATA_PATH: Path to directory containing experimental data
    
    \b
    Examples:
      navigraph discover ./data/
      navigraph discover /path/to/experiments/ --format json
      navigraph discover ./sessions/ --format tree
    
    \b
    Expected file structure:
      • Video files: *.mp4, *.avi, *.mov
      • DeepLabCut files: *DLC*.h5
      • Matching names: video basename should appear in H5 filename
    """
    # Load plugins for session discovery
    from ..plugins import data_sources, shared_resources, analyzers
    
    try:
        click.echo(f"🔍 Discovering sessions in: {data_path}")
        
        # Create minimal config for discovery
        config = OmegaConf.create({
            'stream_path': str(data_path),
            'keypoint_detection_file_path': str(data_path),
            'experiment_output_path': '/tmp/navigraph_discovery'
        })
        
        # Use experiment runner for discovery
        runner = ExperimentRunner(config)
        sessions = runner.discover_sessions()
        
        if not sessions:
            click.echo("⚠️  No sessions found in the specified directory")
            click.echo("💡 Make sure the directory contains:")
            click.echo("   • Video files (.mp4, .avi)")  
            click.echo("   • DeepLabCut files (.h5)")
            click.echo("   • Matching filenames (video basename in H5 filename)")
            return
        
        click.echo(f"✅ Found {len(sessions)} sessions:")
        
        if format == 'table':
            # Table format
            click.echo()
            click.echo("Session Name".ljust(30) + "Video File".ljust(40) + "H5 File")
            click.echo("-" * 90)
            for session in sessions:
                name = session['session_name'][:28]
                video = Path(session['video_file']).name[:38]
                h5 = Path(session['h5_file']).name
                click.echo(f"{name:<30} {video:<40} {h5}")
        
        elif format == 'json':
            # JSON format
            import json
            click.echo(json.dumps(sessions, indent=2))
        
        elif format == 'tree':
            # Tree format
            click.echo()
            for i, session in enumerate(sessions, 1):
                prefix = "├── " if i < len(sessions) else "└── "
                click.echo(f"{prefix}{session['session_name']}")
                click.echo(f"    ├── 🎥 {Path(session['video_file']).name}")
                click.echo(f"    └── 📊 {Path(session['h5_file']).name}")
            
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


if __name__ == '__main__':
    cli()