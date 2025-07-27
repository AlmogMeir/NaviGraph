# NaviGraph

**NaviGraph** (_Navigation on the Graph_) is a flexible, extensible Python framework for behavioral analysis that supports diverse data sources, custom graph structures, and extensible metrics. Originally designed for maze paradigms, it has been redesigned as a generic behavioral analysis platform.

## 🚀 Key Features

- **🔌 Plugin-Based Architecture**: Extensible system supporting custom data sources, analyzers, and visualizers
- **📊 Multiple Data Sources**: Built-in support for DeepLabCut, camera tracking, EEG, calcium imaging (Miniscope)
- **🎨 Rich Visualization System**: Theme-based visualizations with publication-ready presets
- **📈 Structured Analysis Results**: Comprehensive metadata tracking and cross-session statistics
- **⚙️ Configuration-Driven**: YAML-based configuration with factory patterns for reproducibility
- **🎯 Type-Safe**: Comprehensive type hints and validation throughout

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/NaviGraph.git
cd NaviGraph

# Install dependencies
pip install -r requirements.txt

# Or install with Poetry (recommended)
poetry install
```

## 🏃 Quick Start

### Basic Usage

```python
from navigraph.core import ExperimentRunner, VisualizationConfig, ColorTheme

# Configure your experiment
config = {
    'data_sources': [
        {
            'name': 'tracking_data',
            'type': 'deeplabcut',
            'file_pattern': r'.*DLC.*\.h5$',
            'config': {'bodypart': 'nose', 'likelihood_threshold': 0.9}
        }
    ],
    'analyzers': [
        {'name': 'spatial_metrics', 'type': 'spatial_metrics'},
        {'name': 'exploration', 'type': 'exploration_metrics'}
    ]
}

# Run analysis pipeline
runner = ExperimentRunner(experiment_config=config, sessions=sessions)
results = runner.run_analysis()

# Create visualizations with publication theme
viz_config = VisualizationConfig(theme=ColorTheme.PUBLICATION)
pipeline = VisualizationPipeline(viz_config)
```

### Run Complete Pipeline

```bash
# Run with default configuration
python run.py

# Run with custom configuration
python run.py --config-path=configs --config-name=custom_experiment
```

## 📋 System Requirements

- **Python**: 3.8+
- **Core Dependencies**: pandas, numpy, scipy, matplotlib, loguru
- **Optional**: OpenCV (video processing), networkx (graph analysis)

## 🏗️ Architecture Overview

NaviGraph follows a modular plugin-based architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│   Session Core   │───▶│    Analyzers    │
│                 │    │                  │    │                 │
│ • DeepLabCut    │    │ • Data Integration│    │ • Spatial       │
│ • Camera Track  │    │ • Graph Mapping   │    │ • Navigation    │
│ • EEG/Neural    │    │ • Shared Resources│    │ • Exploration   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  Visualizers    │
                       │                 │
                       │ • Trajectories  │
                       │ • Heatmaps      │
                       │ • Tree Graphs   │
                       │ • Metrics Plots │
                       └─────────────────┘
```

### Core Components

- **Data Sources**: Plugin-based data integration (DeepLabCut, custom trackers, neural data)
- **Shared Resources**: Cross-session resources (maps, calibrations, graph structures)
- **Analyzers**: Behavioral metrics computation with structured results
- **Visualizers**: Publication-ready visualizations with theme support
- **Session Management**: Unified data integration and processing pipeline

## 🔧 Configuration

NaviGraph uses YAML configuration files with Hydra for flexible experiment setup:

```yaml
# configs/experiment.yaml
data_sources:
  - name: deeplabcut
    type: deeplabcut
    file_pattern: ".*DLC.*\\.h5$"
    config:
      bodypart: nose
      likelihood_threshold: 0.9

shared_resources:
  - name: maze_map
    type: map_provider
    config:
      map_path: "shared_resources/maze_config.json"

analyzers:
  - name: spatial_metrics
    type: spatial_metrics
  - name: exploration_metrics  
    type: exploration_metrics

visualization:
  theme: publication
  output_formats: [png, pdf]
  output_path: "results/visualizations"
```

## 🎨 Visualization Themes

NaviGraph includes pre-defined visualization themes:

```python
from navigraph.core import VisualizationConfig, ColorTheme, create_default_configs

# Use built-in themes
presets = create_default_configs()
publication_config = presets["publication"]  # High-DPI, clean styling
presentation_config = presets["presentation"]  # High contrast, large fonts
dark_config = presets["dark"]  # Dark theme
colorblind_config = presets["colorblind_safe"]  # Accessible colors

# Create custom theme
custom_config = VisualizationConfig(
    theme=ColorTheme.DARK,
    output_formats=[OutputFormat.PNG, OutputFormat.SVG],
    plot_settings=PlotSettings(figure_size=(12, 8), dpi=300)
)
```

## 🧩 Plugin Development

Create custom plugins for your specific needs:

### Custom Data Source

```python
from navigraph.core import IDataSource, BasePlugin, register_data_source_plugin

@register_data_source_plugin("my_tracker")
class MyTrackerDataSource(BasePlugin, IDataSource):
    @classmethod
    def from_config(cls, config, logger_instance=None):
        return cls(config, logger_instance)
    
    def integrate_data_into_session(self, current_dataframe, config, shared_resources, logger):
        # Your data integration logic
        return enhanced_dataframe
```

### Custom Analyzer

```python
from navigraph.core import IAnalyzer, BasePlugin, AnalysisResult, register_analyzer_plugin

@register_analyzer_plugin("my_metrics")
class MyMetricsAnalyzer(BasePlugin, IAnalyzer):
    def analyze_session(self, session) -> AnalysisResult:
        # Compute your custom metrics
        metrics = {"my_metric": computed_value}
        return AnalysisResult(
            session_id=session.session_id,
            analyzer_name="my_metrics",
            metrics=metrics,
            metadata=analysis_metadata
        )
```

See `examples/plugin_development_guide.py` for comprehensive examples.

## 📊 Analysis Modes

NaviGraph supports multiple execution modes:

- **`calibrate`**: Calibrate camera-to-maze coordinate transformations
- **`test`**: Validate calibration quality and data integrity
- **`analyze`**: Compute behavioral metrics and generate analysis results
- **`visualize`**: Create visual outputs and publication figures
- **Combined modes**: e.g., `calibrate&analyze` for end-to-end processing

```bash
# Set mode in configuration
python run.py system_running_mode=analyze

# Or combine modes
python run.py system_running_mode=calibrate,analyze,visualize
```

## 📁 Project Structure

```
NaviGraph/
├── navigraph/           # Core framework
│   ├── core/           # Core interfaces and classes
│   │   ├── interfaces.py      # Plugin interfaces
│   │   ├── registry.py        # Plugin registry
│   │   ├── session.py         # Session management
│   │   ├── types.py           # Type definitions
│   │   └── visualization_config.py  # Visualization system
│   └── plugins/        # Built-in plugins
│       ├── data_sources/      # Data integration plugins
│       ├── analyzers/         # Analysis plugins
│       ├── visualizers/       # Visualization plugins
│       └── shared_resources/  # Shared resource plugins
├── configs/            # Configuration files
├── examples/           # Usage examples and tutorials
├── tests/              # Test suite
└── scripts/            # Utility scripts
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run integration tests
pytest tests/test_integration.py -v

# Run specific test
pytest tests/test_plugin_system.py::TestPluginRegistration -v
```

## 📚 Examples

- **`examples/complete_pipeline_example.py`**: End-to-end pipeline demonstration
- **`examples/plugin_development_guide.py`**: Custom plugin development
- **`examples/visualization_config_example.py`**: Visualization customization
- **`examples/`**: Additional usage examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the plugin architecture
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow the plugin interfaces for extensibility
- Include comprehensive type hints
- Write tests for new plugins and features
- Use the established logging patterns
- Document configuration requirements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original maze navigation analysis framework
- DeepLabCut integration patterns
- Scientific Python community for foundational libraries
- Contributors and researchers using NaviGraph

## 📞 Support

- **Documentation**: See `examples/` directory for comprehensive guides
- **Issues**: [GitHub Issues](https://github.com/your-username/NaviGraph/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/NaviGraph/discussions)

---

**NaviGraph**: From specialized maze analysis to flexible behavioral research platform 🧭📊