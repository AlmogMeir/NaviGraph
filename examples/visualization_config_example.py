#!/usr/bin/env python3
"""Example of using NaviGraph's visualization configuration system.

This example demonstrates how to use pre-defined themes and create
custom visualization configurations.
"""

from navigraph.core import (
    VisualizationConfig, ColorTheme, OutputFormat,
    ColorPalette, FontSettings, PlotSettings,
    create_default_configs
)


def demo_preset_themes():
    """Demonstrate using pre-defined visualization themes."""
    print("=== Pre-defined Visualization Themes ===\n")
    
    # Get all preset configurations
    presets = create_default_configs()
    
    for name, config in presets.items():
        print(f"{name.upper()} preset:")
        print(f"  Theme: {config.theme.value}")
        print(f"  Output formats: {[fmt.value for fmt in config.output_formats]}")
        print(f"  Figure size: {config.plot_settings.figure_size}")
        print(f"  DPI: {config.plot_settings.dpi}")
        print()


def demo_custom_configuration():
    """Demonstrate creating a custom visualization configuration."""
    print("=== Custom Visualization Configuration ===\n")
    
    # Create a custom configuration
    custom_config = VisualizationConfig(
        # Output settings
        output_formats=[OutputFormat.PNG, OutputFormat.PDF],
        output_path="/path/to/visualizations",
        filename_pattern="{session_id}_{visualizer}_{timestamp}",
        
        # Use high contrast theme
        theme=ColorTheme.HIGH_CONTRAST,
        
        # Custom plot settings
        plot_settings=PlotSettings(
            figure_size=(14, 10),
            dpi=150,
            marker_size=80,
            line_width=2.5,
            show_grid=True,
            show_legend=True
        ),
        
        # Custom font settings
        fonts=FontSettings(
            family="Helvetica",
            size=14,
            title_size=20,
            label_size=12,
            weight="bold"
        ),
        
        # Visualizer-specific settings
        trajectory_settings={
            "trail_length": 150,
            "show_confidence": True,
            "confidence_threshold": 0.95,
            "interpolate": True
        },
        
        map_settings={
            "show_tile_ids": True,
            "show_grid": True,
            "highlight_current": True,
            "overlay_alpha": 0.8
        }
    )
    
    print("Custom configuration created:")
    print(f"  Output formats: {[fmt.value for fmt in custom_config.output_formats]}")
    print(f"  Theme: {custom_config.theme.value}")
    print(f"  Background color: {custom_config.colors.background}")
    print(f"  Trajectory trail length: {custom_config.trajectory_settings['trail_length']}")
    print()


def demo_theme_colors():
    """Demonstrate color palettes for different themes."""
    print("=== Theme Color Palettes ===\n")
    
    themes = [
        ColorTheme.DEFAULT,
        ColorTheme.DARK,
        ColorTheme.HIGH_CONTRAST,
        ColorTheme.COLORBLIND_SAFE,
        ColorTheme.PUBLICATION
    ]
    
    for theme in themes:
        palette = ColorPalette.from_theme(theme)
        print(f"{theme.value} theme colors:")
        print(f"  Background: RGB{palette.background}")
        print(f"  Foreground: RGB{palette.foreground}")
        print(f"  Primary: RGB{palette.primary}")
        print(f"  Trajectory colors: {len(palette.trajectory)} colors")
        print()


def demo_save_load_configuration():
    """Demonstrate saving and loading configurations."""
    print("=== Save/Load Configuration ===\n")
    
    # Create a configuration
    config = VisualizationConfig(
        theme=ColorTheme.PUBLICATION,
        output_formats=[OutputFormat.PDF, OutputFormat.SVG],
        plot_settings=PlotSettings(dpi=300, figure_size=(8, 6))
    )
    
    # Convert to dictionary
    config_dict = config.to_dict()
    print("Configuration as dictionary:")
    print(f"  Keys: {list(config_dict.keys())}")
    print()
    
    # Create from dictionary
    restored_config = VisualizationConfig.from_dict(config_dict)
    print("Restored configuration:")
    print(f"  Theme: {restored_config.theme.value}")
    print(f"  Output formats: {[fmt.value for fmt in restored_config.output_formats]}")
    print(f"  DPI: {restored_config.plot_settings.dpi}")


def demo_visualizer_specific_config():
    """Demonstrate getting configuration for specific visualizers."""
    print("\n=== Visualizer-Specific Configuration ===\n")
    
    configs = create_default_configs()
    config = configs["presentation"]
    
    visualizers = [
        "trajectory_visualizer",
        "map_visualizer",
        "tree_visualizer",
        "metrics_visualizer"
    ]
    
    for viz_name in visualizers:
        viz_config = config.get_visualizer_config(viz_name)
        print(f"{viz_name} configuration:")
        print(f"  Number of settings: {len(viz_config)}")
        print(f"  Has colors: {'colors' in viz_config}")
        print(f"  Has fonts: {'fonts' in viz_config}")
        print(f"  Output formats: {[fmt.value for fmt in viz_config.get('output_formats', [])]}")
        print()


def main():
    """Run all demonstration functions."""
    print("NaviGraph Visualization Configuration Examples")
    print("=" * 50)
    print()
    
    demo_preset_themes()
    demo_custom_configuration()
    demo_theme_colors()
    demo_save_load_configuration()
    demo_visualizer_specific_config()
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()