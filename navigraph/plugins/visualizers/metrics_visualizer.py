"""Metrics visualizer plugin for NaviGraph.

This plugin creates static plots and visualizations from analysis results,
including learning curves, statistics plots, and comparative visualizations.
"""

from typing import Dict, Any, Optional, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from ...core.interfaces import IVisualizer, Logger
from ...core.base_plugin import BasePlugin
from ...core.registry import register_visualizer_plugin


@register_visualizer_plugin("metrics_visualizer")
class MetricsVisualizer(BasePlugin, IVisualizer):
    """Visualizes analysis metrics as static plots.
    
    Features:
    - Learning curves over sessions
    - Distribution plots for metrics
    - Comparative bar charts
    - Correlation heatmaps
    - Customizable plot styling
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], logger_instance = None):
        """Factory method to create metrics visualizer from configuration."""
        instance = cls(config, logger_instance)
        instance.initialize()
        return instance
    
    def _validate_config(self) -> None:
        """Validate metrics visualizer configuration."""
        # All config keys are optional with sensible defaults
        pass
    
    def visualize(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        shared_resources: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> Optional[str]:
        """Create static plots from analysis metrics.
        
        Args:
            data: DataFrame with analysis results (metrics as columns, sessions as rows)
            config: Visualization-specific configuration
            shared_resources: Not used for metrics visualization
            output_path: Directory to save visualization outputs
            **kwargs: Additional parameters including:
                - plot_type: Type of plot to create
                - session_groups: Dictionary mapping session IDs to group labels
                
        Returns:
            Path to created visualization file, or None if failed
        """
        try:
            # Get visualization settings with defaults
            viz_config = {
                'plot_type': kwargs.get('plot_type', config.get('plot_type', 'learning_curve')),
                'figure_size': config.get('figure_size', (10, 6)),
                'dpi': config.get('dpi', 300),
                'style': config.get('style', 'whitegrid'),
                'color_palette': config.get('color_palette', 'Set2'),
                'title': config.get('title', 'Analysis Results'),
                'xlabel': config.get('xlabel', None),
                'ylabel': config.get('ylabel', None),
                'legend': config.get('legend', True),
                'save_formats': config.get('save_formats', ['png', 'pdf'])
            }
            
            # Set plot style
            sns.set_style(viz_config['style'])
            plt.rcParams['figure.figsize'] = viz_config['figure_size']
            plt.rcParams['figure.dpi'] = viz_config['dpi']
            
            # Create appropriate plot based on type
            plot_functions = {
                'learning_curve': self._create_learning_curve,
                'distribution': self._create_distribution_plot,
                'comparison': self._create_comparison_plot,
                'correlation': self._create_correlation_heatmap,
                'session_timeline': self._create_session_timeline,
                'metric_summary': self._create_metric_summary
            }
            
            plot_func = plot_functions.get(viz_config['plot_type'])
            if not plot_func:
                self.logger.error(f"Unknown plot type: {viz_config['plot_type']}")
                return None
            
            # Create plot
            fig = plot_func(data, viz_config, kwargs)
            
            # Save plot
            output_files = []
            base_filename = f"{viz_config['plot_type']}_{viz_config['title'].replace(' ', '_').lower()}"
            
            for fmt in viz_config['save_formats']:
                output_file = Path(output_path) / f"{base_filename}.{fmt}"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                fig.savefig(
                    output_file,
                    format=fmt,
                    dpi=viz_config['dpi'],
                    bbox_inches='tight'
                )
                output_files.append(str(output_file))
                self.logger.info(f"Saved {viz_config['plot_type']} plot to: {output_file}")
            
            plt.close(fig)
            
            # Return first saved file path
            return output_files[0] if output_files else None
            
        except Exception as e:
            self.logger.error(f"Metrics visualization failed: {str(e)}")
            return None
    
    def _create_learning_curve(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> plt.Figure:
        """Create learning curve plot showing metric evolution over sessions."""
        fig, ax = plt.subplots()
        
        # Get metric columns (exclude non-numeric)
        metric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Plot each metric
        colors = sns.color_palette(config['color_palette'], len(metric_columns))
        
        for idx, metric in enumerate(metric_columns):
            # Extract values and handle lists/arrays in cells
            values = []
            sessions = []
            
            for session_idx, value in data[metric].items():
                if isinstance(value, (list, np.ndarray)):
                    # If value is array, take mean
                    values.append(np.mean(value))
                elif pd.notna(value):
                    values.append(value)
                else:
                    continue
                sessions.append(session_idx)
            
            if values:
                ax.plot(
                    sessions, values,
                    marker='o',
                    label=metric.replace('_', ' ').title(),
                    color=colors[idx % len(colors)],
                    linewidth=2,
                    markersize=8
                )
        
        # Customize plot
        ax.set_title(config['title'], fontsize=16, fontweight='bold')
        ax.set_xlabel(config['xlabel'] or 'Session', fontsize=12)
        ax.set_ylabel(config['ylabel'] or 'Metric Value', fontsize=12)
        
        if config['legend']:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def _create_distribution_plot(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> plt.Figure:
        """Create distribution plots for metrics."""
        # Get metric to plot
        metric_name = kwargs.get('metric_name')
        if not metric_name or metric_name not in data.columns:
            # Use first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found for distribution plot")
            metric_name = numeric_cols[0]
        
        # Extract values
        values = []
        for val in data[metric_name]:
            if isinstance(val, (list, np.ndarray)):
                values.extend(val)
            elif pd.notna(val):
                values.append(val)
        
        # Create subplots for histogram and box plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config['figure_size'])
        
        # Histogram
        ax1.hist(
            values, 
            bins=30, 
            color=sns.color_palette(config['color_palette'])[0],
            alpha=0.7,
            edgecolor='black'
        )
        ax1.set_title(f"Distribution of {metric_name.replace('_', ' ').title()}", fontsize=14)
        ax1.set_xlabel('Value', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Box plot with session groups if provided
        session_groups = kwargs.get('session_groups', {})
        if session_groups:
            # Group data by session groups
            grouped_data = []
            group_labels = []
            
            for group_name in set(session_groups.values()):
                group_sessions = [
                    s for s, g in session_groups.items() if g == group_name
                ]
                group_values = []
                for session in group_sessions:
                    if session in data.index:
                        val = data.loc[session, metric_name]
                        if isinstance(val, (list, np.ndarray)):
                            group_values.extend(val)
                        elif pd.notna(val):
                            group_values.append(val)
                
                if group_values:
                    grouped_data.append(group_values)
                    group_labels.append(group_name)
            
            if grouped_data:
                ax2.boxplot(grouped_data, labels=group_labels)
                ax2.set_xticklabels(group_labels, rotation=45)
        else:
            ax2.boxplot([values])
        
        ax2.set_ylabel('Value', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(config['title'], fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _create_comparison_plot(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> plt.Figure:
        """Create comparison bar plots for metrics across groups."""
        fig, ax = plt.subplots()
        
        # Get session groups
        session_groups = kwargs.get('session_groups', {})
        if not session_groups:
            # Create default groups based on session index
            n_sessions = len(data)
            mid_point = n_sessions // 2
            session_groups = {
                idx: 'Early' if i < mid_point else 'Late'
                for i, idx in enumerate(data.index)
            }
        
        # Calculate mean metrics per group
        group_metrics = {}
        for group_name in set(session_groups.values()):
            group_sessions = [
                s for s, g in session_groups.items() if g == group_name
            ]
            group_data = data.loc[data.index.intersection(group_sessions)]
            
            # Calculate means for numeric columns
            means = {}
            for col in group_data.select_dtypes(include=[np.number]).columns:
                values = []
                for val in group_data[col]:
                    if isinstance(val, (list, np.ndarray)):
                        values.append(np.mean(val))
                    elif pd.notna(val):
                        values.append(val)
                if values:
                    means[col] = np.mean(values)
            
            group_metrics[group_name] = means
        
        # Create grouped bar plot
        metrics_df = pd.DataFrame(group_metrics).T
        metrics_df.plot(
            kind='bar',
            ax=ax,
            color=sns.color_palette(config['color_palette'], len(metrics_df.columns))
        )
        
        ax.set_title(config['title'], fontsize=16, fontweight='bold')
        ax.set_xlabel(config['xlabel'] or 'Group', fontsize=12)
        ax.set_ylabel(config['ylabel'] or 'Mean Value', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        if config['legend']:
            ax.legend(
                title='Metrics',
                bbox_to_anchor=(1.05, 1),
                loc='upper left'
            )
        
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        return fig
    
    def _create_correlation_heatmap(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> plt.Figure:
        """Create correlation heatmap between metrics."""
        # Prepare data for correlation
        numeric_data = pd.DataFrame()
        
        for col in data.select_dtypes(include=[np.number]).columns:
            values = []
            for val in data[col]:
                if isinstance(val, (list, np.ndarray)):
                    values.append(np.mean(val))
                elif pd.notna(val):
                    values.append(val)
                else:
                    values.append(np.nan)
            numeric_data[col] = values
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=config['figure_size'])
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(config['title'], fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _create_session_timeline(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> plt.Figure:
        """Create timeline visualization of metrics across sessions."""
        fig, ax = plt.subplots(figsize=(config['figure_size'][0], config['figure_size'][1] * 0.6))
        
        # Get first numeric metric for timeline
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for timeline")
        
        metric_name = kwargs.get('metric_name', numeric_cols[0])
        
        # Extract values
        sessions = []
        values = []
        for idx, val in data[metric_name].items():
            if isinstance(val, (list, np.ndarray)):
                values.append(np.mean(val))
            elif pd.notna(val):
                values.append(val)
            else:
                continue
            sessions.append(str(idx))
        
        # Create timeline scatter plot
        y_positions = range(len(sessions))
        colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
        
        scatter = ax.scatter(
            values, y_positions,
            c=values,
            cmap='viridis',
            s=200,
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )
        
        # Add session labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(sessions)
        
        # Customize
        ax.set_xlabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Session', fontsize=12)
        ax.set_title(config['title'], fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Value', fontsize=10)
        
        plt.tight_layout()
        
        return fig
    
    def _create_metric_summary(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> plt.Figure:
        """Create summary dashboard with multiple metric visualizations."""
        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(config['figure_size'][0] * 1.5, config['figure_size'][1] * 1.5))
        axes = axes.flatten()
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:4]  # Max 4 metrics
        
        for idx, (ax, metric) in enumerate(zip(axes, numeric_cols)):
            # Extract values
            values = []
            for val in data[metric]:
                if isinstance(val, (list, np.ndarray)):
                    values.extend(val)
                elif pd.notna(val):
                    values.append(val)
            
            if not values:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric.replace('_', ' ').title())
                continue
            
            # Create violin plot
            parts = ax.violinplot(
                [values],
                positions=[1],
                showmeans=True,
                showmedians=True
            )
            
            # Color the violin
            for pc in parts['bodies']:
                pc.set_facecolor(sns.color_palette(config['color_palette'])[idx])
                pc.set_alpha(0.7)
            
            # Add summary statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.text(
                0.02, 0.98,
                f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove unused subplots
        for idx in range(len(numeric_cols), 4):
            fig.delaxes(axes[idx])
        
        plt.suptitle(config['title'], fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @property
    def supported_formats(self) -> List[str]:
        """List of supported output formats."""
        return ['png', 'pdf', 'svg', 'jpg']