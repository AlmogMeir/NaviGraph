# Clean Unified Architecture Plan
## Simple, Direct Migration Strategy for NaviGraph

### Core Principles
1. **One base class**: `NaviGraphPlugin` only
2. **One registry**: Manages plugins, analysis functions, visualization functions
3. **Three decorators**: `@register_plugin`, `@register_analysis`, `@register_visualizer`
4. **Two orchestrators**: `SessionAnalyzer` and `SessionVisualizer` managed by `Session`
5. **Delete the cruft**: Remove `interfaces.py`, `base_plugin.py` entirely

---

## 1. Simplified Registry (One Registry to Rule Them All)

```python
# navigraph/core/registry.py
from typing import Dict, Type, Callable, Any
import inspect
from loguru import logger

class UnifiedRegistry:
    """Single registry for plugins, analysis functions, and visualizers."""
    
    def __init__(self):
        self.plugins: Dict[str, Type] = {}
        self.analysis_functions: Dict[str, Callable] = {}
        self.visualizers: Dict[str, Callable] = {}
    
    def register_plugin(self, name: str, plugin_class: Type) -> None:
        """Register a plugin (data source type)."""
        self.plugins[name] = plugin_class
        logger.info(f"Registered plugin: {name}")
    
    def register_analysis(self, name: str, func: Callable) -> None:
        """Register an analysis function with validation."""
        # Validate function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        required = ['dataframe', 'shared_resources']
        if not all(p in params for p in required):
            raise ValueError(
                f"Analysis function '{name}' must have parameters: "
                f"dataframe, shared_resources, **config. Got: {params}"
            )
        
        self.analysis_functions[name] = func
        logger.info(f"Registered analysis function: {name}")
    
    def register_visualizer(self, name: str, func: Callable) -> None:
        """Register a visualizer function with validation."""
        # Validate function signature
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        required = ['frame', 'frame_data', 'shared_resources']
        if not all(p in params for p in required):
            raise ValueError(
                f"Visualizer '{name}' must have parameters: "
                f"frame, frame_data, shared_resources, **config. Got: {params}"
            )
        
        self.visualizers[name] = func
        logger.info(f"Registered visualizer: {name}")
    
    def get_plugin(self, name: str) -> Type:
        if name not in self.plugins:
            raise ValueError(f"Plugin '{name}' not found. Available: {list(self.plugins.keys())}")
        return self.plugins[name]
    
    def get_analysis(self, name: str) -> Callable:
        if name not in self.analysis_functions:
            raise ValueError(f"Analysis '{name}' not found. Available: {list(self.analysis_functions.keys())}")
        return self.analysis_functions[name]
    
    def get_visualizer(self, name: str) -> Callable:
        if name not in self.visualizers:
            raise ValueError(f"Visualizer '{name}' not found. Available: {list(self.visualizers.keys())}")
        return self.visualizers[name]

# Global instance
registry = UnifiedRegistry()

# Simple decorator functions
def register_plugin(name: str):
    """Decorator to register plugins."""
    def decorator(cls):
        registry.register_plugin(name, cls)
        return cls
    return decorator

def register_analysis(name: str):
    """Decorator to register analysis functions."""
    def decorator(func):
        registry.register_analysis(name, func)
        return func
    return decorator

def register_visualizer(name: str):
    """Decorator to register visualizer functions."""
    def decorator(func):
        registry.register_visualizer(name, func)
        return func
    return decorator
```

---

## 2. Session Analyzer (Orchestrates Analysis)

```python
# navigraph/core/session_analyzer.py
from typing import Dict, Any, List
import pandas as pd
from loguru import logger
from .registry import registry

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
    
    def run_analyses(self, dataframe: pd.DataFrame, shared_resources: Dict[str, Any]) -> Dict[str, Any]:
        """Run all configured analyses.
        
        Args:
            dataframe: Session dataframe with all plugin data
            shared_resources: Shared resources (graph, mapping, etc.)
            
        Returns:
            Dict of analysis results by name
        """
        results = {}
        
        for analysis_name, analysis_config in self.config.items():
            # Skip disabled analyses
            if not analysis_config.get('enabled', True):
                logger.debug(f"Skipping disabled analysis: {analysis_name}")
                continue
            
            try:
                # Get analysis function from registry
                analysis_func = registry.get_analysis(analysis_name)
                
                # Run analysis with config
                logger.info(f"Running analysis: {analysis_name}")
                result = analysis_func(
                    dataframe=dataframe,
                    shared_resources=shared_resources,
                    **analysis_config
                )
                
                results[analysis_name] = result
                logger.info(f"✓ {analysis_name}: {len(result)} metrics")
                
            except ValueError as e:
                logger.error(f"Analysis '{analysis_name}' not found: {e}")
            except Exception as e:
                logger.error(f"Analysis '{analysis_name}' failed: {e}")
                results[analysis_name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all analysis results."""
        summary = {
            'total_analyses': len(self.results),
            'successful': sum(1 for r in self.results.values() if 'error' not in r),
            'failed': sum(1 for r in self.results.values() if 'error' in r),
            'metrics': {}
        }
        
        # Flatten all metrics
        for analysis_name, result in self.results.items():
            if 'error' not in result:
                for metric_name, value in result.items():
                    summary['metrics'][f"{analysis_name}.{metric_name}"] = value
        
        return summary
```

---

## 3. Session Visualizer (Orchestrates Visualization)

```python
# navigraph/core/session_visualizer.py
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from loguru import logger
from .registry import registry

class SessionVisualizer:
    """Orchestrates visualization pipeline for a session."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with visualization configuration.
        
        Args:
            config: Dict with structure:
                {
                    'visualizations': {
                        'pipeline': ['trajectory', 'map_overlay', 'metrics'],
                        'output': {
                            'enabled': True,
                            'path': './output',
                            'format': 'mp4',
                            'fps': 30
                        },
                        'visualizer_configs': {
                            'trajectory': {'color': [0, 255, 0]},
                            'map_overlay': {'opacity': 0.7}
                        }
                    }
                }
        """
        self.config = config.get('visualizations', {})
        self.pipeline = self.config.get('pipeline', [])
        self.output_config = self.config.get('output', {})
        self.visualizer_configs = self.config.get('visualizer_configs', {})
    
    def process_video(
        self, 
        video_path: str,
        dataframe: pd.DataFrame, 
        shared_resources: Dict[str, Any],
        output_name: str = "output"
    ) -> Optional[str]:
        """Process video through visualization pipeline.
        
        Args:
            video_path: Path to input video
            dataframe: Session dataframe
            shared_resources: Shared resources
            output_name: Name for output file
            
        Returns:
            Path to output video if created, None otherwise
        """
        if not self.output_config.get('enabled', True):
            logger.info("Video output disabled")
            return None
        
        # Load visualizer functions
        visualizers = []
        for viz_name in self.pipeline:
            try:
                viz_func = registry.get_visualizer(viz_name)
                visualizers.append((viz_name, viz_func))
                logger.info(f"Loaded visualizer: {viz_name}")
            except ValueError as e:
                logger.error(f"Visualizer '{viz_name}' not found: {e}")
        
        if not visualizers:
            logger.warning("No valid visualizers in pipeline")
            return None
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
        
        # Get video properties
        fps = self.output_config.get('fps', cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output
        output_path = Path(self.output_config.get('path', './output'))
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_format = self.output_config.get('format', 'mp4')
        output_file = output_path / f"{output_name}.{output_format}"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
        
        logger.info(f"Processing {total_frames} frames through {len(visualizers)} visualizers")
        
        # Process frames
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get frame data from dataframe
            if frame_idx < len(dataframe):
                frame_data = dataframe.iloc[frame_idx]
            else:
                frame_data = pd.Series()
            
            # Apply visualizer pipeline (each modifies the frame)
            for viz_name, viz_func in visualizers:
                viz_config = self.visualizer_configs.get(viz_name, {})
                try:
                    frame = viz_func(
                        frame=frame,
                        frame_data=frame_data,
                        shared_resources=shared_resources,
                        **viz_config
                    )
                except Exception as e:
                    logger.error(f"Visualizer '{viz_name}' failed on frame {frame_idx}: {e}")
            
            # Write processed frame
            writer.write(frame)
            
            # Progress logging
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                logger.debug(f"Progress: {progress:.1f}%")
            
            frame_idx += 1
        
        # Cleanup
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
        logger.info(f"✓ Video saved to: {output_file}")
        return str(output_file)
```

---

## 4. Updated Session Class

```python
# navigraph/core/session.py (additions)
from .session_analyzer import SessionAnalyzer
from .session_visualizer import SessionVisualizer

class Session:
    """Session with integrated analyzer and visualizer orchestration."""
    
    def __init__(self, session_configuration, logger_instance):
        # ... existing init code ...
        
        # Initialize orchestrators
        self.analyzer = SessionAnalyzer(session_configuration)
        self.visualizer = SessionVisualizer(session_configuration)
    
    def run_analyses(self) -> Dict[str, Any]:
        """Run all configured analyses through SessionAnalyzer.
        
        Returns:
            Dict of analysis results
        """
        if self._integrated_dataframe is None:
            self._integrated_dataframe = self._execute_plugins()
        
        return self.analyzer.run_analyses(
            dataframe=self._integrated_dataframe,
            shared_resources=self.shared_resources
        )
    
    def create_visualization(self, video_path: str, output_name: str = None) -> Optional[str]:
        """Create visualization through SessionVisualizer.
        
        Args:
            video_path: Path to input video
            output_name: Name for output file (defaults to session_id)
            
        Returns:
            Path to created video or None
        """
        if self._integrated_dataframe is None:
            self._integrated_dataframe = self._execute_plugins()
        
        output_name = output_name or self.session_id
        
        return self.visualizer.process_video(
            video_path=video_path,
            dataframe=self._integrated_dataframe,
            shared_resources=self.shared_resources,
            output_name=output_name
        )
```

---

## 5. Example Analysis Functions

```python
# navigraph/plugins/analyzers/spatial_metrics.py
from navigraph.core.registry import register_analysis
import pandas as pd
import numpy as np

@register_analysis("spatial_metrics")
def analyze_spatial_metrics(dataframe: pd.DataFrame, shared_resources: dict, **config) -> dict:
    """Compute spatial navigation metrics.
    
    Expected config:
        - compute_distance: bool
        - compute_visits: bool
        - target_tiles: list of tile IDs
    """
    results = {}
    
    if 'tile_id' in dataframe.columns:
        # Unique tiles visited
        results['unique_tiles'] = dataframe['tile_id'].nunique()
        
        # Visit counts for target tiles
        if config.get('target_tiles'):
            for tile_id in config['target_tiles']:
                results[f'visits_tile_{tile_id}'] = (dataframe['tile_id'] == tile_id).sum()
    
    if config.get('compute_distance', True) and 'keypoints_x' in dataframe.columns:
        # Total distance traveled
        dx = dataframe['keypoints_x'].diff()
        dy = dataframe['keypoints_y'].diff()
        distances = np.sqrt(dx**2 + dy**2)
        results['total_distance_pixels'] = distances.sum()
    
    return results

@register_analysis("timing_metrics")
def analyze_timing(dataframe: pd.DataFrame, shared_resources: dict, **config) -> dict:
    """Compute timing-related metrics."""
    results = {}
    
    # Get FPS from shared resources
    stream_info = shared_resources.get('stream_info', {})
    fps = stream_info.get('fps', 30)
    
    results['duration_seconds'] = len(dataframe) / fps
    results['total_frames'] = len(dataframe)
    
    # Time to reach target
    target_tile = config.get('target_tile')
    if target_tile and 'tile_id' in dataframe.columns:
        first_arrival = dataframe[dataframe['tile_id'] == target_tile].index.min()
        if pd.notna(first_arrival):
            results['time_to_target'] = first_arrival / fps
    
    return results
```

---

## 6. Example Visualizer Functions

```python
# navigraph/plugins/visualizers/trajectory.py
from navigraph.core.registry import register_visualizer
import cv2
import pandas as pd
import numpy as np

@register_visualizer("trajectory")
def visualize_trajectory(frame: np.ndarray, frame_data: pd.Series, shared_resources: dict, **config) -> np.ndarray:
    """Draw current position marker on frame.
    
    Config:
        - color: [B, G, R] color values
        - radius: marker radius in pixels
        - thickness: -1 for filled, positive for outline
    """
    color = config.get('color', [0, 255, 0])
    radius = config.get('radius', 5)
    thickness = config.get('thickness', -1)
    
    # Get position from frame data
    if 'keypoints_x' in frame_data and 'keypoints_y' in frame_data:
        x = frame_data['keypoints_x']
        y = frame_data['keypoints_y']
        
        if pd.notna(x) and pd.notna(y):
            cv2.circle(frame, (int(x), int(y)), radius, color, thickness)
    
    return frame

@register_visualizer("map_overlay")
def visualize_map(frame: np.ndarray, frame_data: pd.Series, shared_resources: dict, **config) -> np.ndarray:
    """Overlay map on frame.
    
    Config:
        - opacity: float between 0 and 1
        - position: 'top_left', 'top_right', 'bottom_left', 'bottom_right'
        - size_ratio: float, proportion of frame size
    """
    # Get map from shared resources
    map_data = shared_resources.get('map_image')
    if map_data is None:
        return frame
    
    opacity = config.get('opacity', 0.7)
    position = config.get('position', 'bottom_right')
    size_ratio = config.get('size_ratio', 0.3)
    
    # Calculate overlay size
    h, w = frame.shape[:2]
    map_h, map_w = map_data.shape[:2]
    
    # Resize map
    new_w = int(w * size_ratio)
    new_h = int(new_w * map_h / map_w)
    resized_map = cv2.resize(map_data, (new_w, new_h))
    
    # Determine position
    positions = {
        'top_left': (0, 0),
        'top_right': (w - new_w, 0),
        'bottom_left': (0, h - new_h),
        'bottom_right': (w - new_w, h - new_h)
    }
    x, y = positions.get(position, (w - new_w, h - new_h))
    
    # Overlay with transparency
    roi = frame[y:y+new_h, x:x+new_w]
    result = cv2.addWeighted(roi, 1-opacity, resized_map, opacity, 0)
    frame[y:y+new_h, x:x+new_w] = result
    
    return frame

@register_visualizer("tile_highlight")  
def visualize_tile(frame: np.ndarray, frame_data: pd.Series, shared_resources: dict, **config) -> np.ndarray:
    """Highlight current tile on map overlay."""
    # Implementation here
    return frame
```

---

## 7. Configuration Format

```yaml
# Clean, simple configuration
experiment_path: ./data/experiment1

# Data source plugins (keep as is)
plugins:
  - name: pose_tracking
    type: pose_tracking
    file_pattern: ".*\\.h5$"
  - name: calibration
    type: calibration
    shared: true
  - name: graph_location
    type: graph_location

# Analysis configuration
analyses:
  spatial_metrics:
    enabled: true
    compute_distance: true
    target_tiles: [5, 10, 15]
  
  timing_metrics:
    enabled: true
    target_tile: 5

# Visualization configuration  
visualizations:
  pipeline: [trajectory, map_overlay, tile_highlight]
  output:
    enabled: true
    path: ./output/videos
    format: mp4
    fps: 30
  visualizer_configs:
    trajectory:
      color: [0, 255, 0]
      radius: 5
    map_overlay:
      opacity: 0.7
      position: bottom_right
    tile_highlight:
      highlight_color: [255, 0, 0]
```

---

## 8. Migration Steps (Clean & Simple)

### Step 1: Clean House (Day 1)
```bash
# Delete redundant files
rm navigraph/core/interfaces.py
rm navigraph/core/base_plugin.py

# Update imports in all files to remove references
grep -r "from.*interfaces import" navigraph/
grep -r "from.*base_plugin import" navigraph/
# Fix all imports to use navigraph_plugin instead
```

### Step 2: Update Registry (Day 1)
1. Replace current registry.py with simplified version
2. Keep backward compatibility temporarily:
   ```python
   # Temporary aliases
   register_data_source_plugin = lambda name: register_plugin(name)
   register_analyzer_plugin = lambda name: register_analysis(name)
   register_visualizer_plugin = lambda name: register_visualizer(name)
   ```

### Step 3: Create Orchestrators (Day 2)
1. Create `session_analyzer.py`
2. Create `session_visualizer.py`
3. Update `session.py` to use them

### Step 4: Migrate Analyzers (Day 3)
1. Convert each analyzer to simple function with `@register_analysis`
2. Remove class boilerplate, keep core logic
3. Test each function independently

### Step 5: Migrate Visualizers (Day 4)
1. Convert each visualizer to simple function with `@register_visualizer`
2. Remove IVisualizer interface usage
3. Test pipeline with multiple visualizers

### Step 6: Final Cleanup (Day 5)
1. Remove temporary aliases from registry
2. Update all decorators to use new names
3. Update configs to new format
4. Run full test suite

---

## 9. Benefits of This Approach

### Simplicity
- **3 decorators** instead of complex class hierarchies
- **Functions over classes** for analysis and visualization
- **Clear separation**: Plugins augment data, functions analyze/visualize

### Maintainability  
- **Less code**: ~60% reduction in boilerplate
- **Obvious flow**: Session → Orchestrator → Functions
- **Easy debugging**: Each function is independent

### Flexibility
- **Mix and match**: Any combination of visualizers in pipeline
- **Easy to extend**: Just add a decorated function
- **Config-driven**: Everything controlled from YAML

### Performance
- **Lazy loading**: Functions loaded only when needed
- **Pipeline optimization**: Process frame once through all visualizers
- **Parallel analysis**: Could easily parallelize independent analyses

---

## 10. What We're NOT Doing

1. **NOT** creating new base classes or interfaces
2. **NOT** using multiple inheritance
3. **NOT** creating complex plugin hierarchies
4. **NOT** keeping backward compatibility forever
5. **NOT** over-engineering for hypothetical future needs

---

## Summary

This plan creates a clean, simple architecture:
- `NaviGraphPlugin` for data sources (provide + augment)
- Decorated functions for analysis and visualization
- `SessionAnalyzer` and `SessionVisualizer` as orchestrators
- One unified registry managing everything
- Delete all the old cruft

The migration is straightforward, can be done incrementally, and results in a much simpler, more maintainable codebase.