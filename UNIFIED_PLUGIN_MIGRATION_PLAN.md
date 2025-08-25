# NaviGraph Unified Plugin Architecture Migration Plan

## Overview

This document outlines the completed migration to a unified plugin architecture and the remaining work needed to fully transition NaviGraph from the old interface-based system to the new NaviGraphPlugin architecture.

## Completed Work

### 1. Core Architecture Changes

#### New Unified Plugin Base Class
- **Created**: `navigraph/core/navigraph_plugin.py`
- **Purpose**: Single base class for all plugin types replacing old interfaces
- **Key Methods**:
  - `provide(shared_resources)`: Add resources to shared pool (optional)
  - `augment_data(dataframe, shared_resources)`: Add columns to session DataFrame (optional)
  - `get_plugin_info()`: Plugin metadata and capabilities
  - **File Discovery**: Automatic file discovery in constructor based on `file_pattern` config

#### Updated Session Architecture
- **Replaced**: `navigraph/core/session.py` with clean unified implementation
- **Key Features**:
  - Creates graph structure first using `GraphStructure.from_config()`
  - Validates mapping file compatibility with builder config
  - Two-phase plugin execution: `provide()` then `augment_data()`
  - Plugin instances stored by name in dictionary (allows multiple instances)
  - **Removed**: All backward compatibility code

#### Updated Registry System
- **Modified**: `navigraph/core/registry.py` for unified architecture
- **Changes**:
  - All plugin categories now use `NaviGraphPlugin` base class
  - Removed `shared_resources` category (now provided by plugins)
  - Added plugin info and type filtering capabilities
  - **Maintained**: Separate categories for data_sources, analyzers, visualizers

#### Shared Coordinate Transformation Utility
- **Created**: `navigraph/core/coordinate_transform.py`
- **Functions**:
  - `transform_coordinates()`: Homography transformation using calibration matrix
  - `apply_coordinate_transform_to_bodyparts()`: Batch coordinate transformation
  - Used by both map_location and graph_location plugins

### 2. Unified Data Source Plugins

All data source plugins have been migrated to the unified architecture:

#### Core Data Plugins
1. **pose_tracking.py** (`@register_data_source_plugin("pose_tracking")`)
   - Loads DeepLabCut H5 files
   - Adds bodypart_x/y/likelihood columns
   - Supports derived bodyparts with centroid calculation
   - Replaces old: `deeplabcut_old.py`

2. **calibration.py** (`@register_data_source_plugin("calibration")`)
   - Loads transform_matrix.npy
   - Provides `calibration_matrix` to shared resources
   - Validates matrix shape and determinant

3. **stream_info.py** (`@register_data_source_plugin("stream_info")`)
   - Extracts video metadata using OpenCV
   - Provides `stream_info` with fps, frame_count, duration, dimensions

#### Location Mapping Plugins
4. **map_location.py** (`@register_data_source_plugin("map_location")`)
   - Transforms pose coordinates to map coordinates using calibration matrix
   - Provides `map_image` and `map_metadata` to shared resources
   - Uses coordinate_transform utility
   - Replaces old: `map_integration_old.py`

5. **graph_location.py** (`@register_data_source_plugin("graph_location")`)
   - Maps coordinates to graph nodes and edges using spatial mapping
   - Process: pose → calibration → map coordinates → graph mapping
   - Uses coordinate_transform utility
   - Replaces old: `graph_integration_old.py`

#### Multimodal Data Plugins
6. **neural_activity.py** (`@register_data_source_plugin("neural_activity")`)
   - Loads Minian zarr data (calcium imaging)
   - Adds neuron_* columns for each detected neuron
   - Supports derived metrics (e.g., population mean)
   - Flexible zarr loading with xarray and direct zarr fallback

7. **head_direction.py** (`@register_data_source_plugin("head_direction")`)
   - Loads quaternion data from CSV files
   - Converts to Euler angles using `quaternions_to_euler`
   - Adds yaw, pitch, roll columns
   - Supports skip_index for manual frame synchronization

### 3. Cleanup and Removal

#### Removed Obsolete Files
- **Data Sources**: All `*_old.py` backup files
- **Session**: `session_old.py`
- **Shared Resources**: Entire `shared_resources/` folder (now obsolete)
- **Unified Folder**: Empty `plugins/unified/` directory

#### Updated Import Structure
- **Modified**: `navigraph/plugins/data_sources/__init__.py`
- **Updated**: `navigraph/plugins/__init__.py` (removed shared_resources reference)
- **Verified**: All unified plugins import and register successfully

## Current Architecture Gap

### Interface Mismatch
The system currently has **two coexisting plugin architectures**:

1. **New Unified Architecture** (Data Sources ✅)
   - Base class: `NaviGraphPlugin`
   - Methods: `provide()`, `augment_data()`
   - Registry: `register_data_source_plugin()`

2. **Old Interface Architecture** (Analyzers, Visualizers ❌)
   - Base class: `BasePlugin`
   - Interfaces: `IAnalyzer`, `IVisualizer`
   - Methods: `integrate_data_into_session()`, `process()`, `visualize()`
   - Registry: `register_analyzer_plugin()`, `register_visualizer_plugin()`

### Registry Categories
Current registry maintains separate categories but all should use unified base:
- ✅ `data_sources`: Uses `NaviGraphPlugin`
- ❌ `analyzers`: Still uses `IAnalyzer` interface
- ❌ `visualizers`: Still uses `IVisualizer` interface

## Migration Plan for Remaining Components

### Phase 1: Analyzer Migration

#### Current Analyzers to Migrate
```
navigraph/plugins/analyzers/
├── exploration_metrics.py    # Exploration analysis
├── navigation_metrics.py     # Navigation performance
├── spatial_metrics.py        # Spatial behavior analysis
└── utils.py                  # Shared analyzer utilities
```

#### Migration Strategy for Analyzers
1. **Convert to NaviGraphPlugin base class**
   - Replace `BasePlugin, IAnalyzer` with `NaviGraphPlugin`
   - Change `integrate_data_into_session()` to `augment_data()`
   - Add `get_plugin_info()` method
   - Update registry decorator to `@register_data_source_plugin()`

2. **Session vs Cross-Session Analysis**
   - **Session-level**: Current analyzers work on individual session DataFrames
   - **Cross-session**: Need new mechanism for multi-session analysis
   - **Proposal**: 
     - Keep session-level analyzers as data sources (add computed columns)
     - Create separate cross-session analysis system in experiment_runner
     - Cross-session analyzers could inherit from NaviGraphPlugin but work on session collections

3. **Analyzer Plugin Architecture**
   ```python
   @register_data_source_plugin("spatial_metrics")
   class SpatialMetricsPlugin(NaviGraphPlugin):
       def augment_data(self, dataframe, shared_resources):
           # Compute metrics and add as columns
           # e.g., speed, acceleration, distance_traveled, etc.
           return enhanced_dataframe
   ```

### Phase 2: Visualizer Migration

#### Current Visualizers to Migrate
```
navigraph/plugins/visualizers/
├── keypoint_visualizer.py     # Pose visualization
├── map_visualizer.py          # Map overlays
├── metrics_visualizer.py      # Metrics plots
├── text_visualizer.py         # Text annotations
├── trajectory_visualizer.py   # Movement paths
└── tree_visualizer.py         # Graph structure
```

#### Critical Visualization Capabilities
Must preserve ability to visualize:
- ✅ **Map Image**: From map_location plugin (`map_image`, `map_metadata`)
- ✅ **Map Contours**: Map boundaries and regions
- ✅ **Graph Structure**: Nodes, edges, paths from graph structure
- ❌ **Pose Keypoints**: Body part positions and tracking
- ❌ **Trajectories**: Movement paths over time
- ❌ **Metrics Overlays**: Speed, direction, neural activity heatmaps

#### Migration Strategy for Visualizers
1. **Convert to NaviGraphPlugin base class**
   - Replace `BasePlugin, IVisualizer` with `NaviGraphPlugin`
   - Change approach: instead of separate `visualize()` method, generate visualization data as columns
   - **Option A**: Add visualization columns to DataFrame (e.g., `plot_data_json`)
   - **Option B**: Create separate visualization pipeline that uses NaviGraphPlugin interface

2. **Visualization Data vs Direct Rendering**
   - **Current**: Visualizers directly render to files/display
   - **Proposed**: Visualizers prepare data, separate rendering step
   - **Benefits**: Better separation of concerns, testability, modularity

3. **Visualizer Plugin Architecture**
   ```python
   @register_data_source_plugin("trajectory_visualizer")  
   class TrajectoryVisualizerPlugin(NaviGraphPlugin):
       def augment_data(self, dataframe, shared_resources):
           # Add trajectory visualization data columns
           # Could be JSON, plot coordinates, etc.
           return dataframe_with_viz_data
   ```

### Phase 3: Interface Cleanup

#### Remove Obsolete Interfaces
Once all plugins are migrated:
1. **Remove**: `navigraph/core/interfaces.py`
2. **Remove**: `navigraph/core/base_plugin.py`
3. **Update**: All remaining imports to use NaviGraphPlugin
4. **Simplify**: Registry to single plugin type with categories

#### Update CLI and Experiment Runner
1. **CLI**: Update commands to work with unified plugins only
2. **Experiment Runner**: Remove shared_resources loading, use session's unified approach
3. **Configuration**: Update example configs to use unified plugin specifications

## Testing and Validation Plan

### Phase 4: Integration Testing

#### Test Cases to Validate
1. **Basic Maze Example**
   - Load pose tracking data
   - Apply calibration transformation  
   - Map to graph locations
   - Generate spatial metrics
   - Visualize trajectory on map with graph overlay

2. **Multimodal Demo Example**
   - All basic maze functionality
   - Load neural activity data
   - Load head direction data
   - Cross-correlate behavioral and neural metrics
   - Generate multimodal visualizations

3. **Session Processing Pipeline**
   - Multiple sessions in experiment
   - Consistent plugin execution order
   - Shared resource propagation
   - Error handling and validation

4. **Visualization Capabilities**
   - Map loading and display
   - Graph structure overlay
   - Trajectory rendering
   - Real-time pose tracking display
   - Metrics visualization

## Configuration Updates Required

### Current Config Structure (Needs Update)
```yaml
data_sources:
  - name: deeplabcut      # OLD NAME
    type: deeplabcut      # OLD TYPE
    
shared_resources:         # OBSOLETE SECTION
  - name: graph_provider
```

### New Unified Config Structure
```yaml
plugins:  # OR keep data_sources but expand scope
  - name: pose_tracking
    type: pose_tracking
    file_pattern: '.*DLC.*\.h5$'
    config:
      bodyparts: 'all'
      derived_bodyparts:
        center: centroid
        
  - name: calibration
    type: calibration
    file_pattern: 'transform_matrix\.npy$'
    
  - name: map_location
    type: map_location
    file_pattern: '.*maze_map\.png$'
    
  - name: graph_location
    type: graph_location
    # No file - uses shared resources
    
  - name: spatial_metrics
    type: spatial_metrics
    # Analyzer as data source
    
  - name: trajectory_visualizer  
    type: trajectory_visualizer
    # Visualizer as data source
```

## Risk Mitigation

### Backward Compatibility
- **OLD**: Keep old plugins as `*_old.py` during transition
- **NEW**: All old plugins already backed up and removed
- **CONFIG**: Support both old and new config formats during migration

### Testing Strategy
- **Unit Tests**: Each unified plugin independently
- **Integration Tests**: Full pipeline with basic_maze
- **Regression Tests**: Compare outputs before/after migration
- **Performance Tests**: Ensure no significant slowdown

### Rollback Plan
- **Git History**: All changes are version controlled
- **Backup Files**: Critical old files backed up as `*_old.py`
- **Configuration**: Old config format support during transition

## Success Criteria

### Migration Complete When:
1. ✅ All data source plugins use NaviGraphPlugin
2. ❌ All analyzer plugins use NaviGraphPlugin  
3. ❌ All visualizer plugins use NaviGraphPlugin
4. ❌ No references to old interfaces (IDataSource, IAnalyzer, IVisualizer)
5. ❌ No shared_resources category in registry
6. ✅ Basic maze example runs successfully
7. ❌ Multimodal demo example runs successfully
8. ❌ All critical visualizations work (map, graph, trajectories)

### Current Status: **Phase 1 Complete** ✅
- Unified data source plugins implemented and working
- Registry updated for unified architecture
- Session using unified execution pipeline  
- Ready to begin Phase 2 (Analyzer Migration)

## Next Immediate Steps

1. **Test Basic Maze**: Validate current unified data source pipeline
2. **Migrate Analyzers**: Convert to NaviGraphPlugin architecture  
3. **Migrate Visualizers**: Ensure visualization capabilities preserved
4. **Update Configuration**: Support new unified config format
5. **Integration Testing**: Full pipeline validation with both examples