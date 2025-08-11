# NaviGraph - Graph Setup GUI Complete Documentation

## Overview
The Graph Setup GUI (`navigraph/core/graph/setup_gui_qt.py`) is a sophisticated PyQt5-based interface for mapping graph structures (nodes and edges) to spatial regions on a map image. It supports two primary mapping workflows: **Grid-based Setup** and **Manual Contour Drawing**.

## Architecture Components

### 1. Main Window (`GraphSetupWindow`)
- **Dual-view interface**: Graph structure visualization (top) + Interactive map (bottom)
- **Control panel**: Mode selection and mode-specific controls
- **Status bar**: Real-time feedback and progress tracking
- **Three operation modes**: Grid Setup, Manual Drawing, Test Mode

### 2. Map Widget (`MapWidget`)
- Interactive canvas for placing grids and drawing contours
- Handles mouse events for cell selection and contour drawing
- Visualizes mapped regions with color coding (green for nodes, blue for edges)
- Supports zoom/pan through automatic scaling

### 3. Graph Widget (`GraphWidget`)
- Matplotlib-based graph visualization
- Automatic layout computation (hierarchical for trees, spring layout for general graphs)
- Interactive highlighting of selected elements
- Scales visualization based on graph size

## Workflow 1: Grid-Based Setup

### Flow Description
1. **Mode Selection**: User clicks "📊 Grid Setup" button
2. **Grid Configuration**: 
   - Set grid dimensions (rows × columns)
   - Define cell size (width/height in pixels)
   - Grid structure type (currently rectangle, hexagon planned)
3. **Grid Placement**:
   - Click "🎯 Place Grid" button
   - Click on map to set grid origin point
   - Grid overlay appears on map
4. **Element Mapping**:
   - System queues all graph elements (nodes first, then edges)
   - Current element highlighted in graph view
   - User clicks grid cells to select regions for current element
   - Selected cells turn green
5. **Assignment**:
   - Click "✅ Assign Selected Cells" to map cells to element
   - Cells change color permanently (light green for nodes, light blue for edges)
   - System automatically advances to next element
6. **Navigation Controls**:
   - **Next**: Skip current element
   - **Jump**: Select specific element from dropdown
   - **Undo**: Revert last assignment
   - **Clear All**: Reset all mappings

### Grid Configuration (`GridConfig`)
```python
@dataclass
class GridConfig:
    structure_type: str = 'rectangle'  # Future: 'hexagon'
    rows: int = 8
    cols: int = 8
    cell_width: float = 50.0
    cell_height: float = 50.0
    origin_x: float = 0.0
    origin_y: float = 0.0
```

### Cell Identification
- Cells identified as `cell_{row}_{col}` (e.g., "cell_3_5")
- Each cell becomes a `RectangleRegion` when assigned
- Multiple cells can map to single graph element

## Workflow 2: Manual Contour Drawing

### Flow Description
1. **Mode Selection**: User clicks "✏️ Manual Drawing"
2. **Element Selection**:
   - Current element shown in status bar
   - Element highlighted in graph view
3. **Contour Drawing**:
   - Left-click on map to add contour points
   - Points connected with red lines
   - Current drawing shown in real-time
4. **Contour Controls**:
   - **🗑️ Clear Contour**: Remove current drawing
   - **✅ Commit Contour**: Save contour for current element
5. **Contour Management**:
   - Committed contours listed in sidebar
   - Click contour in list to highlight on map
   - Delete individual contours or clear all
6. **Element Navigation**:
   - **Next**: Move to next element
   - **Jump**: Select specific element
   - **Undo**: Revert last contour

### Contour Storage
- Contours stored as list of (x, y) points
- Converted to `ContourRegion` objects
- Each contour has unique ID: `{element_type}_{element_id}_contour_{index}`

## Test Mode

### Purpose
Validate mappings by testing coordinate-to-element resolution

### Features
- Click map locations to see which graph element is mapped
- Click graph elements to highlight their mapped regions
- Load/save mapping files for testing
- Real-time feedback in results display

## Data Structures

### Spatial Regions (from `regions.py`)

#### Base Class: `SpatialRegion`
- Abstract class for all region types
- Key methods:
  - `contains_point(x, y)`: Check if point is inside
  - `get_bounds()`: Return bounding box
  - `get_center()`: Calculate center point
  - `get_area()`: Calculate region area

#### Region Types
1. **`ContourRegion`**: Polygon defined by point list
2. **`RectangleRegion`**: Axis-aligned rectangle
3. **`CircleRegion`**: Circular region
4. **`GridCell`**: Grid-based rectangular cell
5. **`HexagonalCell`**: Hexagonal grid cell (planned)

### Mapping Structure (`SpatialMapping`)
```python
class SpatialMapping:
    def __init__(self, graph: GraphStructure):
        self.graph = graph
        self.node_to_regions = {}  # node_id -> List[SpatialRegion]
        self.edge_to_regions = {}  # edge_tuple -> List[SpatialRegion]
        self.region_to_node = {}   # region_id -> node_id
        self.region_to_edge = {}   # region_id -> edge_tuple
```

## Color Coding System
- **Light Green (150, 255, 150, 100)**: Node regions
- **Light Blue (150, 150, 255, 100)**: Edge regions
- **Bright Green (0, 255, 0, 100)**: Currently selected cells
- **Yellow (255, 255, 0, 150)**: Temporary highlight
- **Red contour lines**: Active drawing

## File Operations

### Saving Mappings
- Format: Python pickle (`.pkl`)
- Contains complete `SpatialMapping` object
- Preserves all regions and associations

### Loading Mappings
- Restores `SpatialMapping` object
- Automatically visualizes loaded regions
- Updates progress indicators

## Key Event Handlers

### Grid Mode
- `_on_grid_mode()`: Initialize grid setup
- `_on_apply_grid_config()`: Update grid parameters
- `_on_place_grid()`: Start grid placement
- `_on_grid_placed()`: Handle grid origin placement
- `_on_cell_clicked()`: Toggle cell selection
- `_on_assign_cells()`: Map cells to current element

### Manual Mode
- `_on_manual_mode()`: Initialize drawing mode
- `mousePressEvent()`: Add contour points
- `_on_contour_drawn()`: Process completed contour
- `_on_clear_contour()`: Clear current drawing
- `_on_commit_contour()`: Save contour
- `_on_delete_contour()`: Remove selected contour

### Common Controls
- `_on_next_element()`: Advance to next element
- `_on_jump_to_element()`: Jump to specific element
- `_on_undo_last()`: Revert last action
- `_on_toggle_mappings()`: Show/hide all mappings
- `_on_save_mapping()`: Export to file
- `_on_load_mapping()`: Import from file

## State Management

### Mapping History
- Stack-based undo system
- Tracks actions: `assign_cells`, `add_contour`
- Stores element, regions, and cell IDs
- Enables single-step undo

### Element Queue
- Initialized with all nodes and edges
- Processes elements sequentially
- Supports jumping and skipping
- Tracks completion progress

### Visual State
- `show_all_mappings`: Toggle mapped region visibility
- `interaction_mode`: Current mouse interaction type
- `current_element`: Active element being mapped
- `scale_factor`: Map zoom level

## Progress Tracking
- Progress bar shows mapped/total elements
- Percentage completion displayed
- Separate counts for nodes and edges
- Real-time updates after each action

## GUI Layout Hierarchy
```
GraphSetupWindow
├── Control Panel (Left, 380px fixed)
│   ├── Mode Selection (Grid/Manual/Test)
│   ├── Mode Stack (Dynamic content)
│   │   ├── Grid Controls
│   │   ├── Manual Controls
│   │   └── Test Controls
│   └── Display Options
└── View Panel (Right, expandable)
    ├── Graph Widget (Top, 1/4 height)
    └── Map Widget (Bottom, 3/4 height)
```

## Error Handling
- Graceful backend fallback for matplotlib
- Safe undo operations with try-catch
- Mode switching validation
- File I/O error messages

## Keyboard Shortcuts (Planned)
- **V**: Vertex/node mode
- **E**: Edge mode
- **R**: Region mode
- **Delete**: Remove selected
- **Ctrl+Z**: Undo
- **Ctrl+S**: Save mapping
- **Escape**: Cancel current operation

## Performance Optimizations
- Lazy grid generation
- Efficient contour point storage
- Scaled visualization for large graphs
- Batched UI updates

## Future Enhancements
1. **Hexagonal grid support**: Already structured in `GridConfig`
2. **Multi-select elements**: Map multiple elements simultaneously
3. **Region templates**: Save and reuse common region patterns
4. **Auto-mapping**: Suggest regions based on graph topology
5. **Keyboard shortcuts**: Full keyboard navigation
6. **Region editing**: Modify existing regions after creation
7. **Import/export formats**: JSON, XML support beyond pickle

## Integration Points

### CLI Command
```bash
navigraph setup graph config.yaml
```
Launches GUI with graph and map from configuration.

### Python API
```python
from navigraph.core.graph.setup_gui_qt import launch_setup_gui
from navigraph.core.graph.structures import GraphStructure

mapping = launch_setup_gui(graph, map_image)
```

### Output Usage
The created `SpatialMapping` object is used by:
- Navigation metrics analyzers
- Trajectory visualizers
- Graph-based path analysis
- Region occupancy calculations