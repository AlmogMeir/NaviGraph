# NaviGraph GUI Expert Reference

## 🎯 Purpose
This document serves as the **definitive expert reference** for the NaviGraph graph setup GUI system. Every future Claude Code session working with the GUI should read this document to understand the intricate system architecture, particularly the **critical scaling mechanisms** that must be preserved.

---

## 🏗️ Architecture Overview

### Core Components

The GUI system consists of three primary classes:

```python
# navigraph/core/graph/setup_gui_qt.py

1. MapWidget(QWidget)           # Interactive map display & manipulation
2. GraphWidget(QWidget)         # Graph structure visualization  
3. GraphSetupWindow(QMainWindow) # Main orchestrating window
```

### Layout Architecture

```
GraphSetupWindow (QMainWindow)
├── QSplitter (Horizontal) - main_splitter
│   ├── Control Panel (Left) - 350-450px width, manually resizable
│   └── QSplitter (Vertical) - views_splitter  
│       ├── Graph View (Top) - GraphWidget
│       └── Map View (Bottom) - MapWidget (2x larger by default)
```

**🔴 CRITICAL**: The dual-splitter system allows **manual resizing** in both directions:
- **Horizontal splitter**: Controls panel width vs views width
- **Vertical splitter**: Controls graph height vs map height
- **Both respond to resize events with proper scaling**

---

## ⚙️ Critical Scaling System

### 🎨 Image Scaling Pipeline (MapWidget)

The map image scaling uses a **dual-factor system**:

```python
# MapWidget scaling state
self.scale_factor = self.base_scale_factor * self.user_scale_factor

# base_scale_factor: Auto-calculated to fit widget (recalculated on resize)
# user_scale_factor: Manual zoom (mouse wheel, 0.1x to 10x range)
```

#### Scaling Calculation Flow:
```python
def paintEvent(self, event):
    # 1. Auto-fit calculation
    available_width = widget_size.width() - 40  # 20px padding each side
    available_height = widget_size.height() - 40
    scale_x = available_width / image_width
    scale_y = available_height / image_height
    self.base_scale_factor = min(scale_x, scale_y, 1.0)  # Don't upscale
    
    # 2. Combine with user zoom
    self.scale_factor = self.base_scale_factor * self.user_scale_factor
    
    # 3. Apply scaling
    final_width = int(image_width * self.scale_factor)
    final_height = int(image_height * self.scale_factor)
    scaled_pixmap = pixmap.scaled(final_width, final_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
```

**🔴 CRITICAL**: Changes to this scaling system can break:
- Coordinate transformations between screen/image space
- Grid placement accuracy
- Contour drawing precision
- Region highlighting alignment

### 🖼️ Graph Visualization Scaling (GraphWidget)

Graph rendering uses **dynamic sizing based on graph type**:

```python
def draw_graph(self):
    # Node size scaling based on graph size
    node_count = len(self.graph.nodes)
    if node_count > 100:
        node_size, font_size = 400, 9    # Increased from 250, 7
    elif node_count > 50:
        node_size, font_size = 600, 10   # Increased from 400, 8
    elif node_count > 20:
        node_size, font_size = 800, 12   # Increased from 600, 10
    else:
        node_size, font_size = 1000, 14  # Increased from 800, 12
    
    # Special handling for binary trees
    if hasattr(self.graph.builder, 'height'):
        tree_height = self.graph.builder.height
        width = max(8, min(25, 2.5 * tree_height))  # Scale: height 7 ≈ 17.5 width
        figsize = (width, 5)
    else:
        figsize = (12, 8)
```

### 📝 Font Scaling System (Window-level)

Dynamic font scaling responds to window size:

```python
def calculate_font_scale(self):
    base_width = 1600  # Original design width
    current_width = self.width()
    
    # Conservative scaling: 0.85x to 1.15x 
    scale = max(0.85, min(1.15, current_width / base_width))
    
    # Cap scaling on very wide screens (>2000px)
    if current_width > 2000:
        scale = min(scale, 1.0)
    
    return scale

def update_font_sizes(self):
    scale = self.calculate_font_scale()
    base_sizes = {'title': 14, 'normal': 13, 'small': 12, 'tiny': 11}
    scaled_sizes = {k: int(v * scale) for k, v in base_sizes.items()}
    # Updates entire stylesheet with scaled fonts
```

**🔴 CRITICAL**: Font scaling is called on every `resizeEvent()` - performance sensitive.

---

## 🎮 Interaction Modes & State Management

### Mode Architecture

The GUI operates in three distinct modes with complete state isolation:

```python
self.setup_mode = None  # 'grid', 'manual', or 'test'
```

### Grid Mode (`'grid'`)

**Purpose**: Map graph elements to rectangular grid cells

**State Flow**:
1. **Configure Grid** → Grid parameters (rows, cols, cell size)
2. **Place Grid** → Click map to position grid origin
3. **Select Cells** → Click cells to select/deselect
4. **Assign Elements** → Map selected cells to current graph element
5. **Repeat** → Next element in queue

**Key State Variables**:
```python
# Grid configuration
self.grid_config = GridConfig(rows=8, cols=8, cell_width=50.0, cell_height=50.0)

# Grid visual state (MapWidget)
self.grid_enabled = False
self.grid_cells: Dict[str, QRectF] = {}  # cell_id -> rectangle
self.selected_cells: Set[str] = set()    # currently selected cells
self.cell_colors: Dict[str, QColor] = {} # persistent cell colors
self.cell_mappings: Dict[str, Tuple[str, Any]] = {}  # cell_id -> (elem_type, elem_id)
```

### Manual Mode (`'manual'`)

**Purpose**: Draw custom contours around map regions

**State Flow**:
1. **Draw Contour** → Left-click to add points
2. **Commit Contour** → Finish drawing and assign to element
3. **Repeat** → Next element via dropdown selection

**Key State Variables**:
```python
# Contour drawing state (MapWidget)
self.current_contour: List[Tuple[float, float]] = []  # points being drawn
self.completed_contours: List[Tuple] = []             # finished contours
self.contour_mappings: Dict[str, Tuple[str, Any]] = {} # region_id -> (elem_type, elem_id)
```

### Test Mode (`'test'`)

**Purpose**: Test and visualize existing mappings

**Interaction Methods**:
- **Click map** → Find which element is mapped at position
- **Select element** → Highlight all regions mapped to that element

**Key Features**:
- Uses `SpatialMapping.query_point(x, y)` for position-to-element lookup
- Purple highlighting for selected elements
- Clean visualization (mappings/labels disabled by default)

### Mode Switching Behavior

**🔴 CRITICAL**: Mode switches trigger complete UI recreation:

```python
def _recreate_fresh_ui(self):
    """Recreate UI widgets to prevent stale references"""
    # This method destroys and rebuilds the entire control panel
    # Essential to prevent RuntimeError exceptions from deleted widgets
```

**Mode Switch Sequence**:
1. Check for unsaved progress → Confirm with user
2. Clear current mode state → `_reset_mapping_state()`
3. Recreate UI → `_recreate_fresh_ui()`
4. Initialize new mode → Mode-specific setup
5. Set visualization defaults → `_set_mode_specific_defaults()`

---

## 🧠 State Management Deep Dive

### Element Queue System

Graph elements are processed in a systematic order:

```python
def _init_element_queue(self):
    # Numeric sorting for consistent order
    def numeric_sort_key(x):
        try:
            return int(x)
        except (ValueError, TypeError):
            return float('inf')
    
    # Add all nodes (sorted)
    for node in sorted(self.graph.nodes, key=numeric_sort_key):
        self.element_queue.append(('node', node))
        self.all_elements.append(('node', node))
    
    # Add all edges (sorted by both nodes)
    for edge in sorted(self.graph.edges, key=edge_sort_key):
        self.element_queue.append(('edge', edge))
        self.all_elements.append(('edge', edge))
```

### Mapping State

The core mapping is managed by `SpatialMapping` class:

```python
self.mapping = SpatialMapping(graph)  # Core spatial mapping object
self.mapped_elements = {}             # element -> region_ids (UI tracking)
self.mapping_history = []            # Undo stack
```

### Progress Tracking

```python
def _update_progress_display(self):
    stats = self.mapping.validate_mapping()
    progress_value = stats.mapped_nodes + stats.mapped_edges
    total_elements = len(self.graph.nodes) + len(self.graph.edges)
    completion = (progress_value / total_elements) * 100
```

---

## 🎯 Coordinate Transformations

### Screen ↔ Image Space

**MapWidget** handles multiple coordinate spaces:

```python
# Screen to Image coordinates
def mousePressEvent(self, event):
    # Convert screen position to image coordinates
    x = (event.x() - self.offset_x) / self.scale_factor
    y = (event.y() - self.offset_y) / self.scale_factor

# Image to Screen coordinates  
def paintEvent(self, event):
    # Convert image coordinates to screen for drawing
    screen_x = self.offset_x + image_x * self.scale_factor
    screen_y = self.offset_y + image_y * self.scale_factor
```

**🔴 CRITICAL**: All coordinate transforms depend on:
- `self.offset_x, self.offset_y` (panning offset)
- `self.scale_factor` (combined scaling factor)

### Panning & Zooming

**Mouse Wheel Zooming**:
```python
def wheelEvent(self, event):
    # Get mouse position in image coordinates BEFORE zoom
    image_x = (mouse_x - self.offset_x) / self.scale_factor
    image_y = (mouse_y - self.offset_y) / self.scale_factor
    
    # Apply zoom (0.1x to 10x limits)
    zoom_factor = 1.1 ** (event.angleDelta().y() / 120.0)
    self.user_scale_factor *= zoom_factor
    
    # Adjust offsets to keep mouse position fixed
    new_offset_x = mouse_x - image_x * self.scale_factor
    new_offset_y = mouse_y - image_y * self.scale_factor
```

**Right-Click Panning**:
```python
def mouseMoveEvent(self, event):
    if self.panning:
        delta = event.pos() - self.last_pan_point
        self.offset_x += delta.x()
        self.offset_y += delta.y()
        self.last_pan_point = event.pos()
```

---

## 🎨 Rendering Pipeline

### Region Visualization

The system renders multiple types of regions with specific visual encoding:

```python
def paintEvent(self, event):
    # 1. Draw map image (scaled and positioned)
    
    # 2. Draw grid (if enabled)
    #    - Empty cells: gray outline
    #    - Selected cells: bright green fill
    #    - Mapped cells: persistent color fill
    
    # 3. Draw regions from mapping (from SpatialMapping)
    #    - Node regions: light green (150, 255, 150, 100)
    #    - Edge regions: orange (255, 165, 0, 100)
    
    # 4. Draw current contour being drawn (red)
    
    # 5. Draw element highlights (purple)
    
    # 6. Draw test mode indicators
```

### Adaptive Text Rendering

Text sizing adapts to available space:

```python
def _draw_text_in_contour(self, painter, polygon, text):
    # Adaptive mode: Scale font to fit within contour bounds
    available_width = bounding_rect.width() * 0.8
    available_height = bounding_rect.height() * 0.8
    
    # Try font sizes from 20px down to 5px
    for size in range(20, 4, -1):
        font.setPointSize(size)
        metrics = QFontMetrics(font)
        text_rect = metrics.boundingRect(text)
        
        if text_rect.width() <= available_width and text_rect.height() <= available_height:
            break
```

---

## ⚠️ Common Pitfalls & Solutions

### 1. Widget Lifecycle Issues

**Problem**: RuntimeError when accessing deleted widgets after mode switches

**Solution**: Always check widget validity before operations:
```python
try:
    if hasattr(self, 'widget') and self.widget is not None:
        self.widget.isVisible()  # Raises RuntimeError if deleted
        # Safe to use widget
except RuntimeError:
    # Widget deleted, skip operation
    pass
```

### 2. Scaling Consistency

**Problem**: Coordinates become misaligned after scaling changes

**Solution**: Always recalculate dependent coordinates:
```python
def resizeEvent(self, event):
    super().resizeEvent(event)
    # base_scale_factor recalculated in paintEvent()
    self.update()  # Triggers paintEvent with new scaling
```

### 3. State Synchronization

**Problem**: UI state doesn't match underlying data state

**Solution**: Use centralized update methods:
```python
def _update_progress_display(self):
    # Single source of truth for all progress UI elements
    stats = self.mapping.validate_mapping()
    # Update progress bar, labels, button states consistently
```

### 4. Performance Issues

**Problem**: Slow redraws during frequent updates (zoom, pan)

**Solutions**:
- Cache pixmaps where possible
- Use `QPixmap.scaled()` with `Qt.SmoothTransformation` sparingly
- Implement tooltip debouncing
- Prevent recursive drawing with `self._drawing` flag

### 5. Event Handling Conflicts

**Problem**: Multiple event handlers interfering with each other

**Solution**: Proper event propagation and mode checking:
```python
def mousePressEvent(self, event):
    if self.interaction_mode == 'place_grid':
        # Handle grid placement
        return  # Don't propagate
    elif self.interaction_mode == 'draw_contour':
        # Handle contour drawing
        return  # Don't propagate
    # ... other modes
```

---

## 🔧 Key Methods Reference

### MapWidget Critical Methods

```python
# Scaling and positioning
def paintEvent(self, event)           # Main rendering pipeline
def resizeEvent(self, event)          # Recalculate scaling on resize
def reset_zoom(self)                  # Reset to fit-to-window
def wheelEvent(self, event)           # Mouse wheel zoom with fixed point

# Coordinate transformations
def mousePressEvent(self, event)      # Screen → image coordinates
def mouseMoveEvent(self, event)       # Panning + tooltips

# State management
def set_interaction_mode(self, mode)  # Change interaction behavior
def reset_all(self)                   # Clear all visual elements
def set_current_element(self, type, id) # Set element being mapped

# Grid system
def enable_grid(self, x, y)          # Place and enable grid
def _generate_grid(self)             # Create grid cells
def select_cells(self, cell_ids)     # Select specific cells

# Contour system
def add_contour_point(self, x, y)    # Add point to current contour
def finish_current_contour(self)     # Complete contour drawing
def add_contour(self, points, ...)   # Add completed contour
```

### GraphWidget Critical Methods

```python
def draw_graph(self)                 # Render graph with current colors/sizes
def highlight_nodes(self, nodes, color) # Highlight specific nodes
def highlight_edges(self, edges, color) # Highlight specific edges
def clear_highlights(self)           # Remove all highlighting
def resizeEvent(self, event)         # Update graph visualization on resize
```

### GraphSetupWindow Critical Methods

```python
# Mode management
def _on_grid_mode(self, checked)     # Switch to grid mode
def _on_manual_mode(self, checked)   # Switch to manual mode  
def _on_test_mode(self, checked)     # Switch to test mode
def _reset_mapping_state(self)       # Clear all state
def _recreate_fresh_ui(self)         # Rebuild UI widgets

# Scaling system
def calculate_font_scale(self)       # Window size → font scale factor
def update_font_sizes(self)          # Apply scaled fonts to all UI
def resizeEvent(self, event)         # Handle window resize

# Element management
def _init_element_queue(self)        # Setup element processing order
def _select_next_element(self)       # Move to next element
def _populate_element_combos(self)   # Update dropdown selections
def _update_progress_display(self)   # Update progress indicators

# Splitter management
def _set_initial_splitter_sizes(self) # Set proportional splitter sizes
# Note: Splitters handle their own resize behavior
```

---

## 📱 User Interface Behavior

### Splitter Controls (Manual Resize)

**Horizontal Splitter** (`main_splitter`):
- Controls separation between control panel and views
- Control panel: 350-450px width (manually adjustable)
- Views area gets remaining space
- Responds to window resize by maintaining proportions

**Vertical Splitter** (`views_splitter`):
- Controls separation between graph and map views  
- Default ratio: 1:2 (map larger than graph)
- User can manually adjust to any proportion
- Both views scale their content appropriately

### Keyboard Shortcuts

```python
F11           # Toggle fullscreen mode
Ctrl+M        # Toggle maximize window  
R (when focused on map) # Reset zoom to fit image
```

### Mouse Controls

**Map Widget**:
- **Left click**: Mode-specific action (place grid, select cells, draw contour, test)
- **Right drag**: Pan the image
- **Mouse wheel**: Zoom in/out (0.1x to 10x range)
- **Hover**: Show tooltips for mapped regions

**Graph Widget**:
- **Display only**: No direct interaction (selection via dropdowns in test mode)

### Visual Feedback

**Color Coding**:
- **Bright green**: Currently selected cells/contours  
- **Light green**: Mapped node regions
- **Orange**: Mapped edge regions
- **Purple**: Highlighted elements in test mode
- **Yellow**: Region selection highlighting
- **Red**: Current contour being drawn
- **Gray**: Empty grid cells, default text

---

## 🚨 Critical Preservation Points

### What Must NOT Be Changed

1. **Dual-scale system** in MapWidget - breaks coordinate transformations
2. **Widget recreation** on mode switch - prevents RuntimeError crashes  
3. **Font scaling calculation** - maintains UI proportions across screen sizes
4. **Splitter architecture** - allows manual horizontal/vertical adjustment
5. **Coordinate transformation logic** - screen/image space conversions
6. **Element queue ordering** - ensures consistent processing sequence

### Safe Modification Areas

1. **Visual styling** - colors, borders, spacing (within reason)
2. **Button layouts** - positions and groupings (preserving functionality)
3. **Status messages** - text content and formatting
4. **Validation logic** - mapping completeness checks
5. **Progress indicators** - display format and information

### High-Risk Modification Areas  

1. **Scaling calculations** - require extensive testing across window sizes
2. **Event handlers** - can break interaction modes
3. **State management** - can cause inconsistent UI behavior
4. **Memory management** - widget lifecycle and cleanup

---

## 🎓 Session Handoff Checklist

When starting GUI work in a new session:

✅ **Read this document completely**  
✅ **Understand the dual-scale system**  
✅ **Know the three interaction modes**  
✅ **Verify manual splitter controls work**  
✅ **Test image scaling across window sizes**  
✅ **Check coordinate transformation accuracy**  
✅ **Confirm mode switching preserves state**  

Remember: This GUI system is **intricate and delicate**. Small changes can have cascading effects. Always test thoroughly after modifications, especially anything related to scaling, coordinate transformations, or widget lifecycle management.