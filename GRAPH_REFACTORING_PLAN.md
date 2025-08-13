# NaviGraph Graph Architecture Refactoring Plan

## Overview
Transform NaviGraph from a binary tree-specific system to a generic NetworkX graph framework with pluggable builders and visualizations, completely removing dependency on the old frozenset dictionary mapping approach.

**Key Principles:**
- **No Backward Compatibility**: Old mappings will be obsolete - users must recreate mappings using the new system
- **Library Agnostic Output**: Visualizations output images, not library-specific objects
- **Decorator-Based Registry**: Easy configuration access via class names
- **Default Visualizations**: Base class provides fallback visualization for any graph
- **Minimal Implementation**: Users implement maximum 2 methods
- **Auto Validation**: Constructor parameters provide automatic config validation
- **Phased Approach**: Graph structure → GUI → Mapping → Pipeline integration

---

## Phase 1: Core Graph Builder Architecture (Days 1-2)

### 1.1 Abstract Graph Builder Base Class

**File**: `navigraph/core/graph/builders/base.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import networkx as nx
import numpy as np

class GraphBuilder(ABC):
    """Abstract base class for all graph builders."""
    
    @abstractmethod
    def build_graph(self) -> nx.Graph:
        """Build and return NetworkX graph. Must be implemented by subclasses."""
        pass
    
    def get_visualization(self, 
                         highlight_nodes: Optional[List[Any]] = None,
                         highlight_edges: Optional[List[Tuple[Any, Any]]] = None,
                         node_colors: Optional[Dict[Any, str]] = None,
                         edge_colors: Optional[Dict[Tuple[Any, Any], str]] = None,
                         positions: Optional[Dict[Any, Tuple[float, float]]] = None,
                         **kwargs) -> np.ndarray:
        """
        Generate visualization as image array.
        
        Args:
            highlight_nodes: Nodes to highlight in different color
            highlight_edges: Edges to highlight in different color
            node_colors: Custom color for each node {node: color}
            edge_colors: Custom color for each edge {edge: color}
            positions: Node positions {node: (x, y)}. If None, uses default layout
            **kwargs: Additional visualization parameters
            
        Returns:
            Image as numpy array (H, W, 3) in RGB format
        """
        return self._default_visualization(
            highlight_nodes=highlight_nodes,
            highlight_edges=highlight_edges, 
            node_colors=node_colors,
            edge_colors=edge_colors,
            positions=positions,
            **kwargs
        )
    
    def _default_visualization(self, 
                              highlight_nodes: Optional[List[Any]] = None,
                              highlight_edges: Optional[List[Tuple[Any, Any]]] = None,
                              node_colors: Optional[Dict[Any, str]] = None,
                              edge_colors: Optional[Dict[Tuple[Any, Any], str]] = None,
                              positions: Optional[Dict[Any, Tuple[float, float]]] = None,
                              **kwargs) -> np.ndarray:
        """
        Default graph visualization using matplotlib backend.
        Subclasses can override get_visualization() for custom rendering.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from io import BytesIO
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get graph
        graph = self.build_graph()
        
        # Use provided positions or generate default layout
        if positions is None:
            positions = self._get_default_positions(graph)
        
        # Draw edges
        for edge in graph.edges():
            start_pos = positions[edge[0]]
            end_pos = positions[edge[1]]
            
            # Determine edge color
            if edge_colors and edge in edge_colors:
                color = edge_colors[edge]
            elif highlight_edges and edge in highlight_edges:
                color = 'red'
            else:
                color = 'gray'
            
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   color=color, linewidth=2, alpha=0.7)
        
        # Draw nodes
        for node in graph.nodes():
            pos = positions[node]
            
            # Determine node color
            if node_colors and node in node_colors:
                color = node_colors[node]
            elif highlight_nodes and node in highlight_nodes:
                color = 'orange'
            else:
                color = 'lightblue'
            
            circle = patches.Circle(pos, 0.03, facecolor=color, edgecolor='black')
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], str(node), ha='center', va='center', fontsize=8)
        
        # Configure plot
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Convert to numpy array
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        import PIL.Image
        img = PIL.Image.open(buf)
        img_array = np.array(img)[:, :, :3]  # Remove alpha if present
        
        plt.close(fig)
        return img_array
    
    def _get_default_positions(self, graph: nx.Graph) -> Dict[Any, Tuple[float, float]]:
        """Generate default positions using spring layout."""
        if len(graph.nodes()) > 0:
            pos = nx.spring_layout(graph, seed=42)
            return pos
        return {}
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GraphBuilder':
        """Create builder instance from configuration dictionary.
        
        This method enables automatic parameter validation - if required
        parameters are missing from config, the constructor will fail
        with a clear error message.
        """
        return cls(**config)
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Return builder metadata for storage."""
        # Get constructor parameters by inspecting the instance
        import inspect
        
        # Get constructor signature
        sig = inspect.signature(self.__class__.__init__)
        config = {}
        
        # Extract current values of constructor parameters
        for param_name in sig.parameters:
            if param_name != 'self' and hasattr(self, param_name):
                config[param_name] = getattr(self, param_name)
        
        return {
            'builder_class': self.__class__.__name__,
            'config': config
        }
```

### 1.2 Decorator-Based Registry System

**File**: `navigraph/core/graph/builders/registry.py`

```python
from typing import Dict, Type, Any
from .base import GraphBuilder

# Global registry
_GRAPH_BUILDERS: Dict[str, Type[GraphBuilder]] = {}

def register_graph_builder(name: str):
    """Decorator to register graph builder classes."""
    def decorator(cls: Type[GraphBuilder]):
        if not issubclass(cls, GraphBuilder):
            raise ValueError(f"Class {cls} must inherit from GraphBuilder")
        _GRAPH_BUILDERS[name] = cls
        return cls
    return decorator

def get_graph_builder(name: str) -> Type[GraphBuilder]:
    """Get graph builder class by name."""
    if name not in _GRAPH_BUILDERS:
        available = list(_GRAPH_BUILDERS.keys())
        raise ValueError(f"Graph builder '{name}' not found. Available: {available}")
    return _GRAPH_BUILDERS[name]

def list_graph_builders() -> Dict[str, Type[GraphBuilder]]:
    """Get all registered graph builders."""
    return _GRAPH_BUILDERS.copy()

def create_graph_builder(name: str, config: Dict[str, Any]) -> GraphBuilder:
    """Create graph builder instance from name and config.
    
    Uses the from_config class method which automatically validates
    that all required constructor parameters are present in config.
    """
    builder_class = get_graph_builder(name)
    return builder_class.from_config(config)
```

### 1.3 Binary Tree Builder Implementation

**File**: `navigraph/core/graph/builders/binary_tree.py`

```python
import networkx as nx
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from .base import GraphBuilder
from .registry import register_graph_builder

@register_graph_builder("binary_tree")
class BinaryTreeBuilder(GraphBuilder):
    """Builder for binary tree graphs."""
    
    def __init__(self, height: int):
        """Initialize binary tree builder.
        
        Args:
            height: Height of the binary tree (required)
            
        Raises:
            TypeError: If height is not provided
            ValueError: If height < 1
        """
        if height < 1:
            raise ValueError("Tree height must be >= 1")
        
        self.height = height
    
    def build_graph(self) -> nx.Graph:
        """Build binary tree graph."""
        graph = nx.Graph()
        
        # Build tree level by level
        for level in range(self.height):
            num_nodes_at_level = 2 ** level
            
            for node_idx in range(num_nodes_at_level):
                # Create node ID
                if level == 0:
                    node_id = 0
                else:
                    node_id = int(f"{level}{node_idx}")
                
                # Add node with metadata
                graph.add_node(node_id, level=level, position_in_level=node_idx)
                
                # Add edges to children (except for leaf nodes)
                if level < self.height - 1:
                    left_child_idx = node_idx * 2
                    right_child_idx = node_idx * 2 + 1
                    
                    left_child_id = int(f"{level + 1}{left_child_idx}")
                    right_child_id = int(f"{level + 1}{right_child_idx}")
                    
                    graph.add_edge(node_id, left_child_id, child_type='left')
                    graph.add_edge(node_id, right_child_id, child_type='right')
        
        return graph
    
    def get_visualization(self, 
                         highlight_nodes: Optional[List[Any]] = None,
                         highlight_edges: Optional[List[Tuple[Any, Any]]] = None,
                         node_colors: Optional[Dict[Any, str]] = None,
                         edge_colors: Optional[Dict[Tuple[Any, Any], str]] = None,
                         positions: Optional[Dict[Any, Tuple[float, float]]] = None,
                         **kwargs) -> np.ndarray:
        """
        Custom binary tree visualization with hierarchical layout.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from io import BytesIO
        
        # Create figure with specific styling for binary trees
        fig, ax = plt.subplots(figsize=(12, 8))
        
        graph = self.build_graph()
        
        # Use provided positions or calculate hierarchical positions
        if positions is None:
            positions = self._get_hierarchical_positions(graph)
        
        # Draw edges with hierarchical styling
        for edge in graph.edges():
            start_pos = positions[edge[0]]
            end_pos = positions[edge[1]]
            
            if edge_colors and edge in edge_colors:
                color = edge_colors[edge]
            elif highlight_edges and edge in highlight_edges:
                color = 'red'
            else:
                color = 'darkblue'
            
            width = 3 if (highlight_edges and edge in highlight_edges) else 2
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   color=color, linewidth=width, alpha=0.8)
        
        # Draw nodes with level-based styling
        for node in graph.nodes():
            pos = positions[node]
            level = graph.nodes[node]['level']
            
            if node_colors and node in node_colors:
                color = node_colors[node]
            elif highlight_nodes and node in highlight_nodes:
                color = 'orange'
            else:
                # Different colors per level
                colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 
                         'lightpink', 'lightgray', 'lightcyan']
                color = colors[level % len(colors)]
            
            circle = patches.Circle(pos, 0.04, facecolor=color, 
                                  edgecolor='black', linewidth=1.5)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], str(node), ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"Binary Tree (Height: {self.height})", 
                    fontsize=14, fontweight='bold')
        
        # Convert to numpy array
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        import PIL.Image
        img = PIL.Image.open(buf)
        img_array = np.array(img)[:, :, :3]
        
        plt.close(fig)
        return img_array
    
    def _get_hierarchical_positions(self, graph: nx.Graph) -> Dict[Any, Tuple[float, float]]:
        """Calculate hierarchical positions for binary tree."""
        positions = {}
        
        for node in graph.nodes():
            level = graph.nodes[node]['level']
            pos_in_level = graph.nodes[node]['position_in_level']
            
            # Calculate x position to spread nodes evenly at each level
            num_at_level = 2 ** level
            x = (pos_in_level + 0.5) / num_at_level
            y = 1.0 - (level / (self.height - 1)) if self.height > 1 else 0.5
            
            positions[node] = (x, y)
        
        return positions
```

### 1.4 Example Grid Builder

**File**: `navigraph/core/graph/builders/grid.py`

```python
import networkx as nx
from typing import Dict, Any, Tuple, Optional
from .base import GraphBuilder
from .registry import register_graph_builder

@register_graph_builder("grid")
class GridBuilder(GraphBuilder):
    """Builder for grid graphs."""
    
    def __init__(self, rows: int, cols: int, connectivity: int = 4):
        """Initialize grid builder.
        
        Args:
            rows: Number of rows (required)
            cols: Number of columns (required) 
            connectivity: 4 or 8 connectivity (optional, default=4)
            
        Raises:
            TypeError: If rows or cols not provided
            ValueError: If invalid parameters
        """
        if rows < 1 or cols < 1:
            raise ValueError("Rows and cols must be >= 1")
        if connectivity not in [4, 8]:
            raise ValueError("Connectivity must be 4 or 8")
            
        self.rows = rows
        self.cols = cols
        self.connectivity = connectivity
    
    def build_graph(self) -> nx.Graph:
        """Build grid graph."""
        if self.connectivity == 4:
            graph = nx.grid_2d_graph(self.rows, self.cols)
        else:
            # 8-connected grid
            graph = nx.grid_2d_graph(self.rows, self.cols)
            # Add diagonal connections
            for r in range(self.rows):
                for c in range(self.cols):
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            graph.add_edge((r, c), (nr, nc))
        
        return graph
```

---

## Phase 2: GUI Integration (Days 2-3)

### 2.1 Enhanced GraphStructure Class

**File**: `navigraph/core/graph/structures.py` (major refactor)

```python
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
import numpy as np
from .builders.base import GraphBuilder
from .builders.registry import create_graph_builder

class GraphStructure:
    """Enhanced graph structure using builder pattern."""
    
    def __init__(self, builder: GraphBuilder):
        self.builder = builder
        self._graph = None
    
    @classmethod
    def from_config(cls, builder_name: str, config: Dict[str, Any]) -> 'GraphStructure':
        """Create GraphStructure from builder configuration.
        
        Automatic validation: If config is missing required parameters,
        the builder constructor will raise TypeError with clear message.
        """
        builder = create_graph_builder(builder_name, config)
        return cls(builder)
    
    @property
    def graph(self) -> nx.Graph:
        """Get NetworkX graph (cached)."""
        if self._graph is None:
            self._graph = self.builder.build_graph()
        return self._graph
    
    @property
    def nodes(self) -> List[Any]:
        """Get list of all nodes."""
        return list(self.graph.nodes())
    
    @property
    def edges(self) -> List[Tuple[Any, Any]]:
        """Get list of all edges."""
        return list(self.graph.edges())
    
    def get_visualization(self, **kwargs) -> np.ndarray:
        """Get graph visualization as image array."""
        return self.builder.get_visualization(**kwargs)
    
    def invalidate_cache(self):
        """Clear cached graph."""
        self._graph = None
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update builder configuration and invalidate cache."""
        # Create new builder with updated config
        builder_name = self.builder.__class__.__name__.lower().replace('builder', '')
        self.builder = create_graph_builder(builder_name, new_config)
        self.invalidate_cache()
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get builder metadata."""
        return self.builder.metadata
```

### 2.2 GUI Integration Plan

**File**: `navigraph/core/graph/setup_gui_qt.py` (enhancements)

**Key Changes:**
1. **Builder Selection Widget**: Dropdown with all registered builders
2. **Dynamic Configuration Panel**: Auto-generate UI based on builder requirements  
3. **Live Graph Preview**: Show visualization using builder's render method
4. **Automatic Validation**: Constructor failures provide clear error messages

**New GUI Components:**

```python
class GraphBuilderWidget(QWidget):
    """Widget for selecting and configuring graph builders."""
    
    def __init__(self):
        super().__init__()
        self.current_structure = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Builder selection
        self.builder_combo = QComboBox()
        self.populate_builders()
        self.builder_combo.currentTextChanged.connect(self.on_builder_changed)
        
        # Dynamic config panel
        self.config_panel = QWidget()
        self.config_layout = QFormLayout(self.config_panel)
        
        # Preview button
        self.preview_btn = QPushButton("Preview Graph")
        self.preview_btn.clicked.connect(self.show_preview)
        
        # Error display
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: red; font-weight: bold;")
        self.error_label.hide()
        
        layout.addWidget(QLabel("Graph Type:"))
        layout.addWidget(self.builder_combo)
        layout.addWidget(self.config_panel)
        layout.addWidget(self.preview_btn)
        layout.addWidget(self.error_label)
        
        self.setLayout(layout)
        
        # Initialize with first builder
        if self.builder_combo.count() > 0:
            self.on_builder_changed(self.builder_combo.currentText())
    
    def populate_builders(self):
        """Populate combo box with registered builders."""
        from .builders.registry import list_graph_builders
        builders = list_graph_builders()
        self.builder_combo.addItems(builders.keys())
    
    def on_builder_changed(self, builder_name: str):
        """Handle builder selection change."""
        self.update_config_panel(builder_name)
        self.update_graph_structure()
    
    def update_config_panel(self, builder_name: str):
        """Dynamically create configuration UI for selected builder."""
        # Clear existing widgets
        while self.config_layout.count():
            child = self.config_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Get builder class to inspect constructor
        from .builders.registry import get_graph_builder
        import inspect
        
        builder_class = get_graph_builder(builder_name)
        sig = inspect.signature(builder_class.__init__)
        
        # Create UI for each constructor parameter
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # Create appropriate widget based on parameter type and default
            if param.annotation == int or param.default == inspect.Parameter.empty:
                widget = QSpinBox()
                widget.setRange(1, 50)
                
                # Set default value if available
                if param.default != inspect.Parameter.empty:
                    widget.setValue(param.default)
                elif param_name == 'height':
                    widget.setValue(7)  # Binary tree default
                elif param_name in ['rows', 'cols']:
                    widget.setValue(5)  # Grid defaults
                
                widget.valueChanged.connect(self.update_graph_structure)
                
            elif param.annotation == str:
                widget = QLineEdit()
                if param.default != inspect.Parameter.empty:
                    widget.setText(str(param.default))
                widget.textChanged.connect(self.update_graph_structure)
            
            else:
                # Default to line edit for unknown types
                widget = QLineEdit()
                if param.default != inspect.Parameter.empty:
                    widget.setText(str(param.default))
                widget.textChanged.connect(self.update_graph_structure)
            
            # Add to layout with proper label
            label = param_name.replace('_', ' ').title()
            if param.default == inspect.Parameter.empty:
                label += " *"  # Mark required parameters
            
            self.config_layout.addRow(f"{label}:", widget)
    
    def get_current_config(self) -> Dict[str, Any]:
        """Extract configuration from UI widgets."""
        config = {}
        
        for i in range(self.config_layout.rowCount()):
            label_item = self.config_layout.itemAt(i, QFormLayout.LabelRole)
            field_item = self.config_layout.itemAt(i, QFormLayout.FieldRole)
            
            if label_item and field_item and label_item.widget() and field_item.widget():
                label_text = label_item.widget().text()
                param_name = label_text.rstrip(':').rstrip(' *').lower().replace(' ', '_')
                widget = field_item.widget()
                
                if isinstance(widget, QSpinBox):
                    config[param_name] = widget.value()
                elif isinstance(widget, QLineEdit):
                    text = widget.text().strip()
                    if text:  # Only include non-empty values
                        # Try to convert to int if it looks like a number
                        try:
                            config[param_name] = int(text)
                        except ValueError:
                            config[param_name] = text
        
        return config
    
    def update_graph_structure(self):
        """Update the current graph structure based on UI."""
        builder_name = self.builder_combo.currentText()
        config = self.get_current_config()
        
        try:
            self.current_structure = GraphStructure.from_config(builder_name, config)
            self.error_label.hide()
            self.preview_btn.setEnabled(True)
            
        except Exception as e:
            # Show clear error message from constructor
            self.error_label.setText(f"Configuration Error: {str(e)}")
            self.error_label.show()
            self.current_structure = None
            self.preview_btn.setEnabled(False)
    
    def show_preview(self):
        """Show graph preview in popup window."""
        if self.current_structure is None:
            return
        
        try:
            image = self.current_structure.get_visualization()
            
            # Show in popup window
            dialog = GraphPreviewDialog(image, self)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Failed to generate preview: {e}")
```

---

## Phase 3: New Mapping System Integration (Days 3-4)

### 3.1 Enhanced SpatialMapping for Generic Graphs

**File**: `navigraph/core/graph/mapping.py` (enhancements)

```python
import pickle
from pathlib import Path
from typing import Dict, Any

class SpatialMapping:
    """Enhanced spatial mapping supporting any graph type."""
    
    def __init__(self, graph_structure: GraphStructure, unmapped_value=None):
        self.graph_structure = graph_structure
        self.unmapped_value = unmapped_value
        
        # Internal mappings (no more frozensets!)
        self._node_to_regions = {}
        self._edge_to_regions = {}
        self._region_to_element = {}
        self._regions = {}
    
    @property
    def graph(self) -> nx.Graph:
        """Get the underlying NetworkX graph."""
        return self.graph_structure.graph
    
    def get_builder_info(self) -> Dict[str, Any]:
        """Get information about the graph builder."""
        return self.graph_structure.metadata
    
    def save_with_builder_info(self, file_path: Path):
        """Save mapping with builder information for reconstruction."""
        data = {
            'graph_builder': self.graph_structure.metadata,
            'mapping': self,
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load_with_builder_reconstruction(cls, file_path: Path) -> 'SpatialMapping':
        """Load mapping and reconstruct graph using builder information."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct graph structure from builder info
        builder_info = data['graph_builder']
        builder_class_name = builder_info['builder_class']
        builder_config = builder_info['config']
        
        # Convert class name to registry name 
        # BinaryTreeBuilder -> binary_tree
        registry_name = builder_class_name.lower().replace('builder', '')
        
        # Recreate graph structure
        graph_structure = GraphStructure.from_config(registry_name, builder_config)
        
        # Restore mapping
        mapping = data['mapping']
        mapping.graph_structure = graph_structure
        
        return mapping
```

---

## Phase 4: Pipeline Integration (Days 4-5)

### 4.1 Updated Graph Integration Plugin

**File**: `navigraph/plugins/shared_resources/graph_integration.py` (complete rewrite)

```python
from navigraph.core.base_plugin import BasePlugin
from navigraph.core.graph.structures import GraphStructure
from navigraph.core.graph.mapping import SpatialMapping

class GraphIntegrationPlugin(BasePlugin):
    """Graph integration using builder system."""
    
    def process(self, session_data):
        """Process graph integration for session."""
        
        # Get builder configuration from main config
        builder_config = self.config.get('graph_builder', {})
        builder_name = builder_config.get('type', 'binary_tree')
        builder_params = builder_config.get('config', {})
        
        try:
            # Create graph structure (automatic validation)
            graph_structure = GraphStructure.from_config(builder_name, builder_params)
        except Exception as e:
            raise ValueError(f"Failed to create graph structure: {e}")
        
        # Load spatial mapping
        mapping_path = self.config.get('mapping_file')
        if not mapping_path:
            raise ValueError("No mapping file specified in configuration")
        
        try:
            mapping = SpatialMapping.load_with_builder_reconstruction(mapping_path)
        except Exception as e:
            raise ValueError(f"Failed to load mapping: {e}")
        
        # Verify mapping compatibility with current graph
        if not self._verify_mapping_compatibility(mapping, graph_structure):
            mapping_info = mapping.get_builder_info()
            current_info = graph_structure.metadata
            
            raise ValueError(
                f"Loaded mapping is not compatible with current graph structure.\n\n"
                f"Mapping was created for:\n"
                f"  Builder: {mapping_info['builder_class']}\n"
                f"  Config: {mapping_info['config']}\n\n"
                f"Current configuration specifies:\n"
                f"  Builder: {current_info['builder_class']}\n" 
                f"  Config: {current_info['config']}\n\n"
                f"Please recreate the mapping using the setup GUI with the correct graph configuration."
            )
        
        # Store in session
        session_data.shared_data['graph_structure'] = graph_structure
        session_data.shared_data['spatial_mapping'] = mapping
        
        return {'status': 'success', 'graph_type': builder_name}
    
    def _verify_mapping_compatibility(self, mapping: SpatialMapping, 
                                    graph_structure: GraphStructure) -> bool:
        """Verify that mapping is compatible with graph structure."""
        
        # Check if nodes and edges match exactly
        mapping_nodes = set(mapping.graph.nodes())
        structure_nodes = set(graph_structure.nodes)
        
        mapping_edges = set(mapping.graph.edges())
        structure_edges = set(graph_structure.edges)
        
        return (mapping_nodes == structure_nodes and 
                mapping_edges == structure_edges)
```

---

## Configuration Updates

### New Configuration Schema

```yaml
# New graph_builder section replaces old graph section
graph_builder:
  type: binary_tree  # Registered builder name
  config:
    height: 7         # Required parameter for binary_tree
    # Any other builder-specific parameters

# Example for grid builder
# graph_builder:
#   type: grid
#   config:
#     rows: 5           # Required
#     cols: 5           # Required  
#     connectivity: 4   # Optional (defaults to 4)

# Mapping configuration (unchanged)
graph_mapping:
  mapping_file: ./resources/graph_mapping.pkl
```

---

## CLI Updates

### New CLI Command for Listing Builders

**File**: `navigraph/cli/main.py` (addition)

```python
@cli.command('list-graph-builders')
@click.option('--format', '-f', 
              type=click.Choice(['table', 'json', 'simple']),
              default='table',
              help='Output format')
def list_graph_builders(format: str):
    """List all available graph builders.
    
    Display registered graph builders that can be used in configuration.
    Use this to discover available graph types for your experiments.
    
    Examples:
      navigraph list-graph-builders
      navigraph list-graph-builders --format json
    """
    try:
        # Import to ensure builders are registered
        from ..core.graph.builders import binary_tree, grid  # noqa: F401
        from ..core.graph.builders.registry import list_graph_builders
        
        builders = list_graph_builders()
        
        if not builders:
            click.echo("⚠️  No graph builders found")
            return
        
        if format == 'simple':
            for name in builders.keys():
                click.echo(name)
        
        elif format == 'json':
            import json
            import inspect
            
            output = {'available_builders': []}
            
            for name, builder_class in builders.items():
                # Get constructor parameters
                sig = inspect.signature(builder_class.__init__)
                params = {}
                
                for param_name, param in sig.parameters.items():
                    if param_name != 'self':
                        param_info = {
                            'name': param_name,
                            'required': param.default == inspect.Parameter.empty
                        }
                        if param.default != inspect.Parameter.empty:
                            param_info['default'] = param.default
                        params[param_name] = param_info
                
                builder_info = {
                    'name': name,
                    'class': builder_class.__name__,
                    'parameters': params
                }
                
                if builder_class.__doc__:
                    builder_info['description'] = builder_class.__doc__.strip().split('\n')[0]
                
                output['available_builders'].append(builder_info)
            
            output['total_count'] = len(builders)
            click.echo(json.dumps(output, indent=2))
        
        else:  # table format
            import inspect
            
            click.echo()
            click.echo("📊 Available Graph Builders")
            click.echo("=" * 50)
            
            for name, builder_class in builders.items():
                click.echo(f"\n🔹 {name}")
                
                # Show description
                if builder_class.__doc__:
                    doc_lines = builder_class.__doc__.strip().split('\n')
                    if doc_lines:
                        click.echo(f"   {doc_lines[0]}")
                
                # Show parameters
                sig = inspect.signature(builder_class.__init__)
                params = []
                
                for param_name, param in sig.parameters.items():
                    if param_name != 'self':
                        if param.default == inspect.Parameter.empty:
                            params.append(f"{param_name} (required)")
                        else:
                            params.append(f"{param_name} (default: {param.default})")
                
                if params:
                    click.echo(f"   Parameters: {', '.join(params)}")
            
            click.echo()
            click.echo(f"Total: {len(builders)} builders available")
            click.echo()
            click.echo("Usage in config:")
            click.echo("  graph_builder:")
            click.echo("    type: <builder_name>")
            click.echo("    config:")
            click.echo("      <param>: <value>")
            
    except Exception as e:
        click.echo(f"❌ Failed to list graph builders: {str(e)}", err=True)
        sys.exit(1)
```

---

## Q&A Section: Next Steps and Implementation Details

### Q1: How does automatic validation work?

**A**: **Constructor-Based Validation**: Python's built-in parameter checking provides automatic validation:

```python
# Configuration missing required parameter
graph_builder:
  type: binary_tree
  config:
    # height is missing!

# Runtime error (automatic):
TypeError: BinaryTreeBuilder.__init__() missing 1 required positional argument: 'height'

# Configuration with invalid value  
graph_builder:
  type: binary_tree
  config:
    height: -1

# Runtime error (from constructor validation):
ValueError: Tree height must be >= 1
```

**Benefits**:
- No manual validation code needed
- Clear, standard Python error messages
- Automatic discovery of required parameters
- Type checking built-in

### Q2: How does the metadata property work without storing config?

**A**: **Introspection-Based Metadata**: Uses Python's introspection to reconstruct config from instance:

```python
@property
def metadata(self) -> Dict[str, Any]:
    import inspect
    
    # Get constructor signature
    sig = inspect.signature(self.__class__.__init__)
    config = {}
    
    # Extract current values of constructor parameters
    for param_name in sig.parameters:
        if param_name != 'self' and hasattr(self, param_name):
            config[param_name] = getattr(self, param_name)
    
    return {
        'builder_class': self.__class__.__name__,
        'config': config
    }
```

**Example**:
```python
# Builder instance
builder = BinaryTreeBuilder(height=7)

# Metadata automatically extracted
metadata = builder.metadata
# Returns: {'builder_class': 'BinaryTreeBuilder', 'config': {'height': 7}}
```

### Q3: How does the GUI handle parameter discovery?

**A**: **Dynamic UI Generation**: Uses introspection to discover constructor parameters:

```python
import inspect

builder_class = get_graph_builder("binary_tree")
sig = inspect.signature(builder_class.__init__)

for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
        
    # Check if required or optional
    required = param.default == inspect.Parameter.empty
    
    # Create appropriate widget
    if param.annotation == int:
        widget = QSpinBox()
        if not required:
            widget.setValue(param.default)
    
    # Add to UI with required marker
    label = param_name if not required else f"{param_name} *"
    layout.addRow(label, widget)
```

**Result**: GUI automatically shows correct fields for each builder type.

### Q4: How do we ensure the binary tree produces identical graphs?

**A**: **Exact Replication**: The new `BinaryTreeBuilder` generates identical node IDs and connections:

```python
# OLD: Current system node IDs
# Level 0: 0
# Level 1: 10, 11  
# Level 2: 20, 21, 22, 23

# NEW: BinaryTreeBuilder node IDs  
# Level 0: 0
# Level 1: 10, 11
# Level 2: 20, 21, 22, 23

# Same algorithm, same result
```

**Validation**: Run both systems and compare NetworkX graph structure - should be identical.

### Q5: What's the complete error handling flow?

**A**: **Multi-Level Error Handling**:

1. **Configuration Load**: Missing/invalid YAML
2. **Builder Creation**: Missing required parameters → TypeError
3. **Builder Validation**: Invalid parameter values → ValueError  
4. **Graph Building**: Runtime errors in build_graph()
5. **Mapping Load**: File not found, corruption
6. **Compatibility Check**: Graph structure mismatch

**Error Messages**: Each level provides specific, actionable feedback to user.

### Q6: How do we test this systematically?

**A**: **Comprehensive Testing Strategy**:

```python
# Unit Tests
def test_binary_tree_builder():
    builder = BinaryTreeBuilder(height=3)
    graph = builder.build_graph()
    
    # Verify structure
    assert len(graph.nodes()) == 7  # 2^3 - 1
    assert len(graph.edges()) == 6  # n - 1 for tree
    
    # Verify specific nodes exist
    assert 0 in graph.nodes()  # Root
    assert 10 in graph.nodes()  # Level 1
    assert 20 in graph.nodes()  # Level 2

def test_automatic_validation():
    # Missing parameter should raise TypeError
    with pytest.raises(TypeError):
        BinaryTreeBuilder()  # Missing height
    
    # Invalid parameter should raise ValueError
    with pytest.raises(ValueError):
        BinaryTreeBuilder(height=-1)

def test_configuration_loading():
    config = {'height': 5}
    builder = BinaryTreeBuilder.from_config(config)
    assert builder.height == 5

# Integration Tests
def test_gui_parameter_discovery():
    # Test GUI correctly shows required fields
    pass

def test_end_to_end_pipeline():
    # Test complete pipeline with new system
    pass
```

### Q7: What's the migration path for existing users?

**A**: **Clear Migration Steps**:

1. **Update Configuration**: Change `graph:` section to `graph_builder:` section
2. **Recreate Mappings**: Use new GUI to recreate all mappings (old mappings won't work)  
3. **Test Pipeline**: Run analysis and verify identical results
4. **Update Custom Code**: Replace any direct `Graph` usage with `GraphStructure`

**No Automatic Migration**: Clean break ensures no legacy issues.

---

## Implementation Timeline

### Phase 1: Core Architecture (Days 1-2)
- **Day 1**: Base class (no validation), registry, binary tree builder
- **Day 2**: GraphStructure refactor, CLI command, metadata extraction

### Phase 2: GUI Integration (Days 2-3)  
- **Day 2-3**: Dynamic parameter discovery, automatic validation UI
- **Day 3**: Error handling, graph preview integration

### Phase 3: New Mapping (Days 3-4)
- **Day 3-4**: SpatialMapping with builder reconstruction
- **Day 4**: Save/load integration with GUI and pipeline

### Phase 4: Pipeline Integration (Days 4-5)
- **Day 4-5**: Plugin updates, compatibility checking
- **Day 5**: End-to-end testing with clear error messages

### Phase 5: Validation (Days 5-6)
- **Day 5-6**: Binary tree equivalence testing, example recreation
- **Day 6**: Performance testing, documentation

**Total Duration**: 6 days with automatic validation and clear error handling.

---

This refined plan eliminates manual validation and configuration storage, relying instead on Python's built-in mechanisms for cleaner, more maintainable code.