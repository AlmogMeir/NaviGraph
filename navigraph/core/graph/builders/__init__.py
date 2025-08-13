"""Graph builders for NaviGraph.

This module provides a pluggable graph builder system that allows users to create
different types of graph structures with minimal implementation burden.

All graph builders are automatically discovered and registered when this module is imported.
"""

from .base import GraphBuilder
from .registry import register_graph_builder, get_graph_builder, list_graph_builders, get_graph_builder_info

# Auto-discover and import all builders
import os
import importlib
from pathlib import Path

# Get the current directory
current_dir = Path(__file__).parent

# Auto-import all python files in this directory
for file_path in current_dir.glob("*.py"):
    module_name = file_path.stem
    
    # Skip special files
    if module_name in ['__init__', 'base', 'registry']:
        continue
    
    # Import the module to trigger decorator registration
    try:
        importlib.import_module(f'.{module_name}', package='navigraph.core.graph.builders')
    except ImportError as e:
        print(f"Warning: Could not import builder module {module_name}: {e}")

__all__ = [
    'GraphBuilder',
    'register_graph_builder', 
    'get_graph_builder',
    'list_graph_builders',
    'get_graph_builder_info'
]