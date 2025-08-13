"""Graph builder that loads from file."""

import networkx as nx
from pathlib import Path
from typing import Optional

from .base import GraphBuilder
from .registry import register_graph_builder


@register_graph_builder("file_loader")
class FileGraphBuilder(GraphBuilder):
    """Builder that loads graphs from files.
    
    This builder is useful for loading pre-existing graphs saved in various formats.
    It can be configured via config files to load graphs from disk.
    """
    
    def __init__(self, filepath: str, format: str = 'graphml'):
        """Initialize file loader builder.
        
        Args:
            filepath: Path to the graph file
            format: File format ('graphml', 'gexf', 'gml', 'adjlist', 'edgelist')
            
        Raises:
            ValueError: If format is not supported
            FileNotFoundError: If file doesn't exist
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        supported_formats = ['graphml', 'gexf', 'gml', 'adjlist', 'edgelist']
        if format not in supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {supported_formats}")
        
        self.format = format
        self._graph: Optional[nx.Graph] = None
    
    def build_graph(self) -> nx.Graph:
        """Load and return the graph from file.
        
        Returns:
            NetworkX graph loaded from file
        """
        if self._graph is not None:
            return self._graph
        
        if self.format == 'graphml':
            self._graph = nx.read_graphml(str(self.filepath))
        elif self.format == 'gexf':
            self._graph = nx.read_gexf(str(self.filepath))
        elif self.format == 'gml':
            self._graph = nx.read_gml(str(self.filepath))
        elif self.format == 'adjlist':
            self._graph = nx.read_adjlist(str(self.filepath))
        elif self.format == 'edgelist':
            self._graph = nx.read_edgelist(str(self.filepath))
        
        return self._graph