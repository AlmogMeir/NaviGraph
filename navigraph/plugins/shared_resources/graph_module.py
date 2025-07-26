"""Graph module for NaviGraph shared resources.

This module contains the Graph class and related utilities migrated from
the original graph package to make the plugin system self-contained.
"""

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Union, Dict, Any

# Configure matplotlib for non-interactive use
plt.rcParams["figure.figsize"] = (35, 20)


class Graph:
    """Binary tree graph representation for spatial navigation analysis.
    
    This class creates and manages a binary tree structure used for
    topological analysis of navigation patterns in maze experiments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize graph with configuration.
        
        Args:
            config: Dictionary with graph configuration including:
                - verbose: Enable verbose logging
                - graph.height: Height of the binary tree
                - graph.draw: Drawing parameters
                - graph.options: Color and style options
        """
        self._cfg = config
        self.tree = self._build_binary_tree(self.cfg.get('height', 7))
        self._ax = self.set_axes()
        self.draw_base_tree()
    
    @property
    def cfg(self) -> Dict[str, Any]:
        """Get graph configuration."""
        return self._cfg
    
    def set_axes(self):
        """Set up matplotlib axes for graph visualization."""
        _ax = plt.axes()
        _ax.axis('off')
        _ax.figure.tight_layout(pad=0)
        _ax.margins(0)
        return _ax
    
    def _build_binary_tree(self, height: int, weight=lambda p_node, c_node: 1):
        """Build a binary tree as a networkx graph.
        
        Node naming convention: 2-digit number where first digit is level
        (starting from 0) and second digit is node ID within that level.
        
        Args:
            height: Height of the binary tree
            weight: Function to calculate edge weights (default: constant 1)
            
        Returns:
            NetworkX graph representing the binary tree
        """
        g = nx.Graph()
        x_pos = range(2**height)
        get_x_pos = lambda x: np.convolve(x, np.ones(2), 'valid')/2
        
        for level in range(height)[::-1]:
            num_nodes_at_level = 2**level
            
            # Set x position per level
            if level == height - 1:
                x_pos = np.arange(2 ** height)
            else:
                x_pos = get_x_pos(x_pos)[::2]  # rolling mean with step size
            
            # Add nodes and edges with child nodes
            for level_node_id in range(num_nodes_at_level):
                current_node = int(str(level) + str(level_node_id))
                g.add_node(current_node, pos=(x_pos[level_node_id], height - level))
                
                if level == height - 1:
                    continue
                
                left_most_child_id = level_node_id * 2
                right_most_child_id = (level_node_id * 2) + 1
                
                left_most_child = int(str(level + 1) + str(left_most_child_id))
                right_most_child = int(str(level + 1) + str(right_most_child_id))
                
                g.add_edge(current_node, left_most_child, weight=weight(current_node, left_most_child))
                g.add_edge(current_node, right_most_child, weight=weight(current_node, right_most_child))
        
        return g
    
    def draw_base_tree(self):
        """Draw the base tree structure with default styling."""
        options = self.cfg.get('options', {})
        draw_config = self.cfg.get('draw', {})
        
        node_color = [options.get('static_node_color', '#C9D6E8')] * len(self.tree.nodes)
        edge_color = [options.get('static_edge_color', 'k')] * len(self.tree.edges)
        width = [1] * len(self.tree.edges)
        
        nx.draw(self.tree,
                pos=nx.get_node_attributes(self.tree, 'pos'),
                ax=self._ax,
                node_color=node_color,
                edge_color=edge_color,
                width=width,
                with_labels=draw_config.get('with_labels', True),
                font_weight=draw_config.get('font_weight', 'bold'),
                node_size=draw_config.get('node_size', 1000),
                font_size=draw_config.get('font_size', 15))
    
    def draw_tree(self, node_list: List[int] = None, edge_list: List[tuple] = None, 
                  color_mode='current', unique_path: List = None):
        """Draw tree with highlighted nodes and edges.
        
        Args:
            node_list: Nodes to highlight
            edge_list: Edges to highlight
            color_mode: 'current' or 'history' coloring
            unique_path: Path to reward for special coloring
        """
        def set_unique_path_color(itr: List, color_list: List):
            if unique_path is not None:
                options = self.cfg.get('options', {})
                for ind, item in enumerate(itr):
                    if item in unique_path:
                        if color_mode == 'current':
                            if isinstance(item, tuple):
                                selected_color = options.get('dynamic_reward_edge_color', '#7CFC00')
                            elif isinstance(item, int):
                                selected_color = options.get('dynamic_reward_node_color', '#7CFC00')
                            else:
                                raise ValueError('color mode not supported')
                            color_list[ind] = selected_color
                        elif color_mode == 'history':
                            if isinstance(item, tuple):
                                selected_color = options.get('history_reward_edge_color', '#228B22')
                            elif isinstance(item, int):
                                selected_color = options.get('history_reward_node_color', '#228B22')
                            else:
                                raise ValueError('color mode not supported')
                            color_list[ind] = selected_color
                        else:
                            raise ValueError('mode not supported')
            return color_list
        
        options = self.cfg.get('options', {})
        draw_config = self.cfg.get('draw', {})
        
        if node_list is not None:
            if color_mode == 'current':
                color = options.get('dynamic_node_color', '#FF0000')
            elif color_mode == 'history':
                color = options.get('history_node_color', '#8b0000')
            else:
                raise ValueError('color mode not supported')
            
            node_color = [color] * len(node_list)
            node_color = set_unique_path_color(node_list, node_color)
            
            nx.draw_networkx_nodes(self.tree,
                                   pos=nx.get_node_attributes(self.tree, 'pos'),
                                   nodelist=node_list,
                                   ax=self._ax,
                                   node_color=node_color,
                                   node_size=draw_config.get('node_size', 1000))
        
        if edge_list is not None:
            if color_mode == 'current':
                color = options.get('dynamic_edge_color', '#FF0000')
            elif color_mode == 'history':
                color = options.get('history_edge_color', '#8b0000')
            else:
                raise ValueError('color mode not supported')
            
            edge_color = [color] * len(edge_list)
            edge_color = set_unique_path_color(edge_list, edge_color)
            
            nx.draw_networkx_edges(self.tree,
                                   pos=nx.get_node_attributes(self.tree, 'pos'),
                                   edgelist=edge_list,
                                   ax=self._ax,
                                   edge_color=edge_color,
                                   width=options.get('edge_width', 10))
    
    def show_tree(self):
        """Display the tree visualization."""
        self._ax.figure.show()
    
    def tree_fig_to_img(self):
        """Convert tree figure to numpy image array."""
        self._ax.figure.canvas.draw()
        plot_to_image = np.frombuffer(self._ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
        return plot_to_image.reshape(self._ax.figure.canvas.get_width_height()[::-1] + (3,))
    
    def get_tree_location(self, key: int) -> Union[Tuple, int, frozenset]:
        """Get tree location (node/edge) for a given tile ID.
        
        Args:
            key: Tile ID to lookup
            
        Returns:
            Tree location - can be int (node), tuple (edge), or frozenset (both)
        """
        from .graph_dictionary import graph_dict
        return graph_dict.get(key, None)
    
    def get_shortest_path(self, source=None, target=None, weight=None, method='dijkstra'):
        """Get shortest path between two nodes.
        
        Args:
            source: Source node
            target: Target node  
            weight: Edge weight attribute name
            method: Algorithm to use (default: 'dijkstra')
            
        Returns:
            List of nodes in shortest path
        """
        return nx.shortest_path(self.tree, source=source, target=target, weight=weight, method=method)
    
    def get_random_walk(self, source, target, disable_backtrack=False):
        """Generate random walk path from source to target.
        
        Args:
            source: Starting node
            target: Target node
            disable_backtrack: If True, prevent immediate backtracking
            
        Returns:
            List of nodes visited in random walk
        """
        path = [source]
        current_parent = None
        
        while source != target:
            current_neighbors = list(self.tree.adj[source])
            
            if disable_backtrack and current_parent in current_neighbors and len(current_neighbors) > 1:
                current_neighbors.remove(current_parent)
            
            current_parent = source
            source = random.choice(current_neighbors)
            path.append(source)
        
        return path