#!/usr/bin/env python3
"""Generate test graphs for file loading examples.

This script creates various types of random graphs and saves them
in multiple file formats to test the FileGraphBuilder.
"""

import networkx as nx
import numpy as np
from pathlib import Path


def create_test_graphs():
    """Create and save various test graphs."""
    graphs_dir = Path(__file__).parent / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    
    print(f"Creating test graphs in: {graphs_dir}")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Define graphs to create
    graphs = [
        # Erdős–Rényi random graphs
        {
            'name': 'random_erdos_renyi_small',
            'graph': nx.erdos_renyi_graph(n=15, p=0.2, seed=42),
            'description': 'Small Erdős–Rényi graph (15 nodes, p=0.2)'
        },
        {
            'name': 'random_erdos_renyi_medium',
            'graph': nx.erdos_renyi_graph(n=25, p=0.15, seed=42),
            'description': 'Medium Erdős–Rényi graph (25 nodes, p=0.15)'
        },
        
        # Barabási–Albert preferential attachment graphs
        {
            'name': 'random_barabasi_albert',
            'graph': nx.barabasi_albert_graph(n=20, m=2, seed=42),
            'description': 'Barabási–Albert graph (20 nodes, m=2)'
        },
        
        # Watts-Strogatz small-world graphs
        {
            'name': 'random_watts_strogatz',
            'graph': nx.watts_strogatz_graph(n=20, k=4, p=0.3, seed=42),
            'description': 'Watts-Strogatz small-world graph (20 nodes, k=4, p=0.3)'
        },
        
        # Complete graph for comparison
        {
            'name': 'complete_graph',
            'graph': nx.complete_graph(8),
            'description': 'Complete graph (8 nodes)'
        },
        
        # Grid graph
        {
            'name': 'grid_graph',
            'graph': nx.grid_2d_graph(4, 5),
            'description': '2D grid graph (4x5)'
        },
        
        # Path graph
        {
            'name': 'path_graph',
            'graph': nx.path_graph(12),
            'description': 'Path graph (12 nodes)'
        },
        
        # Cycle graph
        {
            'name': 'cycle_graph',
            'graph': nx.cycle_graph(10),
            'description': 'Cycle graph (10 nodes)'
        }
    ]
    
    # File formats to save
    formats = [
        ('graphml', lambda g, p: nx.write_graphml(g, p)),
        ('gexf', lambda g, p: nx.write_gexf(g, p)),
        ('gml', lambda g, p: nx.write_gml(g, p)),
    ]
    
    graph_info = []
    
    for graph_def in graphs:
        name = graph_def['name']
        graph = graph_def['graph']
        description = graph_def['description']
        
        # Add some node attributes for visualization
        pos = nx.spring_layout(graph, seed=42)
        for node, (x, y) in pos.items():
            graph.nodes[node]['x'] = float(x)
            graph.nodes[node]['y'] = float(y)
        
        # Add graph metadata
        graph.graph['name'] = name
        graph.graph['description'] = description
        graph.graph['nodes'] = len(graph.nodes)
        graph.graph['edges'] = len(graph.edges)
        
        print(f"\n📊 {name}: {description}")
        print(f"   Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
        
        # Save in multiple formats
        for format_name, save_func in formats:
            file_path = graphs_dir / f"{name}.{format_name}"
            try:
                save_func(graph, str(file_path))
                print(f"   ✓ Saved as {format_name}: {file_path.name}")
            except Exception as e:
                print(f"   ✗ Failed to save as {format_name}: {e}")
        
        graph_info.append({
            'name': name,
            'description': description,
            'nodes': len(graph.nodes),
            'edges': len(graph.edges),
            'files': [f"{name}.{fmt}" for fmt, _ in formats]
        })
    
    # Create a summary file
    summary_file = graphs_dir / "graph_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Test Graph Files Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for info in graph_info:
            f.write(f"Graph: {info['name']}\n")
            f.write(f"Description: {info['description']}\n")
            f.write(f"Nodes: {info['nodes']}, Edges: {info['edges']}\n")
            f.write(f"Files: {', '.join(info['files'])}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\n✅ Created {len(graphs)} test graphs in {len(formats)} formats")
    print(f"📄 Summary saved to: {summary_file}")
    
    return graph_info


if __name__ == "__main__":
    create_test_graphs()