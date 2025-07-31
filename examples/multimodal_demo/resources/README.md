# Multimodal Demo Resources

This directory contains shared resources for the multimodal demo.

## Maze Map

The maze map (`maze_map.png`) is referenced from the basic maze example:
- **Path**: `../basic_maze/resources/maze_map.png`
- **Description**: Standard 17x17 grid maze layout used for spatial navigation analysis
- **Format**: PNG image with maze structure and tile boundaries

## Usage

The maze map is used by the map integration plugin to:
1. Transform camera coordinates to maze coordinates
2. Detect which maze tile the animal is in at each frame
3. Enable spatial navigation analysis and graph-based pathfinding

## Configuration

The map is configured in `config.yaml` with these parameters:
- `segment_length: 86` pixels per maze segment
- `origin: (47, 40)` top-left corner coordinates
- `grid_size: (17, 17)` maze dimensions
- `pixel_to_meter: 2279.4117647058824` conversion factor

These parameters must match the maze used during the experimental session.