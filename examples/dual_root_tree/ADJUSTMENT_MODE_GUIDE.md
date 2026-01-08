# Adjustment Mode Guide

## Overview

The **Adjustment Mode** in NaviGraph allows you to reuse precise contour mappings across multiple sessions by visually adjusting them to match minor positional changes in your modular maze.

This is ideal for experiments where:
- Your maze structure stays the same (same graph)
- The physical maze shifts slightly between sessions (modular components)
- You want to avoid redrawing all contours from scratch

## Workflow

### 1. Create Base Mapping (One Time)

First, create a precise, detailed mapping using Manual Drawing mode:

```bash
# Launch the GUI with your dual-root tree configuration
uv run navigraph setup graph examples/dual_root_tree/config_dual_root.yaml
```

1. Click **"Manual Drawing"** mode
2. Carefully draw contours for each node and edge
3. Save the mapping as your base template: `base_mapping.pkl`

### 2. Adjust for New Session

For each subsequent session with a slightly shifted maze:

```bash
# Launch the GUI again
uv run navigraph setup graph examples/dual_root_tree/config_dual_root.yaml
```

1. Click **"Adjust Mapping"** mode
2. Click **"Load Base Mapping"** → select your `base_mapping.pkl`
3. (Optional) Click **"Load New Image"** → select your new session's maze photo
4. **Adjust contours:**
   - Click any contour to select it (turns yellow)
   - Drag it to the new position
   - Release to place it
   - Repeat for all shifted contours
5. Click **"Save Adjusted Mapping"** → save as `session2_mapping.pkl`

## Visual Feedback

The adjustment mode provides clear visual cues:

| Color | Meaning |
|-------|---------|
| Light Green (nodes) / Light Orange (edges) | Base contour position (not adjusted) |
| Darker Green (nodes) / Darker Orange (edges) | Adjusted contour (has been moved) |
| Yellow highlight | Currently selected contour |
| Thicker border | Selected or adjusted contour |

## Keyboard Shortcuts

- **Mouse Wheel**: Zoom in/out for precise adjustment
- **Right Click + Drag**: Pan around the image
- **R**: Reset zoom to fit view

## Tips

### For Best Results

1. **Take Consistent Photos**: Try to photograph your maze from the same angle and distance each session
2. **Use Landmarks**: Place visual markers (colored tape) on your maze to help align positions
3. **Adjust in Batches**: Adjust similar regions (e.g., all left-side nodes) together
4. **Test Before Running**: Use "Test Mode" to verify your adjusted mapping is accurate

### Handling Large Shifts

If your maze shifts significantly:

- Use the zoom feature to work on one section at a time
- Consider using 4-5 anchor points and adjusting by sections
- For rotation or scaling, you may need to create a new base mapping

### Saving Strategy

Recommended file naming:
```
base_mapping.pkl           # Your master template
session_2024-01-06.pkl     # Session-specific adjusted mapping
session_2024-01-13.pkl     # Next session
```

## Troubleshooting

**Q: Contours won't select/drag**  
A: Make sure you're in "Adjust Mapping" mode and have loaded a base mapping first

**Q: Image and contours don't align**  
A: Your new session image may have different dimensions. Try loading the original image size

**Q: Can't see base contours**  
A: Ensure "Show All Mappings" checkbox is enabled in Display Options

**Q: Lost my adjustments**  
A: Click "Reset All Adjustments" and start over. Adjustments are only saved when you click "Save Adjusted Mapping"

## Example: Dual-Root Tree Adjustment

For your dual-root tree maze:

1. Create base mapping with all nodes (L0, R0, L10, L11, R10, R11, etc.) carefully outlined
2. For each session:
   - Load base mapping
   - Drag left tree nodes to match left side position
   - Drag right tree nodes to match right side position
   - Drag connecting edge between L0↔R0 if needed
   - Save adjusted mapping
3. Use the adjusted mapping in your analysis pipeline

## Integration with Analysis Pipeline

After creating adjusted mapping:

```bash
# Test the adjusted mapping
uv run navigraph test graph your_config.yaml

# Run full analysis with adjusted mapping
uv run navigraph run your_config.yaml
```

The pipeline will use your adjusted contour positions for all spatial analysis.
