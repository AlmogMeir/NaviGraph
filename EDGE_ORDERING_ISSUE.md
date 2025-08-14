# Edge Tuple Ordering Issue in GUI

## Problem Description

The NaviGraph GUI has an issue with edge tuple ordering when displaying contours for mapped edges. This affects the test mode functionality where selecting a mapped edge should highlight its contour.

## Root Cause

### The Issue
1. **NetworkX Edge Ordering**: NetworkX returns edges in the order they were added to the graph (e.g., `(60, 50)`)
2. **Saved Mapping Ordering**: Previously saved mappings might have edges stored in different order (e.g., `(50, 60)`)  
3. **String-based Lookup**: The GUI uses string conversion for edge lookups in standardized contours:
   - Graph edge `(60, 50)` → lookup string `"60_50"`
   - Saved mapping has edge `(50, 60)` → stored as string `"50_60"`
   - Lookup fails, no contour displayed

### Why String Search Instead of Graph Methods?

The GUI uses string-based lookup in the `_highlight_using_standardized_contours()` method because:
- **Serialization Format**: Standardized contours are stored as JSON-compatible format with string keys
- **Cross-session Compatibility**: Allows loading contours without rebuilding the graph
- **Performance**: Direct dict lookup vs graph traversal

However, this creates the ordering dependency issue.

## Current Solution (Temporary Fix)

Added bidirectional string lookup in `_highlight_using_standardized_contours()`:

```python
# Try original edge string
edge_str = f"{elem_id[0]}_{elem_id[1]}"
contour_list = standardized.get('edges', {}).get(edge_str, [])

# For undirected graphs, try reversed edge if not found
if not contour_list and hasattr(self, 'graph') and not self.graph.graph.is_directed():
    reversed_edge_str = f"{elem_id[1]}_{elem_id[0]}"
    contour_list = standardized.get('edges', {}).get(reversed_edge_str, [])
```

## Better Long-term Solutions

### Option 1: Normalize Edge Strings at Storage Time
- Always store edges as `f"{min(a,b)}_{max(a,b)}"` for undirected graphs
- Requires updating the `_create_standardized_contour_mapping()` method
- Pro: Simple lookup logic
- Con: Assumes node IDs are comparable

### Option 2: Use Graph Methods for Lookup
- Replace string-based lookup with `graph.has_edge(a, b)` calls
- Store mapping between standardized keys and actual graph elements
- Pro: Uses graph structure directly, more robust
- Con: Requires graph instance to be available

### Option 3: Bidirectional Key Storage
- Store both `"a_b"` and `"b_a"` keys pointing to the same contour data
- Doubles storage but makes lookup foolproof
- Pro: No lookup logic changes needed
- Con: Increased memory usage

## Affected Code Locations

### Primary Issue Location
- **File**: `navigraph/core/graph/setup_gui_qt.py`
- **Method**: `_highlight_using_standardized_contours()` (lines ~3350-3365)
- **Function**: Converts edge tuples to strings for standardized contour lookup

### Related Code
- `_create_standardized_contour_mapping()`: Creates the standardized format
- `mapping.py`: Edge region storage and retrieval methods
- Any other string-based edge key generation

## Debugging Process Used

1. **Confirmed mapping works**: `mapping.get_edge_regions()` found the contour
2. **Traced execution path**: Found it uses standardized contours, not fallback method  
3. **Added debug prints**: Showed string lookup mismatch
4. **Implemented bidirectional lookup**: Fixed the immediate issue

## Recommended Next Steps

1. **Clean up debug code** from the current implementation
2. **Decide on long-term solution** (Option 1, 2, or 3 above)
3. **Update standardized contour creation** to use consistent edge ordering
4. **Add unit tests** for edge tuple ordering scenarios
5. **Document edge ordering conventions** in the codebase

## Technical Notes

- This issue only affects **undirected graphs** where `(a,b) == (b,a)`
- **Directed graphs** should preserve exact edge ordering
- **Node mappings** are unaffected (no ordering ambiguity)
- Issue appears in **test mode** when selecting edges from dropdown

## Files Modified (Temporary Fix)

- `navigraph/core/graph/setup_gui_qt.py`: Added bidirectional string lookup
- `navigraph/core/graph/mapping.py`: Added bidirectional region lookup (may be redundant)