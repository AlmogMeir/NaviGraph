# Fix for Frame Offset Bug in Edge-to-Edge Node Inference

## Problem Statement

When inferring a missing node between two consecutive edges, the current algorithm modifies frame lists by popping elements and updating `start_frame`/`end_frame`. This causes the frames list to become out of sync with the frame indices, resulting in:

1. **Incorrect duration calculations** - The time duration is computed from frame indices, but frame counts are from the modified list
2. **Duration-to-frame mismatch** - `duration_time` and `duration_frames` don't correspond correctly
3. **Hard-to-track data quality issues** - Reconstructed paths don't match the original frame ranges

## Root Cause

In the edge-to-edge transition handling (Case 1 in `reconstruct_traversal`):

```python
# PROBLEMATIC CODE (lines 800-806):
stolen_frame_prev = prev_run['frames'].pop()      # Removes last frame
prev_run['end_frame'] = prev_run['frames'][-1]    # Updates to new last

stolen_frame_curr = run['frames'].pop(0)          # Removes first frame
run['start_frame'] = run['frames'][0]  # ← BUG: Moves start_frame forward
```

After `pop(0)`:
- `run['frames']` no longer contains the first frame
- `run['start_frame']` is updated to what was the second frame
- But `run['end_frame']` is unchanged

This means:
- The "frames" list length is now 1 less
- But the frame indices still span the original range
- When `duration_frames` is calculated as `len(run['frames'])`, it's wrong!

## Solution: Separate Frame Bookkeeping from Frame Indices

Instead of modifying the frames list and updating indices post-hoc, we should:

1. **Keep frame indices always accurate** - They should always represent the true span
2. **Calculate durations from indices, not from list length**
3. **Store which frames were stolen as metadata, don't mutate the list**

## Proposed Code Changes

Replace the problematic edge-to-edge handling:

```python
# CURRENT (PROBLEMATIC):
if prev_type == 'edge' and curr_type == 'edge':
    if not are_neighbors(G, prev_loc, current_loc):
        print(f"  Warning: Non-adjacent edges {prev_loc} -> {current_loc}")
    else:
        connecting_node = infer_missing_node_between_edges(prev_loc, current_loc)
        if connecting_node:
            prev_run = corrected_runs[-1]
            
            stolen_frame_prev = None
            if prev_run['frames'] and len(prev_run['frames']) > 1:
                stolen_frame_prev = prev_run['frames'].pop()
                prev_run['end_frame'] = prev_run['frames'][-1]
            
            stolen_frame_curr = None
            if run['frames'] and len(run['frames']) > 1:
                stolen_frame_curr = run['frames'].pop(0)
                run['start_frame'] = run['frames'][0]  # ← OFFSET BUG
```

### Fix Option 1: Update Duration Calculation (Quick Fix)

Keep the frame popping but calculate durations from frame indices:

```python
# FIX v1: Calculate duration from frame indices, not frame list
if prev_type == 'edge' and curr_type == 'edge':
    if not are_neighbors(G, prev_loc, current_loc):
        print(f"  Warning: Non-adjacent edges {prev_loc} -> {current_loc}")
    else:
        connecting_node = infer_missing_node_between_edges(prev_loc, current_loc)
        if connecting_node:
            prev_run = corrected_runs[-1]
            
            # Still steal frames, but don't change how we count them
            stolen_frame_prev_idx = prev_run['end_frame']
            if prev_run['frames'] and len(prev_run['frames']) > 1:
                prev_run['frames'].pop()  # Mutate the list
                prev_run['end_frame'] = prev_run['frames'][-1] if prev_run['frames'] else stolen_frame_prev_idx - 1
            
            stolen_frame_curr_idx = run['start_frame']
            if run['frames'] and len(run['frames']) > 1:
                run['frames'].pop(0)  # Mutate the list
                # IMPORTANT: Don't update run['start_frame']! Keep the original!
                # run['start_frame'] stays as stolen_frame_curr_idx
            
            # Create inferred node with stolen frames
            inferred_frames = [f for f in [stolen_frame_prev_idx, stolen_frame_curr_idx] if f is not None]
            if inferred_frames:
                corrected_runs.append({
                    'location': connecting_node,
                    'start_frame': min(inferred_frames),
                    'end_frame': max(inferred_frames),
                    'frames': inferred_frames,
                    'is_inferred': True,
                    'inference_reason': f'Missing node between edges {prev_loc} and {current_loc}'
                })
                print(f"  Inferred node {connecting_node} between edges {prev_loc} and {current_loc}")
```

### Fix Option 2: Don't Mutate Frames, Fix Indices Only (Better)

Never modify the frames list - only adjust indices:

```python
# FIX v2: Keep frames list intact, only adjust frame indices
if prev_type == 'edge' and curr_type == 'edge':
    if not are_neighbors(G, prev_loc, current_loc):
        print(f"  Warning: Non-adjacent edges {prev_loc} -> {current_loc}")
    else:
        connecting_node = infer_missing_node_between_edges(prev_loc, current_loc)
        if connecting_node:
            prev_run = corrected_runs[-1]
            
            # Conceptually "steal" the last frame from prev edge
            stolen_frame_prev_idx = None
            if prev_run['frames'] and len(prev_run['frames']) > 1:
                stolen_frame_prev_idx = prev_run['end_frame']
                # Adjust prev_run to exclude the last frame
                prev_run['end_frame'] = prev_run['end_frame'] - 1
                # Remove from frames list as well for consistency
                prev_run['frames'].pop()
            
            # Conceptually "steal" the first frame from current edge
            stolen_frame_curr_idx = None
            if run['frames'] and len(run['frames']) > 1:
                stolen_frame_curr_idx = run['start_frame']
                # Adjust current run to exclude the first frame
                run['start_frame'] = run['start_frame'] + 1
                # Remove from frames list as well for consistency
                run['frames'].pop(0)
            
            # Create inferred node with stolen frame indices
            inferred_frames_indices = [f for f in [stolen_frame_prev_idx, stolen_frame_curr_idx] if f is not None]
            if inferred_frames_indices:
                corrected_runs.append({
                    'location': connecting_node,
                    'start_frame': min(inferred_frames_indices),
                    'end_frame': max(inferred_frames_indices),
                    'frames': inferred_frames_indices,  # Store the actual indices that were stolen
                    'is_inferred': True,
                    'inference_reason': f'Missing node between edges {prev_loc} and {current_loc}'
                })
                print(f"  Inferred node {connecting_node} between edges {prev_loc} and {current_loc}")
```

### Fix Option 3: Better Duration Calculation (Most Correct)

Update how `duration_frames` is calculated in the final conversion:

```python
# In the final result_data creation loop:
for run in final_runs:
    if not run['frames']:
        continue
    
    start_frame = run['start_frame']
    end_frame = run['end_frame']
    
    if start_frame is None or end_frame is None:
        print(f"  Warning: Skipping run with None frame indices: {run}")
        continue
    
    start_time = df_work.loc[start_frame, 'time']
    end_time = df_work.loc[end_frame, 'time']
    
    # IMPORTANT FIX: Calculate duration_frames from indices, not from list length
    # This is more reliable when frames have been popped
    duration_frames_from_indices = end_frame - start_frame + 1
    
    result_data.append({
        'location': run['location'],
        'location_type': get_location_type(run['location']),
        'start_frame': start_frame,
        'end_frame': end_frame,
        'duration_frames': duration_frames_from_indices,  # ← Use indices, not len(frames)
        'start_time': start_time,
        'end_time': end_time,
        'duration_time': end_time - start_time,
        'is_inferred': run['is_inferred'],
        'inference_reason': run['inference_reason']
    })
```

## Recommendation

**Use Fix Option 3** (better duration calculation) combined with **Fix Option 2** (don't update `run['start_frame']`):

1. **Never update `run['start_frame']`** after popping - leave it pointing to the original first frame
2. **Calculate `duration_frames` from frame indices** (end_frame - start_frame + 1) instead of list length
3. **This ensures frame continuity** is maintained even when frames are stolen

## Verification

After applying the fix, the debug cell will show:
- ✓ `gap_before = 0` and `gap_after = 0` for all inferred nodes between edges
- ✓ `duration_ratio ≈ 1.0` (actual equals expected)
- ✓ Frame indices are always continuous: `prev_edge.end_frame + 1 = node.start_frame`
