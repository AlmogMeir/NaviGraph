# Frame Offset Bug Analysis

## Location
File: `improved_node_df.ipynb`, Cell 11 (reconstruct_traversal function), Lines 787-814

## The Bug: Edge-to-Edge Inference Frame Offset

### Problem Summary
When inferring a missing node between two consecutive edges, the algorithm "steals" frames from both edges to assign to the inferred node. However, this causes a frame offset in the subsequent edge's start_frame.

### Root Cause Analysis

**Line 800-801 (Stealing from previous edge):**
```python
stolen_frame_prev = prev_run['frames'].pop()  # Take LAST frame
prev_run['end_frame'] = prev_run['frames'][-1] # Update end_frame
```
✓ This is correct - removes last frame, updates end_frame to the new last frame.

**Line 805-806 (Stealing from current edge):**
```python
stolen_frame_curr = run['frames'].pop(0)  # Take FIRST frame  
run['start_frame'] = run['frames'][0]     # ← BUG IS HERE!
```
✗ **The Problem:** After `pop(0)`, `run['frames'][0]` is now the SECOND frame of the original sequence, but the original frame at index 1 was never actually in the original data - it's just what comes after the stolen frame.

### Visual Example:

```
Original state:
Edge B: frames = [103, 104, 105]  (consecutive frame indices)
        start_frame = 103
        end_frame = 105

After pop(0):
Edge B: frames = [104, 105]
        start_frame = run['frames'][0] = 104    ← OFFSET BY 1!
        end_frame = 105 (unchanged)
        
But wait, the inferred node gets frame 103!
Inferred Node: frames = [102, 103]  (assuming 102 stolen from prev edge)
               start_frame = 102
               end_frame = 103

Timeline:
- Previous Edge: end_frame = 104
- Inferred Node: start_frame = 102, end_frame = 103
- Current Edge: start_frame = 104

RESULT: Frames are out of order! Node appears BEFORE the edge that precedes it!
```

### Actual Problem Scenario:

The real issue is that after stealing frame 103 from Edge B for the inferred node, we set Edge B's start_frame to 104. But then when we add the inferred node with frames [102, 103] and the next edge with frames [104, 105], the sequence becomes:

1. Edge A: end_frame = last_of_A
2. Inferred Node: start_frame = 102, end_frame = 103 
3. Edge B: start_frame = 104
4. Edge B data in frames list: [104, 105]

**But wait - there's continuity here (103→104). So what's the real offset?**

### The TRUE Issue: `start_frame` Does Not Match First Frame in `frames` List

After `run['frames'].pop(0)`:
- `run['start_frame']` = first remaining index in frames list (104)
- `run['frames'][0]` = the actual first frame value (104)
- `run['frames']` = [104, 105]

This LOOKS consistent, but the problem is the **duration calculation**:

```python
'duration_frames': len(run['frames']),  # Now 2 instead of 3!
```

But the frame indices are:
```
start_frame = 104
end_frame = 105  (unchanged, still pointing to the last frame of original sequence)
```

So we have:
- Duration based on frames list: 2 frames
- Duration based on frame indices: 105 - 104 + 1 = 2 frames ✓

Wait, that still matches...

### The ACTUAL Offset Issue

I think the real problem is more subtle. When we calculate `duration_time`:

```python
'duration_time': end_time - start_time,
```

The `start_time` and `end_time` are looked up from the original `df_work` dataframe based on `start_frame` and `end_frame`. But if we've popped frames, the mapping is broken!

**Example:**
```
Original df_work:
frame 103: time = 2.575
frame 104: time = 2.600
frame 105: time = 2.625

Edge B tracking in df_work showed frames [103, 104, 105]

After pop(0):
run['frames'] = [104, 105]
run['start_frame'] = 104  ← Updated to what?  
run['end_frame'] = 105

But we SHOULD have:
start_frame = 103 (for correct time lookup)
end_frame = 105

Instead we have:
start_frame = 104 (OFFSET BY +1!)
end_frame = 105 (correct)

So when we do: df_work.loc[start_frame, 'time'] = 2.600 (wrong! should be 2.575)
```

**This causes duration_time to be calculated incorrectly because the start_time is wrong.**

## The Fix

The offset occurs because we update `start_frame` after popping. We should NOT update `start_frame` - we should only pop the frame from the list but keep the frame index the same:

```python
# INCORRECT (current):
stolen_frame_curr = run['frames'].pop(0)
run['start_frame'] = run['frames'][0]  # ← Jumps to next frame!

# CORRECT (should be):
if run['frames']:
    stolen_frame_curr = run['frames'].pop(0)
    # Don't update start_frame - it should still point to frame 103 in our example
    # Even though frame 103 is no longer in the frames list!
    # The duration_frames calculation will now be WRONG though...
```

Actually this reveals an even bigger problem: **the `frames` list and the `start_frame`/`end_frame` indices are out of sync.**

## The Better Fix

Instead of modifying frames in-place, we should:

1. Keep the original `start_frame` and `end_frame` unchanged
2. Only use these indices for lookup and duration calculation
3. Don't pop from the frames list, only track which frames are stolen
4. Calculate `duration_frames` from the frame indices, not the frames list length

```python
# Better approach:
stolen_frame_prev_idx = prev_run['end_frame']  # Just reference, don't pop
prev_run['end_frame'] = prev_run['end_frame'] - 1  # Conceptually "remove" the last frame

stolen_frame_curr_idx = run['start_frame']  # Just reference, don't pop
run['start_frame'] = run['start_frame'] + 1  # Conceptually "remove" the first frame

# Create inferred node
inferred_frames = [stolen_frame_prev_idx, stolen_frame_curr_idx]
corrected_runs.append({
    'location': connecting_node,
    'start_frame': stolen_frame_prev_idx,
    'end_frame': stolen_frame_curr_idx,
    'frames': inferred_frames,  # Just metadata
    'is_inferred': True,
    ...
})
```

## Why This Matters

The offset causes:
1. **Wrong duration_time calculation** - start_time is looked up from wrong frame
2. **Discontinuous frame sequence** - duration_frames doesn't match the actual frame span
3. **Data quality issues** - analysis based on time durations will be inaccurate
4. **Incorrect inference reason tracking** - can't accurately reproduce which frames were assigned where

## Test Case

To verify the bug, check the notebook output for inferred nodes between edges:
- Look for an inferred node where `duration_time / duration_frames` doesn't match the frame rate (should be ~0.025 seconds/frame)
- The difference reveals the offset
