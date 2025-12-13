# SAM3 Instructions for Automated Office Item Segmentation & Inventory

## Objective
Configure a two-stage AI pipeline:
1. **SAM3** performs automatic instance segmentation on all objects in the office scene
2. **Classification AI** analyzes each masked segment to identify what it is and compile an inventory list

## Stage 1: SAM3 Segmentation

### Prompt for SAM3:
"Perform automatic instance segmentation on this environment video. Segment each individual physical object as a separate instance. Do not group objects by category - treat each distinct item as its own segment, even if multiple items appear identical."

### Configuration:
- Use SAM3's automatic mask generation mode (no manual prompting)
- Enable instance-level segmentation, not semantic segmentation
- Each physical object should receive a unique mask ID
- Segment all visible objects in the scene, regardless of type

### Expected Output from SAM3:
- Collection of individual masks (one per physical object)
- Each mask with unique ID
- Mask coordinates/polygons
- Optionally: confidence scores and bounding boxes

## Stage 2: Classification & Inventory

### Prompt for the Reviewing AI (e.g., GPT-4V, Claude, or similar vision model):

"Analyze each segmented region from this office scene. For each masked item:
1. Identify what type of office equipment or furniture it is
2. If you cannot confidently identify an item, label it as 'unidentified object'
3. Count how many instances of each item type exist across all masks
4. Generate an inventory list with item names and quantities

Output format:
- Item name: Quantity
- Item name: Quantity
[etc.]"

## Pipeline Architecture

```
Office Image 
    ↓
SAM3 (automatic instance segmentation)
    ↓
Individual masked segments (mask_1, mask_2, mask_3, ...)
    ↓
Vision AI analyzes each mask
    ↓
Inventory list with item identification and counts
```

## Implementation Notes

1. **No pre-defined item list**: The system should work on any office scene without knowing what items to expect

2. **Mask-by-mask analysis**: Feed each SAM3 mask to the classification AI separately (or all at once with clear mask IDs)

3. **Aggregation logic**: The classification AI should tally items of the same type to provide counts

4. **Handle edge cases**: 
   - Partially visible objects
   - Overlapping items
   - Ambiguous objects

## Example Workflow

**Input**: Photo of office with various equipment

**SAM3 Output**: 
- Mask 1, Mask 2, Mask 3... Mask 15 (15 distinct objects detected)

**Classification AI receives**: 15 masked regions to analyze

**Final Inventory Output**:
- Office chairs: 3
- Computer monitors: 2
- Keyboards: 2
- Desk: 1
- Desk lamp: 1
- Mouse: 2
- Laptop: 1
- Waste bin: 1
- Water bottle: 1
- Notebook: 1

**Total items: 15**

---

This approach makes the system fully automated - users just provide the image, and the AI pipeline handles segmentation, identification, and inventory compilation.