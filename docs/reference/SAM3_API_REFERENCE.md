# SAM 3 (Segment Anything with Concepts) - API Reference

> **Source**: Meta/Facebook Research  
> **Repository**: https://github.com/facebookresearch/sam3  
> **Trust Score**: 8.1 | **Code Snippets Available**: 308

---

## Overview

SAM 3 is a unified foundation model for **promptable segmentation** in images and videos. It can:
- Detect, segment, and track objects using **text or visual prompts**
- Exhaustively segment all instances of **open-vocabulary concepts**
- Process both images and videos with consistent APIs

---

## Installation

```bash
# Install PyTorch and dependencies
pip install torch torchvision opencv-python matplotlib scikit-learn

# Install SAM 3
pip install 'git+https://github.com/facebookresearch/sam3.git'
```

---

## Video Predictor API

### Initialize Video Predictor

```python
from sam3.model_builder import build_sam3_video_predictor
import torch

# Multi-GPU setup
gpus_to_use = range(torch.cuda.device_count())
# Single GPU
# gpus_to_use = [torch.cuda.current_device()]

predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
```

### Start Video Session

```python
response = predictor.handle_request({
    "type": "start_session",
    "resource_path": "/path/to/video.mp4"  # or "/path/to/frames/" directory
})
session_id = response["session_id"]
```

### Add Text Prompt

```python
response = predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,  # Frame to add prompt on
    "text": "person wearing red jacket"  # Natural language description
})

outputs = response["outputs"]
# outputs contains: masks, scores, object_ids
```

### Propagate Through Video

```python
# Streaming propagation - yields results per frame
for result in predictor.handle_stream_request({
    "type": "propagate_in_video",
    "session_id": session_id,
    "propagation_direction": "both",  # "forward", "backward", or "both"
    "start_frame_index": 0,
    "max_frame_num_to_track": None  # None = all frames
}):
    frame_idx = result["frame_index"]
    frame_outputs = result["outputs"]
    
    for obj_id, obj_data in frame_outputs.items():
        mask = obj_data["mask"]  # Binary mask (H, W)
        score = obj_data.get("score", 1.0)
```

### Reset Session (Change Prompts)

```python
predictor.handle_request({
    "type": "reset_session",
    "session_id": session_id,
})
```

### Close Session

```python
predictor.handle_request({
    "type": "close_session",
    "session_id": session_id,
})
```

---

## Image Processor API

### Initialize Image Model

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Initialize with BPE vocabulary
bpe_path = "assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=0.5)
```

### Process Image with Text Prompt

```python
# Load and set image
image = Image.open("photo.jpg")
state = processor.set_image(image)

# Apply text prompt
state = processor.set_text_prompt(state=state, prompt="a dog")

# Extract results
masks = state["masks"]   # bool tensor [N, H, W]
boxes = state["boxes"]   # float tensor [N, 4] - [x0, y0, x1, y1]
scores = state["scores"] # float tensor [N]

# Filter by confidence
high_conf_idx = scores > 0.7
filtered_masks = masks[high_conf_idx].cpu().numpy()
```

### Reset Prompts

```python
processor.reset_all_prompts(state)
```

---

## Alternative Video API (SAM2-compatible)

For compatibility with SAM 2 video tasks:

```python
from sam3.model_builder import build_sam3_video_predictor

predictor = build_sam3_video_predictor(gpus_to_use=[0])

# Initialize inference state
inference_state = predictor.init_state(video_path="video.mp4")

# Add box prompt
_, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    box=np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32),
)

# Propagate through video
video_segments = {}
for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
    inference_state, 
    start_frame_idx=0, 
    max_frame_num_to_track=300,
    reverse=False
):
    video_segments[frame_idx] = {
        obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
        for i, obj_id in enumerate(obj_ids)
    }

# Clear tracking
predictor.clear_all_points_in_video(inference_state)
```

---

## Request Types Summary

| Request Type | Description |
|-------------|-------------|
| `start_session` | Initialize video session with video path |
| `add_prompt` | Add text prompt on specific frame |
| `propagate_in_video` | Track objects across all frames |
| `reset_session` | Clear prompts, keep video loaded |
| `close_session` | Free GPU resources |

---

## Best Practices

1. **Text Prompts**: Keep prompts short and concrete ("person", "chair", "dog") rather than long sentences
2. **Resolution Limits**: Implement max resolution caps (e.g., 720p) for acceptable inference times
3. **Frame Limits**: Cap video length (e.g., 30 seconds) to manage GPU memory
4. **GPU Memory**: Use `close_session` to free resources between videos
5. **Multi-GPU**: Use `build_sam3_video_predictor(gpus_to_use=[0, 1])` for better throughput
6. **TF32 Optimization**: Enable for Ampere GPUs:
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```

---

## Output Formats

### Mask Output
- **Type**: Binary numpy array
- **Shape**: `(H, W)` per object per frame
- **Values**: `True` (object), `False` (background)

### Scores
- **Type**: Float
- **Range**: 0.0 - 1.0
- **Usage**: Filter low-confidence detections

### Object IDs
- **Type**: Integer
- **Purpose**: Track same object across frames
