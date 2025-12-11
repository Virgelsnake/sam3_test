# SAM 3 Video Segmentation - Code Examples

## Complete Video Tracking Pipeline

```python
import torch
from sam3.model_builder import build_sam3_video_predictor

# Enable TF32 for better performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Initialize predictor
predictor = build_sam3_video_predictor(gpus_to_use=[0])

# Start session
response = predictor.handle_request({
    "type": "start_session",
    "resource_path": "/path/to/video.mp4"
})
session_id = response["session_id"]

# Add text prompt
response = predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "text": "person"
})

# Collect all frame masks
all_masks = {}
for result in predictor.handle_stream_request({
    "type": "propagate_in_video",
    "session_id": session_id,
    "propagation_direction": "both",
    "start_frame_index": 0,
    "max_frame_num_to_track": None
}):
    frame_idx = result["frame_index"]
    all_masks[frame_idx] = result["outputs"]

# Cleanup
predictor.handle_request({
    "type": "close_session",
    "session_id": session_id,
})
```

## Processing Multiple Prompts

```python
# Reset session to change prompts (keep video loaded)
predictor.handle_request({
    "type": "reset_session",
    "session_id": session_id,
})

# Add new prompt
response = predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "text": "chair"
})

# Propagate again...
```

## Export Masks to Video

```python
import cv2
import numpy as np

def masks_to_video(masks_dict, original_video_path, output_path, fps=30):
    """Convert per-frame masks to a video file."""
    cap = cv2.VideoCapture(original_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in masks_dict:
            # Combine all object masks
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            for obj_id, obj_data in masks_dict[frame_idx].items():
                mask = obj_data["mask"]
                combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
            
            # Create overlay
            overlay = frame.copy()
            overlay[combined_mask > 0] = [0, 255, 0]  # Green overlay
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
```

## Create Grayscale Mask Video

```python
def create_mask_video(masks_dict, width, height, output_path, fps=30):
    """Create a grayscale mask-only video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    for frame_idx in sorted(masks_dict.keys()):
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for obj_id, obj_data in masks_dict[frame_idx].items():
            mask = obj_data["mask"].astype(np.uint8) * 255
            combined_mask = np.maximum(combined_mask, mask)
        out.write(combined_mask)
    
    out.release()
```

## Extract Video Frames

```python
def extract_frames(video_path, max_frames=None):
    """Extract frames from video for processing."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return frames

def get_video_info(video_path):
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return info
```

## Error Handling Pattern

```python
import logging

logger = logging.getLogger(__name__)

async def process_video_job(video_path: str, prompt: str) -> dict:
    """Process a video segmentation job with error handling."""
    session_id = None
    
    try:
        # Validate video
        info = get_video_info(video_path)
        if info["duration"] > 60:
            raise ValueError("Video exceeds 60 second limit")
        if info["height"] > 1080:
            raise ValueError("Video exceeds 1080p resolution limit")
        
        # Initialize predictor
        predictor = build_sam3_video_predictor(gpus_to_use=[0])
        
        # Start session
        response = predictor.handle_request({
            "type": "start_session",
            "resource_path": video_path
        })
        session_id = response["session_id"]
        
        # Add prompt
        response = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt
        })
        
        if not response.get("outputs"):
            return {"status": "error", "message": f"No objects found for prompt: {prompt}"}
        
        # Propagate
        masks = {}
        for result in predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both",
            "start_frame_index": 0,
            "max_frame_num_to_track": None
        }):
            masks[result["frame_index"]] = result["outputs"]
        
        return {"status": "success", "masks": masks, "frame_count": len(masks)}
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        return {"status": "error", "message": str(e)}
        
    finally:
        if session_id:
            try:
                predictor.handle_request({
                    "type": "close_session",
                    "session_id": session_id,
                })
            except:
                pass
```
