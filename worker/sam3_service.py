"""
SAM3 Service - Wrapper for SAM 3 video segmentation.

Provides a clean interface for video segmentation with text prompts.
Supports multi-object detection and dynamic inventory tracking.
"""

import os
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import torch


# Common object categories for comprehensive detection
OBJECT_CATEGORIES = [
    # Furniture
    "chair", "office chair", "desk", "table", "sofa", "couch", "cabinet", 
    "shelf", "bookshelf", "drawer", "filing cabinet",
    # Electronics
    "monitor", "computer", "laptop", "keyboard", "mouse", "phone", 
    "television", "tv", "printer", "speaker",
    # People
    "person", "people",
    # Office items
    "lamp", "plant", "cup", "mug", "bottle", "book", "paper", "pen",
    "notebook", "whiteboard", "clock", "trash can", "bin",
    # Bags/containers
    "bag", "backpack", "box", "container",
]


@dataclass
class TrackedObject:
    """Represents a tracked object across frames."""
    obj_id: int
    category_prompt: str  # The prompt that detected this object
    first_seen_frame: int
    last_seen_frame: int
    bbox_history: list = field(default_factory=list)  # List of (x, y, w, h) tuples
    confidence_history: list = field(default_factory=list)
    is_active: bool = True


@dataclass 
class FrameInventory:
    """Inventory snapshot for a specific frame."""
    frame_index: int
    objects: dict  # obj_id -> TrackedObject
    masks: dict  # obj_id -> np.ndarray (individual mask)
    combined_mask: Optional[np.ndarray] = None
    scene_changed: bool = False
    change_reason: str = ""


class SAM3Service:
    """Service class for SAM 3 video segmentation with dynamic inventory tracking."""

    def __init__(self, device: str = "cuda:0"):
        """
        Initialize the SAM3 service.

        Args:
            device: Device to run inference on (e.g., "cuda:0").
        """
        self.device = device
        self.predictor = None
        self._initialized = False
        
        # Tracking state
        self.tracked_objects: dict[int, TrackedObject] = {}
        self.frame_inventories: dict[int, FrameInventory] = {}
        self.next_obj_id = 0
        self.current_session_id = None
        
        # Scene change detection thresholds
        self.object_exit_threshold = 0.3  # Object considered exited if IoU < this
        self.object_enter_threshold = 0.1  # Min mask area ratio for new object
        self.reinventory_cooldown = 30  # Min frames between re-inventories

    def initialize(self) -> None:
        """
        Initialize the SAM 3 model.

        This loads the model weights and prepares for inference.
        Should be called once before processing videos.
        """
        if self._initialized:
            return

        # Enable TF32 for faster computation on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Authenticate with HuggingFace if token is available
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)

        # Build the video predictor
        from sam3.model_builder import build_sam3_video_predictor

        # Extract GPU index from device string
        gpu_idx = 0
        if ":" in self.device:
            gpu_idx = int(self.device.split(":")[1])

        self.predictor = build_sam3_video_predictor(gpus_to_use=[gpu_idx])
        self._initialized = True

    def process_video(
        self,
        video_path: str,
        prompt: str,
        progress_callback: Optional[callable] = None,
    ) -> dict:
        """
        Process a video with SAM 3 segmentation using multi-category tracking.

        Since SAM3's add_prompt replaces tracked objects, we run separate
        propagation passes for each object category and merge the results.

        Args:
            video_path: Path to the input video file.
            prompt: Text prompt (used as context, but all objects are detected).
            progress_callback: Optional callback for progress updates.

        Returns:
            dict: Results containing:
                - masks: dict[int, np.ndarray] - Frame index to combined mask
                - individual_masks: dict[int, dict[int, np.ndarray]] - Frame -> obj_id -> mask
                - frame_count: int - Total frames processed
                - objects_detected: int - Total unique objects found
                - inventory_snapshots: list[dict] - Inventory at each scene change
                - tracked_objects: dict - Full tracking data for all objects
        """
        if not self._initialized:
            self.initialize()

        # Reset tracking state for new video
        self.tracked_objects = {}
        self.frame_inventories = {}
        self.next_obj_id = 0

        if progress_callback:
            progress_callback(5, "Scanning for object categories...")

        # Step 1: Quick scan to find which categories have objects
        categories_found = self._scan_for_categories(video_path, prompt)
        
        if not categories_found:
            return {
                "masks": {},
                "individual_masks": {},
                "frame_count": 0,
                "objects_detected": 0,
                "inventory_snapshots": [],
                "tracked_objects": {},
                "message": "No objects found in video",
            }

        print(f"[SAM3] Found objects in {len(categories_found)} categories: {list(categories_found.keys())}")

        # Step 2: Run separate propagation for each category and merge
        all_individual_masks = {}  # frame_idx -> {obj_id -> mask}
        all_combined_masks = {}    # frame_idx -> combined mask
        frame_count = 0
        
        total_categories = len(categories_found)
        for cat_idx, (category, obj_count) in enumerate(categories_found.items()):
            if progress_callback:
                base_progress = 10 + int((cat_idx / total_categories) * 70)
                progress_callback(base_progress, f"Tracking '{category}' ({obj_count} objects)...")
            
            print(f"[SAM3] Running propagation for category '{category}' ({obj_count} objects)")
            
            # Run full propagation for this category
            category_result = self._propagate_single_category(
                video_path, category, cat_idx, progress_callback, base_progress, total_categories
            )
            
            if category_result:
                frame_count = max(frame_count, category_result["frame_count"])
                
                # Merge masks into combined results
                for frame_idx, obj_masks in category_result["individual_masks"].items():
                    if frame_idx not in all_individual_masks:
                        all_individual_masks[frame_idx] = {}
                    all_individual_masks[frame_idx].update(obj_masks)

        # Build combined masks from individual masks
        for frame_idx, obj_masks in all_individual_masks.items():
            masks_list = [m for m in obj_masks.values() if m is not None and m.sum() > 0]
            if masks_list:
                all_combined_masks[frame_idx] = self._combine_masks(masks_list)

        if progress_callback:
            progress_callback(90, "Finalizing results...")

        # Build inventory snapshot
        inventory_snapshots = [{
            "frame_index": 0,
            "reason": "Initial inventory",
            "objects": {
                obj_id: obj.category_prompt 
                for obj_id, obj in self.tracked_objects.items()
            },
            "object_count": len(self.tracked_objects),
        }]

        print(f"[SAM3] Tracking complete: {len(self.tracked_objects)} unique objects, {len(all_combined_masks)} frames")

        return {
            "masks": all_combined_masks,
            "individual_masks": all_individual_masks,
            "frame_count": frame_count,
            "objects_detected": len(self.tracked_objects),
            "inventory_snapshots": inventory_snapshots,
            "tracked_objects": {
                obj_id: {
                    "category": obj.category_prompt,
                    "first_frame": obj.first_seen_frame,
                    "last_frame": obj.last_seen_frame,
                    "is_active": obj.is_active,
                }
                for obj_id, obj in self.tracked_objects.items()
            },
        }

    def _scan_for_categories(self, video_path: str, user_prompt: str) -> dict:
        """Quick scan to find which categories have objects in the video."""
        categories_found = {}
        
        # Build category list
        categories_to_try = []
        if user_prompt and user_prompt.lower() not in ["generate an inventory of detected items", "inventory"]:
            categories_to_try.append(user_prompt)
        categories_to_try.extend(OBJECT_CATEGORIES)
        
        # Limit to most common categories for speed
        priority_categories = [
            "chair", "desk", "table", "monitor", "computer", "laptop",
            "keyboard", "mouse", "person", "phone", "lamp", "plant",
            "bottle", "cup", "bag", "book"
        ]
        categories_to_try = [c for c in priority_categories if c in OBJECT_CATEGORIES]
        
        print(f"[SAM3] Scanning for {len(categories_to_try)} object categories...")
        
        session_id = self._create_session(video_path)
        try:
            for category in categories_to_try:
                try:
                    response = self.predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": category
                    })
                    
                    outputs = response.get("outputs", {})
                    obj_ids = outputs.get("out_obj_ids", [])
                    if isinstance(obj_ids, np.ndarray):
                        obj_ids = obj_ids.tolist()
                    
                    if obj_ids:
                        # Verify masks are not empty
                        binary_masks = outputs.get("out_binary_masks")
                        if binary_masks is not None:
                            if hasattr(binary_masks, 'cpu'):
                                binary_masks = binary_masks.cpu().numpy()
                            valid_count = sum(1 for m in binary_masks if m.sum() > 0)
                            if valid_count > 0:
                                categories_found[category] = valid_count
                                print(f"[SAM3] Found {valid_count} '{category}' object(s)")
                except Exception as e:
                    continue
        finally:
            self._close_session(session_id)
        
        return categories_found

    def _propagate_single_category(
        self, 
        video_path: str, 
        category: str, 
        category_index: int,
        progress_callback: Optional[callable],
        base_progress: int,
        total_categories: int
    ) -> Optional[dict]:
        """Run full propagation for a single object category."""
        session_id = self._create_session(video_path)
        
        try:
            # Add prompt for this category
            response = self.predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": category
            })
            
            outputs = response.get("outputs", {})
            obj_ids = outputs.get("out_obj_ids", [])
            if isinstance(obj_ids, np.ndarray):
                obj_ids = obj_ids.tolist()
            
            if not obj_ids:
                return None
            
            # Map SAM object IDs to our tracking IDs
            sam_id_to_our_id = {}
            for sam_obj_id in obj_ids:
                our_obj_id = self.next_obj_id
                self.next_obj_id += 1
                sam_id_to_our_id[sam_obj_id] = our_obj_id
                
                # Create tracked object
                self.tracked_objects[our_obj_id] = TrackedObject(
                    obj_id=our_obj_id,
                    category_prompt=category,
                    first_seen_frame=0,
                    last_seen_frame=0,
                    bbox_history=[],
                    confidence_history=[],
                    is_active=True
                )
            
            # Propagate through video
            individual_masks = {}  # frame_idx -> {our_obj_id -> mask}
            frame_count = 0
            
            for result in self.predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
                "max_frame_num_to_track": 500
            }):
                frame_idx = result["frame_index"]
                frame_outputs = result.get("outputs", {})
                frame_count = max(frame_count, frame_idx + 1)
                
                # Extract masks
                out_obj_ids = frame_outputs.get("out_obj_ids", [])
                binary_masks = frame_outputs.get("out_binary_masks")
                
                if isinstance(out_obj_ids, np.ndarray):
                    out_obj_ids = out_obj_ids.tolist()
                if binary_masks is not None and hasattr(binary_masks, 'cpu'):
                    binary_masks = binary_masks.cpu().numpy()
                
                if frame_idx not in individual_masks:
                    individual_masks[frame_idx] = {}
                
                if binary_masks is not None:
                    for i, sam_obj_id in enumerate(out_obj_ids):
                        if sam_obj_id in sam_id_to_our_id and i < len(binary_masks):
                            our_obj_id = sam_id_to_our_id[sam_obj_id]
                            mask = binary_masks[i]
                            if mask.sum() > 0:
                                individual_masks[frame_idx][our_obj_id] = mask.astype(np.float32)
                                self.tracked_objects[our_obj_id].last_seen_frame = frame_idx
                
                # Progress update
                if progress_callback and frame_idx % 20 == 0:
                    cat_progress = base_progress + int((frame_idx / 200) * (70 / total_categories))
                    progress_callback(min(cat_progress, 85), f"Tracking '{category}' frame {frame_idx}...")
            
            return {
                "individual_masks": individual_masks,
                "frame_count": frame_count,
            }
            
        except Exception as e:
            print(f"[SAM3] Error propagating category '{category}': {e}")
            return None
        finally:
            self._close_session(session_id)

    def _create_session(self, video_path: str) -> str:
        """
        Create a new video processing session.

        Args:
            video_path: Path to the video file.

        Returns:
            str: Session ID for this video.
        """
        # SAM 3 uses handle_request API
        response = self.predictor.handle_request({
            "type": "start_session",
            "resource_path": video_path
        })
        return response["session_id"]

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute Intersection over Union between two masks.

        Args:
            mask1: First binary mask.
            mask2: Second binary mask.

        Returns:
            IoU score between 0 and 1.
        """
        if mask1.shape != mask2.shape:
            return 0.0
        
        intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5).sum()
        union = np.logical_or(mask1 > 0.5, mask2 > 0.5).sum()
        
        if union == 0:
            return 0.0
        return intersection / union

    def _combine_masks(self, masks: list[np.ndarray]) -> np.ndarray:
        """
        Combine multiple masks into a single mask using logical OR.

        Args:
            masks: List of binary masks.

        Returns:
            Combined mask.
        """
        if not masks:
            return None
        
        if len(masks) == 1:
            return masks[0].astype(np.float32)
        
        # Ensure all masks have same shape
        target_shape = masks[0].shape
        valid_masks = [m for m in masks if m.shape == target_shape]
        
        if not valid_masks:
            return None
        
        stacked = np.stack(valid_masks, axis=0)
        return stacked.any(axis=0).astype(np.float32)

    def _close_session(self, session_id: str) -> None:
        """
        Close a video processing session and free resources.

        Args:
            session_id: The session ID to close.
        """
        # SAM 3 uses handle_request API to close session
        if self.predictor is not None:
            try:
                self.predictor.handle_request({
                    "type": "close_session",
                    "session_id": session_id
                })
            except Exception:
                pass

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
