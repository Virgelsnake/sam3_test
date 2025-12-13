"""
SAM3 Service - Wrapper for SAM 3 video segmentation.

Provides a clean interface for video segmentation with text prompts.
Supports multi-object detection and dynamic inventory tracking.
Uses GPT-4V for open-vocabulary object identification.
"""

import os
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import torch

# Import classification service for GPT-4V identification
from classification_service import ClassificationService


# Fallback object categories if GPT-4V identification fails
FALLBACK_CATEGORIES = [
    "chair", "desk", "table", "monitor", "computer", "laptop",
    "keyboard", "mouse", "person", "phone", "lamp", "plant",
]

# How often to re-scan for new objects (in frames)
RESCAN_INTERVAL = 30  # Re-scan every 30 frames (~1 second at 30fps)


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
        
        # GPT-4V classifier for open-vocabulary identification
        self.classifier = ClassificationService()
        
        # Tracking state
        self.tracked_objects: dict[int, TrackedObject] = {}
        self.frame_inventories: dict[int, FrameInventory] = {}
        self.next_obj_id = 0
        self.current_session_id = None
        self.known_categories: set = set()  # Categories we've already detected
        
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
        Process a video with SAM 3 segmentation using GPT-4V open-vocabulary detection.

        Flow:
        1. Use GPT-4V to identify objects in frame 0 (open vocabulary)
        2. Run SAM3 propagation for each identified category
        3. Every N frames, re-scan with GPT-4V for new objects
        4. If new objects found, run additional propagation passes

        Args:
            video_path: Path to the input video file.
            prompt: Text prompt (used as context for GPT-4V).
            progress_callback: Optional callback for progress updates.

        Returns:
            dict: Results containing masks, tracking data, and inventory.
        """
        if not self._initialized:
            self.initialize()

        # Reset tracking state for new video
        self.tracked_objects = {}
        self.frame_inventories = {}
        self.next_obj_id = 0
        self.known_categories = set()

        # Extract all frames for re-scanning
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if not frames:
            return {"masks": {}, "individual_masks": {}, "frame_count": 0, "objects_detected": 0, 
                    "inventory_snapshots": [], "tracked_objects": {}, "message": "Could not read video"}

        if progress_callback:
            progress_callback(5, "Using GPT-4V to identify objects...")

        # Step 1: Initial scan on frame 0 using GPT-4V
        categories_found = self._scan_for_categories(video_path, prompt, frame=frames[0])
        self.known_categories = set(categories_found.keys())
        
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

        print(f"[SAM3] Initial scan found {len(categories_found)} categories: {list(categories_found.keys())}")

        # Step 2: Run propagation for initial categories
        all_individual_masks = {}
        all_combined_masks = {}
        frame_count = 0
        inventory_snapshots = []
        
        total_categories = len(categories_found)
        for cat_idx, (category, obj_count) in enumerate(categories_found.items()):
            if progress_callback:
                base_progress = 10 + int((cat_idx / total_categories) * 50)
                progress_callback(base_progress, f"Tracking '{category}' ({obj_count} objects)...")
            
            print(f"[SAM3] Running propagation for category '{category}' ({obj_count} objects)")
            
            category_result = self._propagate_single_category(
                video_path, category, cat_idx, progress_callback, base_progress, total_categories
            )
            
            if category_result:
                frame_count = max(frame_count, category_result["frame_count"])
                for frame_idx, obj_masks in category_result["individual_masks"].items():
                    if frame_idx not in all_individual_masks:
                        all_individual_masks[frame_idx] = {}
                    all_individual_masks[frame_idx].update(obj_masks)

        # Add initial inventory snapshot
        inventory_snapshots.append({
            "frame_index": 0,
            "reason": "Initial inventory",
            "objects": {obj_id: obj.category_prompt for obj_id, obj in self.tracked_objects.items()},
            "object_count": len(self.tracked_objects),
        })

        # Step 3: Periodic re-scanning for new objects
        rescan_frames = list(range(RESCAN_INTERVAL, len(frames), RESCAN_INTERVAL))
        if rescan_frames:
            print(f"[SAM3] Will re-scan at frames: {rescan_frames}")
        
        for rescan_idx, rescan_frame in enumerate(rescan_frames):
            if progress_callback:
                progress_callback(60 + int((rescan_idx / len(rescan_frames)) * 25), 
                                f"Re-scanning frame {rescan_frame} for new objects...")
            
            # Use GPT-4V to identify objects in this frame
            new_categories = self._scan_for_new_categories(
                video_path, prompt, frames[rescan_frame], rescan_frame
            )
            
            if new_categories:
                print(f"[SAM3] Frame {rescan_frame}: Found {len(new_categories)} NEW categories: {list(new_categories.keys())}")
                
                # Run propagation for new categories (starting from this frame)
                for category, obj_count in new_categories.items():
                    category_result = self._propagate_single_category(
                        video_path, category, 0, None, 0, 1, start_frame=rescan_frame
                    )
                    
                    if category_result:
                        for frame_idx, obj_masks in category_result["individual_masks"].items():
                            if frame_idx not in all_individual_masks:
                                all_individual_masks[frame_idx] = {}
                            all_individual_masks[frame_idx].update(obj_masks)
                
                # Add inventory snapshot for new objects
                inventory_snapshots.append({
                    "frame_index": rescan_frame,
                    "reason": f"New objects detected: {list(new_categories.keys())}",
                    "objects": {obj_id: obj.category_prompt for obj_id, obj in self.tracked_objects.items()},
                    "object_count": len(self.tracked_objects),
                })

        # Build combined masks from individual masks
        for frame_idx, obj_masks in all_individual_masks.items():
            masks_list = [m for m in obj_masks.values() if m is not None and m.sum() > 0]
            if masks_list:
                all_combined_masks[frame_idx] = self._combine_masks(masks_list)

        if progress_callback:
            progress_callback(90, "Finalizing results...")

        print(f"[SAM3] Tracking complete: {len(self.tracked_objects)} unique objects, {len(all_combined_masks)} frames")

        return {
            "masks": all_combined_masks,
            "individual_masks": all_individual_masks,
            "frame_count": len(all_combined_masks),
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

    def _scan_for_new_categories(self, video_path: str, user_prompt: str, frame: np.ndarray, frame_idx: int) -> dict:
        """
        Re-scan a frame for NEW objects not already being tracked.
        
        Args:
            video_path: Path to video file.
            user_prompt: User's prompt.
            frame: Frame to scan.
            frame_idx: Frame index.
            
        Returns:
            dict: NEW categories found (not in self.known_categories).
        """
        # Use GPT-4V to identify objects
        try:
            self.classifier.initialize()
            gpt4v_categories = self.classifier.identify_objects_in_frame(
                frame, 
                context=user_prompt if user_prompt else "inventory scan"
            )
        except Exception as e:
            print(f"[SAM3] GPT-4V re-scan failed at frame {frame_idx}: {e}")
            return {}
        
        # Filter to only NEW categories
        new_categories = {}
        for cat in gpt4v_categories:
            cat_normalized = cat.lower().strip()
            # Check if this is truly new (not a variant of existing)
            is_new = True
            for known in self.known_categories:
                if cat_normalized in known or known in cat_normalized:
                    is_new = False
                    break
            
            if is_new and cat_normalized not in self.known_categories:
                # Verify with SAM3 that objects exist
                session_id = self._create_session(video_path)
                try:
                    response = self.predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": frame_idx,
                        "text": cat_normalized
                    })
                    outputs = response.get("outputs", {})
                    obj_ids = outputs.get("out_obj_ids", [])
                    if isinstance(obj_ids, np.ndarray):
                        obj_ids = obj_ids.tolist()
                    
                    if obj_ids:
                        binary_masks = outputs.get("out_binary_masks")
                        if binary_masks is not None:
                            if hasattr(binary_masks, 'cpu'):
                                binary_masks = binary_masks.cpu().numpy()
                            valid_count = sum(1 for m in binary_masks if m.sum() > 0)
                            if valid_count > 0:
                                new_categories[cat_normalized] = valid_count
                                self.known_categories.add(cat_normalized)
                finally:
                    self._close_session(session_id)
        
        return new_categories

    def _scan_for_categories(self, video_path: str, user_prompt: str, frame: np.ndarray = None) -> dict:
        """
        Use GPT-4V to identify objects in the frame, then verify with SAM3.
        
        Args:
            video_path: Path to video file.
            user_prompt: User's prompt (used as context).
            frame: Optional frame to scan. If None, extracts frame 0.
            
        Returns:
            dict: Category -> object count mapping.
        """
        categories_found = {}
        
        # Step 1: Use GPT-4V to identify objects (open-vocabulary)
        if frame is None:
            # Extract frame 0 for identification
            import cv2
            cap = cv2.VideoCapture(video_path)
            ret, frame_bgr = cap.read()
            cap.release()
            if ret:
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                frame = None
        
        gpt4v_categories = []
        if frame is not None:
            try:
                self.classifier.initialize()
                gpt4v_categories = self.classifier.identify_objects_in_frame(
                    frame, 
                    context=user_prompt if user_prompt else "inventory scan"
                )
                print(f"[SAM3] GPT-4V identified categories: {gpt4v_categories}")
            except Exception as e:
                print(f"[SAM3] GPT-4V identification failed: {e}, using fallback categories")
                gpt4v_categories = []
        
        # Step 2: Build category list - GPT-4V results first, then fallbacks
        categories_to_try = []
        
        # Add GPT-4V identified categories
        for cat in gpt4v_categories:
            # Normalize the category name for SAM3
            cat_normalized = cat.lower().strip()
            if cat_normalized and cat_normalized not in categories_to_try:
                categories_to_try.append(cat_normalized)
        
        # Add fallback categories if GPT-4V didn't find much
        if len(categories_to_try) < 3:
            for cat in FALLBACK_CATEGORIES:
                if cat not in categories_to_try:
                    categories_to_try.append(cat)
        
        # Add user prompt if it's specific
        if user_prompt and user_prompt.lower() not in ["generate an inventory of detected items", "inventory"]:
            if user_prompt.lower() not in categories_to_try:
                categories_to_try.insert(0, user_prompt.lower())
        
        print(f"[SAM3] Scanning for {len(categories_to_try)} object categories: {categories_to_try[:10]}...")
        
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
        total_categories: int,
        start_frame: int = 0
    ) -> Optional[dict]:
        """Run full propagation for a single object category.
        
        Args:
            start_frame: Frame index to start detection from (for new objects appearing later).
        """
        session_id = self._create_session(video_path)
        
        try:
            # Add prompt for this category at the start frame
            response = self.predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": start_frame,
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
                    first_seen_frame=start_frame,
                    last_seen_frame=start_frame,
                    bbox_history=[],
                    confidence_history=[],
                    is_active=True
                )
            
            # Propagate through video (forward from start_frame)
            individual_masks = {}  # frame_idx -> {our_obj_id -> mask}
            frame_count = 0
            
            for result in self.predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": start_frame,
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
