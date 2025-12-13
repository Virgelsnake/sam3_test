"""
SAM3 Service - Wrapper for SAM 3 video segmentation.

Provides a clean interface for video segmentation with text prompts.
"""

import os
from typing import Optional

import numpy as np
import torch


class SAM3Service:
    """Service class for SAM 3 video segmentation."""

    def __init__(self, device: str = "cuda:0"):
        """
        Initialize the SAM3 service.

        Args:
            device: Device to run inference on (e.g., "cuda:0").
        """
        self.device = device
        self.predictor = None
        self._initialized = False

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
        Process a video with SAM 3 segmentation using a text prompt.

        Args:
            video_path: Path to the input video file.
            prompt: Text prompt describing the object to segment.
            progress_callback: Optional callback for progress updates.
                              Called with (progress_percent, message).

        Returns:
            dict: Results containing:
                - masks: dict[int, np.ndarray] - Frame index to mask mapping
                - frame_count: int - Total frames processed
                - objects_detected: int - Number of objects found
        """
        if not self._initialized:
            self.initialize()

        if progress_callback:
            progress_callback(10, "Creating video session...")

        # Create a session for this video
        session_id = self._create_session(video_path)

        try:
            if progress_callback:
                progress_callback(20, "Adding text prompt...")

            # Add the text prompt to identify objects
            prompt_result = self._add_text_prompt(session_id, prompt)
            objects_detected = prompt_result.get("objects_detected", 0)

            if objects_detected == 0:
                return {
                    "masks": {},
                    "frame_count": 0,
                    "objects_detected": 0,
                    "message": f"No objects matching '{prompt}' found in video",
                }

            if progress_callback:
                progress_callback(30, f"Found {objects_detected} object(s). Propagating masks...")

            # Propagate masks through the video
            masks = self._propagate_masks(session_id, progress_callback)

            if progress_callback:
                progress_callback(90, "Finalizing results...")

            return {
                "masks": masks,
                "frame_count": len(masks),
                "objects_detected": objects_detected,
            }

        finally:
            # Always close the session
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

    def _add_text_prompt(self, session_id: str, prompt: str) -> dict:
        """
        Add a text prompt to identify objects in the video.

        Args:
            session_id: The session ID.
            prompt: Text description of the object to segment.

        Returns:
            dict: Result containing objects_detected count.
        """
        # SAM 3 uses handle_request API with add_prompt type
        response = self.predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt
        })

        outputs = response.get("outputs", {})
        
        # SAM3 returns outputs with 'out_obj_ids' containing detected object IDs
        obj_ids = outputs.get("out_obj_ids", [])
        if isinstance(obj_ids, np.ndarray):
            obj_ids = obj_ids.tolist()
        objects_detected = len(obj_ids) if obj_ids else 0
        
        print(f"[SAM3] add_prompt response keys: {list(response.keys())}")
        print(f"[SAM3] outputs type: {type(outputs)}")
        print(f"[SAM3] outputs keys: {list(outputs.keys()) if isinstance(outputs, dict) else 'N/A'}")
        print(f"[SAM3] detected obj_ids from add_prompt: {obj_ids}")

        # If no objects found with the given prompt, try common object categories
        if objects_detected == 0:
            print(f"[SAM3] No objects found with prompt '{prompt}', trying specific categories...")
            
            # Try common office/general object categories
            fallback_prompts = [
                "person", "chair", "desk", "monitor", "computer", 
                "keyboard", "mouse", "phone", "cup", "bottle",
                "laptop", "table", "lamp", "plant", "book"
            ]
            
            all_obj_ids = set()
            for fallback_prompt in fallback_prompts:
                try:
                    fb_response = self.predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": fallback_prompt
                    })
                    fb_outputs = fb_response.get("outputs", {})
                    fb_obj_ids = fb_outputs.get("out_obj_ids", [])
                    if isinstance(fb_obj_ids, np.ndarray):
                        fb_obj_ids = fb_obj_ids.tolist()
                    
                    if fb_obj_ids:
                        print(f"[SAM3] Found {len(fb_obj_ids)} object(s) with prompt '{fallback_prompt}'")
                        all_obj_ids.update(fb_obj_ids)
                except Exception as e:
                    print(f"[SAM3] Error with fallback prompt '{fallback_prompt}': {e}")
            
            objects_detected = len(all_obj_ids)
            print(f"[SAM3] Total unique objects found with fallback prompts: {objects_detected}")

        return {
            "objects_detected": objects_detected,
            "initial_outputs": outputs,
        }

    def _propagate_masks(
        self,
        session_id: str,
        progress_callback: Optional[callable] = None,
    ) -> dict[int, np.ndarray]:
        """
        Propagate masks through all video frames.

        Args:
            session_id: The session ID.
            progress_callback: Optional progress callback.

        Returns:
            dict[int, np.ndarray]: Mapping of frame index to combined mask.
        """
        masks = {}

        # SAM 3 uses handle_stream_request for propagation
        # Limit to 300 frames (~10s at 30fps) to prevent OOM on A10G
        for result in self.predictor.handle_stream_request({
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",  # Changed from "both" to reduce memory
            "start_frame_index": 0,
            "max_frame_num_to_track": 300  # Limit frames to prevent OOM
        }):
            frame_idx = result["frame_index"]
            frame_outputs = result.get("outputs", {})

            if frame_outputs:
                # SAM3 returns: {'out_obj_ids', 'out_probs', 'out_boxes_xywh', 'out_binary_masks', 'frame_stats'}
                # Extract binary masks and combine them
                binary_masks = frame_outputs.get("out_binary_masks")
                
                if binary_masks is not None and isinstance(binary_masks, np.ndarray):
                    # binary_masks shape is typically (N, H, W) where N is number of objects
                    if binary_masks.ndim == 3:
                        # Combine all object masks into one using logical OR
                        combined_mask = binary_masks.any(axis=0).astype(np.float32)
                    elif binary_masks.ndim == 2:
                        # Single mask
                        combined_mask = binary_masks.astype(np.float32)
                    else:
                        combined_mask = None

                if combined_mask is not None:
                    masks[frame_idx] = combined_mask

            # Update progress (30% to 90% range)
            if progress_callback and frame_idx % 10 == 0:
                progress = min(30 + (frame_idx * 0.5), 85)
                progress_callback(int(progress), f"Processing frame {frame_idx}...")

        return masks

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
