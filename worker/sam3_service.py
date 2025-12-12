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
        # SAM 3 uses the video path as the session identifier
        # Initialize the predictor state for this video
        self.predictor.init_state(video_path)
        return video_path

    def _add_text_prompt(self, session_id: str, prompt: str) -> dict:
        """
        Add a text prompt to identify objects in the video.

        Args:
            session_id: The session ID (video path).
            prompt: Text description of the object to segment.

        Returns:
            dict: Result containing objects_detected count.
        """
        # SAM 3 uses add_new_prompt_with_text for text-based segmentation
        # This identifies objects in the first frame matching the text
        frame_idx = 0  # Start from first frame
        obj_ids, masks = self.predictor.add_new_prompt_with_text(
            frame_idx=frame_idx,
            text=prompt,
        )

        return {
            "objects_detected": len(obj_ids) if obj_ids is not None else 0,
            "initial_masks": masks,
        }

    def _propagate_masks(
        self,
        session_id: str,
        progress_callback: Optional[callable] = None,
    ) -> dict[int, np.ndarray]:
        """
        Propagate masks through all video frames.

        Args:
            session_id: The session ID (video path).
            progress_callback: Optional progress callback.

        Returns:
            dict[int, np.ndarray]: Mapping of frame index to combined mask.
        """
        masks = {}

        # Propagate through the video
        # SAM 3's propagate_in_video yields (frame_idx, obj_ids, mask_logits)
        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video():
            # Convert logits to binary mask
            # mask_logits shape: (num_objects, 1, H, W)
            if mask_logits is not None and len(mask_logits) > 0:
                # Combine all object masks into one
                combined_mask = (mask_logits > 0).cpu().numpy()

                # If multiple objects, combine with OR
                if combined_mask.ndim == 4:
                    combined_mask = combined_mask.squeeze(1)  # Remove channel dim
                    combined_mask = combined_mask.any(axis=0)  # Combine objects

                masks[frame_idx] = combined_mask.astype(np.float32)

            # Update progress (30% to 90% range)
            if progress_callback and frame_idx % 10 == 0:
                # Estimate progress based on frame index
                # This is approximate since we don't know total frames upfront
                progress = min(30 + (frame_idx * 0.5), 85)
                progress_callback(int(progress), f"Processing frame {frame_idx}...")

        return masks

    def _close_session(self, session_id: str) -> None:
        """
        Close a video processing session and free resources.

        Args:
            session_id: The session ID to close.
        """
        # Reset predictor state
        if self.predictor is not None:
            self.predictor.reset_state()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
