"""
Video processing utilities for SAM3 worker.

Handles video download, frame extraction, and output video creation.
"""

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def download_video(url: str, output_path: Optional[str] = None) -> str:
    """
    Download video from a URL to a local file.

    Args:
        url: URL to download from (e.g., Supabase signed URL).
        output_path: Optional path to save the video. If None, creates a temp file.

    Returns:
        str: Path to the downloaded video file.
    """
    import httpx

    if output_path is None:
        # Create temp file with .mp4 extension
        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

    with httpx.Client(timeout=300.0) as client:
        response = client.get(url)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

    return output_path


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.

    Args:
        video_path: Path to the video file.

    Returns:
        dict: Video info including fps, frame_count, width, height, duration.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def extract_frames(video_path: str, max_frames: Optional[int] = None) -> list[np.ndarray]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the video file.
        max_frames: Optional maximum number of frames to extract.

    Returns:
        list[np.ndarray]: List of frames as RGB numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    return frames


def create_mask_video(
    masks: dict[int, np.ndarray],
    output_path: str,
    fps: float,
    width: int,
    height: int,
) -> str:
    """
    Create a grayscale mask video from segmentation masks.

    Args:
        masks: Dictionary mapping frame index to mask array.
        output_path: Path to save the output video.
        fps: Frames per second for output video.
        width: Video width.
        height: Video height.

    Returns:
        str: Path to the created video.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    # Sort by frame index
    for frame_idx in sorted(masks.keys()):
        mask = masks[frame_idx]

        # Ensure mask is 2D and uint8
        if mask.ndim > 2:
            mask = mask.squeeze()
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # Resize if needed
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        out.write(mask)

    out.release()
    return output_path


def create_composite_video(
    frames: list[np.ndarray],
    masks: dict[int, np.ndarray],
    output_path: str,
    fps: float,
    color: tuple[int, int, int] = (0, 255, 0),
    opacity: float = 0.5,
) -> str:
    """
    Create a composite video with mask overlay on original frames.

    Args:
        frames: List of original video frames (RGB).
        masks: Dictionary mapping frame index to mask array.
        output_path: Path to save the output video.
        fps: Frames per second for output video.
        color: RGB color for the mask overlay.
        opacity: Opacity of the mask overlay (0-1).

    Returns:
        str: Path to the created video.
    """
    if not frames:
        raise ValueError("No frames provided")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    for frame_idx, frame in enumerate(frames):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if frame_idx in masks:
            mask = masks[frame_idx]

            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask.squeeze()

            # Resize mask if needed
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # Create colored overlay
            overlay = np.zeros_like(frame_bgr)
            # Convert RGB color to BGR
            overlay[mask > 0.5] = (color[2], color[1], color[0])

            # Blend overlay with original frame
            mask_3d = np.stack([mask] * 3, axis=-1)
            frame_bgr = np.where(
                mask_3d > 0.5,
                cv2.addWeighted(frame_bgr, 1 - opacity, overlay, opacity, 0),
                frame_bgr,
            )

        out.write(frame_bgr.astype(np.uint8))

    out.release()
    return output_path


def upload_to_supabase(
    file_path: str,
    bucket: str,
    storage_path: str,
    supabase_url: str,
    supabase_key: str,
) -> str:
    """
    Upload a file to Supabase storage.

    Args:
        file_path: Local path to the file.
        bucket: Supabase storage bucket name.
        storage_path: Path within the bucket.
        supabase_url: Supabase project URL.
        supabase_key: Supabase service role key.

    Returns:
        str: Public URL of the uploaded file.
    """
    from supabase import create_client

    client = create_client(supabase_url, supabase_key)

    with open(file_path, "rb") as f:
        content = f.read()

    # Determine content type
    content_type = "video/mp4"
    if file_path.endswith(".webm"):
        content_type = "video/webm"

    client.storage.from_(bucket).upload(
        path=storage_path,
        file=content,
        file_options={"content-type": content_type},
    )

    # Get public URL
    url = client.storage.from_(bucket).get_public_url(storage_path)
    return url


def cleanup_temp_files(*paths: str) -> None:
    """
    Remove temporary files.

    Args:
        *paths: Paths to files to remove.
    """
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
