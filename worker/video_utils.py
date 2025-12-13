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
    
    # Convert to H.264 for browser compatibility
    output_path = convert_to_h264(output_path)
    
    return output_path


def convert_to_h264(input_path: str) -> str:
    """
    Convert video to H.264 codec for browser compatibility.
    
    Args:
        input_path: Path to the input video.
        
    Returns:
        str: Path to the converted video (same as input, overwritten).
    """
    import subprocess
    
    temp_output = input_path + ".h264.mp4"
    
    try:
        # Convert using ffmpeg with H.264 codec
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                temp_output
            ],
            check=True,
            capture_output=True,
        )
        
        # Replace original with converted
        os.replace(temp_output, input_path)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
        # If conversion fails, keep original
        if os.path.exists(temp_output):
            os.remove(temp_output)
    except FileNotFoundError:
        print("FFmpeg not found, keeping original codec")
    
    return input_path


def create_composite_video(
    frames: list[np.ndarray],
    masks: dict[int, np.ndarray],
    output_path: str,
    fps: float,
    color: tuple[int, int, int] = (0, 255, 0),
    opacity: float = 0.5,
    individual_masks: dict[int, dict[int, np.ndarray]] = None,
    tracked_objects: dict = None,
) -> tuple[str, dict[str, str]]:
    """
    Create a composite video with mask overlay on original frames.

    Args:
        frames: List of original video frames (RGB).
        masks: Dictionary mapping frame index to combined mask array.
        output_path: Path to save the output video.
        fps: Frames per second for output video.
        color: RGB color for the mask overlay (used if no individual_masks).
        opacity: Opacity of the mask overlay (0-1).
        individual_masks: Optional dict[frame_idx][obj_id] -> mask for multi-color rendering.
        tracked_objects: Optional tracking data for color assignment by category.

    Returns:
        tuple: (path to created video, dict mapping category to hex color)
    """
    if not frames:
        raise ValueError("No frames provided")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    # Define distinct colors for different objects/categories (BGR format)
    CATEGORY_COLORS = {
        "chair": (0, 255, 0),      # Green
        "office chair": (0, 255, 0),
        "desk": (255, 165, 0),     # Orange
        "office desk": (255, 165, 0),
        "table": (255, 165, 0),
        "monitor": (255, 0, 255),  # Magenta
        "computer monitor": (255, 0, 255),
        "computer": (255, 255, 0), # Cyan
        "desktop computer": (255, 255, 0),
        "laptop": (255, 255, 0),
        "keyboard": (0, 255, 255), # Yellow
        "computer keyboard": (0, 255, 255),
        "mouse": (128, 0, 255),    # Pink
        "computer mouse": (128, 0, 255),
        "person": (0, 0, 255),     # Red
        "phone": (255, 128, 0),    # Light blue
        "lamp": (0, 128, 255),     # Orange-ish
        "plant": (0, 128, 0),      # Dark green
        "bottle": (128, 128, 255), # Light pink
        "cup": (128, 255, 128),    # Light green
    }
    
    # Fallback colors for objects without category match
    FALLBACK_COLORS = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Pink
        (255, 128, 0),  # Light blue
        (0, 128, 255),  # Orange
        (128, 255, 0),  # Teal
        (255, 0, 128),  # Purple
        (0, 255, 128),  # Spring green
    ]
    
    # Build object ID to color mapping and category color mapping for UI
    obj_colors = {}
    category_colors_hex = {}  # category -> hex color for frontend
    
    def bgr_to_hex(bgr: tuple) -> str:
        """Convert BGR tuple to hex color string."""
        b, g, r = bgr
        return f"#{r:02X}{g:02X}{b:02X}"
    
    if tracked_objects:
        for obj_id, obj_data in tracked_objects.items():
            category = obj_data.get("category", "").lower()
            if category in CATEGORY_COLORS:
                obj_colors[int(obj_id)] = CATEGORY_COLORS[category]
                category_colors_hex[category] = bgr_to_hex(CATEGORY_COLORS[category])
            else:
                # Use fallback color based on object ID
                fallback_color = FALLBACK_COLORS[int(obj_id) % len(FALLBACK_COLORS)]
                obj_colors[int(obj_id)] = fallback_color
                category_colors_hex[category] = bgr_to_hex(fallback_color)

    for frame_idx, frame in enumerate(frames):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Use individual masks if available for multi-color rendering
        if individual_masks and frame_idx in individual_masks:
            frame_obj_masks = individual_masks[frame_idx]
            
            for obj_id, mask in frame_obj_masks.items():
                if mask is None:
                    continue
                    
                # Ensure mask is 2D
                if mask.ndim > 2:
                    mask = mask.squeeze()

                # Resize mask if needed
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

                # Get color for this object
                obj_color = obj_colors.get(int(obj_id), FALLBACK_COLORS[int(obj_id) % len(FALLBACK_COLORS)])

                # Create colored overlay for this object
                overlay = np.zeros_like(frame_bgr)
                overlay[mask > 0.5] = obj_color

                # Blend overlay with frame
                mask_3d = np.stack([mask] * 3, axis=-1)
                frame_bgr = np.where(
                    mask_3d > 0.5,
                    cv2.addWeighted(frame_bgr, 1 - opacity, overlay, opacity, 0),
                    frame_bgr,
                )
        
        # Fall back to combined mask with single color
        elif frame_idx in masks:
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
    
    # Convert to H.264 for browser compatibility
    output_path = convert_to_h264(output_path)
    
    return output_path, category_colors_hex


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
