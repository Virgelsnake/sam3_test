"""
SAM3 Video Segmentation Worker - Modal Deployment

This module defines the Modal app for GPU-accelerated video segmentation
using SAM 3 (Segment Anything Model 3).

Deploy with: modal deploy worker/modal_app.py
Test with: modal run worker/modal_app.py
"""

import os
import tempfile
from typing import Optional

import modal

# Modal app definition
app = modal.App("sam3-video-segmentation")

# Build the container image with all dependencies
sam3_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "build-essential",
    )
    .pip_install(
        # Core ML
        "torch",
        "torchvision",
        "einops",
        # Video processing
        "decord",
        "opencv-python-headless",
        "scipy",
        "Pillow",
        "matplotlib",
        # SAM 3 dependencies
        "pycocotools",
        "cython",
        "hydra-core",
        "omegaconf",
        "submitit",
        "accelerate",
        "huggingface_hub",
        # Supabase client
        "supabase",
        # HTTP client
        "httpx",
    )
    .pip_install(
        # Install SAM 3 from GitHub
        "sam3 @ git+https://github.com/facebookresearch/sam3.git"
    )
    .pip_install("openai")
    .add_local_file("worker/sam3_service.py", "/root/sam3_service.py")
    .add_local_file("worker/video_utils.py", "/root/video_utils.py")
    .add_local_file("worker/classification_service.py", "/root/classification_service.py")
)

# Secrets for HuggingFace, Supabase, and OpenAI
hf_secret = modal.Secret.from_name("huggingface-secret")
supabase_secret = modal.Secret.from_name("supabase-secret")
openai_secret = modal.Secret.from_name("openai-secret")

@app.function(
    gpu="A100-80GB",  # Need 80GB - SAM3 memory grows with frames processed
    image=sam3_image,
    timeout=900,  # Increased timeout for longer videos
    secrets=[hf_secret, supabase_secret, openai_secret],
    retries=1,
)
def process_video_job(
    job_id: str,
    video_url: str,
    prompt: str,
    callback_url: Optional[str] = None,
) -> dict:
    """
    Process a video segmentation job.

    This function:
    1. Downloads the video from Supabase
    2. Runs SAM 3 inference with the text prompt
    3. Creates mask and composite output videos
    4. Uploads results to Supabase
    5. Optionally calls back to the API with results

    Args:
        job_id: Unique identifier for this job.
        video_url: Signed URL to download the input video.
        prompt: Text prompt describing the object to segment.
        callback_url: Optional URL to POST results to.

    Returns:
        dict: Processing results including output URLs.
    """
    import httpx

    from sam3_service import SAM3Service
    from classification_service import ClassificationService
    from video_utils import (
        cleanup_temp_files,
        create_composite_video,
        create_mask_video,
        download_video,
        extract_frames,
        get_video_info,
        upload_to_supabase,
    )

    print(f"[Job {job_id}] Starting video segmentation")
    print(f"[Job {job_id}] Prompt: {prompt}")

    # Get Supabase credentials from environment
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    temp_files = []

    try:
        # Step 1: Download video
        print(f"[Job {job_id}] Downloading video...")
        video_path = download_video(video_url)
        temp_files.append(video_path)

        # Get video info
        video_info = get_video_info(video_path)
        print(f"[Job {job_id}] Video: {video_info['frame_count']} frames, {video_info['fps']:.1f} fps")

        # Step 2: Extract frames
        print(f"[Job {job_id}] Extracting frames...")
        frames = extract_frames(video_path)

        # Step 3: Initialize SAM 3 and process
        print(f"[Job {job_id}] Initializing SAM 3...")
        sam3 = SAM3Service()
        sam3.initialize()

        def progress_callback(progress: int, message: str):
            print(f"[Job {job_id}] [{progress}%] {message}")

        print(f"[Job {job_id}] Running segmentation with dynamic inventory tracking...")
        result = sam3.process_video(video_path, prompt, progress_callback)

        masks = result["masks"]
        individual_masks = result.get("individual_masks", {})
        objects_detected = result["objects_detected"]
        frame_count = result["frame_count"]
        inventory_snapshots = result.get("inventory_snapshots", [])
        tracked_objects = result.get("tracked_objects", {})

        if objects_detected == 0:
            return {
                "job_id": job_id,
                "status": "completed",
                "objects_detected": 0,
                "frame_count": 0,
                "message": f"No objects found in video",
            }

        # Log tracking results
        print(f"[Job {job_id}] Tracking complete: {objects_detected} unique objects detected")
        print(f"[Job {job_id}] Inventory snapshots: {len(inventory_snapshots)}")
        for snapshot in inventory_snapshots:
            print(f"[Job {job_id}]   Frame {snapshot['frame_index']}: {snapshot['objects']} ({snapshot['reason']})")

        # Step 4: Classify/verify detected objects using OpenAI GPT-4V
        print(f"[Job {job_id}] Running object classification with GPT-4V...")
        inventory = {}
        try:
            classifier = ClassificationService()
            classifier.initialize()
            
            classification_result = classifier.classify_video_sample(
                frames=frames,
                all_masks=masks,
                context=prompt,  # Use user's prompt as context
                individual_masks=individual_masks,
                tracked_objects=tracked_objects,
            )
            
            inventory = classification_result.get("inventory", {})
            print(f"[Job {job_id}] Classification complete. Inventory: {inventory}")
        except Exception as e:
            print(f"[Job {job_id}] Classification failed, using tracking data: {e}")
            # Fall back to tracking-based inventory
            if tracked_objects:
                for obj_id, obj_data in tracked_objects.items():
                    category = obj_data.get("category", "unknown").lower()
                    inventory[category] = inventory.get(category, 0) + 1

        # Step 5: Create output videos
        print(f"[Job {job_id}] Creating mask video...")
        mask_video_path = tempfile.mktemp(suffix="_mask.mp4")
        temp_files.append(mask_video_path)
        create_mask_video(
            masks,
            mask_video_path,
            video_info["fps"],
            video_info["width"],
            video_info["height"],
        )

        print(f"[Job {job_id}] Creating composite video...")
        composite_video_path = tempfile.mktemp(suffix="_composite.mp4")
        temp_files.append(composite_video_path)
        create_composite_video(
            frames,
            masks,
            composite_video_path,
            video_info["fps"],
            color=(0, 255, 0),  # Green overlay (fallback)
            opacity=0.5,
            individual_masks=individual_masks,
            tracked_objects=tracked_objects,
        )

        # Step 6: Upload results to Supabase
        print(f"[Job {job_id}] Uploading results...")
        mask_url = upload_to_supabase(
            mask_video_path,
            "outputs",
            f"{job_id}/mask.mp4",
            supabase_url,
            supabase_key,
        )

        composite_url = upload_to_supabase(
            composite_video_path,
            "outputs",
            f"{job_id}/composite.mp4",
            supabase_url,
            supabase_key,
        )

        result = {
            "job_id": job_id,
            "status": "completed",
            "mask_video_url": mask_url,
            "composite_video_url": composite_url,
            "frame_count": frame_count,
            "objects_detected": objects_detected,
            "inventory": inventory,
            "inventory_snapshots": inventory_snapshots,
            "tracked_objects": tracked_objects,
        }

        # Step 7: Update job directly in Supabase (more reliable than callback)
        print(f"[Job {job_id}] Updating job in database...")
        try:
            from supabase import create_client
            from datetime import datetime
            
            supabase_client = create_client(supabase_url, supabase_key)
            update_data = {
                "status": "completed",
                "mask_video_url": mask_url,
                "composite_video_url": composite_url,
                "frame_count": frame_count,
                "objects_detected": objects_detected,
                "inventory": inventory,
                "completed_at": datetime.utcnow().isoformat(),
            }
            supabase_client.table("jobs").update(update_data).eq("id", job_id).execute()
            print(f"[Job {job_id}] Database updated successfully")
        except Exception as e:
            print(f"[Job {job_id}] Database update failed: {e}")
            
            # Fallback: try callback
            if callback_url:
                print(f"[Job {job_id}] Trying callback to {callback_url}")
                try:
                    with httpx.Client(timeout=30.0) as client:
                        client.post(callback_url, json=result)
                except Exception as cb_error:
                    print(f"[Job {job_id}] Callback also failed: {cb_error}")

        print(f"[Job {job_id}] Completed successfully!")
        return result

    except Exception as e:
        error_result = {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
        }

        # Try to send error callback
        if callback_url:
            try:
                with httpx.Client(timeout=30.0) as client:
                    client.post(callback_url, json=error_result)
            except Exception:
                pass

        raise

    finally:
        # Cleanup temp files
        cleanup_temp_files(*temp_files)


@app.function(gpu="A100-80GB", image=sam3_image, timeout=120, secrets=[hf_secret])
def health_check() -> dict:
    """
    Health check function to verify the worker is operational.

    Returns:
        dict: Health status including GPU info.
    """
    import torch

    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
    }


@app.local_entrypoint()
def main(
    video_url: str = "",
    prompt: str = "person",
    job_id: str = "test-job",
):
    """
    Local entrypoint for testing the worker.

    Args:
        video_url: URL to a test video.
        prompt: Text prompt for segmentation.
        job_id: Test job ID.
    """
    print("=" * 60)
    print("SAM3 Video Segmentation Worker")
    print("=" * 60)

    # Run health check first
    print("\n📋 Running health check...")
    health = health_check.remote()
    print(f"Health: {health}")

    if not video_url:
        print("\n⚠️  No video URL provided. Use --video-url to test processing.")
        print("Example:")
        print("  modal run worker/modal_app.py --video-url 'https://...' --prompt 'person'")
        return

    # Process the video
    print(f"\n📋 Processing video with prompt: '{prompt}'")
    result = process_video_job.remote(job_id, video_url, prompt)
    print(f"\nResult: {result}")
