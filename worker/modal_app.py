"""
SAM3 Video Segmentation Worker - Modal Deployment

This module defines the Modal app for GPU-accelerated video segmentation
using SAM 3 (Segment Anything Model 3).

Deploy with: modal deploy worker/modal_app.py
Test with: modal run worker/modal_app.py
"""

import os
import tempfile
import time
from typing import Optional

import modal

# Modal GPU pricing (as of Dec 2024) - $/hour
GPU_PRICING = {
    "A10G": 0.60,        # 24GB VRAM
    "A100-40GB": 2.78,   # 40GB VRAM (A100 PCIe)
    "A100-80GB": 3.22,   # 80GB VRAM (A100 SXM)
}

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
    gpu="A10G",  # 24GB VRAM - sufficient for most videos
    image=sam3_image,
    timeout=1800,  # 30 minute timeout
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
        composite_video_path, category_colors = create_composite_video(
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
            "inventory_colors": category_colors,  # Hex colors for each category
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
                "inventory_colors": category_colors,
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


# =============================================================================
# GPU BENCHMARK FUNCTIONS
# =============================================================================
# These functions run the same processing on different GPU types for cost analysis

def _run_benchmark(gpu_type: str, video_url: str, prompt: str) -> dict:
    """
    Internal benchmark runner - measures processing time and calculates cost.
    
    Returns timing data and cost analysis for the given GPU type.
    """
    import torch
    
    from sam3_service import SAM3Service
    from video_utils import download_video, extract_frames, get_video_info, cleanup_temp_files
    
    start_time = time.time()
    temp_files = []
    
    # Track individual phase timings
    timings = {}
    
    # Phase 1: Download
    phase_start = time.time()
    video_path = download_video(video_url)
    temp_files.append(video_path)
    timings["download"] = time.time() - phase_start
    
    # Get video info
    video_info = get_video_info(video_path)
    
    # Phase 2: Extract frames
    phase_start = time.time()
    frames = extract_frames(video_path)
    timings["frame_extraction"] = time.time() - phase_start
    
    # Phase 3: Initialize SAM3
    phase_start = time.time()
    sam3 = SAM3Service()
    sam3.initialize()
    timings["model_init"] = time.time() - phase_start
    
    # Phase 4: Run segmentation (the main GPU-intensive work)
    phase_start = time.time()
    result = sam3.process_video(video_path, prompt, lambda p, m: None)
    timings["segmentation"] = time.time() - phase_start
    
    # Cleanup
    cleanup_temp_files(*temp_files)
    
    # Calculate totals
    total_time = time.time() - start_time
    gpu_time = timings["model_init"] + timings["segmentation"]  # GPU-bound time
    
    # Cost calculation
    hourly_rate = GPU_PRICING.get(gpu_type, 0)
    cost = (gpu_time / 3600) * hourly_rate
    
    return {
        "gpu_type": gpu_type,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "video_info": {
            "frame_count": video_info["frame_count"],
            "fps": video_info["fps"],
            "duration_seconds": video_info["frame_count"] / video_info["fps"],
        },
        "objects_detected": result["objects_detected"],
        "timings": {
            "download_seconds": round(timings["download"], 2),
            "frame_extraction_seconds": round(timings["frame_extraction"], 2),
            "model_init_seconds": round(timings["model_init"], 2),
            "segmentation_seconds": round(timings["segmentation"], 2),
            "total_seconds": round(total_time, 2),
            "gpu_time_seconds": round(gpu_time, 2),
        },
        "cost": {
            "hourly_rate_usd": hourly_rate,
            "job_cost_usd": round(cost, 4),
            "cost_per_frame_usd": round(cost / video_info["frame_count"], 6),
        },
        "performance": {
            "frames_per_second": round(video_info["frame_count"] / timings["segmentation"], 2),
            "seconds_per_frame": round(timings["segmentation"] / video_info["frame_count"], 3),
        },
    }


@app.function(
    gpu="A10G",
    image=sam3_image,
    timeout=1800,
    secrets=[hf_secret, supabase_secret, openai_secret],
)
def benchmark_a10g(video_url: str, prompt: str = "Generate an inventory of detected items") -> dict:
    """Benchmark on A10G GPU (24GB VRAM) - $0.60/hour"""
    return _run_benchmark("A10G", video_url, prompt)


@app.function(
    gpu="A100-40GB",
    image=sam3_image,
    timeout=1800,
    secrets=[hf_secret, supabase_secret, openai_secret],
)
def benchmark_a100_40gb(video_url: str, prompt: str = "Generate an inventory of detected items") -> dict:
    """Benchmark on A100-40GB GPU - $2.78/hour"""
    return _run_benchmark("A100-40GB", video_url, prompt)


@app.function(
    gpu="A100-80GB",
    image=sam3_image,
    timeout=1800,
    secrets=[hf_secret, supabase_secret, openai_secret],
)
def benchmark_a100_80gb(video_url: str, prompt: str = "Generate an inventory of detected items") -> dict:
    """Benchmark on A100-80GB GPU - $3.22/hour"""
    return _run_benchmark("A100-80GB", video_url, prompt)


@app.function(gpu="A10G", image=sam3_image, timeout=120, secrets=[hf_secret])
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
    prompt: str = "Generate an inventory of detected items",
    job_id: str = "test-job",
    benchmark: bool = False,
):
    """
    Local entrypoint for testing the worker.

    Args:
        video_url: URL to a test video.
        prompt: Text prompt for segmentation.
        job_id: Test job ID.
        benchmark: If True, run GPU cost/performance benchmark on all 3 GPU types.
    """
    import json
    
    print("=" * 60)
    print("SAM3 Video Segmentation Worker")
    print("=" * 60)

    if not video_url:
        print("\n⚠️  No video URL provided.")
        print("\nUsage:")
        print("  # Normal processing:")
        print("  modal run worker/modal_app.py --video-url 'https://...' --prompt 'inventory'")
        print("\n  # GPU Benchmark (compares A10G, A100-40GB, A100-80GB):")
        print("  modal run worker/modal_app.py --video-url 'https://...' --benchmark")
        return

    if benchmark:
        print("\n" + "=" * 60)
        print("GPU COST/PERFORMANCE BENCHMARK")
        print("=" * 60)
        print(f"\nVideo URL: {video_url}")
        print(f"Prompt: {prompt}")
        print("\nRunning benchmarks on all 3 GPU types...")
        print("This will take some time as each GPU processes the full video.\n")
        
        results = []
        
        # Run benchmarks sequentially (can't run same video in parallel due to download)
        print("\n[1/3] Benchmarking A10G (24GB, $0.60/hr)...")
        try:
            r1 = benchmark_a10g.remote(video_url, prompt)
            results.append(r1)
            print(f"      ✓ Completed in {r1['timings']['total_seconds']}s, cost: ${r1['cost']['job_cost_usd']:.4f}")
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            results.append({"gpu_type": "A10G", "error": str(e)})
        
        print("\n[2/3] Benchmarking A100-40GB (40GB, $2.78/hr)...")
        try:
            r2 = benchmark_a100_40gb.remote(video_url, prompt)
            results.append(r2)
            print(f"      ✓ Completed in {r2['timings']['total_seconds']}s, cost: ${r2['cost']['job_cost_usd']:.4f}")
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            results.append({"gpu_type": "A100-40GB", "error": str(e)})
        
        print("\n[3/3] Benchmarking A100-80GB (80GB, $3.22/hr)...")
        try:
            r3 = benchmark_a100_80gb.remote(video_url, prompt)
            results.append(r3)
            print(f"      ✓ Completed in {r3['timings']['total_seconds']}s, cost: ${r3['cost']['job_cost_usd']:.4f}")
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            results.append({"gpu_type": "A100-80GB", "error": str(e)})
        
        # Print comparison table
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        valid_results = [r for r in results if "error" not in r]
        
        if valid_results:
            print(f"\n{'GPU Type':<15} {'Time (s)':<12} {'Cost ($)':<12} {'FPS':<10} {'$/Frame':<12}")
            print("-" * 60)
            
            for r in valid_results:
                print(f"{r['gpu_type']:<15} "
                      f"{r['timings']['gpu_time_seconds']:<12.1f} "
                      f"${r['cost']['job_cost_usd']:<11.4f} "
                      f"{r['performance']['frames_per_second']:<10.2f} "
                      f"${r['cost']['cost_per_frame_usd']:<11.6f}")
            
            # Find the winner
            cheapest = min(valid_results, key=lambda x: x['cost']['job_cost_usd'])
            fastest = min(valid_results, key=lambda x: x['timings']['gpu_time_seconds'])
            
            print("\n" + "-" * 60)
            print(f"💰 CHEAPEST: {cheapest['gpu_type']} at ${cheapest['cost']['job_cost_usd']:.4f}/job")
            print(f"⚡ FASTEST:  {fastest['gpu_type']} at {fastest['timings']['gpu_time_seconds']:.1f}s")
            
            if cheapest['gpu_type'] == fastest['gpu_type']:
                print(f"\n🏆 RECOMMENDATION: {cheapest['gpu_type']} is both cheapest AND fastest!")
            else:
                # Calculate value ratio
                time_ratio = cheapest['timings']['gpu_time_seconds'] / fastest['timings']['gpu_time_seconds']
                cost_ratio = fastest['cost']['job_cost_usd'] / cheapest['cost']['job_cost_usd']
                print(f"\n📊 ANALYSIS:")
                print(f"   {fastest['gpu_type']} is {time_ratio:.1f}x faster")
                print(f"   {cheapest['gpu_type']} is {cost_ratio:.1f}x cheaper")
                print(f"\n🏆 RECOMMENDATION: {cheapest['gpu_type']} (best value for cost-conscious workloads)")
        
        # Save full results to file
        print("\n\nFull results saved to: benchmark_results.json")
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
    else:
        # Normal processing mode
        print("\n📋 Running health check...")
        health = health_check.remote()
        print(f"Health: {health}")

        print(f"\n📋 Processing video with prompt: '{prompt}'")
        result = process_video_job.remote(job_id, video_url, prompt)
        print(f"\nResult: {result}")
