#!/usr/bin/env python3
"""
GPU Benchmark Script for SAM3 Video Segmentation

This script:
1. Uploads a local video to Supabase storage
2. Gets a signed URL for the video
3. Runs the GPU benchmark on all 3 GPU types (A10G, A100-40GB, A100-80GB)
4. Outputs a cost/performance comparison

Usage:
    python scripts/run_gpu_benchmark.py /path/to/video.mov

Requirements:
    - SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables
    - Modal CLI authenticated
"""

import json
import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

def upload_video_to_supabase(video_path: str) -> str:
    """Upload video to Supabase and return a signed URL."""
    from supabase import create_client
    
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    
    client = create_client(supabase_url, supabase_key)
    
    # Generate unique filename
    import uuid
    filename = f"benchmark/{uuid.uuid4()}{Path(video_path).suffix}"
    
    print(f"Uploading {video_path} to Supabase...")
    
    with open(video_path, "rb") as f:
        video_data = f.read()
    
    # Upload to 'uploads' bucket
    result = client.storage.from_("uploads").upload(
        filename,
        video_data,
        file_options={"content-type": "video/quicktime"}
    )
    
    # Get signed URL (valid for 1 hour)
    signed_url = client.storage.from_("uploads").create_signed_url(
        filename,
        3600  # 1 hour expiry
    )
    
    print(f"Video uploaded successfully!")
    return signed_url["signedURL"]


def run_benchmark(video_url: str):
    """Run the Modal GPU benchmark."""
    print("\nStarting GPU benchmark...")
    print("This will run the same video through 3 different GPU types.")
    print("Expected time: 15-30 minutes total\n")
    
    cmd = [
        "modal", "run", "worker/modal_app.py",
        "--video-url", video_url,
        "--benchmark"
    ]
    
    # Run in the project root
    project_root = Path(__file__).parent.parent
    
    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output
    for line in process.stdout:
        print(line, end="")
    
    process.wait()
    return process.returncode


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_gpu_benchmark.py /path/to/video.mov")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("SAM3 GPU COST/PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"\nVideo: {video_path}")
    print(f"Size: {os.path.getsize(video_path) / 1024 / 1024:.1f} MB")
    
    # Step 1: Upload video
    try:
        video_url = upload_video_to_supabase(video_path)
        print(f"Signed URL: {video_url[:80]}...")
    except Exception as e:
        print(f"\nError uploading video: {e}")
        print("\nMake sure SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set.")
        sys.exit(1)
    
    # Step 2: Run benchmark
    exit_code = run_benchmark(video_url)
    
    if exit_code == 0:
        print("\n✓ Benchmark completed successfully!")
        print("Results saved to benchmark_results.json")
    else:
        print(f"\n✗ Benchmark failed with exit code {exit_code}")
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
