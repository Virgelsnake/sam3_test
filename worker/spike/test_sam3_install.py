"""
SAM 3 Installation Test
========================
Tests that SAM 3 can be installed and loaded on Modal GPU.

This is Phase 1.2 of the tech spike - verifying SAM 3 works.

Run with: modal run worker/spike/test_sam3_install.py
"""

import modal

app = modal.App("sam3-install-test")

# Build image with SAM 3 dependencies
# This may take a few minutes on first run as it installs PyTorch and SAM 3
# Use a simpler approach - install SAM 3 first with its deps, then add missing
sam3_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg", 
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "build-essential",  # For compiling pycocotools
    )
    # Install core dependencies first
    .pip_install(
        "torch",
        "torchvision", 
        "einops",
        "decord",
        "opencv-python-headless",
        "scipy",
        "Pillow",
        "matplotlib",
        "pycocotools",      # COCO dataset tools required by SAM 3
        "cython",           # Required for pycocotools
        "hydra-core",       # Config management often used in Meta projects
        "omegaconf",        # Required by hydra
        "submitit",         # Job submission for Meta projects
        "accelerate",       # HuggingFace accelerate
    )
    # Install SAM 3 from GitHub (will install its own deps)
    .pip_install(
        "sam3 @ git+https://github.com/facebookresearch/sam3.git"
    )
)


# HuggingFace secret for gated model access
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(gpu="T4", image=sam3_image, timeout=300)
def test_sam3_import():
    """Test that SAM 3 can be imported."""
    import torch
    
    print("=" * 60)
    print("🔍 SAM 3 IMPORT TEST")
    print("=" * 60)
    
    # Check PyTorch and CUDA first
    print(f"\n✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Try importing SAM 3 components
    print("\n📦 Importing SAM 3 modules...")
    
    try:
        import sam3
        print(f"✅ sam3 imported successfully")
        print(f"   Location: {sam3.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import sam3: {e}")
        return {"status": "error", "message": str(e)}
    
    try:
        from sam3.model_builder import build_sam3_video_predictor
        print("✅ build_sam3_video_predictor imported")
    except ImportError as e:
        print(f"❌ Failed to import build_sam3_video_predictor: {e}")
        return {"status": "error", "message": str(e)}
    
    try:
        from sam3.model_builder import build_sam3_image_model
        print("✅ build_sam3_image_model imported")
    except ImportError as e:
        print(f"⚠️ build_sam3_image_model not available: {e}")
    
    print("\n" + "=" * 60)
    print("✅ SAM 3 IMPORT TEST PASSED!")
    print("=" * 60)
    
    return {
        "status": "success",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "sam3_location": sam3.__file__
    }


@app.function(gpu="T4", image=sam3_image, timeout=600, secrets=[hf_secret])
def test_sam3_model_load():
    """Test that SAM 3 video predictor can be loaded."""
    import torch
    import time
    import os
    
    print("=" * 60)
    print("🔍 SAM 3 MODEL LOAD TEST")
    print("=" * 60)
    
    # Login to HuggingFace using the secret
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("\n🔐 Authenticating with HuggingFace...")
        from huggingface_hub import login
        login(token=hf_token)
        print("✅ HuggingFace authentication successful!")
    else:
        print("⚠️ No HF_TOKEN found. Model download may fail.")
    
    # Enable TF32 for faster computation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print("\n📥 Loading SAM 3 Video Predictor...")
    print("   (This may take 30-60 seconds on first load)")
    
    start_time = time.time()
    
    try:
        from sam3.model_builder import build_sam3_video_predictor
        
        # Build the predictor with GPU 0
        predictor = build_sam3_video_predictor(gpus_to_use=[0])
        
        load_time = time.time() - start_time
        
        print(f"\n✅ Model loaded successfully!")
        print(f"   Load time: {load_time:.1f} seconds")
        
        # Check GPU memory usage
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        
        print(f"\n📊 GPU Memory Usage:")
        print(f"   Allocated: {memory_allocated:.2f} GB")
        print(f"   Reserved: {memory_reserved:.2f} GB")
        
        print("\n" + "=" * 60)
        print("✅ SAM 3 MODEL LOAD TEST PASSED!")
        print("=" * 60)
        
        return {
            "status": "success",
            "load_time_seconds": round(load_time, 1),
            "memory_allocated_gb": round(memory_allocated, 2),
            "memory_reserved_gb": round(memory_reserved, 2)
        }
        
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.local_entrypoint()
def main(full: bool = False):
    """
    Run SAM 3 installation tests.
    
    Args:
        full: If True, also test model loading (slower, uses more GPU time)
    """
    print("\n" + "=" * 60)
    print("🚀 SAM 3 INSTALLATION TEST")
    print("=" * 60)
    print("\nThis test verifies SAM 3 is correctly installed on Modal.")
    print("First run may take 5-10 minutes to build the container image.\n")
    
    # Test 1: Import test
    print("📋 Test 1: SAM 3 Import Test...")
    print("-" * 40)
    import_result = test_sam3_import.remote()
    print(f"\nResult: {import_result}")
    
    if import_result["status"] != "success":
        print("\n❌ Import test failed. Cannot proceed.")
        return
    
    # Test 2: Model load test (optional)
    if full:
        print("\n" + "=" * 60)
        print("📋 Test 2: SAM 3 Model Load Test...")
        print("-" * 40)
        print("⚠️  This will download model weights (~2GB) and load to GPU.")
        print("    Expected time: 1-2 minutes\n")
        
        load_result = test_sam3_model_load.remote()
        print(f"\nResult: {load_result}")
    else:
        print("\n💡 Tip: Run with --full to also test model loading:")
        print("   modal run worker/spike/test_sam3_install.py --full")
    
    print("\n" + "=" * 60)
    print("🎉 SAM 3 INSTALLATION TESTS COMPLETED!")
    print("=" * 60)
