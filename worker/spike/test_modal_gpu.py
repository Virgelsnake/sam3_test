"""
Modal GPU Test Script
=====================
Verifies that Modal is correctly configured and can access GPUs.

Run with: modal run worker/spike/test_modal_gpu.py
"""

import modal

# Create a Modal app
app = modal.App("sam3-gpu-test")

# Define a simple image with PyTorch
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "numpy"
)


@app.function(gpu="T4", image=image, timeout=120)
def test_gpu():
    """Test that GPU is accessible and working."""
    import subprocess
    import torch
    
    print("=" * 60)
    print("🔍 MODAL GPU TEST")
    print("=" * 60)
    
    # 1. Check nvidia-smi
    print("\n📊 NVIDIA-SMI Output:")
    print("-" * 40)
    subprocess.run(["nvidia-smi"], check=True)
    
    # 2. Check PyTorch CUDA
    print("\n🔥 PyTorch CUDA Check:")
    print("-" * 40)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 3. Quick GPU computation test
        print("\n⚡ GPU Computation Test:")
        print("-" * 40)
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        
        # Time matrix multiplication
        import time
        start = time.time()
        for _ in range(100):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"100x matrix multiplications (1000x1000): {elapsed:.3f}s")
        print(f"Average per operation: {elapsed/100*1000:.2f}ms")
    
    print("\n" + "=" * 60)
    print("✅ GPU TEST PASSED!")
    print("=" * 60)
    
    return {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "status": "success"
    }


@app.function(image=image)
def test_cpu():
    """Basic CPU test to verify Modal works."""
    import numpy as np
    
    print("🖥️ CPU Test: Creating random array...")
    arr = np.random.randn(1000, 1000)
    result = np.mean(arr)
    print(f"✅ CPU test passed! Mean: {result:.6f}")
    
    return {"status": "success", "test": "cpu"}


@app.local_entrypoint()
def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🚀 STARTING MODAL TESTS")
    print("=" * 60)
    
    # Test 1: Basic CPU test
    print("\n📋 Test 1: CPU Function...")
    cpu_result = test_cpu.remote()
    print(f"Result: {cpu_result}")
    
    # Test 2: GPU test
    print("\n📋 Test 2: GPU Function...")
    gpu_result = test_gpu.remote()
    print(f"Result: {gpu_result}")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS COMPLETED!")
    print("=" * 60)
    print("\nYou're ready to build the SAM 3 worker!")
