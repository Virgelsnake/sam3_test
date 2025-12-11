# Modal Complete Guide for SAM 3 Video Segmentation

> **Purpose**: This guide helps developers (including juniors) set up and use Modal for GPU-powered SAM 3 inference.

---

## Table of Contents

1. [What is Modal?](#1-what-is-modal)
2. [Why We Use Modal](#2-why-we-use-modal)
3. [Prerequisites](#3-prerequisites)
4. [Installation & Setup](#4-installation--setup)
5. [Core Concepts Explained](#5-core-concepts-explained)
6. [Your First Modal Function](#6-your-first-modal-function)
7. [GPU Functions](#7-gpu-functions)
8. [Container Images](#8-container-images)
9. [SAM 3 Implementation](#9-sam-3-implementation)
10. [Deployment](#10-deployment)
11. [Monitoring & Debugging](#11-monitoring--debugging)
12. [Costs & Pricing](#12-costs--pricing)
13. [Common Issues & Solutions](#13-common-issues--solutions)
14. [Quick Reference](#14-quick-reference)

---

## 1. What is Modal?

Modal is a **serverless cloud platform** that lets you run Python code on powerful GPUs without managing any infrastructure.

### In Simple Terms

Think of Modal like this:
- You write a Python function on your laptop
- Modal runs that function on a powerful GPU in the cloud
- You pay only for the seconds you use
- No servers to set up, no Docker to configure manually

### Key Benefits

| Traditional GPU Setup | Modal |
|----------------------|-------|
| Buy/rent a GPU server | No server needed |
| Install CUDA drivers | Pre-installed |
| Configure Docker | Automatic |
| Manage scaling | Auto-scales |
| Pay 24/7 | Pay per second |

---

## 2. Why We Use Modal

### The Problem

Our Hostinger VPS does **not have a GPU**. SAM 3 requires a GPU (NVIDIA CUDA) to run in reasonable time:

| Hardware | Time per 30s Video |
|----------|-------------------|
| CPU Only | 30-60 minutes ❌ |
| GPU (T4) | 60-90 seconds ✅ |

### The Solution

Modal provides on-demand GPU access:

```
Your Laptop → Modal Cloud → NVIDIA GPU → Results back to you
```

---

## 3. Prerequisites

Before starting, make sure you have:

### Required

- [ ] **Python 3.10 or higher** installed
- [ ] **pip** (Python package manager)
- [ ] **A Modal account** (free tier available)
- [ ] **Internet connection**

### Check Your Python Version

Open terminal and run:

```bash
python --version
# Should show: Python 3.10.x or higher
```

If your version is too old, install a newer Python from [python.org](https://python.org).

---

## 4. Installation & Setup

### Step 1: Install Modal

Open your terminal and run:

```bash
pip install modal
```

**Verify installation:**

```bash
modal --version
# Should show: modal version X.X.X
```

### Step 2: Create a Modal Account

1. Go to [modal.com](https://modal.com)
2. Click "Sign Up" (use GitHub or email)
3. Complete the registration

### Step 3: Authenticate Your Computer

Run this command:

```bash
modal setup
```

This will:
1. Open your browser
2. Ask you to log in to Modal
3. Connect your computer to your Modal account

**Success message:**

```
✓ Token stored in ~/.modal.toml
✓ You are now authenticated!
```

### Step 4: Verify Everything Works

Create a test file called `hello_modal.py`:

```python
import modal

app = modal.App("hello-test")

@app.function()
def hello():
    return "Hello from Modal cloud!"

@app.local_entrypoint()
def main():
    result = hello.remote()
    print(result)
```

Run it:

```bash
modal run hello_modal.py
```

**Expected output:**

```
Hello from Modal cloud!
```

🎉 **Congratulations!** Modal is now set up correctly.

---

## 5. Core Concepts Explained

### 5.1 App

An **App** is a container for your Modal functions. Think of it as a project name.

```python
import modal

# Create an app called "my-project"
app = modal.App("my-project")
```

### 5.2 Function

A **Function** is Python code that runs in Modal's cloud.

```python
@app.function()
def my_function():
    return "I run in the cloud!"
```

### 5.3 Remote Execution

`.remote()` tells Modal to run the function in the cloud, not on your computer.

```python
# Runs in the cloud
result = my_function.remote()

# Runs on your computer (for testing)
result = my_function.local()
```

### 5.4 Image

An **Image** defines what software is installed in the cloud container.

```python
# Start with a basic Linux image
image = modal.Image.debian_slim()

# Add Python packages
image = image.pip_install("numpy", "opencv-python")
```

### 5.5 Local Entrypoint

The `@app.local_entrypoint()` decorator marks the function that runs on YOUR computer and orchestrates cloud functions.

```python
@app.local_entrypoint()
def main():
    # This runs on your laptop
    result = my_cloud_function.remote()  # This runs in the cloud
    print(result)  # This runs on your laptop
```

---

## 6. Your First Modal Function

### Complete Example

Create `first_function.py`:

```python
import modal

# Step 1: Create an app
app = modal.App("my-first-app")

# Step 2: Define a function that runs in the cloud
@app.function()
def add_numbers(a: int, b: int) -> int:
    print(f"Adding {a} + {b} in the cloud...")
    return a + b

# Step 3: Create the entry point (runs on your computer)
@app.local_entrypoint()
def main():
    # Call the cloud function
    result = add_numbers.remote(5, 3)
    print(f"Result: {result}")
```

### Run It

```bash
modal run first_function.py
```

**Output:**

```
Adding 5 + 3 in the cloud...
Result: 8
```

### What Happened?

1. Modal uploaded your code to the cloud
2. Created a container with Python
3. Ran `add_numbers(5, 3)` in that container
4. Returned the result to your computer
5. Shut down the container

---

## 7. GPU Functions

### Basic GPU Function

Add `gpu="T4"` to run on a GPU:

```python
import modal

app = modal.App("gpu-example")

@app.function(gpu="T4")
def check_gpu():
    import subprocess
    subprocess.run(["nvidia-smi"], check=True)
    return "GPU is working!"

@app.local_entrypoint()
def main():
    result = check_gpu.remote()
    print(result)
```

### Available GPU Types

| GPU | Memory | Best For | Cost |
|-----|--------|----------|------|
| `T4` | 16GB | Basic ML, Testing | $ (cheapest) |
| `L4` | 24GB | Medium workloads | $$ |
| `A10G` | 24GB | Production inference | $$ |
| `A100-40GB` | 40GB | Large models | $$$ |
| `A100-80GB` | 80GB | Very large models | $$$$ |
| `H100` | 80GB | Fastest, latest | $$$$$ |

### GPU Syntax Options

```python
# Single GPU (string)
@app.function(gpu="T4")

# Specific memory size
@app.function(gpu="A100-80GB")

# Multiple GPUs
@app.function(gpu="A100:2")  # 2x A100 GPUs

# Any available GPU (for testing)
@app.function(gpu="any")
```

### PyTorch with GPU Example

```python
import modal

app = modal.App("pytorch-gpu")

# Install PyTorch in the container
image = modal.Image.debian_slim().pip_install("torch")

@app.function(gpu="T4", image=image)
def pytorch_test():
    import torch
    
    # Check CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Create a tensor on GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = torch.matmul(x, y)
    
    return f"Matrix multiplication done on GPU! Shape: {z.shape}"

@app.local_entrypoint()
def main():
    result = pytorch_test.remote()
    print(result)
```

---

## 8. Container Images

### What is an Image?

An **Image** is like a recipe for the container. It defines:
- Operating system (usually Debian Linux)
- Python version
- Installed packages
- System dependencies

### Building Images

#### Method 1: pip_install (Most Common)

```python
image = modal.Image.debian_slim().pip_install(
    "torch",
    "numpy",
    "opencv-python",
    "Pillow"
)
```

#### Method 2: From requirements.txt

```python
image = modal.Image.debian_slim().pip_install_from_requirements(
    "requirements.txt"
)
```

#### Method 3: With System Packages

```python
image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libgl1")  # System packages
    .pip_install("opencv-python", "torch")  # Python packages
)
```

#### Method 4: From Docker Registry

```python
image = modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04",
    add_python="3.11"
)
```

#### Method 5: With GPU During Build

Some packages need GPU access during installation:

```python
image = modal.Image.debian_slim().pip_install(
    "bitsandbytes",  # Needs GPU to compile
    gpu="T4"  # Provide GPU during build
)
```

### Using Images in Functions

```python
import modal

app = modal.App("image-example")

# Define the image
my_image = modal.Image.debian_slim().pip_install("numpy", "pandas")

# Use the image in a function
@app.function(image=my_image)
def process_data():
    import numpy as np
    import pandas as pd
    return "Packages are available!"
```

### Conditional Imports

Import packages only inside the container:

```python
import modal

app = modal.App("example")
image = modal.Image.debian_slim().pip_install("numpy")

# This runs when the image is built
with image.imports():
    import numpy as np  # Only available in Modal container

@app.function(image=image)
def use_numpy():
    # np is now available
    arr = np.array([1, 2, 3])
    return arr.sum()
```

---

## 9. SAM 3 Implementation

### Complete SAM 3 Worker

Create `worker/modal_app.py`:

```python
import modal
import os

# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "sam3-video-segmentation"
GPU_TYPE = "T4"  # Use "A10G" for faster processing
TIMEOUT_SECONDS = 600  # 10 minutes max per job

# ============================================================
# CONTAINER IMAGE
# ============================================================

# Build an image with all SAM 3 dependencies
sam3_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "opencv-python-headless",
        "numpy",
        "Pillow",
        "supabase",
        "httpx",
    )
    .pip_install(
        "sam3 @ git+https://github.com/facebookresearch/sam3.git"
    )
    .entrypoint([])  # Remove verbose NVIDIA logging
)

# ============================================================
# MODAL APP
# ============================================================

app = modal.App(APP_NAME)

# ============================================================
# SAM 3 VIDEO PROCESSOR CLASS
# ============================================================

@app.cls(
    image=sam3_image,
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SECONDS,
    # Keep container warm for 5 minutes after last request
    container_idle_timeout=300,
)
class SAM3Processor:
    """Handles video segmentation using SAM 3."""
    
    @modal.enter()
    def initialize(self):
        """Called once when container starts. Load the model here."""
        import torch
        from sam3.model_builder import build_sam3_video_predictor
        
        print("🚀 Initializing SAM 3 Video Predictor...")
        
        # Enable TF32 for faster computation on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Build the predictor
        self.predictor = build_sam3_video_predictor(gpus_to_use=[0])
        
        print("✅ SAM 3 Predictor ready!")
    
    @modal.method()
    def process_video(
        self,
        video_path: str,
        prompt: str,
        callback_url: str = None
    ) -> dict:
        """
        Process a video with SAM 3 segmentation.
        
        Args:
            video_path: Path or URL to the video file
            prompt: Text prompt (e.g., "person", "chair")
            callback_url: Optional URL to POST results to
            
        Returns:
            dict with mask data or output paths
        """
        import cv2
        import numpy as np
        import tempfile
        import os
        
        print(f"📹 Processing video: {video_path}")
        print(f"📝 Prompt: {prompt}")
        
        try:
            # Download video if it's a URL
            local_video = self._download_video(video_path)
            
            # Get video info
            cap = cv2.VideoCapture(local_video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            print(f"📊 Video: {frame_count} frames, {width}x{height}, {fps} FPS")
            
            # Start SAM 3 session
            response = self.predictor.handle_request({
                "type": "start_session",
                "resource_path": local_video
            })
            session_id = response["session_id"]
            
            try:
                # Add text prompt
                response = self.predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt
                })
                
                objects_found = len(response.get("outputs", {}).get("object_ids", []))
                print(f"🔍 Found {objects_found} objects matching '{prompt}'")
                
                if objects_found == 0:
                    return {
                        "status": "warning",
                        "message": f"No objects found for prompt: {prompt}",
                        "objects_detected": 0
                    }
                
                # Propagate through video
                all_masks = {}
                for result in self.predictor.handle_stream_request({
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "both",
                    "start_frame_index": 0,
                    "max_frame_num_to_track": None
                }):
                    frame_idx = result["frame_index"]
                    all_masks[frame_idx] = result["outputs"]
                    
                    if frame_idx % 30 == 0:
                        print(f"⏳ Processed frame {frame_idx}/{frame_count}")
                
                print(f"✅ Segmentation complete! {len(all_masks)} frames processed")
                
                # Generate output videos
                mask_video_path = self._create_mask_video(
                    all_masks, width, height, fps
                )
                composite_video_path = self._create_composite_video(
                    local_video, all_masks, fps
                )
                
                # Upload results (implement based on your storage)
                # mask_url = self._upload_to_storage(mask_video_path)
                # composite_url = self._upload_to_storage(composite_video_path)
                
                return {
                    "status": "success",
                    "frame_count": len(all_masks),
                    "objects_detected": objects_found,
                    # "mask_video_url": mask_url,
                    # "composite_video_url": composite_url,
                }
                
            finally:
                # Always close the session
                self.predictor.handle_request({
                    "type": "close_session",
                    "session_id": session_id
                })
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _download_video(self, video_path: str) -> str:
        """Download video from URL to local temp file."""
        import tempfile
        import httpx
        
        if video_path.startswith("http"):
            print(f"⬇️ Downloading video...")
            response = httpx.get(video_path, timeout=60)
            response.raise_for_status()
            
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False
            )
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name
        else:
            return video_path
    
    def _create_mask_video(
        self, 
        masks: dict, 
        width: int, 
        height: int, 
        fps: float
    ) -> str:
        """Create grayscale mask video."""
        import cv2
        import numpy as np
        import tempfile
        
        output_path = tempfile.NamedTemporaryFile(
            suffix="_mask.mp4", delete=False
        ).name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)
        
        for frame_idx in sorted(masks.keys()):
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            
            for obj_id, obj_data in masks[frame_idx].items():
                mask = obj_data.get("mask", np.zeros((height, width)))
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                combined_mask = np.maximum(
                    combined_mask, 
                    (mask > 0).astype(np.uint8) * 255
                )
            
            out.write(combined_mask)
        
        out.release()
        return output_path
    
    def _create_composite_video(
        self,
        original_video: str,
        masks: dict,
        fps: float,
        overlay_color: tuple = (0, 255, 0),  # Green
        opacity: float = 0.4
    ) -> str:
        """Create video with mask overlay."""
        import cv2
        import numpy as np
        import tempfile
        
        cap = cv2.VideoCapture(original_video)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = tempfile.NamedTemporaryFile(
            suffix="_composite.mp4", delete=False
        ).name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in masks:
                # Combine all object masks
                combined_mask = np.zeros((height, width), dtype=np.uint8)
                for obj_id, obj_data in masks[frame_idx].items():
                    mask = obj_data.get("mask", np.zeros((height, width)))
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
                    combined_mask = np.logical_or(
                        combined_mask, mask > 0
                    ).astype(np.uint8)
                
                # Create overlay
                overlay = frame.copy()
                overlay[combined_mask > 0] = overlay_color
                frame = cv2.addWeighted(
                    frame, 1 - opacity, 
                    overlay, opacity, 
                    0
                )
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        return output_path


# ============================================================
# SIMPLE FUNCTION (Alternative to Class)
# ============================================================

@app.function(image=sam3_image, gpu=GPU_TYPE, timeout=TIMEOUT_SECONDS)
def process_video_simple(video_url: str, prompt: str) -> dict:
    """
    Simple function version (no persistent model).
    Use SAM3Processor class for better performance.
    """
    processor = SAM3Processor()
    processor.initialize()
    return processor.process_video(video_url, prompt)


# ============================================================
# LOCAL ENTRYPOINT (For Testing)
# ============================================================

@app.local_entrypoint()
def main(
    video: str = "https://example.com/sample.mp4",
    prompt: str = "person"
):
    """Test the SAM 3 processor locally."""
    print(f"🎬 Testing SAM 3 with prompt: '{prompt}'")
    
    processor = SAM3Processor()
    result = processor.process_video.remote(video, prompt)
    
    print(f"📊 Result: {result}")
    return result


# ============================================================
# WEB ENDPOINT (For HTTP API Access)
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

web_app = FastAPI()

class SegmentationRequest(BaseModel):
    video_url: str
    prompt: str
    callback_url: str = None

@app.function(image=sam3_image)
@modal.asgi_app()
def api():
    """Expose SAM 3 as a web API."""
    
    @web_app.post("/segment")
    async def segment_video(request: SegmentationRequest):
        processor = SAM3Processor()
        result = processor.process_video.remote(
            request.video_url,
            request.prompt,
            request.callback_url
        )
        return result
    
    @web_app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    return web_app
```

### Running the SAM 3 Worker

#### Test Locally

```bash
# Test with sample video
modal run worker/modal_app.py --video "https://example.com/video.mp4" --prompt "person"
```

#### Deploy to Modal

```bash
# Deploy the app (stays running)
modal deploy worker/modal_app.py
```

After deployment, you'll get URLs like:

```
✓ App deployed!
├── 🔗 https://your-workspace--sam3-video-segmentation.modal.run
└── 🔗 https://your-workspace--sam3-video-segmentation-api.modal.run
```

---

## 10. Deployment

### Development Mode: `modal run`

```bash
# Runs once and exits
modal run my_app.py
```

- Good for testing
- Shuts down after completion
- Code changes require re-running

### Serving Mode: `modal serve`

```bash
# Runs continuously, hot-reloads on code changes
modal serve my_app.py
```

- Good for development
- Auto-reloads when you save files
- Press Ctrl+C to stop

### Production Mode: `modal deploy`

```bash
# Deploy to Modal cloud permanently
modal deploy my_app.py
```

- Runs 24/7 in the cloud
- Scales automatically
- Update by running `modal deploy` again

### Viewing Deployments

```bash
# List all deployed apps
modal app list

# Stop a deployed app
modal app stop my-app-name
```

---

## 11. Monitoring & Debugging

### Modal Dashboard

Visit [modal.com/apps](https://modal.com/apps) to see:
- Running apps
- Function logs
- Resource usage
- Costs

### View Logs

```bash
# Stream logs from a deployed app
modal app logs my-app-name
```

### Debug Locally

Add print statements - they appear in Modal logs:

```python
@app.function()
def my_function():
    print("Starting...")  # Visible in logs
    # ... work ...
    print("Done!")
    return result
```

### Timeouts

Set appropriate timeouts:

```python
@app.function(
    timeout=600,  # 10 minutes max
    container_idle_timeout=300  # Keep warm for 5 min
)
def long_running_task():
    pass
```

---

## 12. Costs & Pricing

### GPU Pricing (Approximate)

| GPU | $/hour | $/second |
|-----|--------|----------|
| T4 | ~$0.59 | ~$0.000164 |
| A10G | ~$1.10 | ~$0.000306 |
| A100-40GB | ~$3.00 | ~$0.000833 |
| A100-80GB | ~$4.00 | ~$0.001111 |
| H100 | ~$5.00 | ~$0.001389 |

### Estimating Costs

For a 30-second video taking 90 seconds to process on T4:

```
90 seconds × $0.000164/second = $0.01476 per video
```

At 1000 videos/month: **~$15/month**

### Free Tier

Modal offers free credits for new users. Check current offers at [modal.com/pricing](https://modal.com/pricing).

### Cost Tips

1. **Use T4 for testing** - Cheapest GPU
2. **Use container_idle_timeout** - Keeps warm containers, faster restarts
3. **Batch requests** - Process multiple in one container session
4. **Set timeouts** - Prevent runaway costs

---

## 13. Common Issues & Solutions

### Issue: "No module named 'modal'"

**Solution:** Install Modal

```bash
pip install modal
```

### Issue: "You are not authenticated"

**Solution:** Run setup again

```bash
modal setup
```

### Issue: "Container failed to start"

**Possible causes:**
- Package installation failed
- Out of memory during build

**Solution:** Check image definition and logs

### Issue: "CUDA out of memory"

**Solutions:**
1. Use a GPU with more memory (e.g., A100 instead of T4)
2. Process smaller videos
3. Reduce batch sizes in your code

### Issue: "Timeout exceeded"

**Solution:** Increase timeout

```python
@app.function(timeout=1200)  # 20 minutes
def long_task():
    pass
```

### Issue: "Import error in container"

**Solution:** Ensure packages are in the image

```python
image = modal.Image.debian_slim().pip_install("missing-package")

@app.function(image=image)
def my_func():
    import missing_package  # Now available
```

---

## 14. Quick Reference

### Command Cheat Sheet

```bash
# Install Modal
pip install modal

# Authenticate
modal setup

# Run a script
modal run my_app.py

# Serve with hot-reload
modal serve my_app.py

# Deploy to production
modal deploy my_app.py

# List apps
modal app list

# View logs
modal app logs APP_NAME

# Stop an app
modal app stop APP_NAME
```

### Code Templates

#### Basic Function

```python
import modal

app = modal.App("my-app")

@app.function()
def hello(name: str) -> str:
    return f"Hello, {name}!"

@app.local_entrypoint()
def main():
    print(hello.remote("World"))
```

#### GPU Function

```python
import modal

app = modal.App("gpu-app")
image = modal.Image.debian_slim().pip_install("torch")

@app.function(gpu="T4", image=image)
def gpu_task():
    import torch
    return torch.cuda.is_available()
```

#### Web Endpoint

```python
import modal

app = modal.App("web-app")

@app.function()
@modal.fastapi_endpoint()
def web(name: str = "World"):
    return {"message": f"Hello, {name}!"}
```

#### Class with Setup

```python
import modal

app = modal.App("class-app")

@app.cls(gpu="T4")
class MyProcessor:
    @modal.enter()
    def setup(self):
        # Run once when container starts
        self.model = load_model()
    
    @modal.method()
    def process(self, data):
        return self.model.predict(data)
```

---

## Next Steps

1. ✅ Complete the [Installation & Setup](#4-installation--setup)
2. ✅ Run [Your First Modal Function](#6-your-first-modal-function)
3. ✅ Test [GPU Functions](#7-gpu-functions)
4. 🔨 Implement [SAM 3 Worker](#9-sam-3-implementation)
5. 🚀 [Deploy](#10-deployment) to production

---

## Resources

- **Modal Documentation**: [modal.com/docs](https://modal.com/docs)
- **Modal Examples**: [github.com/modal-labs/modal-examples](https://github.com/modal-labs/modal-examples)
- **SAM 3 Repository**: [github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- **Modal Discord**: [discord.gg/modal](https://discord.gg/modal)
