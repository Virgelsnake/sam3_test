# SAM3 GPU Worker

This directory contains the Modal-based GPU worker for SAM 3 video segmentation.

## Architecture

The worker runs on [Modal](https://modal.com), a serverless GPU platform. When a job is created via the FastAPI backend, it triggers the Modal worker which:

1. Downloads the video from Supabase storage
2. Runs SAM 3 inference with the text prompt
3. Creates mask and composite output videos
4. Uploads results back to Supabase
5. Calls back to the API to update job status

## Prerequisites

1. **Modal Account**: Sign up at https://modal.com (free tier available)
2. **Modal CLI**: Install with `pip install modal`
3. **Modal Authentication**: Run `modal setup`

## Setup Modal Secrets

Before deploying, you need to create two secrets in Modal:

### 1. HuggingFace Secret (for SAM 3 model access)

SAM 3 is a gated model on HuggingFace. You need:
1. Create a HuggingFace account at https://huggingface.co
2. Accept the SAM 3 model license at https://huggingface.co/facebook/sam3
3. Create an access token at https://huggingface.co/settings/tokens

Then create the Modal secret:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

### 2. Supabase Secret (for storage access)

```bash
modal secret create supabase-secret \
  SUPABASE_URL=https://your-project.supabase.co \
  SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

## Deployment

### Deploy the Worker

```bash
modal deploy worker/modal_app.py
```

This will:
- Build the container image (first time takes ~5-10 minutes)
- Deploy the `process_video_job` and `health_check` functions
- Make them available for the backend to call

### Test the Worker

Run a quick health check:

```bash
modal run worker/modal_app.py
```

Test with a video (requires a publicly accessible video URL):

```bash
modal run worker/modal_app.py --video-url "https://example.com/video.mp4" --prompt "person"
```

## Local Development

For local testing without deploying:

```bash
# Run the spike tests first
modal run worker/spike/test_modal_gpu.py
modal run worker/spike/test_sam3_install.py --full
```

## Files

- `modal_app.py` - Main Modal app definition with GPU functions
- `sam3_service.py` - SAM 3 wrapper class for video segmentation
- `video_utils.py` - Video processing utilities (download, encode, upload)
- `requirements.txt` - Python dependencies (for reference)
- `spike/` - Tech spike scripts for testing

## Monitoring

Monitor your Modal deployments at: https://modal.com/apps

You can view:
- Function invocations and logs
- GPU usage and costs
- Error rates and retries

## Cost Estimation

Modal charges per-second for GPU usage:
- T4 GPU: ~$0.000164/second (~$0.59/hour)
- Typical video processing: 30-120 seconds

For a 30-second video with ~60 seconds of processing:
- Cost: ~$0.01 per video

## Troubleshooting

### "Model not found" or "Access denied"
- Ensure you've accepted the SAM 3 license on HuggingFace
- Verify your HF_TOKEN is correct in the Modal secret

### "CUDA out of memory"
- The T4 has 16GB VRAM, sufficient for most videos
- For very long videos, consider reducing resolution or frame rate

### "Timeout exceeded"
- Default timeout is 600 seconds (10 minutes)
- Very long videos may need increased timeout in `modal_app.py`
