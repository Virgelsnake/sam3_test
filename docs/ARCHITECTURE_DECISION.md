# Architecture Decision Record: GPU Provider

## Context

SAM 3 (Segment Anything Model 3) requires **NVIDIA GPU with CUDA** for viable inference performance:
- GPU: ~50-200ms per frame
- CPU: ~30-60 seconds per frame (impractical)

**Hostinger VPS does NOT offer GPU instances.**

## Your Current Hostinger Resources

| Resource | Specification |
|----------|--------------|
| VPS ID | 1120808 |
| Plan | KVM 2 |
| CPUs | 2 |
| RAM | 8GB |
| Disk | 100GB |
| OS | Ubuntu 24.04 with n8n |
| IP | 72.61.147.147 |
| Location | Manchester, UK |
| Firewall | Ports 22, 80, 443, 3001, 3002 |

## Decision: Hybrid Architecture

Split the workload across providers based on their strengths:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NETLIFY (Frontend)                          │
│  React SPA - Upload UI, Status, Preview, Download                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  HOSTINGER VPS (API + Queue)                        │
│  ┌──────────────────────┐    ┌──────────────────────┐             │
│  │   FastAPI Server     │◀──▶│   Redis Queue        │             │
│  │   - Job management   │    │   - Job dispatch     │             │
│  │   - File handling    │    │   - Status tracking  │             │
│  └──────────────────────┘    └──────────────────────┘             │
│  Deployed via Docker Compose using Hostinger API                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU PROVIDER (SAM 3 Worker)                      │
│  Options (see comparison below):                                    │
│  - Modal (serverless Python + GPU)                                  │
│  - RunPod (serverless GPU containers)                               │
│  - Replicate (hosted SAM API)                                       │
│  - AWS/GCP Spot GPU instances                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         SUPABASE (Storage + DB)                     │
│  - PostgreSQL for job metadata                                      │
│  - Storage buckets for videos                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## GPU Provider Comparison

### Option A: Modal (RECOMMENDED for MVP)

| Aspect | Details |
|--------|---------|
| **Pricing** | ~$0.000463/sec for T4 GPU (~$1.67/hr) |
| **Cold Start** | ~10-30 seconds (can keep warm) |
| **Integration** | Native Python, decorator-based |
| **Pros** | Easy deployment, auto-scaling, pay-per-second |
| **Cons** | Cold starts, vendor lock-in |

```python
# Example Modal deployment
import modal

app = modal.App("sam3-worker")
image = modal.Image.debian_slim().pip_install("sam3", "torch", "opencv-python")

@app.function(gpu="T4", image=image, timeout=300)
def process_video(video_bytes: bytes, prompt: str) -> dict:
    # SAM 3 inference here
    pass
```

### Option B: RunPod (Serverless)

| Aspect | Details |
|--------|---------|
| **Pricing** | ~$0.00031/sec for RTX 3090 |
| **Cold Start** | Variable, depends on availability |
| **Integration** | HTTP API, Docker containers |
| **Pros** | Cheap GPUs, flexible |
| **Cons** | More setup required |

### Option C: Replicate (Hosted API)

| Aspect | Details |
|--------|---------|
| **Pricing** | Per-prediction (~$0.0023/sec) |
| **Cold Start** | Minimal with popular models |
| **Integration** | Simple HTTP API |
| **Pros** | Zero infrastructure, instant |
| **Cons** | SAM 3 may not be available yet |

### Option D: AWS/GCP Spot Instances

| Aspect | Details |
|--------|---------|
| **Pricing** | ~$0.30-0.50/hr for T4 (spot) |
| **Cold Start** | Minutes (VM boot) |
| **Integration** | SSH, requires orchestration |
| **Pros** | Full control, can be cheapest at scale |
| **Cons** | Complex setup, spot interruptions |

## Recommendation

**For MVP: Use Modal**

1. **Fastest time-to-market**: Python-native, minimal infrastructure
2. **Pay-per-use**: No idle GPU costs during development
3. **Auto-scaling**: Handles traffic spikes automatically
4. **Easy migration**: Can switch to self-hosted later

**For Production at Scale: Consider RunPod or dedicated GPU**

## Updated Component Responsibilities

### Hostinger VPS (Your existing server)
- **FastAPI Server**: Job CRUD, file upload handling
- **Redis**: Job queue, status tracking
- **Nginx**: Reverse proxy, SSL termination
- Deployment via Docker Compose (Hostinger API supported)

### Modal/RunPod (GPU Provider)
- **SAM 3 Worker**: Video segmentation inference
- Pulls jobs from queue or called via HTTP
- Uploads results to Supabase Storage

### Integration Patterns

**Pattern 1: Queue-based (Recommended)**
```
API (Hostinger) → Redis Queue → Worker polls → Modal GPU
```

**Pattern 2: Direct HTTP call**
```
API (Hostinger) → HTTP POST → Modal endpoint → Webhook callback
```

## Cost Estimates (30s 720p video)

| Provider | Inference Time | Cost per Job |
|----------|---------------|--------------|
| Modal T4 | ~60-90 seconds | ~$0.03-0.04 |
| RunPod 3090 | ~30-45 seconds | ~$0.01-0.02 |
| AWS T4 Spot | ~60-90 seconds | ~$0.01-0.02 |

At 1000 jobs/month: **$10-40/month** for GPU compute.

## Implementation Impact

This architecture change affects:

1. **Worker deployment**: Not on Hostinger, on GPU provider
2. **API design**: May call external GPU service vs local queue
3. **Latency**: +network round trip to GPU provider
4. **Cold starts**: Initial request may be slower

## Next Steps

1. Sign up for Modal (free tier available)
2. Create SAM 3 tech spike on Modal
3. Deploy API + Redis on Hostinger via Docker Compose
4. Connect the pieces

---

## Alternative: CPU-Only with Smaller Model

If GPU cost is a concern, consider:
- **MobileSAM**: Lighter model, ~10x faster on CPU
- **FastSAM**: YOLO-based, much faster but less accurate
- Trade-off: Lower quality segmentation

This would allow running entirely on Hostinger VPS but with degraded results.
