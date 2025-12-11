# Product Requirements Document
## SAM3 Video Segmentation Web Application

**Version**: 1.0  
**Last Updated**: December 2024  
**Status**: Draft

---

## 1. Executive Summary

A browser-based video segmentation tool that enables users to upload short video clips and segment objects using natural language prompts. Powered by Meta's SAM 3 (Segment Anything with Concepts), the application produces per-frame mask videos that users can preview and download.

### Key Differentiator
**Open-vocabulary text prompts** — users describe objects in plain English ("mask all the chairs", "highlight the dog") without being limited to a fixed class list.

---

## 2. Problem Statement

Video editors, content creators, and researchers often need to isolate specific objects in video footage for:
- Background removal/replacement
- Visual effects compositing
- Object tracking analysis
- Privacy masking (faces, license plates)
- Training data preparation

Current solutions require either:
- Manual frame-by-frame masking (time-consuming)
- Fixed-class models (limited vocabulary)
- Complex software installation (accessibility barrier)

---

## 3. Target Users

| Persona | Need | Technical Level |
|---------|------|-----------------|
| Video Editor | Quick object isolation for compositing | Medium |
| Content Creator | Background removal, effects | Low-Medium |
| ML Researcher | Training data annotation | High |
| Privacy Officer | PII masking in footage | Low |

---

## 4. Core User Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Upload     │───▶│   Enter      │───▶│    Run       │───▶│   Preview    │
│   Video      │    │   Prompt     │    │ Segmentation │    │  & Download  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Detailed Flow

1. **Upload**: User uploads MP4/WEBM video (≤30s, ≤720p)
2. **Preview**: System shows first frame thumbnail and video metadata
3. **Prompt**: User enters natural language prompt (e.g., "mask all the people")
4. **Submit**: User clicks "Run" button
5. **Process**: System queues job, shows progress indicator
6. **Result**: User sees split-view preview (original | masked)
7. **Download**: User downloads mask video and/or composite video

---

## 5. Functional Requirements

### 5.1 Frontend (React + Vite)

| ID | Requirement | Priority |
|----|-------------|----------|
| F1 | Drag-and-drop video upload widget | P0 |
| F2 | File validation (format, size, duration) | P0 |
| F3 | First-frame thumbnail preview | P0 |
| F4 | Text input for natural language prompt | P0 |
| F5 | "Run" button with loading state | P0 |
| F6 | Job status indicator (queued/processing/done/error) | P0 |
| F7 | Split-view video player (original + mask overlay) | P0 |
| F8 | Download buttons (mask video, composite video) | P0 |
| F9 | Error messages with actionable guidance | P1 |
| F10 | Prompt suggestions/examples | P1 |
| F11 | Job history list | P2 |

### 5.2 Backend API (FastAPI)

| ID | Endpoint | Method | Description |
|----|----------|--------|-------------|
| B1 | `/api/health` | GET | Health check |
| B2 | `/api/jobs` | POST | Create segmentation job |
| B3 | `/api/jobs/{job_id}` | GET | Get job status and results |
| B4 | `/api/jobs/{job_id}` | DELETE | Cancel/delete job |
| B5 | `/api/uploads` | POST | Upload video file |

#### POST `/api/jobs` Request
```json
{
  "video_id": "uuid-of-uploaded-video",
  "prompt": "mask all the people",
  "options": {
    "output_format": "mp4",
    "mask_type": "binary",
    "overlay_color": "#00FF00",
    "overlay_opacity": 0.5
  }
}
```

#### GET `/api/jobs/{job_id}` Response
```json
{
  "job_id": "abc123",
  "status": "completed",
  "progress": 100,
  "created_at": "2024-12-11T18:00:00Z",
  "completed_at": "2024-12-11T18:01:30Z",
  "results": {
    "mask_video_url": "/outputs/abc123_mask.mp4",
    "composite_video_url": "/outputs/abc123_composite.mp4",
    "frame_count": 450,
    "objects_detected": 3
  },
  "error": null
}
```

### 5.3 Worker Service (Python + SAM 3)

| ID | Requirement | Priority |
|----|-------------|----------|
| W1 | Load SAM 3 video predictor on startup | P0 |
| W2 | Process video from job queue | P0 |
| W3 | Extract frames from uploaded video | P0 |
| W4 | Initialize SAM 3 session with video | P0 |
| W5 | Add text prompt to session | P0 |
| W6 | Propagate segmentation across all frames | P0 |
| W7 | Generate mask video (grayscale) | P0 |
| W8 | Generate composite video (overlay) | P0 |
| W9 | Upload outputs to storage | P0 |
| W10 | Update job status in database | P0 |
| W11 | Handle errors gracefully | P0 |
| W12 | GPU memory management | P1 |

---

## 6. Technical Architecture

> **⚠️ IMPORTANT**: Hostinger VPS does not offer GPU instances. SAM 3 requires NVIDIA GPU (CUDA) for viable performance. See `ARCHITECTURE_DECISION.md` for full analysis.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NETLIFY (Frontend)                          │
│  React SPA + TailwindCSS + shadcn/ui                                │
│  - Upload widget, Video preview, Job status polling                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              HOSTINGER VPS (API + Queue) - YOUR SERVER              │
│              VPS ID: 1120808 | IP: 72.61.147.147                    │
│              Ubuntu 24.04 | 2 CPU, 8GB RAM, 100GB Disk              │
│  ┌──────────────────────┐    ┌──────────────────────┐             │
│  │   FastAPI Server     │◀──▶│   Redis Queue        │             │
│  │   - Job CRUD API     │    │   - Job dispatch     │             │
│  │   - File handling    │    │   - Status tracking  │             │
│  └──────────────────────┘    └──────────────────────┘             │
│  Deployed via Docker Compose (Hostinger API supported)             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   GPU PROVIDER (SAM 3 Worker)                       │
│  Recommended: Modal (serverless Python + GPU)                       │
│  Alternatives: RunPod, Replicate, AWS/GCP Spot                      │
│  ┌──────────────────────────────────────────────────┐             │
│  │   SAM 3 Inference Worker                         │             │
│  │   - Pulls jobs from Redis or HTTP webhook        │             │
│  │   - GPU-accelerated video segmentation           │             │
│  │   - Uploads results to Supabase                  │             │
│  └──────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         SUPABASE (Storage + DB)                     │
│  ┌──────────────────────┐    ┌──────────────────────┐             │
│  │   PostgreSQL         │    │   Storage Buckets    │             │
│  │   - Jobs table       │    │   - uploads/         │             │
│  │   - Users table      │    │   - outputs/         │             │
│  └──────────────────────┘    └──────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Frontend** | React + Vite + TailwindCSS | Fast builds, modern DX, component ecosystem |
| **UI Components** | shadcn/ui + Lucide icons | Accessible, customizable, beautiful |
| **Frontend Hosting** | Netlify | Easy deploys, CDN, serverless functions |
| **Backend API** | FastAPI (Python) | Async, fast, great typing, OpenAPI docs |
| **Job Queue** | Redis + RQ | Simple, reliable, Python-native |
| **GPU Worker** | Modal or RunPod | Serverless GPU, pay-per-use |
| **ML Model** | SAM 3 (Python) | Official Meta model |
| **Database** | Supabase (PostgreSQL) | Managed, realtime, auth built-in |
| **File Storage** | Supabase Storage | Integrated with DB, signed URLs |
| **API Hosting** | Hostinger VPS + Docker | Your existing server |
| **Containerization** | Docker + Docker Compose | Hostinger API supported |
| **Version Control** | GitHub | CI/CD, collaboration |

---

## 7. Non-Functional Requirements

### 7.1 Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Video upload | <10s for 30s clip | Chunked upload |
| Processing time | <90s for 30s 720p | Single GPU |
| API response | <200ms | Excluding upload |
| First contentful paint | <1.5s | Lighthouse target |

### 7.2 Constraints

| Constraint | Limit | Reason |
|------------|-------|--------|
| Video duration | ≤30 seconds | GPU memory, processing time |
| Video resolution | ≤720p (1280x720) | Memory constraints |
| File size | ≤100MB | Storage costs |
| Formats | MP4, WEBM | Browser compatibility |
| Concurrent jobs | 1 per GPU | Resource management |

### 7.3 Scalability

- **Phase 1**: Single GPU, single worker
- **Phase 2**: Job queue allows horizontal scaling
- **Phase 3**: Multi-GPU workers, load balancing

### 7.4 Security

- Signed upload URLs (expire after 15 min)
- Signed download URLs (expire after 1 hour)
- Rate limiting (10 jobs/hour per IP)
- Input sanitization on prompts
- No PII logging

---

## 8. SAM 3 Integration Details

### Session Lifecycle

```python
# 1. Initialize predictor (on worker startup)
predictor = build_sam3_video_predictor(gpus_to_use=[0])

# 2. Start session with video
response = predictor.handle_request({
    "type": "start_session",
    "resource_path": "/tmp/uploads/video.mp4"
})
session_id = response["session_id"]

# 3. Add text prompt
response = predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "text": "person"  # Keep prompts short!
})

# 4. Propagate through video
masks = {}
for result in predictor.handle_stream_request({
    "type": "propagate_in_video",
    "session_id": session_id,
    "propagation_direction": "both"
}):
    masks[result["frame_index"]] = result["outputs"]

# 5. Close session (free GPU memory)
predictor.handle_request({
    "type": "close_session",
    "session_id": session_id
})
```

### Prompt Guidelines (Display to User)

> **Tips for better results:**
> - Use simple nouns: "person", "chair", "dog"
> - Be specific: "red car" vs just "car"
> - One concept per prompt works best
> - Examples: "bicycle", "tennis ball", "coffee mug"

---

## 9. Database Schema (Supabase)

```sql
-- Jobs table
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(20) DEFAULT 'pending',
    prompt TEXT NOT NULL,
    video_path TEXT NOT NULL,
    mask_video_url TEXT,
    composite_video_url TEXT,
    frame_count INTEGER,
    objects_detected INTEGER,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    ip_address INET
);

-- Index for status queries
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created_at ON jobs(created_at DESC);
```

---

## 10. Error Handling

| Error | User Message | Action |
|-------|-------------|--------|
| Invalid format | "Please upload MP4 or WEBM files only" | Block upload |
| File too large | "Video must be under 100MB" | Block upload |
| Duration exceeded | "Video must be 30 seconds or less" | Block upload |
| No objects found | "No '{prompt}' found in video. Try a different description." | Allow retry |
| GPU timeout | "Processing took too long. Try a shorter video." | Refund job |
| Internal error | "Something went wrong. Please try again." | Log, alert |

---

## 11. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Job completion rate | >95% | Completed / (Completed + Failed) |
| Avg processing time | <60s | For 30s 720p video |
| User satisfaction | >4/5 | Optional feedback |
| Objects detected | >80% accuracy | Manual QA sampling |

---

## 12. Out of Scope (V1)

- User authentication/accounts
- Video editing/trimming
- Multiple prompts per job
- Real-time streaming
- Mobile-native apps
- API access for developers
- Batch processing

---

## 13. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| SAM 3 model updates break API | High | Pin model version, test updates |
| GPU costs exceed budget | Medium | Strict limits, usage monitoring |
| Long queue times | Medium | Clear status, estimated wait |
| Copyright content uploaded | High | Terms of service, no storage >24h |

---

## 14. Timeline Estimate

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Tech Spike | 2-3 days | SAM 3 working on GPU |
| API Design | 1-2 days | OpenAPI spec, DB schema |
| Worker MVP | 3-4 days | Job processing pipeline |
| Frontend MVP | 3-4 days | Upload + status + download |
| Integration | 2-3 days | End-to-end flow |
| Polish | 2-3 days | Error handling, UX |
| **Total** | **~2-3 weeks** | Production-ready MVP |

---

## Appendix A: API Specification

See separate OpenAPI specification document.

## Appendix B: UI Wireframes

See Figma/design files.

## Appendix C: SAM 3 Reference Documentation

See `/docs/reference/SAM3_API_REFERENCE.md`
