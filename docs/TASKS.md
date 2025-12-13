# SAM3 Video Segmentation - Implementation Task List

## Overview

This task list is organized into phases. Complete each phase before moving to the next.
Each task includes estimated time and dependencies.

---

## Phase 0: Environment Setup (Day 1)

### 0.1 Development Environment

- [ ] **T0.1.1** Clone repository and set up local development environment
- [ ] **T0.1.2** Install Python 3.10+ with CUDA support
- [ ] **T0.1.3** Install Node.js 18+ and pnpm
- [ ] **T0.1.4** Set up Docker and Docker Compose
- [ ] **T0.1.5** Create `.env.example` with all required environment variables

### 0.2 External Services Setup

- [x] **T0.2.1** Create Supabase project ✅
  - Project: `sam3-video-segmentation` (efesebfkgooupcnuyeoq)
  - URL: https://efesebfkgooupcnuyeoq.supabase.co
  - Storage buckets: `uploads` (private), `outputs` (public)
  - Jobs table created with migration

- [ ] **T0.2.2** Set up Hostinger VPS
  - Provision VPS with GPU (if available) or CPU instance
  - Install Docker, Docker Compose
  - Configure firewall (ports 80, 443, 8000)
  - Set up SSH keys

- [ ] **T0.2.3** Create Netlify site
  - Connect to GitHub repository
  - Configure build settings for Vite
  - Set up environment variables

- [x] **T0.2.4** Create GitHub repository ✅
  - Repository: https://github.com/Virgelsnake/sam3_test
  - Set up branch protection
  - Configure GitHub Actions for CI/CD

---

## Phase 1: Tech Spike - SAM 3 on GPU Provider (Days 2-4)

**Goal**: Prove SAM 3 video segmentation works with text prompts on Modal (or alternative GPU provider).

> **Note**: Hostinger VPS does not have GPU. We'll use Modal for serverless GPU inference.

### 1.1 Modal Setup

- [x] **T1.1.1** Sign up for Modal account (free tier available) ✅
- [x] **T1.1.2** Install Modal CLI: `pip install modal` ✅ (v1.2.4)
- [x] **T1.1.3** Authenticate: `modal setup` ✅
- [x] **T1.1.4** Create `worker/` directory structure ✅

### 1.2 SAM 3 on Modal

- [x] **T1.2.1** Create `worker/modal_app.py` with GPU function: ✅
  ```python
  import modal
  
  app = modal.App("sam3-worker")
  image = modal.Image.debian_slim(python_version="3.10").pip_install(
      "torch", "torchvision", "opencv-python", "numpy", "Pillow",
      "sam3 @ git+https://github.com/facebookresearch/sam3.git"
  )
  
  @app.function(gpu="T4", image=image, timeout=300)
  def process_video(video_bytes: bytes, prompt: str) -> dict:
      # SAM 3 inference
      pass
  ```

- [x] **T1.2.2** Create `worker/spike/test_sam3_modal.py` spike script ✅
- [x] **T1.2.3** Deploy and test: `modal run worker/spike/test_sam3_modal.py` ✅
- [x] **T1.2.4** Document results: ✅
  - GPU: Tesla T4, CUDA 12.8
  - Model load time: ~60 seconds
  - Memory usage: 3.61 GB allocated
  - HuggingFace auth configured for gated model access

### 1.3 Video Processing Utilities

- [ ] **T1.3.1** Create `worker/utils/video.py`:
  ```python
  def get_video_info(path) -> dict  # duration, fps, resolution
  def validate_video(path, max_duration, max_resolution) -> bool
  def extract_frames(path) -> list[np.ndarray]
  def frames_to_video(frames, output_path, fps)
  def create_mask_video(masks, output_path, fps)
  def create_composite_video(frames, masks, output_path, fps, color, opacity)
  ```

- [ ] **T1.3.2** Write unit tests for video utilities

---

## Phase 2: Backend API (Days 5-7)

### 2.1 Project Structure

- [x] **T2.1.1** Create backend directory structure: ✅
  ```
  backend/
  ├── app/
  │   ├── __init__.py
  │   ├── main.py
  │   ├── config.py
  │   ├── models/
  │   ├── routes/
  │   ├── services/
  │   └── utils/
  ├── requirements.txt
  ├── Dockerfile
  └── docker-compose.yml
  ```

### 2.2 FastAPI Application

- [x] **T2.2.1** Create `backend/app/config.py` - Pydantic settings ✅
- [x] **T2.2.2** Create `backend/app/main.py` - FastAPI app with CORS ✅
- [x] **T2.2.3** Create `backend/app/models/job.py` - Pydantic models: ✅
  - `JobCreate`
  - `JobStatus`
  - `JobResponse`

### 2.3 API Routes

- [x] **T2.3.1** Create `backend/app/routes/health.py`: ✅
  - `GET /api/health` - Returns `{"status": "ok"}`

- [x] **T2.3.2** Create `backend/app/routes/uploads.py`: ✅
  - `POST /api/uploads` - Upload video to Supabase storage
  - Returns signed URL and video_id

- [x] **T2.3.3** Create `backend/app/routes/jobs.py`: ✅
  - `POST /api/jobs` - Create new job
  - `GET /api/jobs/{job_id}` - Get job status
  - `DELETE /api/jobs/{job_id}` - Cancel job

### 2.4 Database Integration

- [x] **T2.4.1** Create `backend/app/services/database.py`: ✅
  - Supabase client initialization
  - CRUD operations for jobs table

- [x] **T2.4.2** Create Supabase migration: ✅ (`backend/migrations/001_create_jobs_table.sql`)
  ```sql
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
      progress INTEGER DEFAULT 0,
      created_at TIMESTAMPTZ DEFAULT now(),
      started_at TIMESTAMPTZ,
      completed_at TIMESTAMPTZ
  );
  ```

### 2.5 Job Queue

- [x] **T2.5.1** Add Redis to `docker-compose.yml` ✅
- [x] **T2.5.2** Create `backend/app/services/queue.py`: ✅
  - Enqueue job function
  - Job status checking

### 2.6 Storage Integration

- [x] **T2.6.1** Create `backend/app/services/storage.py`: ✅
  - Upload file to Supabase
  - Generate signed upload URL
  - Generate signed download URL

### 2.7 Backend Testing

- [x] **T2.7.1** Write API integration tests ✅ (20 tests passing)
- [x] **T2.7.2** Test with Postman/curl ✅
  - Health endpoint: `GET /api/health` → `{"status": "ok"}`
  - All 6 endpoints verified via OpenAPI
- [x] **T2.7.3** Generate OpenAPI documentation ✅
  - Available at: http://localhost:8000/docs

---

## Phase 3: GPU Worker on Modal (Days 8-10)

> **Architecture Note**: Worker runs on Modal (serverless GPU), NOT on Hostinger VPS.

### 3.1 Worker Structure

- [x] **T3.1.1** Create worker directory structure: ✅
  ```
  worker/
  ├── modal_app.py        # Modal deployment
  ├── sam3_service.py     # SAM 3 wrapper
  ├── video_utils.py      # Video processing
  ├── requirements.txt
  └── spike/
      └── test_sam3.py
  ```

### 3.2 Modal App Definition

- [x] **T3.2.1** Create `worker/modal_app.py`: ✅
  ```python
  import modal
  from sam3_service import SAM3Service
  
  app = modal.App("sam3-video-segmentation")
  
  # Pre-built image with SAM 3 and dependencies
  image = modal.Image.debian_slim(python_version="3.10").pip_install(...)
  
  @app.function(gpu="T4", image=image, timeout=600)
  def process_video_job(job_id: str, video_url: str, prompt: str) -> dict:
      # 1. Download video from Supabase
      # 2. Run SAM 3 inference
      # 3. Generate outputs
      # 4. Upload to Supabase
      # 5. Return result URLs
  ```

### 3.3 SAM 3 Service

- [x] **T3.3.1** Create `worker/sam3_service.py`: ✅
  ```python
  class SAM3Service:
      def __init__(self)
      def process_video(self, video_path: str, prompt: str) -> dict
      def _create_session(self, video_path: str) -> str
      def _add_prompt(self, session_id: str, prompt: str) -> dict
      def _propagate(self, session_id: str) -> dict[int, dict]
      def _close_session(self, session_id: str)
  ```

### 3.4 Video Utilities

- [x] **T3.4.1** Create `worker/video_utils.py`: ✅
  - `download_video(url)` - Download from Supabase signed URL
  - `create_mask_video(masks, fps)` - Grayscale mask video
  - `create_composite_video(frames, masks, color, opacity)` - Overlay video
  - `upload_to_supabase(file_path, bucket)` - Upload results

### 3.5 API Integration

- [x] **T3.5.1** Create Modal client service (`backend/app/services/modal_client.py`): ✅
  ```python
  # Option A: Direct HTTP call from FastAPI to Modal
  @app.post("/api/jobs")
  async def create_job(...):
      # Trigger Modal function
      result = modal.Function.lookup("sam3-video-segmentation", "process_video_job")
      call = result.spawn(job_id, video_url, prompt)
  ```

- [x] **T3.5.2** Implement callback webhook for job completion (`POST /api/jobs/{job_id}/complete`): ✅
  ```python
  # Modal calls back to Hostinger API when done
  @app.post("/api/jobs/{job_id}/complete")
  async def job_complete(job_id: str, results: dict):
      # Update job status in database
  ```

### 3.6 Deployment

- [ ] **T3.6.1** Deploy Modal app: `modal deploy worker/modal_app.py`
- [ ] **T3.6.2** Test end-to-end with sample video
- [ ] **T3.6.3** Monitor costs in Modal dashboard

---

## Phase 4: Frontend (Days 11-14)

### 4.1 Project Setup

- [x] **T4.1.1** Create Vite + React + TypeScript project: ✅
  ```bash
  pnpm create vite frontend --template react-ts
  ```

- [x] **T4.1.2** Install dependencies: ✅
  ```bash
  pnpm add tailwindcss postcss autoprefixer
  pnpm add @tanstack/react-query axios
  pnpm add lucide-react
  pnpm add class-variance-authority clsx tailwind-merge
  ```

- [x] **T4.1.3** Set up Tailwind CSS v4 + custom UI components: ✅
  ```bash
  pnpm dlx shadcn@latest init
  pnpm dlx shadcn@latest add button card input progress
  ```

### 4.2 Component Structure

- [x] **T4.2.1** Create directory structure: ✅
  ```
  frontend/src/
  ├── components/
  │   ├── ui/           # shadcn components
  │   ├── VideoUpload.tsx
  │   ├── PromptInput.tsx
  │   ├── JobStatus.tsx
  │   ├── VideoPlayer.tsx
  │   └── DownloadButtons.tsx
  ├── hooks/
  │   ├── useUpload.ts
  │   └── useJob.ts
  ├── services/
  │   └── api.ts
  ├── types/
  │   └── index.ts
  └── App.tsx
  ```

### 4.3 Core Components

- [x] **T4.3.1** Create `VideoUpload.tsx`: ✅
  - Drag-and-drop zone
  - File type validation (MP4, WEBM)
  - Size validation (<100MB)
  - Duration validation (<30s)
  - First frame thumbnail preview

- [x] **T4.3.2** Create `PromptInput.tsx`: ✅
  - Text input field
  - Example prompts dropdown
  - Character limit indicator

- [x] **T4.3.3** Create `JobStatus.tsx`: ✅
  - Status badge (pending/processing/completed/error)
  - Progress bar
  - Error message display

- [x] **T4.3.4** Create `VideoPlayer.tsx`: ✅
  - Split view (original | result)
  - Playback controls
  - Toggle between mask and composite view

- [x] **T4.3.5** Create `DownloadButtons.tsx`: ✅
  - Download mask video button
  - Download composite video button
  - File size indicators

### 4.4 API Integration

- [x] **T4.4.1** Create `services/api.ts`: ✅
  ```typescript
  export const uploadVideo = async (file: File) => Promise<UploadResponse>
  export const createJob = async (videoId: string, prompt: string) => Promise<Job>
  export const getJob = async (jobId: string) => Promise<Job>
  ```

- [x] **T4.4.2** Create `hooks/useUpload.ts` with React Query mutation ✅
- [x] **T4.4.3** Create `hooks/useJob.ts` with polling for status ✅

### 4.5 Main App

- [x] **T4.5.1** Create `App.tsx` with full user flow: ✅
  1. Upload state
  2. Prompt state
  3. Processing state
  4. Results state

### 4.6 Styling & UX

- [x] **T4.6.1** Implement responsive design ✅
- [x] **T4.6.2** Add loading animations ✅
- [x] **T4.6.3** Add error states with retry options ✅
- [x] **T4.6.4** Add success animations ✅

---

## Phase 5: Integration & Deployment (Days 15-17)

### 5.1 Docker Compose

- [ ] **T5.1.1** Create complete `docker-compose.yml`:
  ```yaml
  services:
    api:
      build: ./backend
      ports: ["8000:8000"]
      environment: [...]
    worker:
      build: ./worker
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
    redis:
      image: redis:alpine
  ```

### 5.2 Environment Configuration

- [ ] **T5.2.1** Create `.env.production` template
- [ ] **T5.2.2** Document all environment variables
- [ ] **T5.2.3** Set up secrets in GitHub

### 5.3 Deployment Scripts

- [ ] **T5.3.1** Create `deploy.sh` for Hostinger VPS
- [ ] **T5.3.2** Configure Netlify build:
  ```toml
  [build]
    command = "pnpm build"
    publish = "dist"
  ```

### 5.4 GitHub Actions

- [ ] **T5.4.1** Create `.github/workflows/deploy-backend.yml`
- [ ] **T5.4.2** Create `.github/workflows/deploy-frontend.yml`

### 5.5 End-to-End Testing

- [ ] **T5.5.1** Test complete flow locally
- [ ] **T5.5.2** Test complete flow on staging
- [ ] **T5.5.3** Performance testing with real videos

---

## Phase 6: Polish & Launch (Days 18-20)

### 6.1 Error Handling

- [ ] **T6.1.1** Implement comprehensive error messages
- [ ] **T6.1.2** Add error logging (consider Sentry)
- [ ] **T6.1.3** Add rate limiting

### 6.2 Performance

- [ ] **T6.2.1** Optimize video encoding settings
- [ ] **T6.2.2** Implement chunked uploads for large files
- [ ] **T6.2.3** Add caching headers

### 6.3 Monitoring

- [ ] **T6.3.1** Set up health check endpoint monitoring
- [ ] **T6.3.2** Configure alerts for failures
- [ ] **T6.3.3** Set up basic analytics

### 6.4 Documentation

- [ ] **T6.4.1** Write user guide
- [ ] **T6.4.2** Document API endpoints
- [ ] **T6.4.3** Create troubleshooting guide

### 6.5 Launch Checklist

- [ ] **T6.5.1** Security review
- [ ] **T6.5.2** Terms of service
- [ ] **T6.5.3** Privacy policy
- [ ] **T6.5.4** Final QA pass
- [ ] **T6.5.5** Launch! 🚀

---

## Quick Reference: Key Files to Create

```
sam3_test/
├── docs/
│   ├── PRD.md
│   ├── TASKS.md
│   ├── ARCHITECTURE_DECISION.md    # GPU provider decision
│   └── reference/
│       ├── SAM3_API_REFERENCE.md
│       └── SAM3_VIDEO_EXAMPLES.md
├── backend/                         # Runs on Hostinger VPS
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models/job.py
│   │   ├── routes/jobs.py
│   │   ├── routes/uploads.py
│   │   └── services/
│   │       ├── database.py
│   │       └── modal_client.py      # Calls Modal GPU worker
│   ├── requirements.txt
│   └── Dockerfile
├── worker/                          # Runs on Modal (GPU)
│   ├── modal_app.py                 # Modal deployment
│   ├── sam3_service.py
│   ├── video_utils.py
│   ├── requirements.txt
│   └── spike/
│       └── test_sam3.py
├── frontend/                        # Runs on Netlify
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   ├── hooks/
│   │   └── services/api.ts
│   ├── package.json
│   └── vite.config.ts
├── docker-compose.yml               # For Hostinger deployment
├── .env.example
└── README.md
```

---

## Estimated Total: 15-20 Working Days

| Phase | Days | Cumulative |
|-------|------|------------|
| 0. Setup | 1 | 1 |
| 1. Tech Spike | 3 | 4 |
| 2. Backend | 3 | 7 |
| 3. Worker | 3 | 10 |
| 4. Frontend | 4 | 14 |
| 5. Integration | 3 | 17 |
| 6. Polish | 3 | 20 |

---

## Dependencies Graph

```
Phase 0 (Setup)
    │
    ▼
Phase 1 (Tech Spike) ──────────────────┐
    │                                   │
    ▼                                   │
Phase 2 (Backend)                       │
    │                                   │
    ├───────────────────────┐           │
    ▼                       ▼           ▼
Phase 3 (Worker) ◀──────── Phase 4 (Frontend)
    │                       │
    └───────────┬───────────┘
                ▼
         Phase 5 (Integration)
                │
                ▼
         Phase 6 (Polish)
```
