# SAM3 Video Segmentation Web App

A browser-based tool for promptable video segmentation using Meta's SAM 3 model.

**Upload a video → Enter a prompt → Get segmented output**

## Features

- **Open-vocabulary prompts**: Describe any object in plain English
- **Video processing**: Works with short clips (≤30s, 720p)
- **Multiple outputs**: Mask video + composite overlay video
- **Web-based**: No installation required for end users

## Tech Stack

| Component | Technology | Location |
|-----------|------------|----------|
| Frontend | React + Vite + TailwindCSS + shadcn/ui | Netlify |
| Backend API | FastAPI (Python) | Hostinger VPS |
| Job Queue | Redis | Hostinger VPS |
| GPU Worker | SAM 3 + CUDA | **Modal** (serverless GPU) |
| Database | Supabase (PostgreSQL) | Supabase Cloud |
| File Storage | Supabase Storage | Supabase Cloud |

> **Note**: Hostinger VPS does not offer GPU instances. GPU inference runs on Modal (or RunPod/Replicate).

## Project Structure

```
sam3_test/
├── docs/
│   ├── PRD.md                    # Product Requirements Document
│   ├── TASKS.md                  # Implementation task list
│   └── reference/
│       ├── SAM3_API_REFERENCE.md # SAM 3 API documentation
│       └── SAM3_VIDEO_EXAMPLES.md # Code examples
├── backend/                       # FastAPI server (to be created)
├── worker/                        # SAM 3 processing worker (to be created)
├── frontend/                      # React web app (to be created)
└── docker-compose.yml            # Service orchestration (to be created)
```

## Documentation

- **[PRD.md](docs/PRD.md)** - Full product requirements, architecture, and API design
- **[TASKS.md](docs/TASKS.md)** - Detailed implementation task list with phases
- **[ARCHITECTURE_DECISION.md](docs/ARCHITECTURE_DECISION.md)** - GPU provider decision and rationale

### Reference Guides

- **[MODAL_GUIDE.md](docs/reference/MODAL_GUIDE.md)** - Complete Modal setup & SAM 3 implementation guide
- **[SAM3_API_REFERENCE.md](docs/reference/SAM3_API_REFERENCE.md)** - SAM 3 model API docs
- **[SAM3_VIDEO_EXAMPLES.md](docs/reference/SAM3_VIDEO_EXAMPLES.md)** - SAM 3 code examples

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- NVIDIA GPU with CUDA support (for worker)

### Development Setup

```bash
# Clone repository
git clone <repo-url>
cd sam3_test

# Create environment file
cp .env.example .env
# Edit .env with your credentials

# Start services
docker-compose up -d

# Frontend development
cd frontend
pnpm install
pnpm dev
```

## Architecture Overview

```
┌──────────────┐     ┌────────────────────────┐     ┌──────────────┐
│   Netlify    │────▶│   Hostinger VPS        │────▶│   Supabase   │
│  (Frontend)  │     │   (API + Redis Queue)  │     │  (DB + Files)│
└──────────────┘     │   72.61.147.147        │     └──────────────┘
                     └────────────────────────┘
                                │
                                ▼
                     ┌────────────────────────┐
                     │   Modal (GPU Worker)   │
                     │   Serverless SAM 3     │
                     └────────────────────────┘
```

## User Flow

1. User uploads video (validated: ≤30s, ≤720p, ≤100MB)
2. User enters prompt (e.g., "mask all the people")
3. Job queued and processed by SAM 3 worker
4. User previews result with split-view player
5. User downloads mask or composite video

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/uploads` | POST | Upload video |
| `/api/jobs` | POST | Create segmentation job |
| `/api/jobs/{id}` | GET | Get job status |

## Timeline

~15-20 working days for MVP:
- Phase 0: Setup (1 day)
- Phase 1: Tech Spike (3 days)
- Phase 2: Backend API (3 days)
- Phase 3: Worker Service (3 days)
- Phase 4: Frontend (4 days)
- Phase 5: Integration (3 days)
- Phase 6: Polish (3 days)

## License

[Add license]

## Contributing

[Add contributing guidelines]
