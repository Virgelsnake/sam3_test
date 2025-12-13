# SAM3 Video Segmentation - API Documentation

## Base URL

```
Production: https://api.your-domain.com
Development: http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. Rate limiting is applied per IP address.

## Rate Limits

| Limit | Value |
|-------|-------|
| Requests per minute | 60 |
| Requests per hour | 1000 |

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Requests remaining in window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Endpoints

### Health Check

#### `GET /api/health`

Check if the API is operational.

**Response:**
```json
{
  "status": "ok"
}
```

#### `GET /api/health/worker`

Check if the GPU worker is operational.

**Response:**
```json
{
  "status": "healthy",
  "cuda_available": true,
  "gpu_name": "Tesla T4",
  "cuda_version": "12.1"
}
```

---

### Video Upload

#### `POST /api/uploads`

Upload a video file for processing.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` - Video file (MP4, WebM, MOV)

**Constraints:**
- Maximum file size: 100MB
- Maximum duration: 60 seconds
- Supported formats: MP4, WebM, MOV, AVI, MKV

**Response:**
```json
{
  "video_id": "abc123-def456",
  "url": "https://storage.supabase.co/...",
  "filename": "my_video.mp4",
  "size": 15728640
}
```

**Errors:**
- `400 Bad Request`: Invalid file format or size
- `413 Payload Too Large`: File exceeds size limit

---

### Jobs

#### `POST /api/jobs`

Create a new segmentation job.

**Request:**
```json
{
  "video_id": "abc123-def456",
  "prompt": "person"
}
```

**Response:**
```json
{
  "id": "job-uuid-here",
  "status": "pending",
  "prompt": "person",
  "video_path": "uploads/abc123-def456",
  "progress": 0,
  "mask_video_url": null,
  "composite_video_url": null,
  "frame_count": null,
  "objects_detected": null,
  "error_message": null,
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": null,
  "completed_at": null
}
```

#### `GET /api/jobs/{job_id}`

Get full job details.

**Response:**
```json
{
  "id": "job-uuid-here",
  "status": "completed",
  "prompt": "person",
  "video_path": "uploads/abc123-def456",
  "progress": 100,
  "mask_video_url": "https://storage.supabase.co/.../mask.mp4",
  "composite_video_url": "https://storage.supabase.co/.../composite.mp4",
  "frame_count": 150,
  "objects_detected": 1,
  "error_message": null,
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:05Z",
  "completed_at": "2024-01-15T10:31:30Z"
}
```

#### `GET /api/jobs/{job_id}/status`

Get lightweight job status (for polling).

**Response:**
```json
{
  "id": "job-uuid-here",
  "status": "processing",
  "progress": 45,
  "message": null
}
```

#### `DELETE /api/jobs/{job_id}`

Cancel a pending or processing job.

**Response:** `204 No Content`

**Errors:**
- `400 Bad Request`: Job cannot be cancelled (already completed/failed)
- `404 Not Found`: Job not found

---

### Job Completion Webhook

#### `POST /api/jobs/{job_id}/complete`

Internal endpoint called by the GPU worker when processing completes.

**Request:**
```json
{
  "job_id": "job-uuid-here",
  "status": "completed",
  "mask_video_url": "https://...",
  "composite_video_url": "https://...",
  "frame_count": 150,
  "objects_detected": 1
}
```

---

## Job Status Values

| Status | Description |
|--------|-------------|
| `pending` | Job created, waiting for GPU worker |
| `processing` | GPU worker is processing the video |
| `completed` | Processing finished successfully |
| `failed` | Processing failed (see error_message) |
| `cancelled` | Job was cancelled by user |

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here"
}
```

Common HTTP status codes:
- `400 Bad Request`: Invalid input
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Example: Complete Flow

```bash
# 1. Upload video
curl -X POST http://localhost:8000/api/uploads \
  -F "file=@my_video.mp4"

# Response: {"video_id": "abc123", ...}

# 2. Create job
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"video_id": "abc123", "prompt": "person"}'

# Response: {"id": "job-456", "status": "pending", ...}

# 3. Poll for status
curl http://localhost:8000/api/jobs/job-456/status

# Response: {"status": "processing", "progress": 45, ...}

# 4. Get results when complete
curl http://localhost:8000/api/jobs/job-456

# Response: {"status": "completed", "mask_video_url": "...", ...}
```

## SDKs & Libraries

Currently, we provide a REST API only. Community SDKs welcome!

## OpenAPI Specification

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`
