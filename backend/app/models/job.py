"""Job-related Pydantic models."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class JobState(str, Enum):
    """Possible states for a segmentation job."""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreate(BaseModel):
    """Request model for creating a new job."""

    video_id: str = Field(..., description="ID of the uploaded video in storage")
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text prompt describing the object to segment",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "video_id": "abc123",
                    "prompt": "the person wearing a red jacket",
                }
            ]
        }
    }


class JobStatus(BaseModel):
    """Response model for job status queries."""

    id: UUID
    status: JobState
    progress: int = Field(ge=0, le=100, description="Progress percentage")
    message: Optional[str] = Field(None, description="Status message or error")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "processing",
                    "progress": 45,
                    "message": "Processing frame 45/100",
                }
            ]
        }
    }


class JobResponse(BaseModel):
    """Full job response model with all details."""

    id: UUID
    status: JobState
    prompt: str
    video_path: str
    progress: int = Field(ge=0, le=100)

    # Output URLs (populated when completed)
    mask_video_url: Optional[str] = None
    composite_video_url: Optional[str] = None

    # Metadata
    frame_count: Optional[int] = None
    objects_detected: Optional[int] = None
    inventory: Optional[dict] = None  # Item inventory from classification
    error_message: Optional[str] = None

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "completed",
                    "prompt": "the person wearing a red jacket",
                    "video_path": "uploads/abc123.mp4",
                    "progress": 100,
                    "mask_video_url": "https://storage.example.com/outputs/abc123_mask.mp4",
                    "composite_video_url": "https://storage.example.com/outputs/abc123_composite.mp4",
                    "frame_count": 150,
                    "objects_detected": 1,
                    "created_at": "2024-01-15T10:30:00Z",
                    "started_at": "2024-01-15T10:30:05Z",
                    "completed_at": "2024-01-15T10:31:30Z",
                }
            ]
        }
    }
