"""Image batch job Pydantic models."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field

from .job import JobState


class ImageJobCreate(BaseModel):
    """Request model for creating a new image batch job."""

    image_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=12,
        description="List of uploaded image IDs (paths) to process",
    )
    prompt: str = Field(
        default="generate an inventory of detected items",
        max_length=500,
        description="Text prompt describing the inventory context",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image_ids": ["img1.jpg", "img2.jpg", "img3.jpg"],
                    "prompt": "office equipment inventory",
                }
            ]
        }
    }


class ImageJobResponse(BaseModel):
    """Full image batch job response model."""

    id: UUID
    status: JobState
    prompt: str
    job_type: str = "image_batch"
    image_count: int
    image_paths: List[str]
    progress: int = Field(ge=0, le=100)

    # Output URLs (populated when completed)
    composite_images: Optional[List[str]] = None  # URLs to composite images with overlays
    
    # Results
    objects_detected: Optional[int] = None
    inventory: Optional[dict] = None  # Item name -> count
    inventory_colors: Optional[dict] = None  # Item name -> hex color
    user_inventory: Optional[dict] = None  # User-corrected counts
    
    # Per-image breakdown
    per_image_results: Optional[List[dict]] = None  # Details per image
    
    error_message: Optional[str] = None

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {
        "from_attributes": True,
    }


class ImageUploadResponse(BaseModel):
    """Response model for image upload."""
    
    image_id: str
    filename: str
    size_mb: float
    url: str
    path: str
