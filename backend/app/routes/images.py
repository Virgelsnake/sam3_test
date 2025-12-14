"""Image batch upload and job management endpoints."""

import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, Query, status
from pydantic import BaseModel

from ..config import get_settings
from ..models.image_job import ImageJobCreate, ImageJobResponse, ImageUploadResponse
from ..models.job import JobState
from ..services.database import database_service
from ..services.modal_client import modal_client_service
from ..services.queue import queue_service
from ..services.storage import storage_service

router = APIRouter(prefix="/api/images", tags=["images"])


@router.post("/upload", response_model=List[ImageUploadResponse])
async def upload_images(files: List[UploadFile] = File(...)) -> List[ImageUploadResponse]:
    """
    Upload multiple images for batch processing.

    Args:
        files: List of image files to upload (max 12).

    Returns:
        List of ImageUploadResponse with image IDs and URLs.

    Raises:
        HTTPException: If validation fails or upload errors occur.
    """
    settings = get_settings()

    if len(files) > settings.max_images_per_batch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {settings.max_images_per_batch} images allowed per batch",
        )

    if len(files) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 image is required",
        )

    results = []
    batch_id = str(uuid.uuid4())

    for i, file in enumerate(files):
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Filename is required for file {i+1}",
            )

        ext = Path(file.filename).suffix.lower()
        if ext not in settings.allowed_image_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type for {file.filename}. Allowed: {settings.allowed_image_extensions}",
            )

        content = await file.read()
        size_mb = len(content) / (1024 * 1024)

        if size_mb > settings.max_image_size_mb:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {file.filename} too large. Maximum size: {settings.max_image_size_mb}MB",
            )

        # Generate unique image ID within batch
        image_id = str(uuid.uuid4())
        storage_path = f"images/{batch_id}/{image_id}{ext}"

        try:
            # Determine content type
            content_type = file.content_type or "image/jpeg"
            if ext == ".png":
                content_type = "image/png"
            elif ext == ".webp":
                content_type = "image/webp"

            url = await storage_service.upload_video(
                content=content,
                path=storage_path,
                content_type=content_type,
            )

            results.append(ImageUploadResponse(
                image_id=image_id,
                filename=file.filename,
                size_mb=round(size_mb, 2),
                url=url,
                path=storage_path,
            ))

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload {file.filename}: {str(e)}",
            )

    return results


@router.post("/jobs", response_model=ImageJobResponse, status_code=status.HTTP_201_CREATED)
async def create_image_job(job_data: ImageJobCreate) -> ImageJobResponse:
    """
    Create a new image batch inventory job.

    Args:
        job_data: Job creation parameters including image_ids and prompt.

    Returns:
        ImageJobResponse: The created job with its ID and initial status.

    Raises:
        HTTPException: If job creation fails.
    """
    try:
        # Create job in database
        job = await database_service.create_image_job(
            image_ids=job_data.image_ids,
            prompt=job_data.prompt,
        )

        # Add to processing queue
        await queue_service.enqueue_job(str(job.id))

        # Get signed URLs for all images
        settings = get_settings()
        image_urls = []
        for image_path in job.image_paths:
            url = await storage_service.get_video_url(image_path)
            image_urls.append(url)

        # Trigger Modal GPU worker for image batch
        callback_url = f"{settings.api_base_url}/api/images/jobs/{job.id}/complete"

        try:
            await modal_client_service.trigger_image_job(
                job_id=job.id,
                image_urls=image_urls,
                prompt=job_data.prompt,
                callback_url=callback_url,
            )
            await database_service.update_job_status(job.id, "processing")
        except Exception as modal_error:
            await database_service.update_job_status(
                job.id, "failed", error_message=f"Failed to start GPU worker: {str(modal_error)}"
            )

        # Refresh job to get updated status
        job = await database_service.get_image_job(job.id)
        return job

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create image job: {str(e)}",
        )


@router.get("/jobs", response_model=List[ImageJobResponse])
async def list_image_jobs(
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=100),
) -> List[ImageJobResponse]:
    """List image batch jobs with optional status filter."""
    jobs = await database_service.list_image_jobs(status=status_filter, limit=limit)
    return jobs


@router.get("/jobs/{job_id}", response_model=ImageJobResponse)
async def get_image_job(job_id: str) -> ImageJobResponse:
    """Get full details of an image batch job."""
    from uuid import UUID
    
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format",
        )
    
    job = await database_service.get_image_job(job_uuid)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image job {job_id} not found",
        )

    return job


class ImageJobCompleteRequest(BaseModel):
    """Request body for image job completion webhook."""
    job_id: str
    status: str
    composite_images: Optional[List[str]] = None
    objects_detected: Optional[int] = None
    inventory: Optional[dict] = None
    inventory_colors: Optional[dict] = None
    per_image_results: Optional[List[dict]] = None
    error: Optional[str] = None


@router.post("/jobs/{job_id}/complete")
async def image_job_complete(job_id: str, data: ImageJobCompleteRequest) -> dict:
    """
    Webhook endpoint called by Modal worker when image job completes.
    """
    from uuid import UUID
    
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format",
        )

    job = await database_service.get_image_job(job_uuid)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image job {job_id} not found",
        )

    if data.status == "completed":
        await database_service.update_image_job_results(
            job_id=job_uuid,
            composite_images=data.composite_images or [],
            objects_detected=data.objects_detected or 0,
            inventory=data.inventory,
            inventory_colors=data.inventory_colors,
            per_image_results=data.per_image_results,
        )
    elif data.status == "failed":
        await database_service.update_job_status(
            job_id=job_uuid,
            status="failed",
            error_message=data.error,
        )
    else:
        await database_service.update_job_status(
            job_id=job_uuid,
            status=data.status,
        )

    return {"status": "ok", "job_id": str(job_id)}


class UpdateImageInventoryRequest(BaseModel):
    """Request body for updating user-corrected inventory."""
    user_inventory: dict


@router.patch("/jobs/{job_id}/inventory")
async def update_image_inventory(job_id: str, data: UpdateImageInventoryRequest) -> dict:
    """Update user-corrected inventory for an image batch job."""
    from uuid import UUID
    
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format",
        )
    
    job = await database_service.get_image_job(job_uuid)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image job {job_id} not found",
        )
    
    if job.status.value != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only update inventory for completed jobs",
        )
    
    await database_service.update_image_job_user_inventory(job_uuid, data.user_inventory)
    
    return {
        "status": "ok",
        "job_id": str(job_id),
        "user_inventory": data.user_inventory,
    }
