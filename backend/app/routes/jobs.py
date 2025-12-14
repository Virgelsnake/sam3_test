"""Job management endpoints."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel

from ..config import get_settings
from ..models.job import JobCreate, JobResponse, JobStatus
from ..services.database import database_service
from ..services.modal_client import modal_client_service
from ..services.queue import queue_service
from ..services.storage import storage_service

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobCompleteRequest(BaseModel):
    """Request body for job completion webhook."""

    job_id: str
    status: str
    mask_video_url: Optional[str] = None
    composite_video_url: Optional[str] = None
    frame_count: Optional[int] = None
    objects_detected: Optional[int] = None
    inventory: Optional[dict] = None
    error: Optional[str] = None


@router.get("", response_model=List[JobResponse])
async def list_jobs(
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=100),
) -> List[JobResponse]:
    """
    List jobs with optional status filter.

    Args:
        status_filter: Filter by status (completed, failed, processing, pending).
        limit: Maximum number of jobs to return.

    Returns:
        List of jobs ordered by creation date (newest first).
    """
    jobs = await database_service.list_jobs(status=status_filter, limit=limit)
    return jobs


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(job_data: JobCreate) -> JobResponse:
    """
    Create a new video segmentation job.

    Args:
        job_data: Job creation parameters including video_id and prompt.

    Returns:
        JobResponse: The created job with its ID and initial status.

    Raises:
        HTTPException: If job creation fails.
    """
    try:
        # Create job in database
        job = await database_service.create_job(
            video_id=job_data.video_id,
            prompt=job_data.prompt,
        )

        # Add to processing queue (for tracking)
        await queue_service.enqueue_job(str(job.id))

        # Get signed URL for the video
        settings = get_settings()
        video_path = job_data.video_id  # video_id is the storage path
        video_url = await storage_service.get_video_url(video_path)

        # Trigger Modal GPU worker
        # Build callback URL for when job completes
        callback_url = f"{settings.api_base_url}/api/jobs/{job.id}/complete"

        try:
            await modal_client_service.trigger_job(
                job_id=job.id,
                video_url=video_url,
                prompt=job_data.prompt,
                callback_url=callback_url,
            )
            # Update status to processing
            await database_service.update_job_status(job.id, "processing")
        except Exception as modal_error:
            # If Modal fails, mark job as failed but still return it
            await database_service.update_job_status(
                job.id, "failed", error_message=f"Failed to start GPU worker: {str(modal_error)}"
            )

        # Refresh job to get updated status
        job = await database_service.get_job(job.id)
        return job

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}",
        )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: UUID) -> JobResponse:
    """
    Get the full details of a job.

    Args:
        job_id: The UUID of the job to retrieve.

    Returns:
        JobResponse: Full job details including outputs if completed.

    Raises:
        HTTPException: If job is not found.
    """
    job = await database_service.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    # Check if job is processing but results exist in storage (callback may have failed)
    if job.status.value == "processing":
        try:
            mask_url = await storage_service.get_output_url(str(job_id), "mask.mp4")
            composite_url = await storage_service.get_output_url(str(job_id), "composite.mp4")
            
            if mask_url and composite_url:
                # Results exist - update job status
                await database_service.update_job_results(
                    job_id=job_id,
                    mask_video_url=mask_url,
                    composite_video_url=composite_url,
                    frame_count=0,  # Not available from storage
                    objects_detected=0,  # Not available from storage
                )
                # Refresh job data
                job = await database_service.get_job(job_id)
        except Exception:
            # If storage check fails, just return current job state
            pass

    # For completed jobs, refresh signed URLs (they expire after 1 hour)
    if job.status.value == "completed" and job.mask_video_url and job.composite_video_url:
        try:
            fresh_mask_url = await storage_service.get_output_url(str(job_id), "mask.mp4")
            fresh_composite_url = await storage_service.get_output_url(str(job_id), "composite.mp4")
            
            if fresh_mask_url and fresh_composite_url:
                # Return job with fresh URLs (don't update DB, just return fresh)
                from ..models.job import JobResponse
                return JobResponse(
                    id=job.id,
                    status=job.status,
                    prompt=job.prompt,
                    video_path=job.video_path,
                    progress=job.progress,
                    mask_video_url=fresh_mask_url,
                    composite_video_url=fresh_composite_url,
                    frame_count=job.frame_count,
                    objects_detected=job.objects_detected,
                    inventory=job.inventory,
                    inventory_colors=job.inventory_colors,
                    error_message=job.error_message,
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                )
        except Exception:
            pass

    return job


@router.get("/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: UUID) -> JobStatus:
    """
    Get the current status of a job (lightweight endpoint for polling).

    Args:
        job_id: The UUID of the job.

    Returns:
        JobStatus: Current status and progress.

    Raises:
        HTTPException: If job is not found.
    """
    job = await database_service.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return JobStatus(
        id=job.id,
        status=job.status,
        progress=job.progress,
        message=job.error_message,
    )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(job_id: UUID) -> None:
    """
    Cancel a pending or processing job.

    Args:
        job_id: The UUID of the job to cancel.

    Raises:
        HTTPException: If job is not found or cannot be cancelled.
    """
    job = await database_service.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    if job.status in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job.status}",
        )

    await database_service.update_job_status(job_id, "cancelled")
    await queue_service.cancel_job(str(job_id))


@router.post("/{job_id}/complete")
async def job_complete(job_id: UUID, data: JobCompleteRequest) -> dict:
    """
    Webhook endpoint called by Modal worker when job completes.

    This endpoint is called by the GPU worker to report job completion
    or failure. It updates the job status in the database.

    Args:
        job_id: The UUID of the job.
        data: Completion data including status and output URLs.

    Returns:
        dict: Acknowledgment of the update.
    """
    job = await database_service.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    if data.status == "completed" and data.mask_video_url and data.composite_video_url:
        await database_service.update_job_results(
            job_id=job_id,
            mask_video_url=data.mask_video_url,
            composite_video_url=data.composite_video_url,
            frame_count=data.frame_count or 0,
            objects_detected=data.objects_detected or 0,
            inventory=data.inventory,
        )
    elif data.status == "failed":
        await database_service.update_job_status(
            job_id=job_id,
            status="failed",
            error_message=data.error,
        )
    else:
        await database_service.update_job_status(
            job_id=job_id,
            status=data.status,
        )

    return {"status": "ok", "job_id": str(job_id)}


class UpdateInventoryRequest(BaseModel):
    """Request body for updating user-corrected inventory."""
    user_inventory: dict  # Item name -> user-corrected count


@router.patch("/{job_id}/inventory")
async def update_user_inventory(job_id: UUID, data: UpdateInventoryRequest) -> dict:
    """
    Update user-corrected inventory for a job.
    
    This allows users to override AI-detected quantities. The original
    AI inventory is preserved in the 'inventory' field, while user
    corrections are stored in 'user_inventory'.
    
    Args:
        job_id: The UUID of the job.
        data: The user-corrected inventory counts.
    
    Returns:
        dict: Confirmation with updated inventory.
    """
    job = await database_service.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    
    if job.status.value != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only update inventory for completed jobs",
        )
    
    await database_service.update_user_inventory(job_id, data.user_inventory)
    
    return {
        "status": "ok",
        "job_id": str(job_id),
        "user_inventory": data.user_inventory,
    }


@router.delete("/{job_id}/inventory")
async def reset_user_inventory(job_id: UUID) -> dict:
    """
    Reset user-corrected inventory to AI-detected values.
    
    This clears the user_inventory field, reverting to the original
    AI-detected counts.
    
    Args:
        job_id: The UUID of the job.
    
    Returns:
        dict: Confirmation of reset.
    """
    job = await database_service.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    
    await database_service.update_user_inventory(job_id, None)
    
    return {
        "status": "ok",
        "job_id": str(job_id),
        "message": "Inventory reset to AI-detected values",
    }
