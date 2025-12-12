"""Job management endpoints."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from ..models.job import JobCreate, JobResponse, JobStatus
from ..services.database import database_service
from ..services.queue import queue_service

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


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

        # Add to processing queue
        await queue_service.enqueue_job(str(job.id))

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
