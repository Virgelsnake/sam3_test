"""Database service for Supabase operations."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from supabase import create_client, Client

from ..config import get_settings
from ..models.job import JobResponse, JobState


class DatabaseService:
    """Service for database operations using Supabase."""

    _client: Optional[Client] = None

    @property
    def client(self) -> Client:
        """Get or create Supabase client."""
        if self._client is None:
            settings = get_settings()
            self._client = create_client(
                settings.supabase_url,
                settings.supabase_service_role_key,
            )
        return self._client

    async def create_job(self, video_id: str, prompt: str) -> JobResponse:
        """
        Create a new job in the database.

        Args:
            video_id: ID of the uploaded video.
            prompt: Text prompt for segmentation.

        Returns:
            JobResponse: The created job.
        """
        settings = get_settings()
        video_path = f"{settings.upload_bucket}/{video_id}"

        data = {
            "prompt": prompt,
            "video_path": video_path,
            "status": JobState.PENDING.value,
            "progress": 0,
        }

        result = self.client.table("jobs").insert(data).execute()
        job_data = result.data[0]

        return JobResponse(
            id=job_data["id"],
            status=JobState(job_data["status"]),
            prompt=job_data["prompt"],
            video_path=job_data["video_path"],
            progress=job_data["progress"],
            mask_video_url=job_data.get("mask_video_url"),
            composite_video_url=job_data.get("composite_video_url"),
            frame_count=job_data.get("frame_count"),
            objects_detected=job_data.get("objects_detected"),
            error_message=job_data.get("error_message"),
            created_at=job_data["created_at"],
            started_at=job_data.get("started_at"),
            completed_at=job_data.get("completed_at"),
        )

    async def get_job(self, job_id: UUID) -> Optional[JobResponse]:
        """
        Get a job by ID.

        Args:
            job_id: The UUID of the job.

        Returns:
            JobResponse if found, None otherwise.
        """
        result = (
            self.client.table("jobs")
            .select("*")
            .eq("id", str(job_id))
            .execute()
        )

        if not result.data:
            return None

        job_data = result.data[0]

        return JobResponse(
            id=job_data["id"],
            status=JobState(job_data["status"]),
            prompt=job_data["prompt"],
            video_path=job_data["video_path"],
            progress=job_data["progress"],
            mask_video_url=job_data.get("mask_video_url"),
            composite_video_url=job_data.get("composite_video_url"),
            frame_count=job_data.get("frame_count"),
            objects_detected=job_data.get("objects_detected"),
            error_message=job_data.get("error_message"),
            created_at=job_data["created_at"],
            started_at=job_data.get("started_at"),
            completed_at=job_data.get("completed_at"),
        )

    async def update_job_status(
        self,
        job_id: UUID,
        status: str,
        progress: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update job status.

        Args:
            job_id: The UUID of the job.
            status: New status value.
            progress: Optional progress percentage.
            error_message: Optional error message.
        """
        data = {"status": status}

        if progress is not None:
            data["progress"] = progress

        if error_message is not None:
            data["error_message"] = error_message

        if status == JobState.PROCESSING.value:
            data["started_at"] = datetime.utcnow().isoformat()
        elif status in [JobState.COMPLETED.value, JobState.FAILED.value]:
            data["completed_at"] = datetime.utcnow().isoformat()

        self.client.table("jobs").update(data).eq("id", str(job_id)).execute()

    async def update_job_results(
        self,
        job_id: UUID,
        mask_video_url: str,
        composite_video_url: str,
        frame_count: int,
        objects_detected: int,
    ) -> None:
        """
        Update job with processing results.

        Args:
            job_id: The UUID of the job.
            mask_video_url: URL to the mask video.
            composite_video_url: URL to the composite video.
            frame_count: Number of frames processed.
            objects_detected: Number of objects detected.
        """
        data = {
            "status": JobState.COMPLETED.value,
            "progress": 100,
            "mask_video_url": mask_video_url,
            "composite_video_url": composite_video_url,
            "frame_count": frame_count,
            "objects_detected": objects_detected,
            "completed_at": datetime.utcnow().isoformat(),
        }

        self.client.table("jobs").update(data).eq("id", str(job_id)).execute()

    async def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[JobResponse]:
        """
        List jobs with optional filtering.

        Args:
            status: Optional status filter.
            limit: Maximum number of jobs to return.

        Returns:
            List of JobResponse objects.
        """
        query = self.client.table("jobs").select("*").order("created_at", desc=True).limit(limit)

        if status:
            query = query.eq("status", status)

        result = query.execute()

        jobs = []
        for job_data in result.data:
            jobs.append(
                JobResponse(
                    id=job_data["id"],
                    status=JobState(job_data["status"]),
                    prompt=job_data["prompt"],
                    video_path=job_data["video_path"],
                    progress=job_data["progress"],
                    mask_video_url=job_data.get("mask_video_url"),
                    composite_video_url=job_data.get("composite_video_url"),
                    frame_count=job_data.get("frame_count"),
                    objects_detected=job_data.get("objects_detected"),
                    error_message=job_data.get("error_message"),
                    created_at=job_data["created_at"],
                    started_at=job_data.get("started_at"),
                    completed_at=job_data.get("completed_at"),
                )
            )

        return jobs


# Singleton instance
database_service = DatabaseService()
