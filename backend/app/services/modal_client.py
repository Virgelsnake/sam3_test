"""
Modal client service for triggering GPU worker jobs.

This service handles communication between the FastAPI backend
and the Modal GPU worker for video segmentation.
"""

import os
from typing import Optional
from uuid import UUID

import modal

from ..config import get_settings


class ModalClientService:
    """Service for interacting with Modal GPU worker."""

    def __init__(self):
        """Initialize the Modal client service."""
        self._function = None

    @property
    def process_video_function(self):
        """Get the Modal function for processing videos."""
        if self._function is None:
            # Look up the deployed Modal function
            self._function = modal.Function.from_name(
                "sam3-video-segmentation",
                "process_video_job",
            )
        return self._function

    async def trigger_job(
        self,
        job_id: UUID,
        video_url: str,
        prompt: str,
        callback_url: Optional[str] = None,
    ) -> str:
        """
        Trigger a video segmentation job on Modal.

        This spawns an async job on Modal's GPU infrastructure.
        The job will process the video and either:
        - Call back to the callback_url with results
        - Store results in Supabase directly

        Args:
            job_id: Unique identifier for the job.
            video_url: Signed URL to download the input video.
            prompt: Text prompt describing the object to segment.
            callback_url: Optional URL to POST results to when complete.

        Returns:
            str: Modal call ID for tracking the job.
        """
        # Spawn the job asynchronously (non-blocking)
        call = self.process_video_function.spawn(
            job_id=str(job_id),
            video_url=video_url,
            prompt=prompt,
            callback_url=callback_url,
        )

        return call.object_id

    async def get_job_result(self, call_id: str) -> Optional[dict]:
        """
        Get the result of a Modal job if completed.

        Args:
            call_id: The Modal call ID returned from trigger_job.

        Returns:
            dict if job is complete, None if still running.
        """
        try:
            call = modal.functions.FunctionCall.from_id(call_id)
            # Try to get result with a short timeout
            result = call.get(timeout=0.1)
            return result
        except TimeoutError:
            # Job still running
            return None
        except Exception as e:
            # Job failed or other error
            return {"status": "failed", "error": str(e)}

    async def health_check(self) -> dict:
        """
        Check if the Modal worker is healthy.

        Returns:
            dict: Health status from the worker.
        """
        try:
            health_fn = modal.Function.from_name(
                "sam3-video-segmentation",
                "health_check",
            )
            result = health_fn.remote()
            return result
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Singleton instance
modal_client_service = ModalClientService()
