"""Job queue service using Redis."""

from typing import Optional

import redis.asyncio as redis

from ..config import get_settings


class QueueService:
    """Service for job queue operations using Redis."""

    _client: Optional[redis.Redis] = None
    QUEUE_NAME = "sam3:jobs"
    PROCESSING_SET = "sam3:processing"

    @property
    def client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            settings = get_settings()
            self._client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    async def enqueue_job(self, job_id: str) -> None:
        """
        Add a job to the processing queue.

        Args:
            job_id: The UUID of the job to enqueue.
        """
        await self.client.rpush(self.QUEUE_NAME, job_id)

    async def dequeue_job(self, timeout: int = 0) -> Optional[str]:
        """
        Get the next job from the queue.

        Args:
            timeout: Seconds to wait for a job (0 = no wait).

        Returns:
            Job ID if available, None otherwise.
        """
        if timeout > 0:
            result = await self.client.blpop(self.QUEUE_NAME, timeout=timeout)
            return result[1] if result else None
        else:
            return await self.client.lpop(self.QUEUE_NAME)

    async def mark_processing(self, job_id: str) -> None:
        """
        Mark a job as currently being processed.

        Args:
            job_id: The UUID of the job.
        """
        await self.client.sadd(self.PROCESSING_SET, job_id)

    async def mark_complete(self, job_id: str) -> None:
        """
        Remove a job from the processing set.

        Args:
            job_id: The UUID of the job.
        """
        await self.client.srem(self.PROCESSING_SET, job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job by removing it from queue or processing set.

        Args:
            job_id: The UUID of the job to cancel.

        Returns:
            bool: True if job was found and cancelled.
        """
        # Try to remove from queue
        removed = await self.client.lrem(self.QUEUE_NAME, 1, job_id)
        if removed > 0:
            return True

        # Try to remove from processing set
        removed = await self.client.srem(self.PROCESSING_SET, job_id)
        return removed > 0

    async def get_queue_length(self) -> int:
        """
        Get the number of jobs waiting in queue.

        Returns:
            int: Number of queued jobs.
        """
        return await self.client.llen(self.QUEUE_NAME)

    async def get_processing_count(self) -> int:
        """
        Get the number of jobs currently being processed.

        Returns:
            int: Number of processing jobs.
        """
        return await self.client.scard(self.PROCESSING_SET)

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


# Singleton instance
queue_service = QueueService()
