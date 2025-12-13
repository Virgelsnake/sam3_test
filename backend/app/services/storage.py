"""Storage service for Supabase file operations."""

from typing import Optional

from supabase import create_client, Client

from ..config import get_settings


class StorageService:
    """Service for file storage operations using Supabase Storage."""

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

    async def upload_video(
        self,
        content: bytes,
        path: str,
        content_type: str = "video/mp4",
    ) -> str:
        """
        Upload a video file to storage.

        Args:
            content: File content as bytes.
            path: Storage path for the file.
            content_type: MIME type of the file.

        Returns:
            str: Public URL of the uploaded file.
        """
        settings = get_settings()

        self.client.storage.from_(settings.upload_bucket).upload(
            path=path,
            file=content,
            file_options={"content-type": content_type},
        )

        # Get signed URL (valid for 1 hour)
        url = self.client.storage.from_(settings.upload_bucket).create_signed_url(
            path=path,
            expires_in=3600,
        )

        return url["signedURL"]

    async def upload_output(
        self,
        content: bytes,
        path: str,
        content_type: str = "video/mp4",
    ) -> str:
        """
        Upload an output file (mask or composite video).

        Args:
            content: File content as bytes.
            path: Storage path for the file.
            content_type: MIME type of the file.

        Returns:
            str: Public URL of the uploaded file.
        """
        settings = get_settings()

        self.client.storage.from_(settings.output_bucket).upload(
            path=path,
            file=content,
            file_options={"content-type": content_type},
        )

        # Get public URL for outputs
        url = self.client.storage.from_(settings.output_bucket).get_public_url(path)

        return url

    async def get_video_url(self, path: str) -> str:
        """
        Get a signed URL for a video in uploads bucket.

        Args:
            path: Storage path of the file.

        Returns:
            str: Signed URL valid for 1 hour.
        """
        settings = get_settings()

        url = self.client.storage.from_(settings.upload_bucket).create_signed_url(
            path=path,
            expires_in=3600,
        )

        return url["signedURL"]

    async def download_video(self, path: str) -> bytes:
        """
        Download a video file from storage.

        Args:
            path: Storage path of the file.

        Returns:
            bytes: File content.
        """
        settings = get_settings()

        response = self.client.storage.from_(settings.upload_bucket).download(path)

        return response

    async def get_output_url(self, job_id: str, filename: str) -> Optional[str]:
        """
        Get a signed URL for an output file if it exists.

        Args:
            job_id: The job ID.
            filename: The output filename (e.g., 'mask.mp4', 'composite.mp4').

        Returns:
            str: Signed URL if file exists, None otherwise.
        """
        settings = get_settings()
        path = f"{job_id}/{filename}"

        try:
            # Try to create a signed URL - will fail if file doesn't exist
            url = self.client.storage.from_(settings.output_bucket).create_signed_url(
                path=path,
                expires_in=3600,
            )
            return url.get("signedURL")
        except Exception:
            return None


# Singleton instance
storage_service = StorageService()
