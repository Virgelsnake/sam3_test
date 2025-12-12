"""Tests for uploads endpoint."""

import pytest
from unittest.mock import AsyncMock, patch
from io import BytesIO


class TestUploadVideo:
    """Tests for POST /api/uploads endpoint."""

    @patch("app.routes.uploads.storage_service")
    def test_upload_video_success(self, mock_storage, client):
        """Test successful video upload."""
        mock_storage.upload_video = AsyncMock(
            return_value="https://storage.example.com/uploads/test.mp4"
        )

        # Create a small test video file
        video_content = b"fake video content" * 100
        files = {"file": ("test_video.mp4", BytesIO(video_content), "video/mp4")}

        response = client.post("/api/uploads", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "video_id" in data
        assert data["filename"] == "test_video.mp4"
        assert "url" in data
        mock_storage.upload_video.assert_called_once()

    def test_upload_no_file(self, client):
        """Test upload fails without file."""
        response = client.post("/api/uploads")

        assert response.status_code == 422

    def test_upload_invalid_extension(self, client):
        """Test upload fails with invalid file extension."""
        files = {"file": ("test.txt", BytesIO(b"not a video"), "text/plain")}

        response = client.post("/api/uploads", files=files)

        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    @patch("app.routes.uploads.get_settings")
    def test_upload_file_too_large(self, mock_settings, client):
        """Test upload fails when file exceeds size limit."""
        # Mock settings to have 1MB limit
        mock_settings.return_value.max_video_size_mb = 1
        mock_settings.return_value.allowed_video_extensions = [".mp4"]

        # Create file larger than 1MB
        large_content = b"x" * (2 * 1024 * 1024)  # 2MB
        files = {"file": ("large.mp4", BytesIO(large_content), "video/mp4")}

        response = client.post("/api/uploads", files=files)

        assert response.status_code == 400
        assert "too large" in response.json()["detail"]

    def test_upload_allowed_extensions(self, client):
        """Test various allowed video extensions."""
        allowed = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

        for ext in allowed:
            files = {
                "file": (
                    f"test{ext}",
                    BytesIO(b"fake content"),
                    "video/mp4",
                )
            }

            with patch("app.routes.uploads.storage_service") as mock_storage:
                mock_storage.upload_video = AsyncMock(return_value="https://example.com/video")
                response = client.post("/api/uploads", files=files)

            # Should not fail on extension validation
            assert response.status_code in [200, 500]  # 500 if storage fails, but not 400
