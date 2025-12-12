"""Tests for jobs endpoints."""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.models.job import JobState


class TestCreateJob:
    """Tests for POST /api/jobs endpoint."""

    @patch("app.routes.jobs.queue_service")
    @patch("app.routes.jobs.database_service")
    def test_create_job_success(self, mock_db, mock_queue, client, sample_job_data):
        """Test successful job creation."""
        job_id = uuid4()
        mock_db.create_job = AsyncMock(
            return_value=type(
                "Job",
                (),
                {
                    "id": job_id,
                    "status": JobState.PENDING,
                    "prompt": sample_job_data["prompt"],
                    "video_path": f"uploads/{sample_job_data['video_id']}",
                    "progress": 0,
                    "mask_video_url": None,
                    "composite_video_url": None,
                    "frame_count": None,
                    "objects_detected": None,
                    "error_message": None,
                    "created_at": "2024-01-01T00:00:00Z",
                    "started_at": None,
                    "completed_at": None,
                },
            )()
        )
        mock_queue.enqueue_job = AsyncMock()

        response = client.post("/api/jobs", json=sample_job_data)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "pending"
        assert data["prompt"] == sample_job_data["prompt"]
        mock_db.create_job.assert_called_once()
        mock_queue.enqueue_job.assert_called_once()

    def test_create_job_missing_prompt(self, client):
        """Test job creation fails without prompt."""
        response = client.post("/api/jobs", json={"video_id": "test-123"})

        assert response.status_code == 422

    def test_create_job_missing_video_id(self, client):
        """Test job creation fails without video_id."""
        response = client.post("/api/jobs", json={"prompt": "test prompt"})

        assert response.status_code == 422

    def test_create_job_empty_prompt(self, client):
        """Test job creation fails with empty prompt."""
        response = client.post(
            "/api/jobs", json={"video_id": "test-123", "prompt": ""}
        )

        assert response.status_code == 422

    def test_create_job_prompt_too_long(self, client):
        """Test job creation fails with prompt over 500 chars."""
        response = client.post(
            "/api/jobs",
            json={"video_id": "test-123", "prompt": "x" * 501},
        )

        assert response.status_code == 422


class TestGetJob:
    """Tests for GET /api/jobs/{job_id} endpoint."""

    @patch("app.routes.jobs.database_service")
    def test_get_job_success(self, mock_db, client):
        """Test successful job retrieval."""
        job_id = uuid4()
        mock_db.get_job = AsyncMock(
            return_value=type(
                "Job",
                (),
                {
                    "id": job_id,
                    "status": JobState.PROCESSING,
                    "prompt": "test prompt",
                    "video_path": "uploads/test.mp4",
                    "progress": 50,
                    "mask_video_url": None,
                    "composite_video_url": None,
                    "frame_count": None,
                    "objects_detected": None,
                    "error_message": None,
                    "created_at": "2024-01-01T00:00:00Z",
                    "started_at": "2024-01-01T00:00:05Z",
                    "completed_at": None,
                },
            )()
        )

        response = client.get(f"/api/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(job_id)
        assert data["status"] == "processing"
        assert data["progress"] == 50

    @patch("app.routes.jobs.database_service")
    def test_get_job_not_found(self, mock_db, client):
        """Test 404 when job doesn't exist."""
        mock_db.get_job = AsyncMock(return_value=None)

        response = client.get(f"/api/jobs/{uuid4()}")

        assert response.status_code == 404

    def test_get_job_invalid_uuid(self, client):
        """Test 422 with invalid UUID format."""
        response = client.get("/api/jobs/not-a-uuid")

        assert response.status_code == 422


class TestGetJobStatus:
    """Tests for GET /api/jobs/{job_id}/status endpoint."""

    @patch("app.routes.jobs.database_service")
    def test_get_job_status_success(self, mock_db, client):
        """Test successful status retrieval."""
        job_id = uuid4()
        mock_db.get_job = AsyncMock(
            return_value=type(
                "Job",
                (),
                {
                    "id": job_id,
                    "status": JobState.COMPLETED,
                    "progress": 100,
                    "error_message": None,
                },
            )()
        )

        response = client.get(f"/api/jobs/{job_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress"] == 100

    @patch("app.routes.jobs.database_service")
    def test_get_job_status_not_found(self, mock_db, client):
        """Test 404 when job doesn't exist."""
        mock_db.get_job = AsyncMock(return_value=None)

        response = client.get(f"/api/jobs/{uuid4()}/status")

        assert response.status_code == 404


class TestCancelJob:
    """Tests for DELETE /api/jobs/{job_id} endpoint."""

    @patch("app.routes.jobs.queue_service")
    @patch("app.routes.jobs.database_service")
    def test_cancel_job_success(self, mock_db, mock_queue, client):
        """Test successful job cancellation."""
        job_id = uuid4()
        mock_db.get_job = AsyncMock(
            return_value=type("Job", (), {"id": job_id, "status": "pending"})()
        )
        mock_db.update_job_status = AsyncMock()
        mock_queue.cancel_job = AsyncMock()

        response = client.delete(f"/api/jobs/{job_id}")

        assert response.status_code == 204
        mock_db.update_job_status.assert_called_once()
        mock_queue.cancel_job.assert_called_once()

    @patch("app.routes.jobs.database_service")
    def test_cancel_job_not_found(self, mock_db, client):
        """Test 404 when job doesn't exist."""
        mock_db.get_job = AsyncMock(return_value=None)

        response = client.delete(f"/api/jobs/{uuid4()}")

        assert response.status_code == 404

    @patch("app.routes.jobs.database_service")
    def test_cancel_completed_job_fails(self, mock_db, client):
        """Test cannot cancel already completed job."""
        job_id = uuid4()
        mock_db.get_job = AsyncMock(
            return_value=type("Job", (), {"id": job_id, "status": "completed"})()
        )

        response = client.delete(f"/api/jobs/{job_id}")

        assert response.status_code == 400
