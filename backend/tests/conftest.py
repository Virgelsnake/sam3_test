"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_job_data():
    """Sample job creation data."""
    return {
        "video_id": "test-video-123",
        "prompt": "the person wearing a red jacket",
    }
