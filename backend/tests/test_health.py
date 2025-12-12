"""Tests for health check endpoint."""

import pytest


def test_health_check(client):
    """Test that health endpoint returns ok status."""
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_check_method_not_allowed(client):
    """Test that POST to health endpoint is not allowed."""
    response = client.post("/api/health")

    assert response.status_code == 405
