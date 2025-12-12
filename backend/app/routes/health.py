"""Health check endpoint."""

from fastapi import APIRouter

from ..services.modal_client import modal_client_service

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns:
        dict: Status indicating the API is operational.
    """
    return {"status": "ok"}


@router.get("/health/worker")
async def worker_health_check() -> dict:
    """
    Check the health of the Modal GPU worker.

    Returns:
        dict: Worker health status including GPU info.
    """
    try:
        result = await modal_client_service.health_check()
        return result
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
