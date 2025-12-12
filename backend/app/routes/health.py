"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns:
        dict: Status indicating the API is operational.
    """
    return {"status": "ok"}
