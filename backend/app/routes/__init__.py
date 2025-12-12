"""API route modules."""

from .health import router as health_router
from .jobs import router as jobs_router
from .uploads import router as uploads_router

__all__ = ["health_router", "jobs_router", "uploads_router"]
