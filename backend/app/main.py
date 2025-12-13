"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .middleware import error_handler_middleware, RateLimiter
from .routes import health_router, jobs_router, uploads_router
from .services.queue import queue_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    yield
    # Shutdown
    await queue_service.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="API for SAM 3 video segmentation with text prompts",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add error handling middleware
    error_handler_middleware(app)

    # Add rate limiting (60 requests/minute, 1000 requests/hour)
    app.add_middleware(RateLimiter, requests_per_minute=60, requests_per_hour=1000)

    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Register routers
    app.include_router(health_router)
    app.include_router(uploads_router)
    app.include_router(jobs_router)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
