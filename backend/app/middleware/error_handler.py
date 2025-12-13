"""Global error handling middleware."""

import logging
import traceback
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for catching and handling unhandled exceptions."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            # Log the full traceback
            logger.error(
                f"Unhandled exception: {exc}\n"
                f"Path: {request.url.path}\n"
                f"Method: {request.method}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )

            # Return a generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "An internal server error occurred",
                    "error_id": id(exc),  # Can be used to correlate with logs
                },
            )


def error_handler_middleware(app):
    """Add error handler middleware to the app."""
    app.add_middleware(ErrorHandlerMiddleware)
