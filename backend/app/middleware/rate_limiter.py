"""Rate limiting middleware using Redis."""

import time
from typing import Optional

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from ..services.queue import queue_service


class RateLimiter(BaseHTTPMiddleware):
    """
    Rate limiting middleware using Redis sliding window.
    
    Limits requests per IP address within a time window.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"

    async def _check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit.
        
        Returns:
            tuple: (is_allowed, remaining, reset_time)
        """
        redis = queue_service._redis
        if redis is None:
            # If Redis is not available, allow the request
            return True, limit, 0

        now = int(time.time())
        window_start = now - window_seconds

        # Use Redis sorted set for sliding window
        pipe = redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current entries
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiry on the key
        pipe.expire(key, window_seconds)
        
        results = await pipe.execute()
        current_count = results[1]

        remaining = max(0, limit - current_count - 1)
        reset_time = now + window_seconds

        return current_count < limit, remaining, reset_time

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/api/health", "/api/health/worker"]:
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        # Check minute limit
        minute_key = f"rate_limit:minute:{client_ip}"
        allowed, remaining, reset = await self._check_rate_limit(
            minute_key, self.requests_per_minute, 60
        )

        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset),
                    "Retry-After": "60",
                },
            )

        # Check hour limit
        hour_key = f"rate_limit:hour:{client_ip}"
        allowed, remaining_hour, reset_hour = await self._check_rate_limit(
            hour_key, self.requests_per_hour, 3600
        )

        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Hourly rate limit exceeded. Please try again later.",
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_hour),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_hour),
                    "Retry-After": "3600",
                },
            )

        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset)

        return response
