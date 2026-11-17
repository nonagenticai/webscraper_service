"""Request Logging Middleware.

Provides comprehensive request/response logging with timing information and
request tracking. Pulls ``request.state.request_id`` from
:class:`RequestIdMiddleware` (which runs first); falls back to a fresh uuid7
only if that middleware was not wired in.
"""

import logging
import time
import uuid
from collections.abc import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger: logging.Logger = logging.getLogger(__name__)


# The actual probe routes (health router prefix "/health" + /live, /ready).
# The old set listed /healthz, /ready, /readyz (no such routes) and missed the
# real sub-paths, so liveness/readiness probes logged a line on every poll.
_SKIP_LOG_PATHS = frozenset({"/health", "/health/live", "/health/ready"})


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and adding processing-time headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response with timing information."""
        if request.url.path in _SKIP_LOG_PATHS:
            return await call_next(request)

        request_id: str = getattr(request.state, "request_id", "") or str(uuid.uuid7())
        request.state.request_id = request_id

        start_time: float = time.time()

        logger.info(f"Request started: {request.method} {request.url.path} [{request_id}]")

        response: Response = await call_next(request)

        process_time: float = (time.time() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{process_time:.2f}ms"

        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"[{request_id}] - Status: {response.status_code} - "
            f"Time: {process_time:.2f}ms"
        )

        return response
