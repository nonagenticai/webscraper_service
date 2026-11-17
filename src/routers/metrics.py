"""Prometheus metrics endpoint.

Exposes the scrape/cache metrics (services.metrics) in the Prometheus text
exposition format at ``GET /metrics``. The path is outside ``/api`` so the auth
middleware treats it as non-protected (scrapeable without a license token), the
same as ``/health*``.
"""

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["metrics"])


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Return current metrics in Prometheus text exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
