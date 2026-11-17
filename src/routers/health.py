"""Health check router with Kubernetes probes.

Provides health check endpoints for monitoring service status, including
Kubernetes liveness and readiness probes.
"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from config import get_settings
from routers.responses import HEALTH_RESPONSES
from schemas.health import HealthResponse
from services.scraper_manager import get_scraper_manager

logger = logging.getLogger(__name__)
settings = get_settings()

router: APIRouter = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "/live",
    operation_id="health_live",
    summary="Liveness probe",
)
async def liveness_probe() -> dict[str, str]:
    """Kubernetes liveness probe endpoint.

    Returns 200 if the service process is running and responding. Used by
    Kubernetes to determine if the container should be restarted.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(UTC).isoformat(),
        "service": settings.service_name,
    }


@router.get(
    "/ready",
    operation_id="health_ready",
    summary="Readiness probe",
    responses=HEALTH_RESPONSES,
)
async def readiness_probe() -> JSONResponse:
    """Kubernetes readiness probe endpoint.

    Verifies that the service can handle incoming requests by checking
    Playwright availability and browser initialization status. Returns 200 if
    ready to serve traffic, 503 if not ready.
    """
    checks: dict[str, bool] = {
        "playwright_available": False,
        "browser_ready": False,
    }

    try:
        try:
            import importlib.util

            checks["playwright_available"] = importlib.util.find_spec("playwright") is not None
        except ImportError as e:
            logger.warning(f"Playwright not available: {e}")

        scraper_manager = get_scraper_manager()
        checks["browser_ready"] = scraper_manager.is_browser_ready()

        # Readiness has to be DERIVED from every check, not just the cheapest
        # one. Gating on playwright_available alone made this a static 200: the
        # package is present in the image by construction, so a pod whose
        # browser had died still advertised itself as ready and served
        # first-viewport results for every JS page.
        is_ready = checks["playwright_available"] and checks["browser_ready"]

        response_data = {
            "status": "ready" if is_ready else "not_ready",
            "timestamp": datetime.now(UTC).isoformat(),
            "checks": checks,
        }

        if is_ready:
            return JSONResponse(status_code=status.HTTP_200_OK, content=response_data)
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=response_data
            )

    except Exception as e:
        logger.error(f"Readiness probe failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "timestamp": datetime.now(UTC).isoformat(),
                "checks": checks,
                "error": str(e),
            },
        )


@router.get(
    "",
    operation_id="health_check",
    summary="Service health summary",
    response_model=HealthResponse,
)
async def health_check() -> HealthResponse:
    """General health check endpoint with detailed service status.

    Provides comprehensive health information including service version,
    Playwright availability, and browser initialization status.
    """
    try:
        import importlib.util

        playwright_available = importlib.util.find_spec("playwright") is not None

        scraper_manager = get_scraper_manager()
        browser_ready = scraper_manager.is_browser_ready()

        return HealthResponse(
            # Derived, not asserted: reporting "healthy" next to
            # playwright_available=false was a summary that contradicted its own
            # body, and anything alerting on `status` never saw the outage.
            status="healthy" if playwright_available and browser_ready else "degraded",
            timestamp=datetime.now(UTC).isoformat(),
            service=settings.service_name,
            version=settings.version,
            playwright_available=playwright_available,
            browser_ready=browser_ready,
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(UTC).isoformat(),
            service=settings.service_name,
            version=settings.version,
            playwright_available=False,
            browser_ready=False,
        )
