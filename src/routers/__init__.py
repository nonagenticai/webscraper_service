"""API routers for webscraper service.

This package provides FastAPI routers for health check and
web scraping endpoints.
"""

from .health import router as health_router
from .scrape import router as scrape_router

__all__ = ["health_router", "scrape_router"]
