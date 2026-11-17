"""Pydantic schemas for webscraper service.

This package provides request and response models for all API endpoints,
including health checks, single URL scraping, and bulk scraping operations.
"""

from .health import HealthResponse
from .scrape import (
    BulkScrapeRequest,
    BulkScrapeResponse,
    ScrapeRequest,
    ScrapeResponse,
    ScraperType,
    ScrapeStatus,
)

__all__ = [
    "HealthResponse",
    "ScrapeRequest",
    "ScrapeResponse",
    "BulkScrapeRequest",
    "BulkScrapeResponse",
    "ScraperType",
    "ScrapeStatus",
]
