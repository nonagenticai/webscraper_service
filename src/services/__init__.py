"""Scraping services for webscraper service.

This package provides the core scraping functionality including
base abstractions, Playwright integration, and scraper orchestration.
"""

from .base_scraper import BaseScraper, ScrapeConfig, ScrapeResult, ScraperType, ScrapeStatus
from .playwright_scraper import PlaywrightScraper
from .scraper_manager import ScraperManager, get_scraper_manager
from .scrapy_scraper import ScrapyScraper

__all__ = [
    "BaseScraper",
    "ScrapeResult",
    "ScrapeConfig",
    "ScrapeStatus",
    "ScraperType",
    "PlaywrightScraper",
    "ScrapyScraper",
    "ScraperManager",
    "get_scraper_manager",
]
