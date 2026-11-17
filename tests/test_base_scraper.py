"""Tests for base scraper utilities."""

import pytest

from services.base_scraper import BaseScraper, ScrapeConfig, ScrapeResult, ScraperType, ScrapeStatus


class ConcreteScraper(BaseScraper):
    """Concrete implementation for testing abstract BaseScraper."""

    async def scrape(self, url: str, **kwargs) -> ScrapeResult:
        return ScrapeResult(url=url, status=ScrapeStatus.COMPLETED, scraper_used=self.name)

    async def close(self):
        pass


class TestScrapeConfig:
    """Tests for ScrapeConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ScrapeConfig()
        assert config.delay == 0.0
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.follow_links is False
        assert config.max_depth == 1
        assert config.respect_robots_txt is True
        assert config.screenshot is False
        assert config.user_agent is None
        assert config.headers == {}
        assert config.cookies == {}
        assert config.javascript_enabled is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ScrapeConfig(
            delay=2.0,
            timeout=60.0,
            max_retries=5,
            follow_links=True,
            max_depth=3,
            screenshot=True,
            user_agent="TestAgent",
            headers={"X-Custom": "value"},
            cookies={"session": "abc"},
            javascript_enabled=True,
        )
        assert config.delay == 2.0
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.follow_links is True
        assert config.max_depth == 3
        assert config.screenshot is True
        assert config.user_agent == "TestAgent"
        assert config.headers == {"X-Custom": "value"}
        assert config.cookies == {"session": "abc"}
        assert config.javascript_enabled is True

    def test_config_none_headers_cookies(self):
        """Test that None headers/cookies become empty dicts."""
        config = ScrapeConfig(headers=None, cookies=None)
        assert config.headers == {}
        assert config.cookies == {}


class TestScrapeResult:
    """Tests for ScrapeResult dataclass."""

    def test_minimal_result(self):
        """Test creating minimal result."""
        result = ScrapeResult(url="https://example.com")
        assert result.url == "https://example.com"
        assert result.status == ScrapeStatus.COMPLETED
        assert result.html_content is None
        assert result.text_content is None
        assert result.links == []
        assert result.images == []
        assert result.metadata == {}

    def test_full_result(self):
        """Test creating full result with all fields."""
        result = ScrapeResult(
            url="https://example.com",
            html_content="<html></html>",
            text_content="Text",
            title="Title",
            metadata={"key": "value"},
            links=[{"url": "https://link.com"}],
            images=[{"src": "https://img.com/i.png"}],
            screenshot_path="/tmp/shot.png",
            screenshot_base64="base64data",
            status=ScrapeStatus.COMPLETED,
            error=None,
            scraper_used="TestScraper",
            processing_time_seconds=1.5,
        )
        assert result.url == "https://example.com"
        assert result.title == "Title"
        assert len(result.links) == 1
        assert len(result.images) == 1
        assert result.processing_time_seconds == 1.5

    def test_failed_result(self):
        """Test creating failed result."""
        result = ScrapeResult(
            url="https://example.com",
            status=ScrapeStatus.FAILED,
            error="Connection refused",
        )
        assert result.status == ScrapeStatus.FAILED
        assert result.error == "Connection refused"


class TestScraperType:
    """Tests for ScraperType enum."""

    def test_scraper_types(self):
        """Test all scraper type values."""
        assert ScraperType.PLAYWRIGHT.value == "playwright"
        assert ScraperType.SCRAPY.value == "scrapy"
        assert ScraperType.AUTO.value == "auto"


class TestScrapeStatus:
    """Tests for ScrapeStatus enum."""

    def test_scrape_statuses(self):
        """Test all scrape status values."""
        assert ScrapeStatus.QUEUED.value == "queued"
        assert ScrapeStatus.RUNNING.value == "running"
        assert ScrapeStatus.COMPLETED.value == "completed"
        assert ScrapeStatus.FAILED.value == "failed"
        assert ScrapeStatus.PAUSED.value == "paused"
        assert ScrapeStatus.CANCELLED.value == "cancelled"


class TestBaseScraper:
    """Tests for BaseScraper abstract class."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        scraper = ConcreteScraper()
        assert scraper.config is not None
        assert scraper.config.timeout == 30.0
        assert scraper.name == "ConcreteScraper"

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = ScrapeConfig(timeout=60.0, delay=2.0)
        scraper = ConcreteScraper(config)
        assert scraper.config.timeout == 60.0
        assert scraper.config.delay == 2.0

    def test_is_valid_url_valid(self):
        """Test is_valid_url with valid URLs."""
        scraper = ConcreteScraper()
        assert scraper.is_valid_url("https://example.com") is True
        assert scraper.is_valid_url("http://localhost:8080") is True
        assert scraper.is_valid_url("https://example.com/path?query=1") is True

    def test_is_valid_url_invalid(self):
        """Test is_valid_url with invalid URLs."""
        scraper = ConcreteScraper()
        assert scraper.is_valid_url("") is False
        assert scraper.is_valid_url("not-a-url") is False
        assert scraper.is_valid_url("example.com") is False  # Missing scheme
        assert scraper.is_valid_url("://example.com") is False  # Missing scheme

    def test_normalize_url_absolute(self):
        """Test normalize_url with absolute URLs."""
        scraper = ConcreteScraper()
        assert scraper.normalize_url("https://example.com/page") == "https://example.com/page"
        assert (
            scraper.normalize_url("https://example.com/page#section") == "https://example.com/page"
        )

    def test_normalize_url_relative(self):
        """Test normalize_url with relative URLs."""
        scraper = ConcreteScraper()
        base = "https://example.com/dir/"
        assert scraper.normalize_url("/page", base) == "https://example.com/page"
        assert scraper.normalize_url("page.html", base) == "https://example.com/dir/page.html"
        assert scraper.normalize_url("../other", base) == "https://example.com/other"

    def test_normalize_url_without_base(self):
        """Test normalize_url relative URL without base."""
        scraper = ConcreteScraper()
        # Without base, relative URL stays as-is
        assert scraper.normalize_url("/page") == "/page"

    def test_should_scrape_url_valid(self):
        """Test should_scrape_url with valid URL."""
        scraper = ConcreteScraper()
        assert scraper.should_scrape_url("https://example.com") is True

    def test_should_scrape_url_invalid(self):
        """Test should_scrape_url with invalid URL."""
        scraper = ConcreteScraper()
        assert scraper.should_scrape_url("not-a-url") is False

    def test_should_scrape_url_allowed_domains(self):
        """Test should_scrape_url with domain restrictions."""
        scraper = ConcreteScraper()
        allowed = ["example.com", "test.org"]

        assert scraper.should_scrape_url("https://example.com/page", allowed) is True
        assert scraper.should_scrape_url("https://sub.example.com/page", allowed) is True
        assert scraper.should_scrape_url("https://test.org/page", allowed) is True
        assert scraper.should_scrape_url("https://other.com/page", allowed) is False

    def test_extract_domain(self):
        """Test extract_domain method."""
        scraper = ConcreteScraper()
        assert scraper.extract_domain("https://example.com/page") == "example.com"
        assert scraper.extract_domain("https://sub.example.com:8080/page") == "sub.example.com:8080"
        assert scraper.extract_domain("http://localhost:3000") == "localhost:3000"

    def test_extract_domain_invalid(self):
        """Test extract_domain with invalid URL."""
        scraper = ConcreteScraper()
        # Should return empty string for invalid URLs
        assert scraper.extract_domain("") == ""

    @pytest.mark.asyncio
    async def test_scrape_implementation(self):
        """Test that concrete scraper implementation works."""
        scraper = ConcreteScraper()
        result = await scraper.scrape("https://example.com")
        assert result.url == "https://example.com"
        assert result.status == ScrapeStatus.COMPLETED
        assert result.scraper_used == "ConcreteScraper"

    @pytest.mark.asyncio
    async def test_close_implementation(self):
        """Test that close method can be called."""
        scraper = ConcreteScraper()
        await scraper.close()  # Should not raise
