"""Tests for ScrapyScraper (real Scrapy framework scraper)."""

import asyncio
from unittest.mock import patch

import pytest

from services.base_scraper import ScrapeConfig, ScrapeStatus
from services.scrapy_scraper import ScrapyScraper


@pytest.fixture
def scraper():
    return ScrapyScraper(ScrapeConfig(timeout=10))


class TestScrapyScraperInit:
    def test_default_init(self):
        s = ScrapyScraper()
        assert s.name == "ScrapyScraper"

    def test_init_with_config(self):
        config = ScrapeConfig(timeout=60, user_agent="TestBot", follow_links=True, max_depth=3)
        s = ScrapyScraper(config)
        assert s.config.timeout == 60
        assert s.config.user_agent == "TestBot"
        assert s.config.follow_links is True
        assert s.config.max_depth == 3


class TestScrapyScraperScrape:
    @pytest.mark.asyncio
    async def test_scrape_success(self, scraper):
        mock_items = [
            {
                "url": "https://example.com",
                "html_content": "<html><body><h1>Hello</h1></body></html>",
                "text_content": "Hello",
                "title": "Test Page",
                "metadata": {"status_code": 200, "final_url": "https://example.com"},
                "links": [
                    {
                        "text": "About",
                        "url": "https://example.com/about",
                        "rel": None,
                        "type": "internal",
                    }
                ],
                "images": [{"src": "/img.png", "alt": "Img", "title": ""}],
                "depth": 0,
            }
        ]

        with patch("services.scrapy_scraper._run_crawl_in_executor", return_value=(mock_items, [])):
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.COMPLETED
        assert result.url == "https://example.com"
        assert result.title == "Test Page"
        assert result.text_content == "Hello"
        assert result.scraper_used == "ScrapyScraper"
        assert result.processing_time_seconds > 0
        assert len(result.links) == 1
        assert len(result.images) == 1

    @pytest.mark.asyncio
    async def test_scrape_no_items(self, scraper):
        with patch(
            "services.scrapy_scraper._run_crawl_in_executor",
            return_value=([], ["Connection refused"]),
        ):
            result = await scraper.scrape("https://unreachable.example.com")

        assert result.status == ScrapeStatus.FAILED
        assert "Connection refused" in result.error
        assert result.scraper_used == "ScrapyScraper"

    @pytest.mark.asyncio
    async def test_scrape_no_items_no_errors(self, scraper):
        with patch("services.scrapy_scraper._run_crawl_in_executor", return_value=([], [])):
            result = await scraper.scrape("https://empty.example.com")

        assert result.status == ScrapeStatus.FAILED
        assert "No content extracted" in result.error

    @pytest.mark.asyncio
    async def test_scrape_exception(self, scraper):
        with patch(
            "services.scrapy_scraper._run_crawl_in_executor",
            side_effect=RuntimeError("Reactor error"),
        ):
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.FAILED
        assert "Reactor error" in result.error

    @pytest.mark.asyncio
    async def test_scrape_with_follow_links(self):
        config = ScrapeConfig(follow_links=True, max_depth=2)
        s = ScrapyScraper(config)

        mock_items = [
            {
                "url": "https://example.com",
                "html_content": "<html><body>Main</body></html>",
                "text_content": "Main",
                "title": "Main",
                "metadata": {"status_code": 200, "final_url": "https://example.com"},
                "links": [],
                "images": [],
                "depth": 0,
            },
            {
                "url": "https://example.com/about",
                "html_content": "<html><body>About</body></html>",
                "text_content": "About",
                "title": "About",
                "metadata": {"status_code": 200, "final_url": "https://example.com/about"},
                "links": [],
                "images": [],
                "depth": 1,
            },
        ]

        with patch("services.scrapy_scraper._run_crawl_in_executor", return_value=(mock_items, [])):
            result = await s.scrape("https://example.com")

        assert result.status == ScrapeStatus.COMPLETED
        # Returns the primary URL's content
        assert result.title == "Main"

    @pytest.mark.asyncio
    async def test_scrape_with_download_images(self):
        """download_images=True triggers enrich_images_with_content."""
        config = ScrapeConfig(timeout=10, download_images=True)
        s = ScrapyScraper(config)

        mock_items = [
            {
                "url": "https://example.com",
                "html_content": "<html></html>",
                "text_content": "",
                "title": "T",
                "metadata": {"status_code": 200, "final_url": "https://example.com"},
                "links": [],
                "images": [
                    {"src": "https://example.com/img.png", "alt": "Img", "title": ""},
                ],
                "depth": 0,
            }
        ]

        enriched = [
            {
                "src": "https://example.com/img.png",
                "alt": "Img",
                "title": "",
                "content_base64": "abc123",
                "mime_type": "image/png",
                "byte_size": 100,
            }
        ]

        with (
            patch("services.scrapy_scraper._run_crawl_in_executor", return_value=(mock_items, [])),
            patch(
                "services.scrapy_scraper.enrich_images_with_content",
                return_value=enriched,
            ) as mock_enrich,
        ):
            result = await s.scrape("https://example.com", download_images=True)

        mock_enrich.assert_awaited_once()
        assert result.status == ScrapeStatus.COMPLETED
        assert result.images[0]["content_base64"] == "abc123"

    @pytest.mark.asyncio
    async def test_scrape_without_download_images(self, scraper):
        """download_images=False does not trigger enrichment."""
        mock_items = [
            {
                "url": "https://example.com",
                "html_content": "<html></html>",
                "text_content": "",
                "title": "T",
                "metadata": {"status_code": 200, "final_url": "https://example.com"},
                "links": [],
                "images": [{"src": "https://example.com/img.png", "alt": "Img", "title": ""}],
                "depth": 0,
            }
        ]

        with (
            patch("services.scrapy_scraper._run_crawl_in_executor", return_value=(mock_items, [])),
            patch(
                "services.scrapy_scraper.enrich_images_with_content",
            ) as mock_enrich,
        ):
            result = await scraper.scrape("https://example.com")

        mock_enrich.assert_not_awaited()
        assert result.status == ScrapeStatus.COMPLETED
        assert "content_base64" not in result.images[0]


class TestScrapyReactorReuse:
    """Regression for the ReactorNotRestartable bug: every test elsewhere mocks
    _run_crawl, so the real Twisted-reactor path was never exercised. A 2nd real
    crawl in the process must NOT raise ReactorNotRestartable now that each crawl
    runs in its own spawned subprocess."""

    @pytest.mark.asyncio
    async def test_two_sequential_real_crawls_no_reactor_error(self):
        # Un-mocked on purpose. 127.0.0.1:1 refuses fast → each crawl FAILS
        # cleanly (connection refused), but crucially WITHOUT ReactorNotRestartable.
        s = ScrapyScraper(ScrapeConfig(timeout=3, max_retries=0, delay=0.0))
        r1 = await s.scrape("http://127.0.0.1:1/")
        r2 = await s.scrape("http://127.0.0.1:1/")
        for r in (r1, r2):
            assert r.status == ScrapeStatus.FAILED
            assert "ReactorNotRestartable" not in (r.error or "")


class TestBrokenProcessPoolRecovery:
    """Regression for the new failure mode the subprocess fix introduced: with
    ``max_tasks_per_child=1`` a single child hard-crash marks the pool
    permanently ``BrokenProcessPool``, which (pre-fix) FAILed every subsequent
    scrapy crawl until the pod restarts. ``_run_crawl_in_executor`` must recover
    the pool once and retry, and a second break must propagate (no infinite loop)."""

    @pytest.mark.asyncio
    async def test_broken_pool_recovers_once_and_retries(self):
        from concurrent.futures import Future as ConcFuture
        from concurrent.futures.process import BrokenProcessPool

        from services import scrapy_scraper as ss

        crawl_result = ([{"url": "http://x"}], [])

        class _BrokenExecutor:
            shutdown_called = False

            def submit(self, *_a, **_k):
                raise BrokenProcessPool("child died")

            def shutdown(self, *_a, **_k):
                self.shutdown_called = True

        class _GoodExecutor:
            def submit(self, *_a, **_k):
                fut: ConcFuture = ConcFuture()
                fut.set_result(crawl_result)
                return fut

            def shutdown(self, *_a, **_k):
                pass

        broken = _BrokenExecutor()
        good = _GoodExecutor()

        with (
            patch.object(ss, "_executor", broken),
            patch.object(ss, "_make_executor", return_value=good),
        ):
            items, errors = await ss._run_crawl_in_executor("http://x", ScrapeConfig())
            # Module global was swapped to the freshly-made pool, broken one torn down.
            assert ss._executor is good

        assert broken.shutdown_called is True
        assert items == crawl_result[0]

    @pytest.mark.asyncio
    async def test_second_break_propagates_no_infinite_loop(self):
        from concurrent.futures.process import BrokenProcessPool

        from services import scrapy_scraper as ss

        class _BrokenExecutor:
            def submit(self, *_a, **_k):
                raise BrokenProcessPool("child died")

            def shutdown(self, *_a, **_k):
                pass

        with (
            patch.object(ss, "_executor", _BrokenExecutor()),
            patch.object(ss, "_make_executor", return_value=_BrokenExecutor()),
            pytest.raises(BrokenProcessPool),
        ):
            await ss._run_crawl_in_executor("http://x", ScrapeConfig())

    @pytest.mark.asyncio
    async def test_make_executor_raising_leaves_old_pool_recoverable(self):
        """Build-before-teardown: if ``_make_executor()`` raises during recovery
        (the OOM/FD-exhaustion that broke the pool also blocks constructing a new
        one), the old pool must NOT be shut down and ``_executor`` must stay
        pointing at it — otherwise the next ``submit`` hits a dead shut-down pool
        with a non-``BrokenProcessPool`` ``RuntimeError`` that the recovery path
        never catches → permanent brick. Here the next request re-enters
        recovery instead."""
        from concurrent.futures.process import BrokenProcessPool

        from services import scrapy_scraper as ss

        class _BrokenExecutor:
            shutdown_called = False

            def submit(self, *_a, **_k):
                raise BrokenProcessPool("child died")

            def shutdown(self, *_a, **_k):
                self.shutdown_called = True

        broken = _BrokenExecutor()

        def _make_raises():
            raise OSError("cannot allocate memory for a new process pool")

        with (
            patch.object(ss, "_executor", broken),
            patch.object(ss, "_make_executor", side_effect=_make_raises),
        ):
            with pytest.raises(OSError, match="cannot allocate memory"):
                await ss._run_crawl_in_executor("http://x", ScrapeConfig())

            # Asserted while the patch is still active: the old pool was left
            # intact (not torn down) and is still the module global, so a
            # subsequent request re-enters BrokenProcessPool recovery rather than
            # raising the unrecoverable "cannot schedule new futures after shutdown".
            assert broken.shutdown_called is False
            assert ss._executor is broken


class TestScrapyCrawlBudget:
    """The crawl must be bounded in wall-clock AND page count: DOWNLOAD_TIMEOUT
    only bounds a single request, so a follow_links crawl was otherwise unbounded
    (and accumulated every followed page in the child's memory)."""

    def test_run_crawl_sets_closespider_bounds(self):
        from scrapy.settings import Settings as ScrapySettings

        from config import get_settings
        from services import scrapy_scraper as ss

        captured: dict[str, ScrapySettings] = {}

        class _FakeProcess:
            def __init__(self, settings, install_root_handler=True):
                captured["settings"] = settings

            def crawl(self, *_a, **_k):
                pass

            def start(self, *_a, **_k):
                pass

        with patch.object(ss, "CrawlerProcess", _FakeProcess):
            ss._run_crawl("http://example.com", ScrapeConfig())

        settings = captured["settings"]
        assert settings.getfloat("CLOSESPIDER_TIMEOUT") == get_settings().scrapy_crawl_timeout
        assert settings.getint("CLOSESPIDER_PAGECOUNT") == get_settings().scrapy_max_pages
        # Per-response body cap (parity with httpx max_page_bytes) — a multi-GB
        # body must not OOM the crawl subprocess.
        assert settings.getint("DOWNLOAD_MAXSIZE") == get_settings().max_page_bytes
        assert settings.getint("DOWNLOAD_WARNSIZE") == 0

    @pytest.mark.asyncio
    async def test_overall_budget_timeout_returns_failed(self, scraper):
        """If the subprocess wedges past the wait_for budget (the CLOSESPIDER
        in-process bound can't fire on a hung child), scrape() must return FAILED
        and free the manager's concurrency permit — not hang forever."""

        def _raise_timeout(coro, timeout=None):
            coro.close()  # avoid "coroutine was never awaited" warning
            raise TimeoutError

        with patch("services.scrapy_scraper.asyncio.wait_for", new=_raise_timeout):
            result = await scraper.scrape("https://wedged.example.com")

        assert result.status == ScrapeStatus.FAILED
        assert "budget" in (result.error or "").lower()


class TestImageEnrichmentBudget:
    @pytest.mark.asyncio
    async def test_enrichment_timeout_returns_unenriched(self):
        """A hanging image host can't stall the scrape past its budget; images
        are returned un-enriched instead of failing."""
        config = ScrapeConfig(timeout=0.25, download_images=True)  # budget = 0.5s
        s = ScrapyScraper(config)
        mock_items = [
            {
                "url": "https://example.com",
                "html_content": "<html></html>",
                "text_content": "",
                "title": "T",
                "metadata": {"status_code": 200, "final_url": "https://example.com"},
                "links": [],
                "images": [{"src": "https://example.com/img.png", "alt": "", "title": ""}],
                "depth": 0,
            }
        ]

        async def _hang(*_a, **_k):
            await asyncio.sleep(10)

        with (
            patch(
                "services.scrapy_scraper._run_crawl_in_executor",
                return_value=(mock_items, []),
            ),
            patch("services.scrapy_scraper.enrich_images_with_content", side_effect=_hang),
        ):
            result = await s.scrape("https://example.com", download_images=True)

        assert result.status == ScrapeStatus.COMPLETED
        assert "content_base64" not in result.images[0]


class TestResourceCleanup:
    @pytest.mark.asyncio
    async def test_close(self, scraper):
        # Should not raise - no persistent resources
        await scraper.close()

    @pytest.mark.asyncio
    async def test_close_multiple(self, scraper):
        await scraper.close()
        await scraper.close()  # Idempotent


class TestSsrfGuardMiddleware:
    """The downloader middleware that drops requests (incl. redirect hops) to
    blocked hosts."""

    def _mw(self):
        from services.scrapy_scraper import _SsrfGuardMiddleware

        return _SsrfGuardMiddleware()

    def test_blocked_request_raises_ignore(self):
        from scrapy.exceptions import IgnoreRequest

        mw = self._mw()
        req = type("R", (), {"url": "http://169.254.169.254/latest/meta-data/"})()
        with pytest.raises(IgnoreRequest):
            mw.process_request(req, spider=None)

    def test_internal_host_raises_ignore(self):
        from scrapy.exceptions import IgnoreRequest

        mw = self._mw()
        req = type("R", (), {"url": "http://registry.example.com/v2/"})()
        with pytest.raises(IgnoreRequest):
            mw.process_request(req, spider=None)

    def test_public_request_passes(self):
        mw = self._mw()
        req = type("R", (), {"url": "https://93.184.216.34/page"})()
        assert mw.process_request(req, spider=None) is None


class TestScrapyProxyMeta:
    """The spider stamps request.meta['proxy'] from settings.outbound_proxy."""

    def test_request_meta_none_by_default(self, monkeypatch):
        from config import get_settings
        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import WebSpider

        monkeypatch.setattr(get_settings(), "outbound_proxy", None, raising=False)
        spider = WebSpider("https://example.com", ScrapeConfig())
        assert spider._request_meta() is None

    def test_request_meta_carries_proxy(self, monkeypatch):
        from config import get_settings
        from services.base_scraper import ScrapeConfig
        from services.scrapy_scraper import WebSpider

        monkeypatch.setattr(
            get_settings(), "outbound_proxy", "http://proxy.test:3128", raising=False
        )
        spider = WebSpider("https://example.com", ScrapeConfig())
        assert spider._request_meta() == {"proxy": "http://proxy.test:3128"}
