"""Scrapy-based web scraper for efficient crawling.

Runs each Scrapy crawl in a fresh subprocess (one ``CrawlerProcess`` per
child) because Twisted's reactor is a process-global singleton that cannot
be restarted: a second ``CrawlerProcess.start()`` in the same interpreter
raises ``ReactorNotRestartable``. A spawn-based ``ProcessPoolExecutor`` with
``max_tasks_per_child=1`` gives every crawl a pristine interpreter + reactor.
Supports link following, depth control, robots.txt compliance, and retries.
"""

import asyncio
import logging
import multiprocessing as mp
import time
from collections.abc import AsyncIterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any
from urllib.parse import urljoin, urlparse

import scrapy
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import IgnoreRequest
from scrapy.http import Response as ScrapyResponse
from scrapy.settings import Settings as ScrapySettings
from scrapy.signalmanager import dispatcher

from config import get_settings
from services.base_scraper import BaseScraper, ScrapeConfig, ScrapeResult, ScrapeStatus
from services.content_extraction import extract_main_text
from services.curl_cffi_scraper import _origin_of, _without_credentials
from services.image_utils import IMAGE_ENRICH_OVERALL_TIMEOUT, enrich_images_with_content
from services.link_scoring import LinkScorer
from services.url_guard import UrlNotAllowedError, assert_allowed_url

logger = logging.getLogger(__name__)

# Each crawl runs in its OWN spawned subprocess (max_tasks_per_child=1) so the
# Twisted reactor is fresh every time — the only robust way to run Scrapy
# repeatedly in a long-lived service without ReactorNotRestartable. As a bonus,
# the global ``dispatcher.connect`` signal handlers (registered in _run_crawl)
# live and die inside the child, so they can never leak or cross-contaminate
# collectors across crawls. ``spawn`` keeps the child free of any reactor state
# the parent might have touched.
#
# Throughput note: scrapy crawls are bounded by ``max_workers=2`` (each with a
# ~1-2s spawn cost), NOT by the manager's ``scraping_max_concurrent`` semaphore
# — a burst of scrapy requests queues here rather than hanging. The executor is
# left to ``concurrent.futures``' atexit handler for child reaping on shutdown
# (a per-instance close() must not tear down this shared module global).


def _make_executor() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=2,
        mp_context=mp.get_context("spawn"),
        max_tasks_per_child=1,
    )


_executor = _make_executor()
# Guards the one-shot recreation of ``_executor`` after a BrokenProcessPool so
# that a burst of concurrent failures recreates the pool exactly once.
_executor_lock = asyncio.Lock()


class _SsrfGuardMiddleware:
    """Scrapy downloader middleware: SSRF-guard every request, incl. redirects.

    Scrapy follows 30x redirects by default and re-schedules each redirect
    target through ``process_request``, so validating here covers the whole
    chain — not just the start URL the manager already guarded. Runs in the
    Twisted reactor thread, so the sync (blocking-DNS) guard is fine here.
    """

    def __init__(self) -> None:
        self._allowed_hosts = tuple(get_settings().ssrf_allowed_hosts)

    def process_request(self, request: Any, spider: Any) -> None:
        try:
            assert_allowed_url(request.url, self._allowed_hosts)
        except UrlNotAllowedError as e:
            raise IgnoreRequest(str(e)) from e
        return None


def _parse_dim(value: str | None) -> int | None:
    """Parse an HTML width/height attribute into an int, or None if absent/invalid."""
    if not value:
        return None
    try:
        return int(value)
    except ValueError, TypeError:
        return None


class _ResultCollector:
    """Collects scraped items from a Scrapy spider run."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def item_scraped(
        self, item: dict[str, Any], response: ScrapyResponse, spider: scrapy.Spider
    ) -> None:
        self.items.append(dict(item))

    def spider_error(self, failure: Any, response: ScrapyResponse, spider: scrapy.Spider) -> None:
        self.errors.append(str(failure))


def _domain_allowed(url: str, allowed_domains: list[str] | None) -> bool:
    """Whether a followed link is inside the caller's declared crawl scope.

    Mirrors ``BaseScraper.should_scrape_url``'s domain rule (exact host or a
    subdomain of it); the spider is a ``scrapy.Spider``, not a ``BaseScraper``,
    so it cannot reuse the method directly. An empty/None list means the caller
    did not scope the crawl, which stays permissive.
    """
    if not allowed_domains:
        return True
    host = urlparse(url).netloc.split("@")[-1].split(":")[0].lower()
    return any(
        host == allowed.lower() or host.endswith(f".{allowed.lower()}")
        for allowed in allowed_domains
    )


class WebSpider(scrapy.Spider):
    """Generic spider that extracts content matching the ScrapeResult interface."""

    name = "web_spider"

    def __init__(self, start_url: str, config: ScrapeConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.start_urls = [start_url]
        self.config = config
        self._base_domain = urlparse(start_url).netloc
        self._visited: set[str] = set()
        self._current_depth: dict[str, int] = {start_url: 0}
        # Best-first frontier: Scrapy's scheduler dequeues higher `priority`
        # first, so scoring each followed link turns the default blind BFS into
        # a relevance-ordered crawl. This only matters because the crawl is
        # bounded (CLOSESPIDER caps + max_depth) — ordering decides which pages
        # the budget is actually spent on.
        self._scorer = LinkScorer.for_query(config.link_query)
        # Optional ops-level egress proxy (None → direct). Scrapy's default
        # HttpProxyMiddleware honours request.meta["proxy"].
        self._proxy: str | None = get_settings().outbound_proxy or None

    def _request_meta(self) -> dict[str, Any] | None:
        return {"proxy": self._proxy} if self._proxy else None

    def _caller_headers(self) -> dict[str, str]:
        """The caller's request-local headers (User-Agent first, explicit headers win)."""
        headers: dict[str, str] = {}
        if self.config.user_agent:
            headers["User-Agent"] = self.config.user_agent
        if self.config.headers:
            headers.update(self.config.headers)
        return headers

    def _same_origin_as_start(self, url: str) -> bool:
        """Same (scheme, host, port) as the start URL — the origin credentials were given for."""
        try:
            return _origin_of(url) == _origin_of(self.start_urls[0])
        except UrlNotAllowedError:
            return False

    def _follow_headers(self, url: str) -> dict[str, str] | None:
        """Headers for a FOLLOWED request.

        Followed links used to be bare ``Request(url)``s, so every page after the
        start page was fetched without the caller's custom headers (API keys,
        Accept-Language, Authorization, ...). Same-origin follows now carry all of
        them; a follow to any other origin (e.g. an http->https scheme change on the
        same netloc, which the "internal" link test admits) drops the
        credential-bearing ones, with the same semantics as the curl_cffi
        cross-origin redirect path.
        """
        headers = self._caller_headers()
        if not self._same_origin_as_start(url):
            headers = _without_credentials(headers)
        return headers or None

    async def start(self) -> AsyncIterator[scrapy.Request]:
        # Scrapy >= 2.13 drives the spider through ``start()`` and never calls
        # the legacy ``start_requests()``. The inherited default yields a bare
        # ``Request(url)`` that falls back to ``self.parse`` (not defined here —
        # our callback is ``parse_page``) and drops the configured headers,
        # cookies, errback and proxy meta, so it MUST be overridden.
        for url in self.start_urls:
            headers = self._caller_headers()
            cookies = self.config.cookies or {}
            yield scrapy.Request(
                url,
                callback=self.parse_page,
                headers=headers if headers else None,
                cookies=cookies if cookies else None,
                errback=self.errback,
                dont_filter=True,
                meta=self._request_meta(),
            )

    def parse_page(self, response: ScrapyResponse):
        url = response.url
        if url in self._visited:
            return
        self._visited.add(url)

        current_depth = self._current_depth.get(url, 0)

        # Extract content (absent title → None, matching httpx/playwright)
        title = response.css("title::text").get("").strip() or None

        # Clean main-content text (strips nav/header/footer chrome, keeps tables)
        # via the same shared extractor the httpx + Playwright scrapers use, so all
        # three emit identical ``text_content`` instead of chrome-polluted raw text.
        text_content = extract_main_text(response.text, response.url)

        # Extract metadata
        metadata: dict[str, Any] = {}
        for meta in response.css("meta"):
            name = meta.attrib.get("name") or meta.attrib.get("property")
            content = meta.attrib.get("content")
            if name and content:
                metadata[name] = content
        canonical = response.css('link[rel="canonical"]::attr(href)').get()
        if canonical:
            metadata["canonical_url"] = canonical
        # Authoritative transport values LAST so a scraped <meta name="status_code">
        # / <meta name="final_url"> can't clobber them (matches the Playwright path).
        metadata["status_code"] = response.status
        metadata["final_url"] = response.url

        # Extract links
        links: list[dict[str, Any]] = []
        for a in response.css("a[href]"):
            href = a.attrib.get("href", "")
            abs_url = urljoin(response.url, href)
            link_domain = urlparse(abs_url).netloc
            # Absent rel → None (not ""), matching httpx/playwright.
            rel = a.attrib.get("rel") or None
            # Concatenate ALL descendant text nodes so nested-markup anchors
            # (<a><b>Read</b> more</a>) keep their full text like get_text/innerText.
            text = " ".join(t.strip() for t in a.css("::text").getall() if t.strip())
            links.append(
                {
                    "text": text,
                    "url": abs_url,
                    "rel": rel,
                    "type": "internal" if link_domain == self._base_domain else "external",
                }
            )

        # Extract images (resolve relative URLs, read lazy-load attrs, width/height)
        images: list[dict[str, Any]] = []
        seen_srcs: set[str] = set()
        for img in response.css("img"):
            src = (
                img.attrib.get("data-src")
                or img.attrib.get("data-original")
                or img.attrib.get("data-lazy-src")
                or img.attrib.get("src", "")
            )
            if src and not src.startswith(("http://", "https://", "data:")):
                src = urljoin(response.url, src)
            seen_srcs.add(src)

            images.append(
                {
                    "src": src,
                    "alt": img.attrib.get("alt", ""),
                    "title": img.attrib.get("title", ""),
                    "width": _parse_dim(img.attrib.get("width")),
                    "height": _parse_dim(img.attrib.get("height")),
                    "srcset": img.attrib.get("srcset") or None,
                }
            )

        # Extract <picture><source> elements
        for picture in response.css("picture"):
            img_tag = picture.css("img")
            if not img_tag:
                continue
            img_el = img_tag[0]
            src = img_el.attrib.get("data-src") or img_el.attrib.get("src", "")
            if src and not src.startswith(("http://", "https://", "data:")):
                src = urljoin(response.url, src)
            best_srcset = ""
            for source in picture.css("source"):
                ss = source.attrib.get("srcset", "")
                if ss:
                    best_srcset = ss
            if src in seen_srcs:
                # Update existing image's srcset from <source> if missing
                if best_srcset:
                    for existing in images:
                        if existing["src"] == src and not existing.get("srcset"):
                            existing["srcset"] = best_srcset
                            break
                continue
            seen_srcs.add(src)
            images.append(
                {
                    "src": src,
                    "alt": img_el.attrib.get("alt", ""),
                    "title": img_el.attrib.get("title", ""),
                    "width": _parse_dim(img_el.attrib.get("width")),
                    "height": _parse_dim(img_el.attrib.get("height")),
                    "srcset": best_srcset or None,
                }
            )

        yield {
            "url": url,
            "html_content": response.text,
            "text_content": text_content,
            "title": title,
            "metadata": metadata,
            "links": links,
            "images": images,
            "depth": current_depth,
        }

        # Follow links if configured
        if self.config.follow_links and current_depth < self.config.max_depth:
            for link in links:
                link_url = link["url"]
                if (
                    link["type"] == "internal"
                    and link_url not in self._visited
                    and link_url not in self._current_depth
                    # The caller's allowed_domains was accepted, documented and
                    # then never consulted — a crawl a caller believed they had
                    # scoped went wherever the pages pointed.
                    and _domain_allowed(link_url, self.config.allowed_domains)
                ):
                    self._current_depth[link_url] = current_depth + 1
                    score = self._scorer.score(
                        url=link_url,
                        text=link.get("text", ""),
                        rel=link.get("rel"),
                        link_type=link.get("type"),
                    )
                    yield scrapy.Request(
                        link_url,
                        callback=self.parse_page,
                        # Caller cookies need no per-request copy: the start
                        # request's ``cookies=`` seeds Scrapy's domain-scoped
                        # cookie jar, which re-sends them to that host only.
                        headers=self._follow_headers(link_url),
                        errback=self.errback,
                        meta=self._request_meta(),
                        # [0, 1] → integer priority; shallower pages also win ties
                        # so a relevant deep page never starves a relevant shallow one.
                        priority=int(score * 1000) - current_depth,
                    )

    def errback(self, failure: Any):
        logger.error(f"Scrapy request error: {failure}")


def _run_crawl(start_url: str, config: ScrapeConfig) -> tuple[list[dict[str, Any]], list[str]]:
    """Run one Scrapy crawl in a separate process (fresh Twisted reactor per crawl)."""
    collector = _ResultCollector()

    settings = ScrapySettings()
    settings.set("LOG_ENABLED", False)
    settings.set("ROBOTSTXT_OBEY", config.respect_robots_txt)
    settings.set("DOWNLOAD_TIMEOUT", config.timeout)
    settings.set("RETRY_TIMES", config.max_retries)
    settings.set("RETRY_HTTP_CODES", [500, 502, 503, 504, 408, 429])
    settings.set("DOWNLOAD_DELAY", config.delay)
    settings.set("CONCURRENT_REQUESTS", 4)
    settings.set("DEPTH_LIMIT", config.max_depth if config.follow_links else 0)
    settings.set("TELNETCONSOLE_ENABLED", False)
    settings.set("REQUEST_FINGERPRINTER_IMPLEMENTATION", "2.7")

    # Bound the whole crawl, not just each request: a follow_links crawl must not
    # run unbounded in wall-clock or accumulate unbounded pages in this child's
    # memory (only items[0] is ever used). These close the spider gracefully.
    app_settings = get_settings()
    settings.set("CLOSESPIDER_TIMEOUT", app_settings.scrapy_crawl_timeout)
    settings.set("CLOSESPIDER_PAGECOUNT", app_settings.scrapy_max_pages)

    # Per-response body cap (parity with the httpx scraper's max_page_bytes): a
    # server returning a multi-GB body must not OOM the crawl subprocess. Scrapy
    # drops a response past DOWNLOAD_MAXSIZE; WARNSIZE=0 silences the redundant
    # warn (the hard cap already drops it first).
    settings.set("DOWNLOAD_MAXSIZE", app_settings.max_page_bytes)
    settings.set("DOWNLOAD_WARNSIZE", 0)

    # SSRF: guard every request (incl. redirect hops Scrapy follows by default).
    if get_settings().ssrf_protection_enabled:
        settings.set(
            "DOWNLOADER_MIDDLEWARES",
            {"services.scrapy_scraper._SsrfGuardMiddleware": 100},
        )

    if config.user_agent:
        settings.set("USER_AGENT", config.user_agent)

    process = CrawlerProcess(settings, install_root_handler=False)

    dispatcher.connect(collector.item_scraped, signal=signals.item_scraped)
    dispatcher.connect(collector.spider_error, signal=signals.spider_error)

    process.crawl(WebSpider, start_url=start_url, config=config)
    process.start(stop_after_crawl=True, install_signal_handlers=False)

    return collector.items, collector.errors


async def _run_crawl_in_executor(
    start_url: str, config: ScrapeConfig
) -> tuple[list[dict[str, Any]], list[str]]:
    """Await one Scrapy crawl, executed in a fresh subprocess (see ``_executor``).

    Indirection point: ``_run_crawl`` runs in another process (it must stay
    picklable), so unit tests patch THIS awaitable instead to avoid pickling a
    mock across the process boundary.

    Recovers once from ``BrokenProcessPool``: ``max_tasks_per_child=1`` means a
    child hard-crash (segfault in lxml/twisted, OOM-kill, spawn failure) would
    otherwise mark the pool permanently broken, FAILing every subsequent scrapy
    crawl until the pod restarts. On a break we recreate the pool exactly once
    (identity-checked under a lock so concurrent failures don't thrash it) and
    retry; a retry that also breaks propagates → just that one request FAILs.

    Build-before-teardown is deliberate: the very condition that breaks the pool
    (OOM / FD / semaphore exhaustion) can also make ``_make_executor()`` raise.
    We construct the replacement FIRST and only ``shutdown`` the old pool once
    the new one exists, so a construction failure leaves ``_executor`` pointing
    at the (broken-but-not-shut-down) old pool — the next request re-raises
    ``BrokenProcessPool`` and re-enters recovery, instead of hitting a dead
    shut-down pool with a non-``BrokenProcessPool`` ``RuntimeError`` forever.
    """
    global _executor
    loop = asyncio.get_running_loop()
    executor = _executor
    try:
        return await loop.run_in_executor(executor, _run_crawl, start_url, config)
    except BrokenProcessPool:
        logger.error("Scrapy subprocess pool broke (child crash); recreating and retrying once")
        async with _executor_lock:
            if _executor is executor:  # first waiter recreates; others reuse it
                # Construct FIRST: if this raises, _executor is left untouched
                # (the broken-but-not-shut-down old pool) so the next request
                # re-enters recovery instead of hitting a dead shut-down pool.
                new_executor = _make_executor()
                executor.shutdown(wait=False)  # tear down old pool only once the new one exists
                _executor = new_executor
            retry_executor = _executor
        return await loop.run_in_executor(retry_executor, _run_crawl, start_url, config)


class ScrapyScraper(BaseScraper):
    """Scrapy-based scraper for efficient web crawling.

    Uses the Scrapy framework for crawling with link following,
    robots.txt compliance, and automatic retries.

    Best for: bulk scraping, crawling with follow_links=True,
    sites that need robots.txt compliance.
    """

    def __init__(self, config: ScrapeConfig | None = None) -> None:
        super().__init__(config)

    async def scrape(self, url: str, **kwargs: Any) -> ScrapeResult:
        """Scrape a URL using Scrapy.

        Args:
            url: The URL to scrape.
            **kwargs: Additional options (unused, for interface compat).

        Returns:
            ScrapeResult with extracted content.
        """
        start_time = time.time()

        # Request-local config (never self.config — a concurrent request may have
        # mutated the shared instance).
        cfg = self._resolve_config(kwargs)

        # Backstop the in-process CLOSESPIDER_TIMEOUT: if the subprocess wedges
        # (stuck DNS, a hung child) the close-spider setting can't fire, so bound
        # the await too and free the manager's concurrency permit. Budget = crawl
        # timeout + headroom for graceful close + final-request download.
        overall_budget = get_settings().scrapy_crawl_timeout + max(cfg.timeout, 30.0)

        try:
            try:
                items, errors = await asyncio.wait_for(
                    _run_crawl_in_executor(url, cfg), timeout=overall_budget
                )
            except TimeoutError:
                processing_time = time.time() - start_time
                logger.error(
                    f"Scrapy crawl for {url} exceeded the overall budget "
                    f"({overall_budget:.0f}s); returning FAILED"
                )
                return ScrapeResult(
                    url=url,
                    status=ScrapeStatus.FAILED,
                    error=f"crawl exceeded overall budget after {overall_budget:.0f}s",
                    scraper_used=self.name,
                    processing_time_seconds=processing_time,
                )

            processing_time = time.time() - start_time

            if not items:
                error_msg = "; ".join(errors) if errors else "No content extracted"
                logger.error(f"Scrapy failed for {url}: {error_msg}")
                return ScrapeResult(
                    url=url,
                    status=ScrapeStatus.FAILED,
                    error=error_msg,
                    scraper_used=self.name,
                    processing_time_seconds=processing_time,
                )

            # Use the first item (the requested URL's content)
            primary = items[0]
            images = primary.get("images", [])

            # Download image content if requested, bounded so a slow image host
            # can't hang the scrape past the caller's budget (keep un-enriched
            # images on timeout). Parity with the httpx path; the crawl itself
            # already ran in the subprocess above.
            download_images: bool = kwargs.get("download_images", cfg.download_images)
            if download_images and images:
                enrich_budget = min(cfg.timeout * 2, IMAGE_ENRICH_OVERALL_TIMEOUT)
                try:
                    images = await asyncio.wait_for(
                        enrich_images_with_content(images), timeout=enrich_budget
                    )
                except TimeoutError:
                    logger.warning(
                        f"Image enrichment exceeded {enrich_budget:.0f}s for {url}; "
                        "returning images without downloaded content"
                    )

            logger.info(
                f"Successfully scraped {url} in {processing_time:.2f}s "
                f"({len(items)} page(s) crawled)"
            )

            return ScrapeResult(
                url=url,
                html_content=primary.get("html_content"),
                text_content=primary.get("text_content"),
                title=primary.get("title"),
                metadata=primary.get("metadata", {}),
                links=primary.get("links", []),
                images=images,
                status=ScrapeStatus.COMPLETED,
                scraper_used=self.name,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            logger.error(f"Error scraping {url} with Scrapy: {e}", exc_info=True)
            return ScrapeResult(
                url=url,
                status=ScrapeStatus.FAILED,
                error=f"{type(e).__name__}: {e}" if str(e) else repr(e),
                scraper_used=self.name,
                processing_time_seconds=time.time() - start_time,
            )

    async def close(self) -> None:
        """No persistent resources to close for Scrapy (process-per-crawl)."""
        logger.info("ScrapyScraper closed")


__all__ = ["ScrapyScraper"]
