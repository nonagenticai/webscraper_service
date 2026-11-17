"""Lightweight HTTP scraper using httpx and BeautifulSoup.

Provides fast, async scraping for static HTML pages without
requiring a browser. Uses httpx for HTTP requests and
BeautifulSoup with lxml for HTML parsing.
"""

import asyncio
import logging
import time
from typing import Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, Tag

from config import get_settings
from services.base_scraper import BaseScraper, ScrapeConfig, ScrapeResult, ScrapeStatus
from services.content_extraction import extract_main_text
from services.image_utils import IMAGE_ENRICH_OVERALL_TIMEOUT, enrich_images_with_content
from services.url_guard import make_outbound_guard_hook

logger = logging.getLogger(__name__)

# Transient response codes worth retrying — mirrors the Scrapy backend's
# RETRY_HTTP_CODES so httpx doesn't needlessly fall back to Playwright on a
# blip. 4xx other than 408/429 are caller/permanent errors and are NOT retried.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_MAX_BACKOFF_SECONDS = 8.0


class ResponseTooLargeError(Exception):
    """A fetched response body exceeded the configured ``max_page_bytes`` cap.

    Raised mid-stream so an oversized/malicious body is aborted before it is
    fully buffered (and before BeautifulSoup/trafilatura multiply it) — the
    scrape returns FAILED instead of OOMing the pod.
    """


class HttpxScraper(BaseScraper):
    """Lightweight scraper using httpx + BeautifulSoup.

    Suitable for static HTML pages that don't require JavaScript
    rendering. Much faster and lighter than Playwright.

    Holds no persistent client: each ``scrape`` builds a request-local client
    via ``_build_client`` for isolation, closed by its own context manager.
    """

    def __init__(self, config: ScrapeConfig | None = None) -> None:
        super().__init__(config)

    def _build_client(self, cfg: ScrapeConfig, timeout: float) -> httpx.AsyncClient:
        """Build a request-local httpx client from the per-request config.

        A fresh client per request — NOT a cached singleton — so concurrent
        requests can't share each other's User-Agent / headers / cookies /
        timeout. (The previous cached ``self._client`` baked the first request's
        config in permanently, leaking it into every later request.)
        """
        headers: dict[str, str] = {}
        if cfg.user_agent:
            headers["User-Agent"] = cfg.user_agent
        if cfg.headers:
            headers.update(cfg.headers)

        # SSRF: re-validate EVERY hop (httpx follows redirects), not just the
        # initial URL the manager already checked — a public page can 302 to an
        # internal/metadata host. The hook raises UrlNotAllowedError → FAILED.
        settings = get_settings()
        event_hooks = (
            {"request": [make_outbound_guard_hook(settings.ssrf_allowed_hosts)]}
            if settings.ssrf_protection_enabled
            else {}
        )

        return httpx.AsyncClient(
            headers=headers,
            cookies=httpx.Cookies(cfg.cookies) if cfg.cookies else None,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            verify=True,
            event_hooks=event_hooks,
            # Optional ops-level egress proxy (None → direct). The SSRF hook still
            # validates the target host regardless of the proxy.
            proxy=settings.outbound_proxy or None,
        )

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        """Parse a numeric Retry-After header (seconds), capped. None otherwise."""
        raw = response.headers.get("retry-after")
        if not raw:
            return None
        try:
            return min(float(raw), _MAX_BACKOFF_SECONDS)
        except ValueError:
            # HTTP-date form is not honored here; fall back to exponential backoff.
            return None

    @staticmethod
    async def _read_capped(response: httpx.Response, url: str, max_bytes: int) -> str:
        """Read a streamed response body into text, aborting past ``max_bytes``.

        Mirrors the image-download guard (``image_utils.download_image``): never
        buffer the whole body just to discover it's oversized. Decodes with the
        Content-Type charset when present, else UTF-8 with ``errors='replace'``
        (a scraper tradeoff vs ``response.text``'s content-sniffing — acceptable;
        most pages are UTF-8). Raises ``ResponseTooLargeError`` on overflow.
        """
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise ResponseTooLargeError(f"response body exceeded {max_bytes} bytes for {url}")
            chunks.append(chunk)
        body = b"".join(chunks)
        encoding = response.encoding or "utf-8"
        try:
            return body.decode(encoding, errors="replace")
        except LookupError, TypeError:
            # An unknown/invalid charset label — fall back to UTF-8.
            return body.decode("utf-8", errors="replace")

    async def _get_with_retries(
        self, client: httpx.AsyncClient, url: str, max_retries: int, max_bytes: int
    ) -> tuple[httpx.Response, str]:
        """GET ``url`` with bounded retries and a hard response-body cap.

        Streams the response so an oversized/malicious body is bounded at
        ``max_bytes`` (a buffered ``client.get`` would resident the whole body —
        and the per-chunk read timeout never bounds total transfer). Retryable
        responses are never read at all.

        Retries up to ``max_retries`` times with exponential backoff
        (0.5 * 2**n, capped at 8s), honoring a numeric ``Retry-After`` header,
        on transient transport errors and retryable status codes (408/429/5xx).
        Returns the final response together with its decoded body text; permanent
        / final-attempt error responses are returned for the caller's
        ``raise_for_status``. Raises ``ResponseTooLargeError`` past the cap.
        """
        attempt = 0
        while True:
            try:
                async with client.stream("GET", url) as response:
                    retryable = response.status_code in _RETRYABLE_STATUS and attempt < max_retries
                    if not retryable:
                        # Read the (capped) body inside the stream context, then
                        # hand the response + text back. Non-retryable error
                        # statuses still flow to the caller's raise_for_status.
                        html_content = await self._read_capped(response, url, max_bytes)
                        return response, html_content
                    delay = self._retry_after_seconds(response) or min(
                        0.5 * 2**attempt, _MAX_BACKOFF_SECONDS
                    )
                    logger.info(
                        f"httpx got {response.status_code} for {url}; "
                        f"retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                    )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt >= max_retries:
                    raise
                delay = min(0.5 * 2**attempt, _MAX_BACKOFF_SECONDS)
                logger.info(
                    f"httpx transient error ({type(e).__name__}) for {url}; "
                    f"retry {attempt + 1}/{max_retries} in {delay:.1f}s"
                )

            # Reached only on a retry (retryable status, body left unread and the
            # stream closed; or a transient error). Back off, then retry.
            await asyncio.sleep(delay)
            attempt += 1

    async def scrape(self, url: str, **kwargs: Any) -> ScrapeResult:
        """Scrape a URL using httpx + BeautifulSoup.

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
        timeout = float(kwargs.get("timeout") or cfg.timeout)

        try:
            settings = get_settings()
            async with self._build_client(cfg, timeout) as client:
                response, html_content = await self._get_with_retries(
                    client, url, cfg.max_retries, settings.max_page_bytes
                )
                response.raise_for_status()

            download_images: bool = kwargs.get("download_images", cfg.download_images)
            return await self._assemble_result(
                url,
                html_content,
                response.status_code,
                str(response.url),
                cfg,
                timeout,
                download_images,
                start_time,
            )

        except httpx.TimeoutException as e:
            logger.error(f"Timeout scraping {url}: {e}")
            return ScrapeResult(
                url=url,
                status=ScrapeStatus.FAILED,
                error=f"Timeout: {e}",
                scraper_used=self.name,
                processing_time_seconds=time.time() - start_time,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error scraping {url}: {e.response.status_code}")
            return ScrapeResult(
                url=url,
                status=ScrapeStatus.FAILED,
                error=f"HTTP {e.response.status_code}: {e}",
                scraper_used=self.name,
                processing_time_seconds=time.time() - start_time,
            )

        except ResponseTooLargeError as e:
            # Server-controlled oversized body — aborted mid-stream before it
            # could OOM the pod. FAILED (not a truncated/half-parsed 200).
            logger.warning(f"Oversized response scraping {url}: {e}")
            return ScrapeResult(
                url=url,
                status=ScrapeStatus.FAILED,
                error=str(e),
                scraper_used=self.name,
                processing_time_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}", exc_info=True)
            return ScrapeResult(
                url=url,
                status=ScrapeStatus.FAILED,
                error=f"{type(e).__name__}: {e}" if str(e) else repr(e),
                scraper_used=self.name,
                processing_time_seconds=time.time() - start_time,
            )

    def _extract_metadata(
        self, soup: BeautifulSoup, status_code: int, final_url: str, url: str
    ) -> dict[str, Any]:
        """Extract metadata from meta tags.

        ``status_code``/``final_url`` are the authoritative transport values (the
        caller passes them from the concrete HTTP response, httpx or curl_cffi).
        """
        metadata: dict[str, Any] = {}

        for tag in soup.find_all("meta"):
            name = tag.get("name") or tag.get("property")
            content = tag.get("content")
            if name and content:
                metadata[str(name)] = content

        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            metadata["canonical_url"] = str(canonical["href"])

        # Authoritative transport values LAST so a scraped <meta name="status_code">
        # / <meta name="final_url"> can't clobber them (matches the Playwright path).
        metadata["status_code"] = status_code
        metadata["final_url"] = final_url

        return metadata

    async def _assemble_result(
        self,
        url: str,
        html_content: str,
        status_code: int,
        final_url: str,
        cfg: ScrapeConfig,
        timeout: float,
        download_images: bool,
        start_time: float,
    ) -> ScrapeResult:
        """Parse fetched HTML into a ScrapeResult.

        Backend-agnostic post-fetch assembly (parse → title/text/metadata/links/
        images → optional image enrichment) shared by the httpx and curl_cffi
        scrapers so both emit an identical ScrapeResult shape. The fetch itself
        (transport, redirects, retries) is the caller's responsibility.
        """
        soup = BeautifulSoup(html_content, "lxml")
        title = soup.title.string.strip() if soup.title and soup.title.string else None

        text_content = extract_main_text(html_content, url)
        metadata = self._extract_metadata(soup, status_code, final_url, url)
        links = self._extract_links(soup, url)
        images = self._extract_images(soup, url)

        # Download image content if requested. Bound the whole enrichment phase so
        # a slow/stalling image host can't hang an otherwise-good scrape past the
        # caller's budget; on timeout keep the un-enriched image list.
        if download_images and images:
            enrich_budget = min(timeout * 2, IMAGE_ENRICH_OVERALL_TIMEOUT)
            try:
                images = await asyncio.wait_for(
                    enrich_images_with_content(images), timeout=enrich_budget
                )
            except TimeoutError:
                logger.warning(
                    f"Image enrichment exceeded {enrich_budget:.0f}s for {url}; "
                    "returning images without downloaded content"
                )

        processing_time = time.time() - start_time
        logger.info(f"Successfully scraped {url} in {processing_time:.2f}s")

        return ScrapeResult(
            url=url,
            html_content=html_content,
            text_content=text_content,
            title=title,
            metadata=metadata,
            links=links,
            images=images,
            status=ScrapeStatus.COMPLETED,
            scraper_used=self.name,
            processing_time_seconds=processing_time,
        )

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[dict[str, Any]]:
        """Extract all anchor links."""
        links: list[dict[str, Any]] = []
        base_domain = self.extract_domain(base_url)

        for a in soup.find_all("a", href=True):
            href = str(a["href"])
            abs_url = urljoin(base_url, href)
            link_domain = self.extract_domain(abs_url)

            # bs4 returns ``rel`` as a token list; join to preserve ALL tokens
            # (e.g. "nofollow noopener") instead of dropping all but the first,
            # matching the Playwright + Scrapy scrapers. Absent → None.
            rel_val = a.get("rel")
            if isinstance(rel_val, list):
                rel_str: str | None = " ".join(rel_val) if rel_val else None
            else:
                rel_str = str(rel_val) if rel_val else None

            links.append(
                {
                    # Space-join text nodes so nested-markup anchors read
                    # "Read more" (not "Readmore"), matching scrapy/playwright.
                    "text": a.get_text(" ", strip=True),
                    "url": abs_url,
                    "rel": rel_str,
                    "type": "internal" if link_domain == base_domain else "external",
                }
            )

        return links

    @staticmethod
    def _parse_int(value: object) -> int | None:
        """Parse an HTML attribute value to int, or return None."""
        if not value:
            return None
        try:
            return int(str(value))
        except (ValueError, TypeError):  # fmt: skip
            return None

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> list[dict[str, Any]]:
        """Extract all images, resolving relative URLs.

        Handles lazy-loaded images (data-src, data-original, data-lazy-src),
        responsive images (srcset), and <picture> elements.
        Also reads width/height HTML attributes.
        """
        images: list[dict[str, Any]] = []
        seen_srcs: set[str] = set()

        for img in soup.find_all("img"):
            # Prefer lazy-load data attributes over placeholder src
            src = (
                img.get("data-src")
                or img.get("data-original")
                or img.get("data-lazy-src")
                or img.get("src")
                or ""
            )
            src = str(src)
            if src and not src.startswith(("http://", "https://", "data:")):
                src = urljoin(base_url, src)
            srcset = str(img.get("srcset", "") or "")
            seen_srcs.add(src)
            images.append(
                {
                    "src": src,
                    "alt": str(img.get("alt", "")),
                    "title": str(img.get("title", "")),
                    "width": self._parse_int(img.get("width")),
                    "height": self._parse_int(img.get("height")),
                    "srcset": srcset or None,
                }
            )

        # Extract <picture><source> elements
        for picture in soup.find_all("picture"):
            img_tag = picture.find("img")
            if not isinstance(img_tag, Tag):
                continue
            src = img_tag.get("data-src") or img_tag.get("src") or ""
            src = str(src)
            if src and not src.startswith(("http://", "https://", "data:")):
                src = urljoin(base_url, src)
            # Collect srcset from <source> elements
            best_srcset = ""
            for source in picture.find_all("source"):
                ss = source.get("srcset", "")
                if ss:
                    best_srcset = str(ss)
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
                    "alt": str(img_tag.get("alt", "")),
                    "title": str(img_tag.get("title", "")),
                    "width": self._parse_int(img_tag.get("width")),
                    "height": self._parse_int(img_tag.get("height")),
                    "srcset": best_srcset or None,
                }
            )

        return images

    async def close(self) -> None:
        """No persistent resources to close (clients are per-request)."""
        logger.info("HttpxScraper closed")


__all__ = ["HttpxScraper"]
