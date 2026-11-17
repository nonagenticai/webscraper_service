"""Tests for HttpxScraper."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from services.base_scraper import ScrapeConfig, ScrapeStatus
from services.httpx_scraper import HttpxScraper

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="A test page">
    <meta property="og:title" content="OG Title">
    <link rel="canonical" href="https://example.com/canonical">
</head>
<body>
    <script>var x = 1;</script>
    <style>.hidden { display: none; }</style>
    <h1>Hello World</h1>
    <p>Some content here.</p>
    <a href="/about">About</a>
    <a href="https://external.com/page">External</a>
    <img src="/img/logo.png" alt="Logo" title="Site Logo">
    <img src="https://cdn.example.com/photo.jpg" alt="Photo">
</body>
</html>
"""


class _AsyncCtxClient:
    """Wrap a mock client so ``async with _build_client(...) as client`` yields it.

    ``scrape`` now builds a request-local client and enters it as an async context
    manager, so test doubles for ``_build_client`` must support the protocol.
    """

    def __init__(self, client: object) -> None:
        self._client = client

    async def __aenter__(self) -> object:
        return self._client

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _StreamCtx:
    """async-context-manager returned by a mocked ``client.stream(...)``.

    ``_get_with_retries`` now streams (``async with client.stream("GET", url)``)
    and reads the body with a hard cap, so test doubles mock ``client.stream``
    (a sync call returning this ctx) rather than ``client.get``.
    """

    def __init__(self, response: object) -> None:
        self._response = response

    async def __aenter__(self) -> object:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _stream_response(
    body: str | bytes = SAMPLE_HTML,
    status: int = 200,
    *,
    encoding: str = "utf-8",
    url: str = "https://example.com/",
    retry_after: str | None = None,
    chunk_size: int | None = None,
):
    """Build a mock streamed ``httpx.Response`` (status/encoding/url/headers,
    ``aiter_bytes()``, ``raise_for_status()``)."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.encoding = encoding
    resp.url = httpx.URL(url)
    resp.headers = {"retry-after": retry_after} if retry_after else {}

    data = body.encode(encoding) if isinstance(body, str) else body
    chunks = (
        [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
        if chunk_size
        else [data]
    ) or [b""]

    async def _aiter_bytes(*_a, **_k):
        for chunk in chunks:
            yield chunk

    resp.aiter_bytes = _aiter_bytes

    if status >= 400:
        err = httpx.HTTPStatusError("err", request=MagicMock(), response=resp)
        resp.raise_for_status = MagicMock(side_effect=err)
    else:
        resp.raise_for_status = MagicMock(return_value=None)
    return resp


def _stream_client(*responses):
    """A mock AsyncClient whose ``.stream(...)`` yields the given response(s).

    Pass response mocks (from ``_stream_response``) and/or exception instances —
    an exception is raised when ``client.stream(...)`` is called (mirrors a
    transport error opening the connection).
    """
    client = AsyncMock(spec=httpx.AsyncClient)
    if len(responses) == 1:
        only = responses[0]
        if isinstance(only, BaseException):
            # Scalar side_effect → raised on EVERY call (mirrors a transport error
            # that recurs across all retry attempts).
            client.stream = MagicMock(side_effect=only)
        else:
            client.stream = MagicMock(return_value=_StreamCtx(only))
        return client
    # A sequence walked across retries; an exception entry is raised on its call.
    effects = [e if isinstance(e, BaseException) else _StreamCtx(e) for e in responses]
    client.stream = MagicMock(side_effect=effects)
    return client


@pytest.fixture
def scraper():
    return HttpxScraper(ScrapeConfig(timeout=10))


@pytest.fixture
def mock_response():
    """A streamed 200 response carrying SAMPLE_HTML."""
    return _stream_response()


class TestHttpxScraperInit:
    def test_default_init(self):
        s = HttpxScraper()
        assert s.name == "HttpxScraper"
        # No persistent client — clients are built per-request in scrape().
        assert not hasattr(s, "_client")

    def test_init_with_config(self):
        config = ScrapeConfig(timeout=60, user_agent="TestBot")
        s = HttpxScraper(config)
        assert s.config.timeout == 60
        assert s.config.user_agent == "TestBot"


class TestHttpxScraperScrape:
    @pytest.mark.asyncio
    async def test_scrape_success(self, scraper, mock_response):
        mock_client = _stream_client(mock_response)

        with patch.object(scraper, "_build_client", return_value=_AsyncCtxClient(mock_client)):
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.COMPLETED
        assert result.url == "https://example.com"
        assert result.title == "Test Page"
        assert "Hello World" in result.text_content
        assert "var x = 1" not in result.text_content
        assert result.scraper_used == "HttpxScraper"
        assert result.processing_time_seconds > 0

    @pytest.mark.asyncio
    async def test_scrape_timeout(self, scraper):
        mock_client = _stream_client(httpx.TimeoutException("Connection timed out"))

        with patch.object(scraper, "_build_client", return_value=_AsyncCtxClient(mock_client)):
            result = await scraper.scrape("https://slow.example.com")

        assert result.status == ScrapeStatus.FAILED
        assert "Timeout" in result.error
        assert result.scraper_used == "HttpxScraper"

    @pytest.mark.asyncio
    async def test_scrape_http_error(self, scraper):
        mock_client = _stream_client(
            _stream_response(status=404, url="https://example.com/missing")
        )

        with patch.object(scraper, "_build_client", return_value=_AsyncCtxClient(mock_client)):
            result = await scraper.scrape("https://example.com/missing")

        assert result.status == ScrapeStatus.FAILED
        assert "404" in result.error

    @pytest.mark.asyncio
    async def test_scrape_generic_exception(self, scraper):
        mock_client = _stream_client(RuntimeError("Connection reset"))

        with patch.object(scraper, "_build_client", return_value=_AsyncCtxClient(mock_client)):
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.FAILED
        assert "Connection reset" in result.error

    @pytest.mark.asyncio
    async def test_scrape_oversized_body_returns_failed(self, scraper):
        """A server-controlled oversized body is aborted mid-stream → FAILED,
        never buffered whole (no OOM)."""
        big = "<html><body>" + ("x" * 5000) + "</body></html>"
        mock_client = _stream_client(_stream_response(body=big, chunk_size=512))

        stub = MagicMock()
        stub.max_page_bytes = 1000  # smaller than the body
        with (
            patch.object(scraper, "_build_client", return_value=_AsyncCtxClient(mock_client)),
            patch("services.httpx_scraper.get_settings", return_value=stub),
        ):
            result = await scraper.scrape("https://example.com/huge")

        assert result.status == ScrapeStatus.FAILED
        assert "exceeded" in result.error
        assert "1000" in result.error

    @pytest.mark.asyncio
    async def test_scrape_body_at_cap_still_succeeds(self, scraper):
        """A body within the cap streams + parses normally (regression guard for
        the new streaming path)."""
        mock_client = _stream_client(_stream_response(chunk_size=64))

        stub = MagicMock()
        stub.max_page_bytes = 10 * 1024 * 1024
        with (
            patch.object(scraper, "_build_client", return_value=_AsyncCtxClient(mock_client)),
            patch("services.httpx_scraper.get_settings", return_value=stub),
        ):
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.COMPLETED
        assert result.title == "Test Page"
        assert "Hello World" in result.text_content


class TestContentExtraction:
    def test_extract_text(self, scraper):
        from services.content_extraction import extract_main_text

        text = extract_main_text(SAMPLE_HTML, "https://example.com")
        assert "Hello World" in text
        assert "Some content here" in text
        assert "var x = 1" not in text
        assert ".hidden" not in text

    def test_extract_metadata(self, scraper, mock_response):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(SAMPLE_HTML, "lxml")
        metadata = scraper._extract_metadata(
            soup, mock_response.status_code, str(mock_response.url), "https://example.com"
        )
        assert metadata["status_code"] == 200
        assert metadata["description"] == "A test page"
        assert metadata["og:title"] == "OG Title"
        assert metadata["canonical_url"] == "https://example.com/canonical"

    def test_extract_metadata_transport_values_not_clobbered(self, scraper, mock_response):
        """A scraped <meta name="status_code"/"final_url"> must NOT overwrite the
        authoritative transport values (regression: they were set BEFORE the loop)."""
        from bs4 import BeautifulSoup

        html = (
            "<html><head>"
            '<meta name="status_code" content="evil">'
            '<meta name="final_url" content="http://attacker.test/">'
            "</head><body>x</body></html>"
        )
        soup = BeautifulSoup(html, "lxml")
        metadata = scraper._extract_metadata(
            soup, mock_response.status_code, str(mock_response.url), "https://example.com"
        )
        assert metadata["status_code"] == 200  # int from the response, not "evil"
        assert metadata["final_url"] == "https://example.com/"

    def test_extract_links(self, scraper):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(SAMPLE_HTML, "lxml")
        links = scraper._extract_links(soup, "https://example.com")
        assert len(links) == 2

        about_link = next(lnk for lnk in links if "about" in lnk["url"].lower())
        assert about_link["text"] == "About"
        assert about_link["type"] == "internal"

        external_link = next(lnk for lnk in links if "external" in lnk["url"].lower())
        assert external_link["type"] == "external"

    def test_extract_links_multi_token_rel_preserved(self, scraper):
        """Multi-value rel keeps ALL tokens (regression: was dropping all but first)."""
        from bs4 import BeautifulSoup

        html = '<html><body><a href="/p" rel="nofollow noopener">P</a></body></html>'
        soup = BeautifulSoup(html, "lxml")
        links = scraper._extract_links(soup, "https://example.com")
        assert links[0]["rel"] == "nofollow noopener"

    def test_extract_links_absent_rel_is_none(self, scraper):
        """Absent rel → None, consistent with scrapy/playwright."""
        from bs4 import BeautifulSoup

        html = '<html><body><a href="/page">Page</a></body></html>'
        soup = BeautifulSoup(html, "lxml")
        links = scraper._extract_links(soup, "https://example.com")
        assert links[0]["rel"] is None

    def test_extract_links_nested_markup_text_spaced(self, scraper):
        """Nested-markup anchor text is space-joined ("Read more", not "Readmore")."""
        from bs4 import BeautifulSoup

        html = "<html><body><a href='/p'><b>Read</b> more</a></body></html>"
        soup = BeautifulSoup(html, "lxml")
        links = scraper._extract_links(soup, "https://example.com")
        assert links[0]["text"] == "Read more"

    def test_extract_images(self, scraper):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(SAMPLE_HTML, "lxml")
        images = scraper._extract_images(soup, "http://example.com")
        assert len(images) == 2
        assert images[0]["alt"] == "Logo"
        assert images[0]["title"] == "Site Logo"
        # Relative URL resolved
        assert images[0]["src"] == "http://example.com/img/logo.png"
        # Width/height are None for tags without these attributes
        assert images[0]["width"] is None
        assert images[0]["height"] is None
        assert images[0]["srcset"] is None
        assert images[1]["alt"] == "Photo"

    def test_extract_images_with_dimensions(self, scraper):
        from bs4 import BeautifulSoup

        html = '<html><body><img src="http://ex.com/i.png" width="640" height="480"></body></html>'
        soup = BeautifulSoup(html, "lxml")
        images = scraper._extract_images(soup, "http://ex.com")
        assert images[0]["width"] == 640
        assert images[0]["height"] == 480

    def test_extract_images_with_non_numeric_dimensions(self, scraper):
        from bs4 import BeautifulSoup

        html = (
            '<html><body><img src="http://ex.com/i.png" width="auto" height="100%"></body></html>'
        )
        soup = BeautifulSoup(html, "lxml")
        images = scraper._extract_images(soup, "http://ex.com")
        assert images[0]["width"] is None
        assert images[0]["height"] is None

    def test_extract_images_lazy_load(self, scraper):
        from bs4 import BeautifulSoup

        html = """<html><body>
            <img data-src="http://cdn.example.com/lazy.jpg" src="placeholder.gif" alt="Lazy">
            <img data-original="http://cdn.example.com/original.jpg" alt="Original">
            <img data-lazy-src="http://cdn.example.com/lazysrc.jpg" alt="LazySrc">
        </body></html>"""
        soup = BeautifulSoup(html, "lxml")
        images = scraper._extract_images(soup, "http://example.com")
        assert len(images) == 3
        assert images[0]["src"] == "http://cdn.example.com/lazy.jpg"
        assert images[1]["src"] == "http://cdn.example.com/original.jpg"
        assert images[2]["src"] == "http://cdn.example.com/lazysrc.jpg"

    def test_extract_images_srcset(self, scraper):
        from bs4 import BeautifulSoup

        html = '<html><body><img src="http://ex.com/i.png" srcset="http://ex.com/i-2x.png 2x"></body></html>'
        soup = BeautifulSoup(html, "lxml")
        images = scraper._extract_images(soup, "http://ex.com")
        assert images[0]["srcset"] == "http://ex.com/i-2x.png 2x"

    def test_extract_images_picture_element(self, scraper):
        from bs4 import BeautifulSoup

        html = """<html><body>
        <picture>
            <source srcset="http://ex.com/wide.webp" media="(min-width: 800px)">
            <source srcset="http://ex.com/narrow.webp">
            <img src="http://ex.com/fallback.jpg" alt="Responsive">
        </picture>
        </body></html>"""
        soup = BeautifulSoup(html, "lxml")
        images = scraper._extract_images(soup, "http://ex.com")
        # Should have 2: img from the <img> tag (seen first), and <picture> adds srcset
        # The img src is already seen from the <img> loop, so picture updates its srcset
        img_fallback = next(i for i in images if "fallback" in i["src"])
        assert img_fallback["srcset"] == "http://ex.com/narrow.webp"

    def test_extract_images_picture_new_src(self, scraper):
        from bs4 import BeautifulSoup

        html = """<html><body>
        <picture>
            <source srcset="http://ex.com/hero.webp">
            <img data-src="http://ex.com/hero.jpg" alt="Hero">
        </picture>
        </body></html>"""
        soup = BeautifulSoup(html, "lxml")
        images = scraper._extract_images(soup, "http://ex.com")
        # data-src from <picture><img> — the img loop picks up data-src first
        hero = next(i for i in images if "hero" in i["src"])
        assert hero["alt"] == "Hero"

    def test_extract_images_picture_standalone_with_source(self, scraper):
        """Picture element with unique img src adds srcset from <source>."""
        from bs4 import BeautifulSoup

        html = """<html><body>
            <img src="standalone.jpg" alt="Standalone">
            <picture>
                <source srcset="art.webp">
                <img src="art-fallback.jpg" alt="Art">
            </picture>
        </body></html>"""
        soup = BeautifulSoup(html, "lxml")
        images = scraper._extract_images(soup, "http://example.com")
        assert len(images) == 2
        assert images[0]["src"] == "http://example.com/standalone.jpg"
        art_img = next(i for i in images if i["alt"] == "Art")
        assert art_img["srcset"] == "art.webp"

    def test_extract_images_picture_no_img(self, scraper):
        """Picture element without <img> is skipped."""
        from bs4 import BeautifulSoup

        html = """<html><body>
            <picture>
                <source srcset="http://example.com/no-img.webp">
            </picture>
        </body></html>"""
        soup = BeautifulSoup(html, "lxml")
        images = scraper._extract_images(soup, "http://example.com")
        # No <img> inside <picture>, so nothing extracted
        assert len(images) == 0

    def test_extract_images_picture_unique_data_src(self, scraper):
        """Picture with data-src only img adds new image entry (covers lines 255-256)."""
        from bs4 import BeautifulSoup

        # Use data-original on the <img> inside <picture> — the img loop picks it up
        # via data-original, but the picture loop uses data-src or src. Since data-original
        # resolves to different from what picture loop resolves, the picture loop may
        # treat it as a new src if the img has no src or data-src.
        html = """<html><body>
            <picture>
                <source srcset="http://example.com/art.webp">
                <img data-src="http://example.com/picture-only.jpg" alt="PictureOnly">
            </picture>
        </body></html>"""
        soup = BeautifulSoup(html, "lxml")
        images = scraper._extract_images(soup, "http://example.com")
        # The <img> inside <picture> has data-src — picked up by both loops with same value
        # So we get exactly 1 image (deduped by seen_srcs)
        assert len(images) >= 1
        picture_img = next(i for i in images if i["alt"] == "PictureOnly")
        assert picture_img["src"] == "http://example.com/picture-only.jpg"


class TestResourceCleanup:
    @pytest.mark.asyncio
    async def test_close_is_safe_noop(self, scraper):
        # No persistent client to close (clients are per-request); close() must
        # never raise and is idempotent.
        await scraper.close()
        await scraper.close()
        assert not hasattr(scraper, "_client")


class TestClientCreation:
    @pytest.mark.asyncio
    async def test_build_client_with_user_agent(self, scraper):
        cfg = ScrapeConfig(user_agent="CustomBot/1.0")
        async with scraper._build_client(cfg, cfg.timeout) as client:
            assert client.headers.get("user-agent") == "CustomBot/1.0"

    @pytest.mark.asyncio
    async def test_build_client_with_custom_headers(self, scraper):
        cfg = ScrapeConfig(headers={"X-Custom": "value"})
        async with scraper._build_client(cfg, cfg.timeout) as client:
            assert client.headers.get("x-custom") == "value"

    @pytest.mark.asyncio
    async def test_build_client_is_per_request_not_cached(self, scraper):
        cfg = ScrapeConfig()
        async with scraper._build_client(cfg, cfg.timeout) as c1:
            pass
        async with scraper._build_client(cfg, cfg.timeout) as c2:
            pass
        # Fresh client each call — never the cached singleton that leaked the first
        # request's config into every later request.
        assert c1 is not c2
        assert not hasattr(scraper, "_client")

    @pytest.mark.asyncio
    async def test_per_request_config_uses_own_user_agent(self, scraper, mock_response):
        # Two scrapes on ONE instance with different per-request configs must each
        # use their OWN User-Agent — proves no stale cached client bleed.
        captured: list[str | None] = []

        def _build(cfg, timeout):
            client = AsyncMock(spec=httpx.AsyncClient)
            ua = cfg.user_agent

            def _stream(method, url, *a, **k):
                captured.append(ua)
                return _StreamCtx(mock_response)

            client.stream = _stream
            return _AsyncCtxClient(client)

        with patch.object(scraper, "_build_client", side_effect=_build):
            await scraper.scrape("https://example.com", config=ScrapeConfig(user_agent="Bot-A"))
            await scraper.scrape("https://example.com", config=ScrapeConfig(user_agent="Bot-B"))

        assert captured == ["Bot-A", "Bot-B"]

    @pytest.mark.asyncio
    async def test_build_client_attaches_ssrf_hook(self, scraper):
        cfg = ScrapeConfig()
        async with scraper._build_client(cfg, cfg.timeout) as client:
            assert client.event_hooks.get("request")  # per-hop SSRF guard wired


class TestHttpxSsrfRedirectGuard:
    """The event hook must fire on the REDIRECT hop, not just the initial URL."""

    @pytest.mark.asyncio
    async def test_redirect_to_metadata_is_blocked(self):
        from services.url_guard import UrlNotAllowedError, make_outbound_guard_hook

        def handler(request):
            if request.url.host == "93.184.216.34":
                return httpx.Response(302, headers={"location": "http://169.254.169.254/"})
            return httpx.Response(200, text="should-not-reach")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
            event_hooks={"request": [make_outbound_guard_hook()]},
        ) as client:
            with pytest.raises(UrlNotAllowedError):
                await client.get("https://93.184.216.34/")

    @pytest.mark.asyncio
    async def test_public_to_public_redirect_still_succeeds(self):
        from services.url_guard import make_outbound_guard_hook

        def handler(request):
            if request.url.path == "/a":
                return httpx.Response(302, headers={"location": "https://93.184.216.34/b"})
            return httpx.Response(200, text="ok")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
            event_hooks={"request": [make_outbound_guard_hook()]},
        ) as client:
            resp = await client.get("https://93.184.216.34/a")
            assert resp.status_code == 200
            assert resp.text == "ok"


class TestHttpxScraperLiveServer:
    """Tests for HttpxScraper against the local test HTTP server."""

    @pytest.mark.asyncio
    async def test_scrape_live_page(self, base_url):
        """Scrape a real HTML page from the local server and verify full extraction."""
        s = HttpxScraper(ScrapeConfig(timeout=10))
        try:
            result = await s.scrape(f"{base_url}/simple")
            assert result.status == ScrapeStatus.COMPLETED
            assert result.scraper_used == "HttpxScraper"
            assert result.title == "Test Page"
            assert "Hello World" in result.text_content
            assert "<h1>Hello World</h1>" in result.html_content
            assert result.metadata["status_code"] == 200
            assert result.metadata["description"] == "A simple test page"
            assert result.metadata["og:title"] == "OG Test Page"
            assert "canonical_url" in result.metadata
            assert result.processing_time_seconds > 0
            assert result.error is None
            # Links
            assert len(result.links) == 2
            internal = [lnk for lnk in result.links if lnk["type"] == "internal"]
            external = [lnk for lnk in result.links if lnk["type"] == "external"]
            assert len(internal) == 1
            assert internal[0]["text"] == "About"
            assert len(external) == 1
            assert "external.example.com" in external[0]["url"]
            # Images
            assert len(result.images) == 2
            logo = next(i for i in result.images if "logo" in i["src"])
            assert logo["alt"] == "Logo"
            assert logo["title"] == "Site Logo"
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_scrape_live_empty_page(self, base_url):
        """Scrape an empty page and verify graceful handling."""
        s = HttpxScraper(ScrapeConfig(timeout=10))
        try:
            result = await s.scrape(f"{base_url}/empty")
            assert result.status == ScrapeStatus.COMPLETED
            assert result.title == "Empty"
            assert result.links == []
            assert result.images == []
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_scrape_live_server_error(self, base_url):
        """Scrape a 500 page and verify error handling."""
        s = HttpxScraper(ScrapeConfig(timeout=10))
        try:
            result = await s.scrape(f"{base_url}/error500")
            assert result.status == ScrapeStatus.FAILED
            assert "500" in result.error
            assert result.scraper_used == "HttpxScraper"
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_scrape_live_404(self, base_url):
        """Scrape a nonexistent path and verify 404 error handling."""
        s = HttpxScraper(ScrapeConfig(timeout=10))
        try:
            result = await s.scrape(f"{base_url}/no-such-page")
            assert result.status == ScrapeStatus.FAILED
            assert "404" in result.error
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_scrape_live_redirect(self, base_url):
        """Scrape a redirect URL and verify httpx follows it."""
        s = HttpxScraper(ScrapeConfig(timeout=10))
        try:
            result = await s.scrape(f"{base_url}/redirect")
            assert result.status == ScrapeStatus.COMPLETED
            assert result.title == "Test Page"
            assert result.metadata["final_url"].endswith("/simple")
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_scrape_live_with_cookies_and_headers(self, base_url):
        """Scrape with cookies and custom headers -- no errors."""
        config = ScrapeConfig(
            timeout=10,
            cookies={"session": "test123"},
            headers={"X-Custom": "header-value"},
            user_agent="CustomBot/1.0",
        )
        s = HttpxScraper(config)
        try:
            result = await s.scrape(f"{base_url}/simple")
            assert result.status == ScrapeStatus.COMPLETED
            assert result.title == "Test Page"
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_scrape_live_link_page(self, base_url):
        """Scrape a page with multiple links and verify correct count and classification."""
        s = HttpxScraper(ScrapeConfig(timeout=10))
        try:
            result = await s.scrape(f"{base_url}/links")
            assert result.status == ScrapeStatus.COMPLETED
            assert result.title == "Link Page"
            assert len(result.links) == 4
            internal = [lnk for lnk in result.links if lnk["type"] == "internal"]
            external = [lnk for lnk in result.links if lnk["type"] == "external"]
            assert len(internal) == 3
            assert len(external) == 1
            link_texts = {lnk["text"] for lnk in result.links}
            assert "Page 1" in link_texts
            assert "Page 2" in link_texts
            assert "Page 3" in link_texts
            assert "Other" in link_texts
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_scrape_live_timeout(self, base_url):
        """Scrape a slow page with short timeout returns Timeout error."""
        s = HttpxScraper(ScrapeConfig(timeout=1))
        try:
            result = await s.scrape(f"{base_url}/slow")
            assert result.status == ScrapeStatus.FAILED
            assert "Timeout" in result.error
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_scrape_live_with_download_images(self, base_url):
        """Scrape with download_images=True enriches local images with content."""
        config = ScrapeConfig(timeout=10, download_images=True)
        s = HttpxScraper(config)
        try:
            result = await s.scrape(f"{base_url}/simple")
            assert result.status == ScrapeStatus.COMPLETED
            # The local /images/logo.png should have been downloaded
            local_img = next((i for i in result.images if "logo.png" in i["src"]), None)
            assert local_img is not None
            assert "content_base64" in local_img
            assert len(local_img["content_base64"]) > 0
            assert local_img["mime_type"] == "image/png"
            assert local_img["byte_size"] > 0
            # PIL should have filled in dimensions (1x1 red pixel)
            assert local_img["width"] == 1
            assert local_img["height"] == 1
        finally:
            await s.close()


class TestHttpxScraperRetries:
    """Transient-error retry behavior (mirrors Scrapy's RETRY_HTTP_CODES)."""

    @staticmethod
    def _resp(status: int, *, retry_after: str | None = None):
        return _stream_response(status=status, retry_after=retry_after)

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, scraper):
        mock_client = _stream_client(self._resp(503), self._resp(200))

        with (
            patch.object(scraper, "_build_client", return_value=_AsyncCtxClient(mock_client)),
            patch("services.httpx_scraper.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.COMPLETED
        assert mock_client.stream.call_count == 2
        sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_exhausted_returns_failed(self):
        scraper = HttpxScraper(ScrapeConfig(timeout=10, max_retries=2))
        # initial attempt + 2 retries = 3 streamed 503s.
        mock_client = _stream_client(self._resp(503), self._resp(503), self._resp(503))

        with (
            patch.object(scraper, "_build_client", return_value=_AsyncCtxClient(mock_client)),
            patch("services.httpx_scraper.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.FAILED
        assert "503" in result.error
        # initial attempt + max_retries
        assert mock_client.stream.call_count == 3

    @pytest.mark.asyncio
    async def test_4xx_not_retried(self, scraper):
        mock_client = _stream_client(self._resp(404))

        with (
            patch.object(scraper, "_build_client", return_value=_AsyncCtxClient(mock_client)),
            patch("services.httpx_scraper.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            result = await scraper.scrape("https://example.com/missing")

        assert result.status == ScrapeStatus.FAILED
        assert "404" in result.error
        assert mock_client.stream.call_count == 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_transport_error_retried(self, scraper):
        mock_client = _stream_client(httpx.ConnectError("boom"), self._resp(200))

        with (
            patch.object(scraper, "_build_client", return_value=_AsyncCtxClient(mock_client)),
            patch("services.httpx_scraper.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.COMPLETED
        assert mock_client.stream.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_after_header_honored(self, scraper):
        mock_client = _stream_client(self._resp(429, retry_after="2"), self._resp(200))

        with (
            patch.object(scraper, "_build_client", return_value=_AsyncCtxClient(mock_client)),
            patch("services.httpx_scraper.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.COMPLETED
        sleep.assert_awaited_once_with(2.0)


class TestOutboundProxy:
    """settings.outbound_proxy is threaded into the httpx client construction."""

    @staticmethod
    def _spy_client(monkeypatch, captured):
        real_client = httpx.AsyncClient

        def _spy(*args, **kwargs):
            captured["proxy"] = kwargs.get("proxy")
            return real_client(*args, **kwargs)

        monkeypatch.setattr("services.httpx_scraper.httpx.AsyncClient", _spy)

    @pytest.mark.asyncio
    async def test_build_client_passes_proxy(self, scraper, monkeypatch):
        from config import get_settings

        monkeypatch.setattr(
            get_settings(), "outbound_proxy", "http://proxy.test:3128", raising=False
        )
        captured: dict = {}
        self._spy_client(monkeypatch, captured)
        async with scraper._build_client(ScrapeConfig(), timeout=10.0):
            pass
        assert captured["proxy"] == "http://proxy.test:3128"

    @pytest.mark.asyncio
    async def test_build_client_no_proxy_by_default(self, scraper, monkeypatch):
        from config import get_settings

        monkeypatch.setattr(get_settings(), "outbound_proxy", None, raising=False)
        captured: dict = {}
        self._spy_client(monkeypatch, captured)
        async with scraper._build_client(ScrapeConfig(), timeout=10.0):
            pass
        assert captured["proxy"] is None
