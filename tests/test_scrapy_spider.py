"""Tests for WebSpider, _ResultCollector, and _run_crawl internals."""

import asyncio
from unittest.mock import MagicMock, patch

import scrapy
from scrapy.http import HtmlResponse, Request

from services.base_scraper import ScrapeConfig
from services.scrapy_scraper import WebSpider, _ResultCollector, _run_crawl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start(spider: WebSpider) -> list:
    """Drain the spider's async ``start()`` — the hook Scrapy >= 2.13 actually calls."""

    async def _collect() -> list:
        return [r async for r in spider.start()]

    return asyncio.run(_collect())


def _fake_response(url: str, body: str, status: int = 200) -> HtmlResponse:
    """Create a fake Scrapy HtmlResponse for testing spider parse logic."""
    request = Request(url=url)
    return HtmlResponse(
        url=url,
        body=body.encode("utf-8"),
        encoding="utf-8",
        request=request,
        status=status,
    )


SIMPLE_HTML = """
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
    <img src="/img/logo.png" alt="Logo" title="Site Logo" width="200" height="100">
    <img src="https://cdn.example.com/photo.jpg" alt="Photo">
</body>
</html>
"""

LAZY_LOAD_HTML = """
<html><body>
    <img data-src="https://cdn.example.com/lazy.jpg" src="placeholder.gif" alt="Lazy">
    <img data-original="https://cdn.example.com/original.jpg" alt="Original">
    <img data-lazy-src="https://cdn.example.com/lazysrc.jpg" alt="LazySrc">
</body></html>
"""

PICTURE_HTML = """
<html><body>
    <img src="https://example.com/standalone.jpg" alt="Standalone">
    <picture>
        <source srcset="https://example.com/wide.webp" media="(min-width: 800px)">
        <source srcset="https://example.com/narrow.webp">
        <img src="https://example.com/fallback.jpg" alt="Responsive">
    </picture>
</body></html>
"""

PICTURE_NEW_SRC_HTML = """
<html><body>
    <picture>
        <source srcset="https://example.com/hero.webp">
        <img src="https://example.com/unique-picture.jpg" alt="Hero">
    </picture>
</body></html>
"""

PICTURE_NO_IMG_HTML = """
<html><body>
    <picture>
        <source srcset="https://example.com/no-img.webp">
    </picture>
</body></html>
"""

PICTURE_RELATIVE_SRC_HTML = """
<html><body>
    <picture>
        <source srcset="hero.webp">
        <img data-src="/images/hero-relative.jpg" alt="RelativeHero">
    </picture>
</body></html>
"""

IMG_RELATIVE_LAZY_HTML = """
<html><body>
    <img data-src="/images/lazy-relative.jpg" alt="LazyRelative">
    <img data-original="/images/original-relative.jpg" alt="OriginalRelative">
    <img data-lazy-src="/images/lazysrc-relative.jpg" alt="LazySrcRelative">
</body></html>
"""


# ---------------------------------------------------------------------------
# _ResultCollector tests
# ---------------------------------------------------------------------------


class TestResultCollector:
    def test_init(self):
        collector = _ResultCollector()
        assert collector.items == []
        assert collector.errors == []

    def test_item_scraped(self):
        collector = _ResultCollector()
        item = {"url": "https://example.com", "title": "Test"}
        response = MagicMock()
        spider = MagicMock()
        collector.item_scraped(item, response, spider)
        assert len(collector.items) == 1
        assert collector.items[0] == {"url": "https://example.com", "title": "Test"}

    def test_item_scraped_makes_copy(self):
        """Items are stored as copies (dict(item))."""
        collector = _ResultCollector()
        item = {"url": "https://example.com"}
        collector.item_scraped(item, MagicMock(), MagicMock())
        # Mutating the original should not affect stored item
        item["url"] = "changed"
        assert collector.items[0]["url"] == "https://example.com"

    def test_spider_error(self):
        collector = _ResultCollector()
        failure = Exception("Something went wrong")
        collector.spider_error(failure, MagicMock(), MagicMock())
        assert len(collector.errors) == 1
        assert "Something went wrong" in collector.errors[0]

    def test_multiple_items_and_errors(self):
        collector = _ResultCollector()
        collector.item_scraped({"url": "a"}, MagicMock(), MagicMock())
        collector.item_scraped({"url": "b"}, MagicMock(), MagicMock())
        collector.spider_error(Exception("err1"), MagicMock(), MagicMock())
        collector.spider_error(Exception("err2"), MagicMock(), MagicMock())
        assert len(collector.items) == 2
        assert len(collector.errors) == 2


# ---------------------------------------------------------------------------
# WebSpider tests
# ---------------------------------------------------------------------------


class TestWebSpiderInit:
    def test_init_sets_start_urls(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        assert spider.start_urls == ["https://example.com"]
        assert spider._base_domain == "example.com"
        assert spider._visited == set()
        assert spider._current_depth == {"https://example.com": 0}

    def test_init_with_port_in_url(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="http://localhost:8080/page", config=config)
        assert spider._base_domain == "localhost:8080"


class TestWebSpiderStart:
    def test_start_default(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        requests = _start(spider)
        assert len(requests) == 1
        assert requests[0].url == "https://example.com"
        assert requests[0].callback == spider.parse_page
        assert requests[0].errback == spider.errback

    def test_start_with_user_agent(self):
        config = ScrapeConfig(user_agent="CustomBot/1.0")
        spider = WebSpider(start_url="https://example.com", config=config)
        requests = _start(spider)
        assert requests[0].headers.get(b"User-Agent") == b"CustomBot/1.0"

    def test_start_with_headers(self):
        config = ScrapeConfig(headers={"X-Custom": "value"})
        spider = WebSpider(start_url="https://example.com", config=config)
        requests = _start(spider)
        assert requests[0].headers.get(b"X-Custom") == b"value"

    def test_start_with_cookies(self):
        config = ScrapeConfig(cookies={"session": "abc123"})
        spider = WebSpider(start_url="https://example.com", config=config)
        requests = _start(spider)
        assert requests[0].cookies == {"session": "abc123"}

    def test_start_empty_cookies(self):
        config = ScrapeConfig(cookies={})
        spider = WebSpider(start_url="https://example.com", config=config)
        requests = _start(spider)
        # Empty cookies should pass None
        assert len(requests) == 1


class TestWebSpiderParse:
    def test_parse_basic(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", SIMPLE_HTML)

        items = list(spider.parse_page(response))
        assert len(items) == 1
        item = items[0]
        assert item["url"] == "https://example.com"
        assert item["title"] == "Test Page"
        assert "Hello World" in item["text_content"]
        assert "Some content here" in item["text_content"]
        # Script/style text should NOT be in text_content
        assert "var x = 1" not in item["text_content"]
        assert item["depth"] == 0

    def test_parse_strips_chrome_and_keeps_table_rows(self):
        """Scrapy must emit the same clean main-content as httpx/Playwright.

        Regression: the Scrapy spider used to return raw ``body ::text`` (only
        script/style removed), so directory crawls leaked nav/cookie/footer
        chrome into ``text_content`` — the exact bug #32 fixed for the other two
        scrapers but missed on Scrapy. ``extract_main_text`` must strip the
        chrome while linearizing the company table the directory is made of.
        """
        chrome_html = """
        <!DOCTYPE html><html><head><title>Vendor Directory</title></head>
        <body>
          <header><a href="/">DirectoryLogo</a></header>
          <nav>Home Browse Pricing About Contact Login Signup</nav>
          <div class="cookie-bar">We use cookies. Accept all cookies to continue.</div>
          <main>
            <h1>Payment vendors</h1>
            <table>
              <tr><th>Company</th><th>City</th></tr>
              <tr><td>Acme Payments</td><td>Berlin</td></tr>
              <tr><td>Globex Settlement</td><td>Dublin</td></tr>
            </table>
          </main>
          <footer>Copyright 2026 DirectoryLogo Inc. Privacy Terms Sitemap</footer>
        </body></html>
        """
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://dir.example.com", config=config)
        response = _fake_response("https://dir.example.com", chrome_html)

        item = list(spider.parse_page(response))[0]
        text = item["text_content"]

        # Real content survives, including the directory rows linearized as cells.
        assert "Acme Payments | Berlin" in text
        assert "Globex Settlement | Dublin" in text
        # Structural site chrome is stripped (the old raw body::text leaked it).
        assert "Privacy Terms Sitemap" not in text  # <footer>
        assert "Login Signup" not in text  # <nav>
        assert "Accept all cookies" not in text  # <div class="cookie-bar">

    def test_parse_extracts_metadata(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", SIMPLE_HTML)

        items = list(spider.parse_page(response))
        meta = items[0]["metadata"]
        assert meta["status_code"] == 200
        assert meta["final_url"] == "https://example.com"
        assert meta["description"] == "A test page"
        assert meta["og:title"] == "OG Title"
        assert meta["canonical_url"] == "https://example.com/canonical"

    def test_parse_metadata_transport_values_not_clobbered(self):
        """A scraped <meta name="status_code"/"final_url"> must NOT overwrite the
        authoritative transport values (set LAST, matching httpx/playwright)."""
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        html = (
            "<html><head>"
            '<meta name="status_code" content="evil">'
            '<meta name="final_url" content="http://attacker.test/">'
            "</head><body>x</body></html>"
        )
        response = _fake_response("https://example.com", html)

        meta = list(spider.parse_page(response))[0]["metadata"]
        assert meta["status_code"] == 200  # int from the response, not "evil"
        assert meta["final_url"] == "https://example.com"

    def test_parse_extracts_links(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", SIMPLE_HTML)

        items = list(spider.parse_page(response))
        links = items[0]["links"]
        assert len(links) == 2

        about = next(link for link in links if "/about" in link["url"])
        assert about["text"] == "About"
        assert about["type"] == "internal"

        external = next(link for link in links if "external.com" in link["url"])
        assert external["type"] == "external"

    def test_parse_extracts_images(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", SIMPLE_HTML)

        items = list(spider.parse_page(response))
        images = items[0]["images"]
        assert len(images) == 2

        logo = next(i for i in images if "logo" in i["src"])
        assert logo["alt"] == "Logo"
        assert logo["title"] == "Site Logo"
        assert logo["width"] == 200
        assert logo["height"] == 100
        # Relative URL resolved
        assert logo["src"] == "https://example.com/img/logo.png"

        photo = next(i for i in images if "photo" in i["src"])
        assert photo["alt"] == "Photo"
        assert photo["width"] is None
        assert photo["height"] is None

    def test_parse_picture_image_keeps_dimensions(self):
        # Regression: the <picture> branch previously hardcoded width/height=None,
        # dropping dimensions the inner <img> carried (httpx keeps them).
        html = (
            "<html><body><picture>"
            '<source srcset="https://example.com/big.webp" type="image/webp">'
            '<img src="https://example.com/pic.jpg" alt="Pic" width="320" height="240">'
            "</picture></body></html>"
        )
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", html)

        img = list(spider.parse_page(response))[0]["images"][0]
        assert img["src"] == "https://example.com/pic.jpg"
        assert img["width"] == 320
        assert img["height"] == 240
        assert img["srcset"] == "https://example.com/big.webp"

    def test_parse_image_invalid_dimensions(self):
        html = '<html><body><img src="https://example.com/img.png" width="auto" height="100%"></body></html>'
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", html)

        items = list(spider.parse_page(response))
        img = items[0]["images"][0]
        assert img["width"] is None
        assert img["height"] is None

    def test_parse_lazy_load_images(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", LAZY_LOAD_HTML)

        items = list(spider.parse_page(response))
        images = items[0]["images"]
        assert len(images) == 3
        srcs = [i["src"] for i in images]
        assert "https://cdn.example.com/lazy.jpg" in srcs
        assert "https://cdn.example.com/original.jpg" in srcs
        assert "https://cdn.example.com/lazysrc.jpg" in srcs

    def test_parse_picture_element_updates_existing_srcset(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", PICTURE_HTML)

        items = list(spider.parse_page(response))
        images = items[0]["images"]

        # Standalone image
        standalone = next(i for i in images if "standalone" in i["src"])
        assert standalone["alt"] == "Standalone"

        # Fallback image with srcset updated from <picture><source>
        fallback = next(i for i in images if "fallback" in i["src"])
        assert fallback["srcset"] == "https://example.com/narrow.webp"

    def test_parse_picture_element_new_src(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", PICTURE_NEW_SRC_HTML)

        items = list(spider.parse_page(response))
        images = items[0]["images"]
        # The <picture><img> has a unique src not already seen
        hero = next(i for i in images if "unique-picture" in i["src"])
        assert hero["alt"] == "Hero"
        assert hero["srcset"] == "https://example.com/hero.webp"

    def test_parse_picture_element_no_img(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", PICTURE_NO_IMG_HTML)

        items = list(spider.parse_page(response))
        images = items[0]["images"]
        # No <img> inside <picture>, so no images extracted from it
        assert len(images) == 0

    def test_parse_skips_already_visited(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", SIMPLE_HTML)

        # First parse yields items
        items1 = list(spider.parse_page(response))
        assert len(items1) >= 1

        # Second parse of same URL yields nothing (already visited)
        items2 = list(spider.parse_page(response))
        assert len(items2) == 0

    def test_parse_includes_html_content(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", SIMPLE_HTML)

        items = list(spider.parse_page(response))
        assert items[0]["html_content"] == response.text

    def test_parse_follow_links(self):
        html = """
        <html><body>
            <a href="/page1">Page 1</a>
            <a href="https://external.com/nope">External</a>
        </body></html>
        """
        config = ScrapeConfig(follow_links=True, max_depth=2)
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", html)

        results = list(spider.parse_page(response))
        # Should yield the item + follow request for internal link
        items = [r for r in results if isinstance(r, dict)]
        requests = [r for r in results if isinstance(r, scrapy.Request)]

        assert len(items) == 1
        assert len(requests) == 1
        assert requests[0].url == "https://example.com/page1"

    def test_parse_no_follow_when_disabled(self):
        html = '<html><body><a href="/page1">Page 1</a></body></html>'
        config = ScrapeConfig(follow_links=False)
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", html)

        results = list(spider.parse_page(response))
        requests = [r for r in results if isinstance(r, scrapy.Request)]
        assert len(requests) == 0

    def test_parse_no_follow_at_max_depth(self):
        html = '<html><body><a href="/deeper">Deeper</a></body></html>'
        config = ScrapeConfig(follow_links=True, max_depth=1)
        spider = WebSpider(start_url="https://example.com", config=config)
        # Set current depth to max_depth
        spider._current_depth["https://example.com"] = 1
        response = _fake_response("https://example.com", html)

        results = list(spider.parse_page(response))
        requests = [r for r in results if isinstance(r, scrapy.Request)]
        assert len(requests) == 0

    def test_parse_no_follow_already_known_url(self):
        html = '<html><body><a href="/page1">Page 1</a></body></html>'
        config = ScrapeConfig(follow_links=True, max_depth=3)
        spider = WebSpider(start_url="https://example.com", config=config)
        # Pre-mark the link as already in depth tracking
        spider._current_depth["https://example.com/page1"] = 0
        response = _fake_response("https://example.com", html)

        results = list(spider.parse_page(response))
        requests = [r for r in results if isinstance(r, scrapy.Request)]
        assert len(requests) == 0

    def test_parse_no_follow_external_links(self):
        html = '<html><body><a href="https://other.com/page">Other</a></body></html>'
        config = ScrapeConfig(follow_links=True, max_depth=3)
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", html)

        results = list(spider.parse_page(response))
        requests = [r for r in results if isinstance(r, scrapy.Request)]
        assert len(requests) == 0

    def test_parse_image_srcset(self):
        html = '<html><body><img src="https://example.com/i.png" srcset="https://example.com/i-2x.png 2x"></body></html>'
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", html)

        items = list(spider.parse_page(response))
        assert items[0]["images"][0]["srcset"] == "https://example.com/i-2x.png 2x"

    def test_parse_meta_without_name_or_content_skipped(self):
        html = '<html><head><meta charset="utf-8"></head><body></body></html>'
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", html)

        items = list(spider.parse_page(response))
        meta = items[0]["metadata"]
        # Should only have status_code and final_url, no charset entry
        assert "charset" not in meta

    def test_parse_link_rel_attribute(self):
        html = '<html><body><a href="/page" rel="nofollow">Page</a></body></html>'
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", html)

        items = list(spider.parse_page(response))
        link = items[0]["links"][0]
        assert link["rel"] == "nofollow"

    def test_parse_link_rel_multi_token_preserved(self):
        """Multi-value rel keeps ALL tokens (consistent with httpx/playwright)."""
        html = '<html><body><a href="/p" rel="nofollow noopener">P</a></body></html>'
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", html)

        items = list(spider.parse_page(response))
        assert items[0]["links"][0]["rel"] == "nofollow noopener"

    def test_parse_link_absent_rel_is_none(self):
        """Absent rel → None (not ""), consistent with httpx/playwright."""
        html = '<html><body><a href="/page">Page</a></body></html>'
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", html)

        items = list(spider.parse_page(response))
        assert items[0]["links"][0]["rel"] is None

    def test_parse_link_text_nested_markup(self):
        """Nested-markup anchor keeps full text (like get_text/innerText)."""
        html = "<html><body><a href='/p'><b>Read</b> more</a></body></html>"
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", html)

        items = list(spider.parse_page(response))
        assert items[0]["links"][0]["text"] == "Read more"

    def test_parse_absent_title_is_none(self):
        """A page with no <title> yields title=None (not "")."""
        html = "<html><body><p>No title here</p></body></html>"
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", html)

        items = list(spider.parse_page(response))
        assert items[0]["title"] is None

    def test_parse_picture_relative_src(self):
        """Picture element with relative data-src resolves URL via urljoin (covers line 171)."""
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", PICTURE_RELATIVE_SRC_HTML)

        items = list(spider.parse_page(response))
        images = items[0]["images"]
        # The <img data-src="/images/hero-relative.jpg"> inside <picture>
        # is picked up by the <img> loop first (via data-src), with relative
        # URL resolved. The picture loop then finds it already in seen_srcs.
        hero = next(i for i in images if "hero-relative" in i["src"])
        # Relative URL should be resolved to absolute
        assert hero["src"] == "https://example.com/images/hero-relative.jpg"

    def test_parse_img_relative_lazy_load_urls(self):
        """Relative lazy-load URLs (data-src, data-original, data-lazy-src) are resolved."""
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        response = _fake_response("https://example.com", IMG_RELATIVE_LAZY_HTML)

        items = list(spider.parse_page(response))
        images = items[0]["images"]
        assert len(images) == 3
        srcs = [i["src"] for i in images]
        assert "https://example.com/images/lazy-relative.jpg" in srcs
        assert "https://example.com/images/original-relative.jpg" in srcs
        assert "https://example.com/images/lazysrc-relative.jpg" in srcs


class TestWebSpiderFollowHeaders:
    """Followed requests carry the caller's request-local headers; credentials
    (Authorization / Cookie / Proxy-Authorization) never cross the start origin."""

    HEADERS = {
        "X-Api-Key": "k-1",
        "Accept-Language": "nl",
        "Authorization": "Bearer tok",
        "Cookie": "sid=abc",
        "Proxy-Authorization": "Basic cHg=",
    }

    def _follow(self, start_url: str, link: str) -> scrapy.Request:
        config = ScrapeConfig(
            follow_links=True, max_depth=2, user_agent="Bot/2.0", headers=dict(self.HEADERS)
        )
        spider = WebSpider(start_url=start_url, config=config)
        response = _fake_response(start_url, f'<html><body><a href="{link}">L</a></body></html>')
        requests = [r for r in spider.parse_page(response) if isinstance(r, scrapy.Request)]
        assert len(requests) == 1
        return requests[0]

    def test_same_origin_follow_carries_all_caller_headers(self):
        req = self._follow("https://example.com/", "/page1")
        assert req.headers.get(b"X-Api-Key") == b"k-1"
        assert req.headers.get(b"Accept-Language") == b"nl"
        assert req.headers.get(b"User-Agent") == b"Bot/2.0"
        assert req.headers.get(b"Authorization") == b"Bearer tok"
        assert req.headers.get(b"Cookie") == b"sid=abc"
        assert req.headers.get(b"Proxy-Authorization") == b"Basic cHg="

    def test_default_port_spelling_is_same_origin(self):
        config = ScrapeConfig(headers={"Authorization": "Bearer tok"})
        spider = WebSpider(start_url="https://example.com/", config=config)
        assert spider._follow_headers("https://example.com:443/p") == {
            "Authorization": "Bearer tok"
        }
        # A non-default port is a different origin.
        assert spider._follow_headers("https://example.com:8443/p") is None

    def test_cross_origin_follow_drops_credentials_keeps_other_headers(self):
        # http -> https on the same netloc is "internal" (followed) but a different
        # origin, so the credential-bearing headers must not ride along.
        req = self._follow("http://example.com/", "https://example.com/page1")
        assert req.url == "https://example.com/page1"
        assert req.headers.get(b"X-Api-Key") == b"k-1"
        assert req.headers.get(b"Accept-Language") == b"nl"
        assert req.headers.get(b"User-Agent") == b"Bot/2.0"
        assert b"Authorization" not in req.headers
        assert b"Cookie" not in req.headers
        assert b"Proxy-Authorization" not in req.headers

    def test_credential_header_match_is_case_insensitive(self):
        config = ScrapeConfig(
            follow_links=True, max_depth=2, headers={"authorization": "x", "X-K": "v"}
        )
        spider = WebSpider(start_url="http://example.com/", config=config)
        assert spider._follow_headers("https://example.com/p") == {"X-K": "v"}
        assert spider._follow_headers("http://example.com/p") == {
            "authorization": "x",
            "X-K": "v",
        }

    def test_no_caller_headers_follow_has_none(self):
        spider = WebSpider(start_url="https://example.com", config=ScrapeConfig())
        assert spider._follow_headers("https://example.com/p") is None


class TestWebSpiderErrback:
    def test_errback_logs_error(self):
        config = ScrapeConfig()
        spider = WebSpider(start_url="https://example.com", config=config)
        failure = Exception("Request failed")
        # Should not raise
        spider.errback(failure)


# ---------------------------------------------------------------------------
# _run_crawl tests
# ---------------------------------------------------------------------------


class TestRunCrawl:
    def test_run_crawl_with_mocked_process(self):
        """Test _run_crawl sets correct Scrapy settings and calls crawl."""
        config = ScrapeConfig(
            timeout=15,
            max_retries=2,
            delay=0.5,
            follow_links=True,
            max_depth=3,
            respect_robots_txt=False,
            user_agent="TestBot/1.0",
        )

        mock_process = MagicMock()
        mock_process.crawl = MagicMock()
        mock_process.start = MagicMock()

        with patch("services.scrapy_scraper.CrawlerProcess", return_value=mock_process) as mock_cls:
            with patch("services.scrapy_scraper.dispatcher") as mock_dispatcher:
                items, errors = _run_crawl("https://example.com", config)

        # Verify CrawlerProcess was created with settings
        call_args = mock_cls.call_args
        settings = call_args[0][0]
        assert settings.get("DOWNLOAD_TIMEOUT") == 15
        assert settings.get("RETRY_TIMES") == 2
        assert settings.get("DOWNLOAD_DELAY") == 0.5
        assert settings.get("ROBOTSTXT_OBEY") is False
        assert settings.get("DEPTH_LIMIT") == 3
        assert settings.get("USER_AGENT") == "TestBot/1.0"
        assert settings.get("TELNETCONSOLE_ENABLED") is False

        # Verify crawl and start were called
        mock_process.crawl.assert_called_once()
        mock_process.start.assert_called_once_with(
            stop_after_crawl=True, install_signal_handlers=False
        )

        # Verify signal connections
        assert mock_dispatcher.connect.call_count == 2

    def test_run_crawl_no_follow_links(self):
        """When follow_links=False, DEPTH_LIMIT should be 0."""
        config = ScrapeConfig(follow_links=False, max_depth=5)

        mock_process = MagicMock()
        with patch("services.scrapy_scraper.CrawlerProcess", return_value=mock_process) as mock_cls:
            with patch("services.scrapy_scraper.dispatcher"):
                _run_crawl("https://example.com", config)

        settings = mock_cls.call_args[0][0]
        assert settings.get("DEPTH_LIMIT") == 0

    def test_run_crawl_no_user_agent(self):
        """When no user_agent set, USER_AGENT keeps Scrapy default."""
        config = ScrapeConfig(user_agent=None)

        mock_process = MagicMock()
        with patch("services.scrapy_scraper.CrawlerProcess", return_value=mock_process) as mock_cls:
            with patch("services.scrapy_scraper.dispatcher"):
                _run_crawl("https://example.com", config)

        settings = mock_cls.call_args[0][0]
        # Scrapy has its own default USER_AGENT; we just didn't override it
        default_ua = settings.get("USER_AGENT")
        assert "TestBot" not in (default_ua or "")

    def test_run_crawl_returns_collector_data(self):
        """Verify _run_crawl returns items and errors from collector."""
        config = ScrapeConfig()

        mock_process = MagicMock()

        def fake_start(**kwargs):
            pass

        mock_process.start = fake_start

        with patch("services.scrapy_scraper.CrawlerProcess", return_value=mock_process):
            with patch("services.scrapy_scraper.dispatcher"):
                items, errors = _run_crawl("https://example.com", config)

        # With mocked process that doesn't actually crawl, items/errors are empty
        assert items == []
        assert errors == []


class TestWebSpiderBestFirstFrontier:
    """Followed requests are prioritized so a BOUNDED crawl spends its budget well.

    Scrapy dequeues higher `priority` first. Without this the frontier is blind
    BFS, so a depth-limited crawl of a large site burns its cap on whatever
    happened to appear first in the DOM (usually nav and footer chrome).
    """

    CRAWL_HTML = """
    <!DOCTYPE html><html><head><title>Acme</title></head><body>
      <nav><a href="/legal/terms">Terms</a><a href="/x">click here</a></nav>
      <main>
        <a href="/pricing">Pricing</a>
        <a href="/media/deck.pdf">Deck</a>
        <a href="https://twitter.com/acme" rel="nofollow">Follow us</a>
      </main>
    </body></html>
    """

    def _followed(self, link_query=None):
        config = ScrapeConfig(follow_links=True, max_depth=2, link_query=link_query)
        spider = WebSpider(start_url="https://acme.com", config=config)
        response = _fake_response("https://acme.com", self.CRAWL_HTML)
        requests = [
            item for item in spider.parse_page(response) if isinstance(item, scrapy.Request)
        ]
        return {req.url: req.priority for req in requests}

    def test_external_nofollow_link_is_not_followed_at_all(self):
        # Pre-existing behaviour: only internal links are followed.
        assert "https://twitter.com/acme" not in self._followed()

    def test_query_relevant_page_gets_the_highest_priority(self):
        priorities = self._followed(link_query="pricing")
        assert priorities, "expected followed requests"
        assert max(priorities, key=priorities.get) == "https://acme.com/pricing"

    def test_generic_and_non_html_links_rank_below_a_real_page(self):
        priorities = self._followed()
        assert priorities["https://acme.com/pricing"] > priorities["https://acme.com/x"]
        assert (
            priorities["https://acme.com/pricing"] > priorities["https://acme.com/media/deck.pdf"]
        )

    def test_priorities_are_integers(self):
        # Scrapy requires an int priority; a float silently breaks the scheduler.
        for priority in self._followed(link_query="pricing").values():
            assert isinstance(priority, int)
