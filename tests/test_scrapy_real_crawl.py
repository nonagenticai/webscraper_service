"""Regression: a REAL Scrapy crawl through the service's crawl entry point.

Scrapy >= 2.13 drives a spider via ``async def start()`` and never calls the
legacy ``start_requests()``. Its default ``start()`` yields a bare
``Request(url)`` whose missing callback falls back to ``self.parse`` — which
``WebSpider`` does not define (its callback is ``parse_page``). Under 2.19 that
made every crawl return 0 items, and silently dropped the configured headers,
cookies, errback and proxy meta. Unit tests that call ``start_requests()``
directly could not see it, so this one runs the actual crawl subprocess
against a local HTTP server and checks what the server received.
"""

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from services.base_scraper import ScrapeConfig
from services.scrapy_scraper import _run_crawl_in_executor

PAGE = b"""<!DOCTYPE html><html><head><title>Crawl Target</title></head>
<body><main><h1>Hello crawler</h1><p>Real crawl body text.</p></main></body></html>"""


class _RecordingHandler(BaseHTTPRequestHandler):
    seen: list[dict[str, str]] = []

    def do_GET(self):  # noqa: N802
        type(self).seen.append({k.lower(): v for k, v in self.headers.items()})
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def recording_server():
    _RecordingHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", _RecordingHandler.seen
    finally:
        server.shutdown()
        server.server_close()


def test_real_crawl_yields_items_and_sends_configured_headers_and_cookies(recording_server):
    base, seen = recording_server
    config = ScrapeConfig(
        user_agent="RegressionBot/1.0",
        headers={"X-Regression": "start-contract"},
        cookies={"sid": "abc123"},
        respect_robots_txt=False,
        follow_links=False,
        timeout=10,
        max_retries=0,
    )

    items, errors = asyncio.run(_run_crawl_in_executor(f"{base}/page", config))

    assert len(items) >= 1, f"crawl produced no items; errors={errors}"
    assert items[0]["title"] == "Crawl Target"
    assert not any("NotImplementedError" in e for e in errors), errors

    page_requests = [h for h in seen if h]
    assert page_requests, "server received no request"
    first = page_requests[0]
    assert first.get("user-agent") == "RegressionBot/1.0"
    assert first.get("x-regression") == "start-contract"
    assert "sid=abc123" in first.get("cookie", "")


LINKING_PAGE = b"""<!DOCTYPE html><html><head><title>Start</title></head>
<body><main><p>Start page.</p><a href="/next">Next page</a></main></body></html>"""


class _LinkingHandler(BaseHTTPRequestHandler):
    seen: list[tuple[str, dict[str, str]]] = []

    def do_GET(self):  # noqa: N802
        type(self).seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
        body = LINKING_PAGE if self.path == "/start" else PAGE
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def linking_server():
    _LinkingHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LinkingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", _LinkingHandler.seen
    finally:
        server.shutdown()
        server.server_close()


def test_real_crawl_followed_same_origin_link_carries_caller_headers(linking_server):
    """Regression: a FOLLOWED link was a bare ``Request(url)`` — only the start
    request carried the caller's custom headers, so every crawled page after the
    first was fetched without them (an API key / Authorization header gated page
    came back as the anonymous version)."""
    base, seen = linking_server
    config = ScrapeConfig(
        user_agent="RegressionBot/1.0",
        headers={"X-Regression": "follow-contract", "Authorization": "Bearer tok-123"},
        cookies={"sid": "abc123"},
        respect_robots_txt=False,
        follow_links=True,
        max_depth=1,
        timeout=10,
        max_retries=0,
    )

    items, errors = asyncio.run(_run_crawl_in_executor(f"{base}/start", config))

    assert len(items) >= 2, f"follow crawl produced {len(items)} item(s); errors={errors}"
    followed = [h for path, h in seen if path == "/next"]
    assert followed, f"followed link was never requested; seen={[p for p, _ in seen]}"
    headers = followed[0]
    assert headers.get("x-regression") == "follow-contract"
    assert headers.get("authorization") == "Bearer tok-123"
    assert headers.get("user-agent") == "RegressionBot/1.0"
    assert "sid=abc123" in headers.get("cookie", "")
