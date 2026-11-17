"""E2E test fixtures — real FastAPI app + local HTTP server for scraping targets."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

# ── Local HTTP server that serves test pages ─────────────────────────────────

SIMPLE_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="A simple test page">
    <meta property="og:title" content="OG Test Page">
    <link rel="canonical" href="http://localhost:{port}/simple">
</head>
<body>
    <h1>Hello E2E</h1>
    <p>This is a test paragraph with some content.</p>
    <a href="/about">About</a>
    <a href="https://external.example.com/page">External Link</a>
    <img src="/images/logo.png" alt="Logo" title="Site Logo">
    <img src="https://external.example.com/photo.jpg" alt="External Photo">
</body>
</html>"""

MULTI_LINK_HTML = """<!DOCTYPE html>
<html>
<head><title>Link Page</title></head>
<body>
    <h1>Links</h1>
    <a href="/page1">Page 1</a>
    <a href="/page2">Page 2</a>
    <a href="/page3">Page 3</a>
    <a href="https://other.example.com">Other</a>
</body>
</html>"""

EMPTY_HTML = """<!DOCTYPE html>
<html><head><title>Empty</title></head><body></body></html>"""

# 1x1 red PNG pixel (valid image)
RED_PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _TestHandler(BaseHTTPRequestHandler):
    """Serve test HTML pages and images."""

    port: int = 0  # set at class level before starting

    def do_GET(self):
        if self.path == "/simple":
            body = SIMPLE_HTML.replace("{port}", str(self.port)).encode()
            self._respond(200, body, "text/html")
        elif self.path == "/links":
            self._respond(200, MULTI_LINK_HTML.encode(), "text/html")
        elif self.path == "/empty":
            self._respond(200, EMPTY_HTML.encode(), "text/html")
        elif self.path == "/images/logo.png":
            self._respond(200, RED_PIXEL_PNG, "image/png")
        elif self.path == "/slow":
            import time

            time.sleep(10)
            self._respond(200, b"<html><body>slow</body></html>", "text/html")
        elif self.path == "/error500":
            self._respond(500, b"Internal Server Error", "text/plain")
        elif self.path == "/redirect":
            self.send_response(301)
            self.send_header("Location", "/simple")
            self.end_headers()
        else:
            self._respond(404, b"Not Found", "text/plain")

    def do_HEAD(self):
        """HEAD requests — same routing, no body."""
        if self.path in ("/simple", "/links", "/empty", "/images/logo.png", "/redirect"):
            self.send_response(200)
        elif self.path == "/error500":
            self.send_response(500)
        else:
            self.send_response(404)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

    def _respond(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logging."""


@pytest.fixture(scope="session")
def local_server():
    """Start a local HTTP server on a random port for the test session."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TestHandler)
    port = server.server_address[1]
    _TestHandler.port = port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def base_url(local_server):
    """Convenience alias returning the local server base URL."""
    return local_server


@pytest.fixture
async def async_client():
    """httpx AsyncClient pointed at the real FastAPI app (ASGI transport).

    Resets the global scraper manager between tests to avoid stale cached
    clients (e.g. an httpx client with a previous test's timeout value).
    """
    import services.scraper_manager as sm

    # Reset singleton so each test gets a fresh scraper manager
    old = sm._scraper_manager
    sm._scraper_manager = None

    # The result cache is a process-wide singleton (same reset as the root
    # conftest's async_client): without it a COMPLETED response cached by one
    # test is served as an X-Cache HIT to the next test's identical request.
    from services.cache import get_result_cache

    get_result_cache().clear()

    from main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client

    # Cleanup: close the scraper manager we created, restore previous
    if sm._scraper_manager:
        await sm._scraper_manager.close()
    sm._scraper_manager = old
