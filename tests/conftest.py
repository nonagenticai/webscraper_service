"""Pytest configuration and shared fixtures.

Provides:
- A local HTTP server that serves realistic HTML pages, images, and error responses.
- An httpx AsyncClient wired to the real FastAPI app via ASGITransport (no network).
- The auth middleware is configured in bypass mode so scrape tests focus on scraping.
"""

import os

# The test HTTP server binds loopback (127.0.0.1), which the SSRF guard blocks
# by default. Allowlist it for the test environment BEFORE any module triggers
# Settings construction (get_settings is cached on first call). Real SSRF-block
# behavior is covered by test_url_guard + the guard-enabled router tests.
os.environ.setdefault("SSRF_ALLOWED_HOSTS", '["127.0.0.1"]')

# Several router tests assert on the DETAIL of a 422 (e.g. "not a valid CSS
# selector", "JavaScript-capable backend") because that message is the API's
# contract to a caller. main.py decides whether to expose that detail ONCE, at
# import time:
#
#   LEAK_EXCEPTION_DETAILS = os.getenv("LEAK_EXCEPTION_DETAILS", "") == "true"
#                            or settings.environment in {"development", "dev"}
#
# Those tests used to pass only because ENVIRONMENT was UNSET, so Settings fell
# back to its "development" default and the detail branch happened to be on.
# That was ambient luck, not a decision: gitops #2432 began exporting
# ENVIRONMENT=test to every python-service consumer, the environment arm went
# false, and four tests here went red on main with no code change in this repo.
#
# Set the PURPOSE-BUILT switch instead of leaning on the environment name, so
# the assertions no longer depend on which ENVIRONMENT the CI archetype happens
# to export. setdefault (not assignment) keeps a caller's explicit override —
# e.g. a run that deliberately wants the production-shaped generic envelope.
# This is a test-process default only; nothing that ships is loosened.
os.environ.setdefault("LEAK_EXCEPTION_DETAILS", "true")

import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

# ── 1x1 red PNG pixel (valid image) ─────────────────────────────────────────
RED_PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

# ── HTML fixtures ────────────────────────────────────────────────────────────

SIMPLE_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
    <meta name="description" content="A simple test page">
    <meta property="og:title" content="OG Test Page">
    <link rel="canonical" href="http://localhost:{port}/simple">
</head>
<body>
    <h1>Hello World</h1>
    <p>This is a test paragraph.</p>
    <a href="/about">About</a>
    <a href="https://external.example.com/page">External Link</a>
    <img src="/images/logo.png" alt="Logo" title="Site Logo">
    <img src="https://cdn.example.com/photo.jpg" alt="Photo">
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


# ── Local HTTP server ────────────────────────────────────────────────────────


class _TestHandler(BaseHTTPRequestHandler):
    """Serve test HTML pages and images."""

    port: int = 0

    def do_GET(self):  # noqa: N802
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
        elif self.path == "/about":
            self._respond(
                200,
                b"<html><head><title>About</title></head><body>About page</body></html>",
                "text/html",
            )
        else:
            self._respond(404, b"Not Found", "text/plain")

    def do_HEAD(self):  # noqa: N802
        if self.path in ("/simple", "/links", "/empty", "/images/logo.png", "/redirect", "/about"):
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

    Resets the global scraper manager between tests and bypasses auth.
    """
    import services.scraper_manager as sm

    old = sm._scraper_manager
    sm._scraper_manager = None

    # The result cache is a process-wide singleton; clear it between tests so a
    # cached response from one test can't satisfy another's request (X-Cache HIT).
    from services.cache import get_result_cache

    get_result_cache().clear()

    # Temporarily bypass auth for scraping tests — only on the auth middleware.
    # Capture the previous value and RESTORE it on teardown: bypass_auth is read
    # from the AuthMiddleware instance built from these kwargs, so leaving it True
    # leaks auth-bypass into every later test that shares this app (e.g. the
    # auth-enforcement integration tests), making the suite order-dependent.
    from auth.middleware import AuthMiddleware
    from main import app

    _auth_mws = [m for m in app.user_middleware if hasattr(m, "cls") and m.cls is AuthMiddleware]
    _prev_bypass = [m.kwargs.get("bypass_auth", False) for m in _auth_mws]
    for m in _auth_mws:
        m.kwargs["bypass_auth"] = True
    app.middleware_stack = None  # force a rebuild so the new kwarg takes effect

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client

    for m, prev in zip(_auth_mws, _prev_bypass):
        m.kwargs["bypass_auth"] = prev
    app.middleware_stack = None  # rebuild with auth restored for subsequent tests

    if sm._scraper_manager:
        await sm._scraper_manager.close()
    sm._scraper_manager = old


@pytest.fixture
def license_auth_headers():
    """Auth headers with license token."""
    return {"X-Tenant-Id": "Bearer test-license-token"}
