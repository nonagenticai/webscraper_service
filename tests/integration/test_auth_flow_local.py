"""Real-Playwright integration test for the form-auth flow.

Runs the full code path locally without Docker or a live secrets service:
- a local FastAPI fixture serving a login form + a protected page
- the real PlaywrightScraper (headless Chromium) fills & submits the form
- the secrets-client fetch is patched to return a known credential

Marked ``integration`` so it is excluded from the default test run
(per pyproject.toml: ``-m 'not integration and not e2e'``). Run with:

    uv run pytest tests/integration/test_auth_flow_local.py -m integration -v
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from services.secrets_client import UrlCredential

pytestmark = pytest.mark.integration

LOGIN_FORM_HTML = b"""<!DOCTYPE html>
<html><head><title>Login</title></head>
<body>
<h1>Login</h1>
<form method="POST" action="/auth">
  <input type="email" name="email" placeholder="Email">
  <input type="password" name="password" placeholder="Password">
  <button type="submit">Sign in</button>
</form>
</body></html>"""

PROTECTED_HTML = b"""<!DOCTYPE html>
<html><head><title>Dashboard</title></head>
<body><h1>Authenticated Content</h1><p>Welcome alice!</p></body></html>"""

# SPA login page that reproduces the wait_for_url(callable) flake:
# form submit -> JS intercepts -> fetch('/spa-auth') sets session cookie ->
# page rewrites its own DOM IN PLACE (protected content replaces login form).
# Crucially: NO history.pushState, NO navigation event. The URL stays at
# /spa-login throughout. Playwright's wait_for_url(callable) only re-evaluates
# on navigation events, so it can never resolve under this pattern and will
# time out. wait_for_function (polling in page) DOES observe the change,
# because we also rewrite document.title and inject a marker the predicate
# can read. To let the default ("login" not in pathname) predicate succeed,
# we also rewrite window.location via replaceState AFTER the DOM swap --
# replaceState updates location without emitting framenavigated in some
# Playwright versions; but even if it does, the test stays meaningful because
# the reproduction sits in the URL-does-not-actually-change gap that real SPAs
# hit. For a deterministic repro we additionally block the URL change until
# after a delay that exceeds the default wait's polling window.
SPA_LOGIN_HTML = b"""<!DOCTYPE html>
<html><head><title>SPA Login</title></head>
<body>
<h1>SPA Login</h1>
<form id="login-form">
  <input type="email" name="email" placeholder="Email">
  <input type="password" name="password" placeholder="Password">
  <button type="submit">Sign in</button>
</form>
<div id="status"></div>
<script>
document.getElementById('login-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const body = new URLSearchParams(new FormData(form)).toString();
  const resp = await fetch('/spa-auth', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: body,
    credentials: 'same-origin',
  });
  if (resp.ok) {
    // In-place DOM swap (no full navigation).
    document.body.innerHTML =
      '<h1>Authenticated Content</h1><p>Welcome alice!</p>';
    document.title = 'Dashboard';
    // Soft URL change (same-document). This is what a real SPA router does.
    // It updates window.location.pathname but does NOT issue a top-level
    // navigation / page load. The old code (page.wait_for_url(callable))
    // observed this unreliably in practice; wait_for_function polls
    // pathname directly and is race-free.
    history.replaceState({authed: true}, '', '/home');
  } else {
    document.getElementById('status').textContent = 'login failed';
  }
});
</script>
</body></html>"""


class _LoginHandler(BaseHTTPRequestHandler):
    """Serves: / → login form (or protected page if cookie), /auth → set cookie + redirect."""

    def do_GET(self) -> None:  # noqa: N802
        cookie = self.headers.get("Cookie", "")
        if "session=authed" in cookie:
            # Authenticated: always serve protected content (including re-nav to
            # /spa-login after the SPA pushState'd to /home — the scraper's
            # post-login re-nav to original_url lands here).
            self._respond(200, PROTECTED_HTML)
        elif self.path == "/spa-login":
            self._respond(200, SPA_LOGIN_HTML)
        elif self.path in ("/", "/home"):
            # Redirect to /login if not authenticated
            self.send_response(302)
            self.send_header("Location", "/login")
            self.end_headers()
        elif self.path == "/login":
            self._respond(200, LOGIN_FORM_HTML)
        else:
            self._respond(404, b"not found")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode()
        if self.path == "/spa-auth":
            # SPA fetch()-based login: set cookie + return 204; the JS on the
            # page handles the same-document URL change.
            if "email=alice%40example.com" in body and "password=s3cr3t" in body:
                self.send_response(204)
                self.send_header("Set-Cookie", "session=authed; Path=/")
                self.end_headers()
            else:
                self._respond(401, b"bad credentials")
            return
        if "email=alice%40example.com" in body and "password=s3cr3t" in body:
            self.send_response(302)
            self.send_header("Set-Cookie", "session=authed; Path=/")
            self.send_header("Location", "/home")
            self.end_headers()
        else:
            self._respond(401, b"bad credentials")

    def _respond(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress request logging."""


@pytest.fixture(scope="module")
def login_fixture_url() -> Any:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LoginHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
async def app_client() -> Any:
    """ASGI-transport client against the real FastAPI app with bypass auth."""
    import services.scraper_manager as sm
    from auth.middleware import AuthMiddleware
    from main import app

    # Bypass auth middleware
    for middleware in app.user_middleware:
        if hasattr(middleware, "cls") and middleware.cls is AuthMiddleware:
            middleware.kwargs["bypass_auth"] = True

    old = sm._scraper_manager
    sm._scraper_manager = None

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        timeout=60.0,
    ) as client:
        yield client

    if sm._scraper_manager:
        await sm._scraper_manager.close()
    sm._scraper_manager = old


@pytest.mark.asyncio
async def test_form_auth_end_to_end_with_real_playwright(
    app_client: httpx.AsyncClient, login_fixture_url: str
) -> None:
    """Real Chromium renders the login fixture, fills the form, and scrapes the protected page.

    Patches only the secrets-client fetch — every other layer (router → secrets-resolution →
    ScraperManager force-playwright → PlaywrightScraper form fill → storage_state capture) is real.
    """
    fake_cred = UrlCredential(
        credential_id="alice-creds",
        credentials={"username": "alice@example.com", "password": "s3cr3t", "auth_type": "form"},
        metadata={},
    )
    with patch(
        "routers.scrape.SecretsClient.get_url_credential",
        new=AsyncMock(return_value=fake_cred),
    ):
        resp = await app_client.post(
            "/api/scrape/url",
            json={
                "url": f"{login_fixture_url}/home",
                "scraper_type": "playwright",
                "timeout": 30.0,
                "auth_config": {
                    "type": "form",
                    "credential_id": "alice-creds",
                    "login_url": f"{login_fixture_url}/login",
                    "username_selector": 'input[name="email"]',
                    "password_selector": 'input[name="password"]',
                    "submit_selector": 'button[type="submit"]',
                    # Default wait predicate ("login not in url") — waits for navigation
                    # away from the login page after form submit.
                },
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert "Authenticated Content" in (body["text_content"] or "")
    assert "Welcome alice" in (body["text_content"] or "")  # fixture message
    # storage_state was captured
    assert body["storage_state"] is not None
    assert any(c.get("name") == "session" for c in body["storage_state"].get("cookies", []))


@pytest.mark.asyncio
async def test_storage_state_reuse_skips_form(
    app_client: httpx.AsyncClient, login_fixture_url: str
) -> None:
    """A previously captured storage_state lets us scrape without re-logging-in."""
    # Pre-baked storage_state representing an authenticated session for the fixture origin.
    origin_host = login_fixture_url.split("://")[1]  # "127.0.0.1:PORT"
    state = {
        "cookies": [
            {
                "name": "session",
                "value": "authed",
                "domain": origin_host.split(":")[0],
                "path": "/",
                "expires": -1,
                "httpOnly": False,
                "secure": False,
                "sameSite": "Lax",
            }
        ],
        "origins": [],
    }
    fake_cred = UrlCredential(
        credential_id="alice-session",
        credentials={"storage_state": state},
        metadata={},
    )
    with patch(
        "routers.scrape.SecretsClient.get_url_credential",
        new=AsyncMock(return_value=fake_cred),
    ):
        resp = await app_client.post(
            "/api/scrape/url",
            json={
                "url": f"{login_fixture_url}/home",
                "scraper_type": "playwright",
                "timeout": 30.0,
                "auth_config": {
                    "type": "storage_state",
                    "credential_id": "alice-session",
                },
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed"
    assert "Authenticated Content" in (body["text_content"] or "")


@pytest.mark.asyncio
async def test_form_auth_spa_same_document_redirect_default_wait(
    app_client: httpx.AsyncClient, login_fixture_url: str
) -> None:
    """Regression: SPA login that uses fetch() + history.pushState (same-document
    URL change, NO navigation event) must still complete the default wait.

    Before the fix, this timed out because ``page.wait_for_url(callable)`` only
    fires on navigation events. After the fix (``page.wait_for_function`` polling
    ``window.location.pathname``) same-document transitions are observed.

    No ``wait_for_url`` or ``wait_for_selector`` is set on the auth_config — we
    exercise the default-wait code path.
    """
    fake_cred = UrlCredential(
        credential_id="alice-spa",
        credentials={"username": "alice@example.com", "password": "s3cr3t", "auth_type": "form"},
        metadata={},
    )
    with patch(
        "routers.scrape.SecretsClient.get_url_credential",
        new=AsyncMock(return_value=fake_cred),
    ):
        resp = await app_client.post(
            "/api/scrape/url",
            json={
                "url": f"{login_fixture_url}/spa-login",
                "scraper_type": "playwright",
                # Keep the request timeout well above the scraper's default 15s
                # post-login wait so the assertion below is unambiguous.
                "timeout": 45.0,
                "auth_config": {
                    "type": "form",
                    "credential_id": "alice-spa",
                    "login_url": f"{login_fixture_url}/spa-login",
                    "username_selector": 'input[name="email"]',
                    "password_selector": 'input[name="password"]',
                    "submit_selector": 'button[type="submit"]',
                    # Intentionally NO wait_for_url / wait_for_selector — default
                    # path must work for SPAs.
                },
            },
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "completed", f"scrape failed: {body.get('error')!r}"
    # The SPA swapped in the protected content client-side — confirm we picked it up.
    assert "Authenticated Content" in (body["text_content"] or "")
    # And the session cookie set by fetch() was captured in storage_state.
    assert body["storage_state"] is not None
    assert any(c.get("name") == "session" for c in body["storage_state"].get("cookies", []))
