"""Tests for PlaywrightScraper auth flow (form / storage_state / bearer / basic)."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from services.base_scraper import AuthConfig, ScrapeConfig, ScrapeStatus
from services.playwright_scraper import PlaywrightScraper


def _make_locator() -> MagicMock:
    """Build a mock that mimics ``page.locator(...).first.fill/click``."""
    loc = MagicMock()
    loc.first = loc
    loc.fill = AsyncMock()
    loc.click = AsyncMock()
    return loc


def _make_playwright_mocks(
    page_url: str = "https://target.example.com/login",
    storage_state_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mock_pw = MagicMock()
    mock_browser = MagicMock()
    mock_page = MagicMock()
    mock_context = MagicMock()
    mock_response = MagicMock()
    mock_response.status = 200
    mock_locator = _make_locator()

    mock_page.goto = AsyncMock(return_value=mock_response)
    mock_page.content = AsyncMock(return_value="<html><body>authed</body></html>")
    mock_page.title = AsyncMock(return_value="Dashboard")
    # Text now comes from extract_main_text(page.content()); evaluate is called
    # for metadata, links, images (in that order).
    mock_page.evaluate = AsyncMock(side_effect=[{"meta": "data"}, [], []])
    mock_page.set_extra_http_headers = AsyncMock()
    mock_page.wait_for_selector = AsyncMock()
    mock_page.wait_for_url = AsyncMock()
    mock_page.wait_for_function = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.screenshot = AsyncMock(return_value=b"png")
    mock_page.route = AsyncMock()
    mock_page.close = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_locator)
    mock_page.url = page_url
    mock_page.context = mock_context

    mock_context.add_cookies = AsyncMock()
    mock_context.close = AsyncMock()
    mock_context.storage_state = AsyncMock(
        return_value=storage_state_payload
        or {"cookies": [{"name": "sid", "value": "abc"}], "origins": []}
    )
    mock_context.new_page = AsyncMock(return_value=mock_page)

    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()
    mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_pw.stop = AsyncMock()

    return {
        "playwright": mock_pw,
        "browser": mock_browser,
        "context": mock_context,
        "page": mock_page,
        "locator": mock_locator,
        "response": mock_response,
    }


@pytest.fixture
def pw_mocks() -> dict[str, Any]:
    return _make_playwright_mocks()


@contextmanager
def _patched_playwright(pw: dict[str, Any]) -> Iterator[None]:
    """Mock ``async_playwright().start()`` to return our mocked playwright."""
    with patch("services.playwright_scraper.async_playwright") as mock_async_pw:
        mock_async_pw.return_value.start = AsyncMock(return_value=pw["playwright"])
        yield


class TestPlaywrightNoAuthRegression:
    """``auth_config=None`` must not invoke any auth primitives."""

    @pytest.mark.asyncio
    async def test_no_auth_config_skips_login(self, pw_mocks: dict[str, Any]) -> None:
        scraper = PlaywrightScraper(config=ScrapeConfig())

        with _patched_playwright(pw_mocks):
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.COMPLETED
        pw_mocks["locator"].fill.assert_not_called()
        pw_mocks["locator"].click.assert_not_called()
        pw_mocks["browser"].new_context.assert_not_called()
        assert result.storage_state is None


class TestPlaywrightFormAuth:
    @pytest.mark.asyncio
    async def test_form_auth_fills_default_selectors_and_submits(
        self, pw_mocks: dict[str, Any]
    ) -> None:
        """Default-selector happy path: fill username, password, click submit."""
        pw_mocks["page"].url = "https://target.example.com/login"
        auth = AuthConfig(type="form", username="alice", password="s3cr3t")
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            result = await scraper.scrape("https://target.example.com/home")

        # Two fills (username + password) and one click.
        assert pw_mocks["locator"].fill.call_count == 2
        assert pw_mocks["locator"].click.call_count == 1
        # Credentials were filled in order.
        fill_values = [call.args[0] for call in pw_mocks["locator"].fill.call_args_list]
        assert fill_values == ["alice", "s3cr3t"]
        # Selectors used are the defaults (contain common email/password patterns).
        selector_args = [call.args[0] for call in pw_mocks["page"].locator.call_args_list]
        assert any('input[name="email"]' in sel for sel in selector_args)
        assert any('input[type="password"]' in sel for sel in selector_args)
        assert any('button[type="submit"]' in sel for sel in selector_args)
        assert result.status == ScrapeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_form_auth_with_custom_selectors(self, pw_mocks: dict[str, Any]) -> None:
        auth = AuthConfig(
            type="form",
            username="u",
            password="p",
            username_selector="#custom-user",
            password_selector="#custom-pass",
            submit_selector="#custom-submit",
        )
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            await scraper.scrape("https://target.example.com/home")

        selectors = [call.args[0] for call in pw_mocks["page"].locator.call_args_list]
        assert "#custom-user" in selectors
        assert "#custom-pass" in selectors
        assert "#custom-submit" in selectors

    @pytest.mark.asyncio
    async def test_form_auth_navigates_to_explicit_login_url(
        self, pw_mocks: dict[str, Any]
    ) -> None:
        pw_mocks["page"].url = "https://target.example.com/home"  # not on /login yet
        auth = AuthConfig(
            type="form",
            username="u",
            password="p",
            login_url="https://target.example.com/auth",
        )
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            await scraper.scrape("https://target.example.com/home")

        goto_urls = [call.args[0] for call in pw_mocks["page"].goto.call_args_list]
        assert "https://target.example.com/auth" in goto_urls

    @pytest.mark.asyncio
    async def test_form_auth_login_url_navigated_before_target(
        self, pw_mocks: dict[str, Any]
    ) -> None:
        """When login_url is set, the scraper must navigate to login_url BEFORE the target.

        Regression test for the bug where ``page.goto(target_url)`` ran first and
        unauthenticated requests landed on the site's own /login page, causing the
        scraper to fill the form against the wrong page (or capture the login page
        as the result).
        """
        pw_mocks["page"].url = "https://target.example.com/home"
        auth = AuthConfig(
            type="form",
            username="u",
            password="p",
            login_url="https://target.example.com/auth",
        )
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            await scraper.scrape("https://target.example.com/home")

        goto_urls = [call.args[0] for call in pw_mocks["page"].goto.call_args_list]
        # Both URLs must have been visited.
        assert "https://target.example.com/auth" in goto_urls
        assert "https://target.example.com/home" in goto_urls
        # Critically: login_url must come BEFORE the first target navigation.
        first_auth_idx = goto_urls.index("https://target.example.com/auth")
        first_home_idx = goto_urls.index("https://target.example.com/home")
        assert first_auth_idx < first_home_idx, (
            f"login_url must be navigated before target_url; got order: {goto_urls}"
        )
        # The form must have been filled+submitted (not skipped).
        assert pw_mocks["locator"].fill.call_count == 2
        assert pw_mocks["locator"].click.call_count == 1

    @pytest.mark.asyncio
    async def test_form_auth_login_url_handles_redirect_to_login(
        self, pw_mocks: dict[str, Any]
    ) -> None:
        """Even if the site redirects target → /login, login_url path fills form first.

        Simulates the live loginfix scenario: GET /home redirects to /login. With
        login_url set, the fix navigates to login_url first, so the form gets
        filled+submitted and the final navigation lands on /home authenticated.
        """
        nav_state = {"url": "https://target.example.com/auth"}

        async def goto_side_effect(url: str, **_kwargs: Any) -> MagicMock:
            # When asked to go to /home unauthenticated, simulate redirect to /login.
            # When asked to go to login_url, just land there.
            if url == "https://target.example.com/auth":
                nav_state["url"] = "https://target.example.com/auth"
            elif url == "https://target.example.com/home":
                nav_state["url"] = "https://target.example.com/home"
            pw_mocks["page"].url = nav_state["url"]
            return pw_mocks["response"]

        pw_mocks["page"].goto = AsyncMock(side_effect=goto_side_effect)
        # After form submit / wait_for_function, treat us as authenticated.
        pw_mocks["page"].wait_for_function = AsyncMock()

        auth = AuthConfig(
            type="form",
            username="u",
            password="p",
            login_url="https://target.example.com/auth",
        )
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            result = await scraper.scrape("https://target.example.com/home")

        goto_urls = [call.args[0] for call in pw_mocks["page"].goto.call_args_list]
        # First goto must be login_url, not the target.
        assert goto_urls[0] == "https://target.example.com/auth", (
            f"first goto must be login_url; got: {goto_urls}"
        )
        # Form was filled.
        assert pw_mocks["locator"].fill.call_count == 2
        assert pw_mocks["locator"].click.call_count == 1
        # Eventually, target was navigated.
        assert "https://target.example.com/home" in goto_urls
        assert result.status == ScrapeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_form_auth_wait_for_url(self, pw_mocks: dict[str, Any]) -> None:
        auth = AuthConfig(
            type="form",
            username="u",
            password="p",
            wait_for_url="**/dashboard",
            wait_timeout_ms=20000,
        )
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            await scraper.scrape("https://target.example.com/home")

        pw_mocks["page"].wait_for_url.assert_called()
        first_call = pw_mocks["page"].wait_for_url.call_args_list[0]
        assert first_call.args[0] == "**/dashboard"
        assert first_call.kwargs.get("timeout") == 20000

    @pytest.mark.asyncio
    async def test_form_auth_wait_for_selector(self, pw_mocks: dict[str, Any]) -> None:
        auth = AuthConfig(
            type="form",
            username="u",
            password="p",
            wait_for_selector=".dashboard-header",
        )
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            await scraper.scrape("https://target.example.com/home")

        wait_selector_calls = pw_mocks["page"].wait_for_selector.call_args_list
        selectors_waited = [call.args[0] for call in wait_selector_calls]
        assert ".dashboard-header" in selectors_waited

    @pytest.mark.asyncio
    async def test_form_auth_default_wait_uses_wait_for_function(
        self, pw_mocks: dict[str, Any]
    ) -> None:
        """With no wait_for_* set, the default path uses ``page.wait_for_function``.

        Regression test for SPA login flows: ``page.wait_for_url(callable)`` only fires
        on navigation events, so SPAs that set cookies via ``fetch()`` (no navigation,
        same-document URL change) would time out. The fix is to poll in JS via
        ``wait_for_function`` so same-document transitions are observed.
        """
        pw_mocks["page"].wait_for_function = AsyncMock()
        auth = AuthConfig(type="form", username="u", password="p", wait_timeout_ms=9000)
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            await scraper.scrape("https://target.example.com/home")

        # Default path must NOT call wait_for_url with a callable (broken for SPAs).
        for call in pw_mocks["page"].wait_for_url.call_args_list:
            if call.args:
                assert not callable(call.args[0]), (
                    "default wait must not use wait_for_url(callable) — "
                    "Playwright only fires it on navigation events, breaking SPA logins"
                )

        # Default path MUST call wait_for_function with a JS expression inspecting
        # window.location, and honor the configured timeout.
        pw_mocks["page"].wait_for_function.assert_called_once()
        call = pw_mocks["page"].wait_for_function.call_args
        js_expr = call.args[0]
        assert isinstance(js_expr, str)
        assert "window.location" in js_expr
        assert "login" in js_expr.lower()
        assert call.kwargs.get("timeout") == 9000

    @pytest.mark.asyncio
    async def test_form_auth_captures_storage_state(self, pw_mocks: dict[str, Any]) -> None:
        captured = {"cookies": [{"name": "sid", "value": "xyz"}], "origins": []}
        pw_mocks["context"].storage_state = AsyncMock(return_value=captured)
        auth = AuthConfig(type="form", username="u", password="p")
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            result = await scraper.scrape("https://target.example.com/home")

        assert result.storage_state == captured

    @pytest.mark.asyncio
    async def test_form_auth_wait_timeout_returns_failed(self, pw_mocks: dict[str, Any]) -> None:
        # Default path now polls via wait_for_function; simulate that timing out.
        pw_mocks["page"].wait_for_function = AsyncMock(
            side_effect=PlaywrightTimeoutError("timeout")
        )
        auth = AuthConfig(type="form", username="u", password="p")
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            result = await scraper.scrape("https://target.example.com/home")

        assert result.status == ScrapeStatus.FAILED
        assert result.error is not None
        assert "timeout" in result.error.lower() or "login" in result.error.lower()

    @pytest.mark.asyncio
    async def test_form_auth_renavigates_to_original_url_after_login(
        self, pw_mocks: dict[str, Any]
    ) -> None:
        """After login redirects us to /dashboard, the scraper must navigate back to the original URL."""
        # Simulate: initial goto → ends on /login; wait_for_url sets us on /dashboard; re-nav required.
        goto_calls = []

        async def goto_side_effect(url: str, **_kwargs: Any) -> MagicMock:
            goto_calls.append(url)
            if url == "https://target.example.com/home":
                pw_mocks["page"].url = "https://target.example.com/login"
            elif url == "https://target.example.com/home" and goto_calls.count(url) > 1:
                pw_mocks["page"].url = "https://target.example.com/home"
            return pw_mocks["response"]

        async def wait_side_effect(*_args: Any, **_kwargs: Any) -> None:
            pw_mocks["page"].url = "https://target.example.com/dashboard"

        pw_mocks["page"].goto = AsyncMock(side_effect=goto_side_effect)
        # Default path uses wait_for_function (SPA-safe polling); simulate that
        # completing and landing us on /dashboard.
        pw_mocks["page"].wait_for_function = AsyncMock(side_effect=wait_side_effect)
        pw_mocks["page"].wait_for_url = AsyncMock(side_effect=wait_side_effect)
        auth = AuthConfig(type="form", username="u", password="p")
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            await scraper.scrape("https://target.example.com/home")

        # Expect goto called at least twice — original URL appears both pre- and post-login.
        assert goto_calls.count("https://target.example.com/home") >= 2


class TestPlaywrightStorageStateAuth:
    @pytest.mark.asyncio
    async def test_storage_state_injection_uses_explicit_context(
        self, pw_mocks: dict[str, Any]
    ) -> None:
        state = {"cookies": [{"name": "sid", "value": "cached"}], "origins": []}
        auth = AuthConfig(type="storage_state", storage_state=state)
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            result = await scraper.scrape("https://target.example.com/home")

        # new_context must be called with the storage_state payload.
        pw_mocks["browser"].new_context.assert_called_once()
        ctx_kwargs = pw_mocks["browser"].new_context.call_args.kwargs
        assert ctx_kwargs["storage_state"] == state
        # no form interaction
        pw_mocks["locator"].fill.assert_not_called()
        assert result.status == ScrapeStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_storage_state_captures_current_state_in_result(
        self, pw_mocks: dict[str, Any]
    ) -> None:
        fresh = {"cookies": [{"name": "sid", "value": "still-valid"}], "origins": []}
        pw_mocks["context"].storage_state = AsyncMock(return_value=fresh)
        auth = AuthConfig(type="storage_state", storage_state={"cookies": [], "origins": []})
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            result = await scraper.scrape("https://target.example.com/home")

        assert result.storage_state == fresh


class TestPlaywrightBearerAuth:
    @pytest.mark.asyncio
    async def test_bearer_sets_authorization_header(self, pw_mocks: dict[str, Any]) -> None:
        auth = AuthConfig(type="bearer", token="jwt-xyz")
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            await scraper.scrape("https://target.example.com/home")

        pw_mocks["page"].set_extra_http_headers.assert_called()
        headers_arg = pw_mocks["page"].set_extra_http_headers.call_args.args[0]
        assert headers_arg.get("Authorization") == "Bearer jwt-xyz"
        # No form interaction
        pw_mocks["locator"].fill.assert_not_called()


class TestPlaywrightBasicAuth:
    @pytest.mark.asyncio
    async def test_basic_sets_http_credentials_on_context(self, pw_mocks: dict[str, Any]) -> None:
        auth = AuthConfig(type="basic", username="admin", password="pw")
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))

        with _patched_playwright(pw_mocks):
            await scraper.scrape("https://target.example.com/home")

        pw_mocks["browser"].new_context.assert_called_once()
        ctx_kwargs = pw_mocks["browser"].new_context.call_args.kwargs
        assert ctx_kwargs["http_credentials"] == {"username": "admin", "password": "pw"}


class TestPlaywrightResourceLifecycle:
    """Regression: a creation failure on the auth path must not leak the context."""

    @pytest.mark.asyncio
    async def test_context_closed_when_new_page_fails(self, pw_mocks: dict[str, Any]) -> None:
        # storage_state auth → an auth BrowserContext is created; if new_page()
        # then raises, the context must still be closed (not orphaned).
        auth = AuthConfig(type="storage_state", storage_state={"cookies": [], "origins": []})
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth))
        pw_mocks["context"].new_page = AsyncMock(side_effect=RuntimeError("boom"))

        with _patched_playwright(pw_mocks):
            result = await scraper.scrape("https://example.com")

        assert result.status == ScrapeStatus.FAILED
        pw_mocks["browser"].new_context.assert_awaited_once()
        pw_mocks["context"].close.assert_awaited_once()


class TestPlaywrightLoginTimeout:
    """The login-page LOAD uses the request nav budget, not auth.wait_timeout_ms."""

    @pytest.mark.asyncio
    async def test_login_goto_uses_request_nav_timeout(self, pw_mocks: dict[str, Any]) -> None:
        pw_mocks["page"].url = "https://target.example.com/home"
        auth = AuthConfig(
            type="form",
            username="u",
            password="p",
            login_url="https://target.example.com/login",
            wait_timeout_ms=15000,
        )
        scraper = PlaywrightScraper(config=ScrapeConfig(auth_config=auth, timeout=30.0))

        with _patched_playwright(pw_mocks):
            result = await scraper.scrape("https://target.example.com/home")

        assert result.status == ScrapeStatus.COMPLETED
        login_gotos = [
            c
            for c in pw_mocks["page"].goto.call_args_list
            if c.args and c.args[0] == "https://target.example.com/login"
        ]
        assert login_gotos, "expected a navigation to the login_url"
        # 30000 (request nav budget), NOT 15000 (auth condition-wait budget).
        assert login_gotos[0].kwargs["timeout"] == 30000
