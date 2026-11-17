"""Tests for the scrape router auth integration.

Verifies that a request containing an ``auth_config`` triggers:
1. A fetch to the secrets service using the caller's forwarded license token.
2. A service-layer ``AuthConfig`` populated from the fetched credential payload.
3. ``storage_state`` in the response when the scraper captures one.
4. Informative HTTP errors on credential-not-found / missing-field failures.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from services.base_scraper import ScrapeResult, ScrapeStatus
from services.secrets_client import CredentialNotFoundError, UrlCredential


def _cred(credentials: dict[str, Any], cid: str = "cred-1") -> UrlCredential:
    return UrlCredential(credential_id=cid, credentials=credentials, metadata={})


class TestRouterAuthIntegration:
    @pytest.mark.asyncio
    async def test_scrape_with_form_auth_fetches_credential_and_threads_it(
        self, async_client, base_url
    ) -> None:
        captured_auth: list[Any] = []

        async def fake_scrape_url(url, **kwargs):
            config = kwargs["config"]
            captured_auth.append(config.auth_config)
            return ScrapeResult(
                url=url,
                html_content="<html>authed</html>",
                text_content="authed",
                title="Dashboard",
                metadata={"final_url": url},
                status=ScrapeStatus.COMPLETED,
                scraper_used="PlaywrightScraper",
                processing_time_seconds=0.2,
                storage_state={"cookies": [{"name": "sid", "value": "x"}], "origins": []},
            )

        with (
            patch(
                "routers.scrape.SecretsClient.get_url_credential",
                new=AsyncMock(return_value=_cred({"username": "alice", "password": "s3cr3t"})),
            ) as mock_fetch,
            patch("routers.scrape.get_scraper_manager") as mock_get_manager,
        ):
            mock_manager = mock_get_manager.return_value
            mock_manager.scrape_url = AsyncMock(side_effect=fake_scrape_url)

            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": f"{base_url}/simple",
                    "auth_config": {"type": "form", "credential_id": "cred-1"},
                },
            )

        assert resp.status_code == 200
        mock_fetch.assert_awaited_once()
        # The service-layer AuthConfig should carry the resolved credentials.
        assert captured_auth[0] is not None
        assert captured_auth[0].type == "form"
        assert captured_auth[0].username == "alice"
        assert captured_auth[0].password == "s3cr3t"

        body = resp.json()
        assert body["storage_state"] == {
            "cookies": [{"name": "sid", "value": "x"}],
            "origins": [],
        }

    @pytest.mark.asyncio
    async def test_scrape_with_bearer_auth_resolves_token(self, async_client, base_url) -> None:
        captured_auth: list[Any] = []

        async def fake_scrape_url(url, **kwargs):
            config = kwargs["config"]
            captured_auth.append(config.auth_config)
            return ScrapeResult(
                url=url,
                status=ScrapeStatus.COMPLETED,
                scraper_used="HttpxScraper",
                processing_time_seconds=0.1,
            )

        with (
            patch(
                "routers.scrape.SecretsClient.get_url_credential",
                new=AsyncMock(return_value=_cred({"auth_token": "jwt-xyz"})),
            ),
            patch("routers.scrape.get_scraper_manager") as mock_get_manager,
        ):
            mock_manager = mock_get_manager.return_value
            mock_manager.scrape_url = AsyncMock(side_effect=fake_scrape_url)

            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": f"{base_url}/simple",
                    "auth_config": {"type": "bearer", "credential_id": "cred-1"},
                },
            )

        assert resp.status_code == 200
        assert captured_auth[0].token == "jwt-xyz"

    @pytest.mark.asyncio
    async def test_scrape_with_storage_state_auth_resolves_state(
        self, async_client, base_url
    ) -> None:
        captured_auth: list[Any] = []
        state = {"cookies": [{"name": "sid", "value": "cached"}], "origins": []}

        async def fake_scrape_url(url, **kwargs):
            config = kwargs["config"]
            captured_auth.append(config.auth_config)
            return ScrapeResult(
                url=url,
                status=ScrapeStatus.COMPLETED,
                scraper_used="PlaywrightScraper",
                processing_time_seconds=0.1,
            )

        with (
            patch(
                "routers.scrape.SecretsClient.get_url_credential",
                new=AsyncMock(return_value=_cred({"storage_state": state})),
            ),
            patch("routers.scrape.get_scraper_manager") as mock_get_manager,
        ):
            mock_manager = mock_get_manager.return_value
            mock_manager.scrape_url = AsyncMock(side_effect=fake_scrape_url)

            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": f"{base_url}/simple",
                    "auth_config": {"type": "storage_state", "credential_id": "cred-1"},
                },
            )

        assert resp.status_code == 200
        assert captured_auth[0].storage_state == state

    @pytest.mark.asyncio
    async def test_scrape_missing_credential_returns_404(self, async_client, base_url) -> None:
        with patch(
            "routers.scrape.SecretsClient.get_url_credential",
            new=AsyncMock(side_effect=CredentialNotFoundError("cred-1")),
        ):
            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": f"{base_url}/simple",
                    "auth_config": {"type": "form", "credential_id": "cred-1"},
                },
            )

        assert resp.status_code == 404
        assert "cred-1" in resp.text

    @pytest.mark.asyncio
    async def test_scrape_missing_required_credential_field_returns_400(
        self, async_client, base_url
    ) -> None:
        """Form auth requires username + password; surface a clear 400 if missing."""
        with patch(
            "routers.scrape.SecretsClient.get_url_credential",
            new=AsyncMock(return_value=_cred({"username": "u"})),  # no password
        ):
            resp = await async_client.post(
                "/api/scrape/url",
                json={
                    "url": f"{base_url}/simple",
                    "auth_config": {"type": "form", "credential_id": "cred-1"},
                },
            )

        assert resp.status_code == 400
        body = resp.json()
        assert "password" in body["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_scrape_without_auth_unchanged(self, async_client, base_url) -> None:
        """Baseline — unauth request still works end-to-end against local server."""
        resp = await async_client.post(
            "/api/scrape/url",
            json={"url": f"{base_url}/simple", "scraper_type": "httpx"},
        )
        assert resp.status_code == 200
        assert resp.json()["storage_state"] is None
