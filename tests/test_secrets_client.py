"""Tests for SecretsClient — fetches URL credentials from the secrets service.

Covers token forwarding (license-token bearer vs trusted-network tenant-id),
404 → CredentialNotFoundError, generic errors → SecretsServiceError, and response parsing.
"""

import httpx
import pytest
import respx

from services.secrets_client import (
    CredentialNotFoundError,
    SecretsClient,
    SecretsServiceError,
    UrlCredential,
)

BASE_URL = "http://secrets.test.local:8543"


@pytest.fixture
def client() -> SecretsClient:
    return SecretsClient(base_url=BASE_URL, timeout=5.0)


@pytest.mark.asyncio
@respx.mock
async def test_get_url_credential_forwards_bearer_license_token(client: SecretsClient) -> None:
    route = respx.get(f"{BASE_URL}/url-credentials/abc123").mock(
        return_value=httpx.Response(
            200,
            json={
                "credential_id": "abc123",
                "user_id": "tenant-1",
                "identifier": "specra-dev",
                "url": "https://dev.specralabs.com",
                "credentials": {"username": "u", "password": "p", "auth_type": "form"},
                "metadata": {"website": "specralabs"},
                "created_at": "2026-04-17T00:00:00Z",
            },
        )
    )

    result = await client.get_url_credential(
        "abc123", license_token="lic-xyz", tenant_id="tenant-1"
    )

    assert route.called
    assert route.calls.last.request.headers["X-Tenant-Id"] == "Bearer lic-xyz"
    assert result.credential_id == "abc123"
    assert result.credentials["username"] == "u"
    assert result.credentials["password"] == "p"


@pytest.mark.asyncio
@respx.mock
async def test_get_url_credential_trusted_network_uses_plain_tenant_id(
    client: SecretsClient,
) -> None:
    """When no license token is available (trusted-network caller), forward plain tenant-id."""
    route = respx.get(f"{BASE_URL}/url-credentials/abc123").mock(
        return_value=httpx.Response(
            200,
            json={
                "credential_id": "abc123",
                "user_id": "tenant-1",
                "credentials": {"username": "u", "password": "p"},
                "metadata": {},
            },
        )
    )

    await client.get_url_credential("abc123", license_token=None, tenant_id="tenant-1")

    assert route.called
    assert route.calls.last.request.headers["X-Tenant-Id"] == "tenant-1"


@pytest.mark.asyncio
@respx.mock
async def test_get_url_credential_404_raises_credential_not_found(client: SecretsClient) -> None:
    respx.get(f"{BASE_URL}/url-credentials/missing").mock(
        return_value=httpx.Response(404, json={"detail": "not found"})
    )

    with pytest.raises(CredentialNotFoundError) as exc:
        await client.get_url_credential("missing", license_token="lic", tenant_id="t")

    assert "missing" in str(exc.value)


@pytest.mark.asyncio
@respx.mock
async def test_get_url_credential_401_raises_service_error(client: SecretsClient) -> None:
    respx.get(f"{BASE_URL}/url-credentials/abc123").mock(
        return_value=httpx.Response(401, json={"detail": "unauthorized"})
    )

    with pytest.raises(SecretsServiceError) as exc:
        await client.get_url_credential("abc123", license_token="bad", tenant_id="t")

    assert "401" in str(exc.value)


@pytest.mark.asyncio
@respx.mock
async def test_get_url_credential_network_error_raises_service_error(
    client: SecretsClient,
) -> None:
    respx.get(f"{BASE_URL}/url-credentials/abc123").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    with pytest.raises(SecretsServiceError):
        await client.get_url_credential("abc123", license_token="lic", tenant_id="t")


@pytest.mark.asyncio
@respx.mock
async def test_get_url_credential_parses_all_fields(client: SecretsClient) -> None:
    respx.get(f"{BASE_URL}/url-credentials/full").mock(
        return_value=httpx.Response(
            200,
            json={
                "credential_id": "full",
                "user_id": "tenant-1",
                "identifier": "my-site",
                "url": "https://example.com",
                "credentials": {
                    "username": "alice",
                    "password": "secret",
                    "auth_token": "jwt-xyz",
                    "cookies": {"session": "abc"},
                    "auth_type": "form",
                },
                "metadata": {"website": "example", "purpose": "scraping"},
                "created_at": "2026-04-17T00:00:00Z",
                "expires_at": "2026-04-24T00:00:00Z",
            },
        )
    )

    result = await client.get_url_credential("full", license_token="lic", tenant_id="t")

    assert isinstance(result, UrlCredential)
    assert result.credential_id == "full"
    assert result.identifier == "my-site"
    assert result.url == "https://example.com"
    assert result.credentials["auth_token"] == "jwt-xyz"
    assert result.credentials["cookies"] == {"session": "abc"}
    assert result.metadata["website"] == "example"


@pytest.mark.asyncio
@respx.mock
async def test_get_url_credential_tolerates_missing_optional_fields(
    client: SecretsClient,
) -> None:
    """Minimal payload — only required fields present."""
    respx.get(f"{BASE_URL}/url-credentials/min").mock(
        return_value=httpx.Response(
            200,
            json={
                "credential_id": "min",
                "user_id": "t",
                "credentials": {"username": "u", "password": "p"},
                "metadata": {},
            },
        )
    )

    result = await client.get_url_credential("min", license_token="lic", tenant_id="t")

    assert result.credential_id == "min"
    assert result.url is None
    assert result.identifier is None


@pytest.mark.asyncio
@respx.mock
async def test_get_url_credential_tolerates_null_metadata(client: SecretsClient) -> None:
    """Secrets service returns metadata as Optional[Dict] — JSON null must not 500."""
    respx.get(f"{BASE_URL}/url-credentials/nullmeta").mock(
        return_value=httpx.Response(
            200,
            json={
                "credential_id": "nullmeta",
                "user_id": "tenant-1",
                "identifier": "specra-dev",
                "url": "https://dev.specralabs.com",
                "credentials": {"username": "u", "password": "p", "auth_type": "form"},
                "metadata": None,
                "created_at": "2026-04-22T00:00:00Z",
            },
        )
    )

    result = await client.get_url_credential("nullmeta", license_token="lic", tenant_id="tenant-1")

    assert result.credential_id == "nullmeta"
    assert result.metadata == {}


@pytest.mark.asyncio
@respx.mock
async def test_invalid_json_body_raises_secrets_error(client: SecretsClient) -> None:
    """A 2xx with a non-JSON body must surface as SecretsServiceError (→ 502),
    not leak a raw JSONDecodeError as a 500."""
    respx.get(f"{BASE_URL}/url-credentials/bad1").mock(
        return_value=httpx.Response(200, content=b"not json at all")
    )
    with pytest.raises(SecretsServiceError):
        await client.get_url_credential("bad1", license_token=None, tenant_id="t1")


@pytest.mark.asyncio
@respx.mock
async def test_unexpected_shape_raises_secrets_error(client: SecretsClient) -> None:
    """A 2xx whose JSON is missing required fields → SecretsServiceError, not 500."""
    respx.get(f"{BASE_URL}/url-credentials/bad2").mock(
        return_value=httpx.Response(200, json={"unexpected": "shape"})
    )
    with pytest.raises(SecretsServiceError):
        await client.get_url_credential("bad2", license_token=None, tenant_id="t1")
