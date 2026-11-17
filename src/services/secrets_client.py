"""Client for the secrets service — fetches URL credentials for authenticated scraping.

The secrets service at ``secrets.example.com:8543`` stores per-tenant
website credentials. This client forwards the caller's license token (or plain
tenant-id for trusted-network callers) so the secrets service enforces tenant
ownership of the requested credential.
"""

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class UrlCredential(BaseModel):
    """Response payload from ``GET /url-credentials/{id}``."""

    credential_id: str
    url: str | None = None
    identifier: str | None = None
    credentials: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_null_metadata(cls, v: Any) -> Any:
        # Secrets service returns metadata as Optional[Dict], so JSON null is valid upstream.
        return {} if v is None else v


class SecretsServiceError(Exception):
    """Generic failure talking to the secrets service (non-404)."""


class CredentialNotFoundError(SecretsServiceError):
    """The requested credential_id does not exist (or has expired)."""

    def __init__(self, credential_id: str) -> None:
        super().__init__(f"Credential not found: {credential_id}")
        self.credential_id = credential_id


class SecretsClient:
    """Async httpx-based client for the secrets service."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def get_url_credential(
        self,
        credential_id: str,
        *,
        license_token: str | None,
        tenant_id: str,
    ) -> UrlCredential:
        """Fetch a URL credential by id.

        Args:
            credential_id: Credential identifier returned by the secrets service.
            license_token: Caller's license bearer token, forwarded as
                ``X-Tenant-Id: Bearer <token>``. If ``None``, falls back to
                plain ``X-Tenant-Id: <tenant_id>`` (trusted-network path).
            tenant_id: Tenant identifier used when no license token is available.

        Raises:
            CredentialNotFoundError: On 404.
            SecretsServiceError: On any other non-2xx status or network failure.
        """
        if license_token:
            headers = {"X-Tenant-Id": f"Bearer {license_token}"}
        else:
            headers = {"X-Tenant-Id": tenant_id}

        url = f"{self.base_url}/url-credentials/{credential_id}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("Secrets service network error for %s: %s", credential_id, exc)
            raise SecretsServiceError(f"secrets service unreachable: {exc}") from exc

        if resp.status_code == 404:
            raise CredentialNotFoundError(credential_id)
        if resp.status_code >= 400:
            raise SecretsServiceError(
                f"secrets service returned {resp.status_code} for credential {credential_id}"
            )

        # A 2xx with an empty/malformed body or an unexpected shape must not leak
        # a raw JSONDecodeError/ValidationError as a generic 500 — surface it as
        # a SecretsServiceError (→ 502 at the router) like other upstream faults.
        try:
            return UrlCredential.model_validate(resp.json())
        except ValueError as exc:
            logger.warning(
                "Secrets service returned an invalid body for %s: %s", credential_id, exc
            )
            raise SecretsServiceError(
                f"secrets service returned an invalid body for credential {credential_id}"
            ) from exc


__all__ = ["SecretsClient", "UrlCredential", "CredentialNotFoundError", "SecretsServiceError"]
