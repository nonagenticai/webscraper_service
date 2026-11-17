"""FastAPI dependencies for authentication.

The license-token middleware (``auth/middleware.py``) populates
``request.state.user`` with a dict and ``request.state.tenant_id`` with a
string. This module exposes that identity as a FastAPI dependency so route
handlers can declare ``user: dict = Depends(get_current_user)``.

When auth is fully disabled (``ENABLE_AUTH=false``), returns a synthetic
trusted-network identity so internal smoke tests can hit the service without a
license server.
"""

from __future__ import annotations

import os

from fastapi import HTTPException, Request, status

from .tenant import DEFAULT_TENANT_ID

ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"

# DEFAULT_TENANT_ID is imported, never re-read from the environment here: this
# module used to keep its own `os.getenv("DEFAULT_TENANT_ID", "n8n-dev")`, so a
# change to the middleware's copy would leave the ENABLE_AUTH=false identity
# billing a different tenant than the trusted-network fallback. See
# auth/tenant.py for why that default is a real billable tenant, not a
# placeholder.

_ANONYMOUS_USER: dict[str, str] = {
    "user_id": DEFAULT_TENANT_ID,
    "tenant_id": DEFAULT_TENANT_ID,
    "sub": DEFAULT_TENANT_ID,
}


async def get_current_user(request: Request) -> dict[str, str]:
    """Return the authenticated user dict for this request."""
    if not ENABLE_AUTH:
        return _ANONYMOUS_USER

    user = getattr(request.state, "user", None)
    if isinstance(user, dict):
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )
