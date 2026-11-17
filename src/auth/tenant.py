"""Tenant defaults — the ONE place this service reads ``DEFAULT_TENANT_ID``.

The value used to be read independently in three modules
(``auth/middleware.py``, ``auth/dependencies.py``, ``config.py``), each with its
own copy of the ``"n8n-dev"`` literal. Three readers of one env var is three
chances to drift, and a drifted tenant default does not crash anything — it
silently bills someone. Everything below is resolved ONCE, at import, and
imported from here; ``config.Settings.default_tenant_id`` takes its default from
this module too, so the literal exists exactly once in the tree.

⚠️ ``"n8n-dev"`` LOOKS like a placeholder and is NOT one. billing_service's
``canonical_tenant_id`` resolves that slug to a real tenant UUID and bills it,
so a request served under this default lands on a real customer's invoice. The
live Deployment overrides it with a real tenant UUID anyway
(``DEFAULT_TENANT_ID: 0195f48b-d000-7000-8000-000000000001`` in
gitops-production ``clusters/strato/services/webscraper/deployment.yaml``) —
which is a *different* real customer, so mis-attribution is the failure mode
either way. The value is deliberately NOT changed here: changing the default
and adding the gate below in one step would make any billing regression
unattributable to one of the two.

⚠️ And it must never become the nil UUID ``00000000-...-000000000000``.
billing_service quarantines nil-tenant events as malformed into a dead-letter
stream that skips BOTH the token rollup and the budget debit
(``billing_service/src/services/_cost_schema.py``). That converts mis-billing
into un-billing, which is strictly worse: the spend stops landing on the wrong
line and starts not existing at all.
"""

from __future__ import annotations

import logging
import os
import uuid

from prometheus_client import Counter

logger = logging.getLogger(__name__)

DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "n8n-dev")

# The trusted-network default-tenant fallback: serve a header-less request from
# a peer inside TRUSTED_CIDRS as DEFAULT_TENANT_ID instead of 401'ing it.
#
# ⚠️ DEFAULT IS ON, AND MUST STAY ON IN THIS CHANGE. TRUSTED_CIDRS defaults to
# the whole pod CIDR (10.0.0.0/8), so every in-cluster caller currently reaches
# this service with no credential at all and is served under the default tenant.
# Flipping this to false today 401s those live callers. The flip is a SEPARATE,
# MEASURED gitops step: it may only happen once the TENANT_FALLBACK warning
# below has gone quiet, which is the evidence that every caller has been issued
# a license token. Modelled on kag's ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT
# (``kag/src/core/config.py``), which carries the same default for the same
# reason.
#
# Note this gates ONLY the DEFAULT-tenant arm. A trusted peer that names its own
# tenant in X-Tenant-Id is still honoured when the gate is closed — that traffic
# is already attributed, so it is not what the burn-down is about.
ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT = (
    os.getenv("ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT", "true").lower() == "true"
)

# The burn-down counter. Defined here rather than in ``services/metrics.py``
# because ``services/__init__.py`` eagerly imports the whole scraper stack
# (playwright, scrapy, bs4, trafilatura); making the auth path depend on that is
# both a layering inversion and a large import cost on a module every request
# touches. ``routers/metrics.py`` serves ``generate_latest()`` over the DEFAULT
# registry, so a counter registered from anywhere is exposed at ``/metrics``.
#
# Unlabelled on purpose: the interesting dimension is the peer IP, and peer IPs
# are unbounded cardinality in a pod-CIDR-wide trusted range. The per-peer
# breakdown belongs in the log line below, which Loki can aggregate; Prometheus
# gets the scalar so an alert can watch it reach zero.
TENANT_FALLBACK_TOTAL = Counter(
    "webscraper_tenant_fallback_total",
    "Requests served under DEFAULT_TENANT_ID via the trusted-network fallback "
    "with no credential presented.",
)


def _is_uuid(value: str) -> bool:
    """True when ``value`` parses as a UUID of any version (the fleet uses uuid7)."""
    try:
        uuid.UUID(value)
    except ValueError, AttributeError, TypeError:
        return False
    return True


def log_tenant_fallback(peer: str | None, path: str) -> None:
    """Record one request served under ``DEFAULT_TENANT_ID`` with no credential.

    THIS IS THE POINT OF THE GATE. Without a line naming the peer, nobody can
    enumerate which callers still have to be issued a license token, so
    ``ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT`` can never be closed — the flip
    would be a blind 401 of unknown production traffic. WARNING (not debug, not
    info) because the pre-existing ``logger.debug`` on this same branch never
    reached Loki: this service sets LOG_LEVEL=INFO in the live Deployment, so
    the debug line was emitted into nothing for the whole life of the fallback.

    When this stops being emitted, the switch is safe to flip.
    """
    TENANT_FALLBACK_TOTAL.inc()
    logger.warning(
        "TENANT_FALLBACK peer=%s path=%s — served as DEFAULT_TENANT_ID with no "
        "credential presented; this caller needs a license token before "
        "ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT can be closed",
        peer or "unknown",
        path,
    )


# Startup shape check. Deliberately a WARNING and NOT a raise: the module
# default ``"n8n-dev"`` is itself not a UUID, so refusing a non-UUID at import
# would brick every local/dev boot and the entire test suite (neither sets
# DEFAULT_TENANT_ID), while the only environment that would benefit — production
# — already supplies a real UUID. A hard refusal here buys nothing and breaks
# the boots that are fine.
if not _is_uuid(DEFAULT_TENANT_ID):
    logger.warning(
        "DEFAULT_TENANT_ID=%r is not a canonical tenant UUID. This is NOT a "
        "harmless placeholder: billing resolves such a slug to a real tenant and "
        "bills it. Set DEFAULT_TENANT_ID to the intended tenant UUID.",
        DEFAULT_TENANT_ID,
    )


__all__ = [
    "ALLOW_TRUSTED_NETWORK_DEFAULT_TENANT",
    "DEFAULT_TENANT_ID",
    "TENANT_FALLBACK_TOTAL",
    "log_tenant_fallback",
]
