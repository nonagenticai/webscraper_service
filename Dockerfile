FROM registry.example.com/library/python:3.14-slim AS base

ENV UV_SYSTEM_PYTHON=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/usr/local/bin:$PATH

WORKDIR /app

# Bring in uv binary (multi-arch)
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

# ──────────────────────────────────────────────────────────────────────────
# Builder stage - install build deps and create a deps-only .venv
#
# The .venv is deliberately deps-only (--no-install-project): the app runs
# from /app/src via PYTHONPATH, not from an installed package (no
# [project.scripts], no importlib.metadata version reads, nothing imports the
# `webscraper_service` name). Baking src into the venv would make the ~1GB
# venv layer change on every code commit and force a full re-push to
# registry.example.com. Keeping the venv deps-only ties it to uv.lock alone.
# ──────────────────────────────────────────────────────────────────────────
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    ca-certificates \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests only - this layer is invalidated solely by a
# dependency change (pyproject.toml / uv.lock), never by application code.
COPY README.md pyproject.toml uv.lock* ./

# Install dependencies only (no project) - cache-stable across code changes.
RUN uv sync --frozen --no-dev --no-install-project

# ──────────────────────────────────────────────────────────────────────────
# Production stage - minimal runtime image
#
# Layer order is stable → volatile so a code-only commit leaves the heavy
# layers (venv ~1GB, playwright chromium ~1GB) byte-identical: their blobs
# already exist in registry.example.com and `buildx --push` skips them, leaving
# only the small src layer to upload. Anything that changes on every commit
# (COPY src) MUST come last.
# ──────────────────────────────────────────────────────────────────────────
FROM base AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Trust internal CA for pypi.example.com
COPY internal-ca.crt /usr/local/share/ca-certificates/internal-ca.crt
RUN update-ca-certificates

# Deps-only venv (changes only when uv.lock changes, not on code commits)
COPY --from=builder /app/.venv /app/.venv

# Install Playwright browsers + system deps. Depends only on the venv's
# pinned playwright version, NOT on application source, so this ~1GB layer is
# cache-stable across code commits. uv/playwright write world-readable trees,
# so the non-root appuser can read/exec them without an (image-doubling)
# recursive chown.
ENV PLAYWRIGHT_BROWSERS_PATH=/app/ms-playwright
RUN /app/.venv/bin/playwright install --with-deps chromium && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user + writable dirs. Stable layer (no src dependency); only
# the dirs the app actually writes to are chowned - the big read-only venv and
# browser trees are left as-is to avoid a multi-GB copy-up layer.
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs /tmp/webscraper_screenshots && \
    chown -R appuser:appuser /app/logs /tmp/webscraper_screenshots

# Runtime env
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PORT=9119 \
    LOG_LEVEL=INFO

# Application code + manifests LAST - the only layer that changes per commit.
# Copied straight from the build context (not the builder) and owned by appuser.
COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser pyproject.toml uv.lock* README.md /app/

USER appuser

EXPOSE 9119

HEALTHCHECK --interval=30s --timeout=15s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:9119/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9119"]
