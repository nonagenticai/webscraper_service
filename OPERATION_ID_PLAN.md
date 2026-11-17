# Operation ID Plan — webscraper-service

This file is the **source of truth** for the OpenAPI `operationId` of every
route on this service. The guardrail test `tests/test_openapi.py` enforces that
the live OpenAPI schema matches this table exactly — a route added or renamed
without updating this file fails CI.

Naming convention: `verb_noun` (snake_case, globally unique across the fleet).

## Root (`src/main.py`)

| Method | Path | Function | operation_id   | Tag    |
|--------|------|----------|----------------|--------|
| GET    | /    | `root`   | `service_root` | Health |

## Health router (`src/routers/health.py`, prefix `/health`)

| Method | Path           | Function          | operation_id     | Tag    |
|--------|----------------|-------------------|------------------|--------|
| GET    | /health        | `health_check`    | `health_check`   | Health |
| GET    | /health/live   | `liveness_probe`  | `health_live`    | Health |
| GET    | /health/ready  | `readiness_probe` | `health_ready`   | Health |

## Scrape router (`src/routers/scrape.py`, prefix `/api/scrape`)

| Method | Path             | Function       | operation_id  | Tag    |
|--------|------------------|----------------|---------------|--------|
| POST   | /api/scrape/url  | `scrape_url`   | `scrape_url`  | Scrape |
| POST   | /api/scrape/bulk | `scrape_bulk`  | `scrape_bulk` | Scrape |
