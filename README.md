# Webscraper Service

> **Retired.** This service has been absorbed into the Rust tool suite at
> [`nonagenticai/conveyor`](https://github.com/nonagenticai/conveyor) (`apps/research`), which serves
> the same `/api/scrape/url` and `/api/scrape/bulk` operations. This repository is published as
> **reference material** for anyone reading the original implementation: it is the source of the
> retired service with private infrastructure references removed, and it carries **no git history**.
> It is not maintained and is not deployed anywhere.

Web scraping API with four backends: Playwright (JS-rendered pages), curl_cffi (Chrome TLS/JA3 impersonation for anti-bot reach), Scrapy (bulk crawling), and httpx+BeautifulSoup (fast static pages). Auto mode selects the best backend per URL. Output is LLM/RAG-ready (Markdown + parsed structured data). Port 9119.

## What Webscraper Service Does

When an upstream service or n8n workflow needs to extract content from a web page, it sends a POST request to this service with a URL and optional configuration. The service accepts single-URL and bulk scraping requests, returning structured content including raw HTML, extracted text, **Markdown (`markdown_content`)**, **embedded structured data (`structured_data` — JSON-LD/microdata/OpenGraph/microformats)**, metadata, links, images, and optional screenshots. Callers never need to decide which scraping engine to use because auto mode handles backend selection transparently.

Before scraping, the service runs a backend selection algorithm in auto mode. It issues a HEAD request to the target URL and inspects the response: 2xx/3xx responses route to the fast httpx+BeautifulSoup path, **403/429 responses (a network-layer block) route to curl_cffi (Chrome impersonation) first** and escalate to Playwright only if curl_cffi also fails, and timeouts or 5xx errors trigger Playwright. If the request specifies link-following or depth greater than 1, Scrapy handles the crawl instead. If an httpx, curl_cffi, or Scrapy scrape fails, the service falls back to Playwright automatically.

The Playwright backend launches a headless Chromium instance that renders JavaScript, waits for selectors, captures screenshots, and extracts content from fully rendered pages. The httpx backend uses BeautifulSoup with lxml for fast static parsing, and integrates trafilatura for article text + Markdown extraction. The curl_cffi backend impersonates a real Chrome TLS/JA3 + HTTP/2 fingerprint to clear anti-bot blocks that reject plain HTTP clients, reusing the httpx parsing path (no JavaScript). The Scrapy backend uses CrawlerRunner for async-compatible crawling with robots.txt compliance, retry middleware, and configurable depth control.

The service is stateless (no database). All scraper instances are managed by a singleton ScraperManager with async locking to prevent race conditions during backend creation. Playwright browser instances are reused across requests and cleaned up on shutdown. An optional short-TTL in-memory result cache returns repeat anonymous scrapes without re-fetching. Bulk scrapes process URLs sequentially to avoid overwhelming target servers. Concurrent scraping is capped at 16 operations by default to prevent resource exhaustion.

## Features

- **Auto backend selection** — issues a HEAD probe and routes each URL to the optimal scraper (httpx, Playwright, or Scrapy) so callers do not need to know which backend fits
- **Playwright JS rendering** — launches headless Chromium to scrape JavaScript-heavy SPAs and dynamic pages that return empty HTML to simple HTTP clients
- **Httpx fast path** — uses httpx with BeautifulSoup and lxml for static pages, avoiding browser overhead and completing in a fraction of the time
- **curl_cffi impersonation** — a TLS/JA3 Chrome-fingerprint backend that clears network-layer anti-bot blocks (403/429) which plain httpx triggers, far more cheaply than a browser; AUTO tries it before escalating to Playwright. No JavaScript
- **Outbound proxy** — an optional ops-level egress proxy (`OUTBOUND_PROXY`) is wired into every backend (httpx, Playwright, Scrapy, curl_cffi); the SSRF guard still validates the target host regardless of the proxy
- **Scrapy crawling** — handles multi-page crawls with link following, depth control, and robots.txt compliance for bulk data extraction
- **Automatic fallback** — retries failed httpx/Scrapy scrapes with Playwright so transient failures or unexpected JS-dependent pages still succeed
- **LLM/RAG-ready output** — returns the main content as Markdown (`markdown_content`) alongside the plain `text_content`, so downstream RAG pipelines get clean, chunkable input without re-deriving it
- **Relevance-scored links** — every entry in `links[]` carries a `score` in [0, 1] from link-quality signals (anchor-text quality, path depth, query noise, same-domain, nofollow, non-HTML target). Pass `link_query` (e.g. `"pricing plans"`) to blend in topical relevance, and `sort_links_by_score` to get them ranked. With `follow_links` the same score orders the Scrapy crawl frontier best-first, so a depth-limited crawl spends its budget on relevant pages instead of nav chrome
- **Declarative CSS extraction** — send an `extraction_schema` (CSS selectors, nested/list fields) and get `extracted_data`: structured rows out of directory listings, results tables, pricing grids and team pages, with no HTML post-parsing on the caller's side. Wire-compatible with crawl4ai's `JsonCssExtractionStrategy` dialect. Runs against the page HTML, so every backend supports it. Selectors are compiled when the request is parsed, so an invalid one is a 422 before anything is scraped rather than an empty result afterwards; hitting an output cap sets `metadata.extraction_truncated`
- **Infinite-scroll handling** — `virtual_scroll` scrolls the page (or a container) before extraction so lazily-loaded feeds yield more than the first viewport. Virtualized lists that *recycle* DOM nodes are accumulated across steps (keyed on each element's stable identity, so re-rendered markup is not double-counted) and merged back into the DOM, so all downstream output (text, markdown, links, images, extracted_data) sees the whole feed rather than the last window. Accumulation requires `container_selector` — without one there is nothing to accumulate from. Scrolling shares the scrape's deadline and reports why it stopped in `metadata.virtual_scroll.stop_reason` (`step_cap` / `exhausted` / `budget` / `scroll_failed` / `capture_cap`), so a truncated feed is never mistaken for a complete one
- **Stealth browser tier** — `stealth` applies browser-level fingerprint patches (`navigator.webdriver` and friends) to the Playwright context. Applied automatically when AUTO escalates to Playwright after curl_cffi was already blocked — the strongest hostile-site signal the router has. `stealth` and `virtual_scroll` require a JavaScript-capable backend: pairing either with an explicit `httpx`/`curl_cffi`/`scrapy` type, or with `follow_links` (which crawls via Scrapy), is rejected with a 422 instead of being silently dropped
- **Structured-data extraction** — parses embedded JSON-LD, microdata, OpenGraph and microformats into `structured_data` (schema.org author, publish date, Article/Product fields) instead of leaving them buried in raw tags
- **Screenshot capture** — returns base64-encoded page screenshots on demand for visual verification or archival
- **Image downloading** — optionally downloads and enriches image metadata (dimensions, MIME type, base64 content) inline in the response
- **Result caching** — an optional short-TTL in-memory cache returns a repeat anonymous scrape without re-fetching (`X-Cache: HIT/MISS/BYPASS`, per-request `no_cache` opt-out, `CACHE_TTL_SECONDS=0` disables); authed scrapes are never cached
- **Prometheus metrics** — `GET /metrics` exposes scrape counts/durations by backend and cache hit/miss in Prometheus exposition format (auth-exempt, like `/health`)
- **Bulk scraping** — accepts lists of URLs in a single request with per-URL error isolation and summary statistics
- **Form-based auth** — resolves a `credential_id` against the secrets service and performs login (fill form, submit, wait) before content extraction; returns Playwright `storage_state` so subsequent requests can reuse the session without re-logging-in
- **License token auth** — validates license tokens via introspection endpoint with TTL cache, trusted-network bypass for internal cluster traffic
- **Concurrent scraping limit** — caps simultaneous operations at 16 (configurable) to prevent resource exhaustion on the host

## Harvested from crawl4ai (not a dependency)

Three capabilities are adapted from [crawl4ai](https://github.com/unclecode/crawl4ai)
(Apache-2.0, see `THIRD_PARTY_NOTICES.md`): link scoring, the declarative CSS
extraction dialect, and virtualized-scroll capture.

crawl4ai is deliberately **not** installed. It hard-pins `unclecode-litellm==1.81.13`
— a personal fork of litellm — as a mandatory top-level dependency, which would ship
an LLM client into a scraper, contradict the platform rule that all LLM traffic goes
through the workflow manager, and add an unpatchable pin under the `pip-audit` gate.
Upstream also classifies only up to Python 3.13 while this service requires 3.14, and
it ships its own browser manager, cache and config that would duplicate ours.

## Authentication

License token authentication via `X-Tenant-Id: Bearer <token>` header. Tokens are validated against the license introspection service. Trusted-network clients (RFC 1918 CIDRs) bypass token requirement.

| Priority | Method | Details |
|----------|--------|---------|
| 1 | License token | `X-Tenant-Id: Bearer <token>` introspected via license-authz (5 min TTL cache) |
| 2 | Trusted network | RFC 1918 CIDRs bypass auth; tenant defaults to `DEFAULT_TENANT_ID` |

Health endpoints (`/`, `/health`, `/health/*`, `/docs`, `/redoc`) and `/metrics` are always exempt (everything outside `/api` is non-protected).

## Quick Start

```bash
uv sync
uv run playwright install chromium
PYTHONPATH=src uv run uvicorn main:app --host 0.0.0.0 --port 9119 --app-dir src
uv run pytest tests/ -v --cov=src
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

### Scrape `/api/scrape`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/url` | Scrape a single URL with backend selection |
| POST | `/bulk` | Scrape multiple URLs with per-URL error isolation |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Detailed health with Playwright and browser status |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe (checks Playwright availability) |

### Metrics

| Method | Path | Description |
|--------|------|-------------|
| GET | `/metrics` | Prometheus exposition: `webscraper_scrapes_total{backend,status}`, `webscraper_scrape_duration_seconds{backend}`, `webscraper_cache_lookups_total{result}` (auth-exempt) |

## Configuration

All configuration via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Listen host |
| `PORT` | `9119` | Listen port |
| `LOG_LEVEL` | `INFO` | Log level |
| `ENVIRONMENT` | `development` | Deployment environment |
| `DEFAULT_TENANT_ID` | `n8n-dev` | Tenant ID for trusted-network requests |
| `LICENSE_INTROSPECT_URL` | _(empty)_ | License validation endpoint. **Required in production** — start-up fails if unset; empty means no token can be validated. |
| `TRUSTED_CIDRS` | `127.0.0.1/32,::1/128,192.168.0.0/16,172.16.0.0/12,10.0.0.0/8` | CIDRs that bypass token auth |
| `CORS_ALLOW_ORIGINS` | `http://localhost:8921,http://localhost:3000,http://127.0.0.1:3000` | Allowed CORS origins (comma-separated) |
| `PLAYWRIGHT_HEADLESS` | `true` | Headless browser mode |
| `PLAYWRIGHT_BROWSERS_PATH` | _(unset)_ | Custom path for Playwright browser binaries |
| `SCRAPING_DEFAULT_DELAY` | `1.0` | Delay between requests (seconds) |
| `SCRAPING_MAX_CONCURRENT` | `16` | Max concurrent scraping operations |
| `SCRAPING_USER_AGENT` | `WebScraper-Service/1.0 (+https://example.com/bot)` | User agent string |
| `OUTBOUND_PROXY` | _(unset)_ | Optional egress proxy applied to all backends (httpx, Playwright, Scrapy, curl_cffi); SSRF guard still validates the target host |
| `CACHE_TTL_SECONDS` | `300` | Result-cache TTL; `0` disables the cache entirely |
| `CACHE_MAX_SIZE` | `128` | Max cached responses (size-bounded for memory safety; entries hold full responses) |
| `SECRETS_SERVICE_URL` | _(empty)_ | Secrets service base URL for `auth_config.credential_id` lookups. **Required in production**; empty disables authed scraping. |
| `SECRETS_SERVICE_TIMEOUT` | `10.0` | Timeout (seconds) for secrets-service calls |

## Authenticated scraping (auth_config)

For login-gated sites, add an `auth_config` block referencing a credential stored in the secrets service. Credentials never travel inline on the wire — the webscraper fetches them using the caller's forwarded license token, and the secrets service enforces tenant ownership.

```bash
# 1. Store the credential once against the secrets service (done by the tenant or their admin).
# 2. Ask webscraper to use it:
curl -sS -X POST https://api.example.com/kag/api/scrape/url \
  -H 'X-Tenant-Id: Bearer <license-token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://dev.specralabs.com",
    "auth_config": {
      "type": "form",
      "credential_id": "specra-dev-user1"
    }
  }'
```

### Auth types

| Type | Required secret fields | Behaviour |
|------|------------------------|-----------|
| `form` | `username`, `password` | Fill the login form (default or custom selectors), submit, wait for post-login state, re-navigate to the originally requested URL |
| `storage_state` | `storage_state` | Inject a previously captured Playwright `storageState` into the browser context — no form interaction |
| `bearer` | `auth_token` | Sets `Authorization: Bearer <token>` on outgoing requests |
| `basic` | `username`, `password` | Sets HTTP Basic auth credentials on the browser context |

### Storage-state reuse

Successful auth returns the captured `storage_state` in the response. Clients can store this (or store it as a `storage_state` secret) and pass `auth_config.type: "storage_state"` on subsequent requests to avoid re-logging-in.

### Bulk behaviour

Bulk scrapes with form auth perform login **once** on the first URL, then reuse the captured `storage_state` for the remaining URLs. If the first-URL login fails, the remaining URLs short-circuit to `FAILED` rather than hammering the target with N failed logins.

### Notes

- `auth_config.type` `form` and `storage_state` force the Playwright backend regardless of the requested `scraper_type`.
- CSS selectors default to common patterns (`input[name="email"]`, `input[type="password"]`, `button[type="submit"]`, etc.). Override per-site via `username_selector` / `password_selector` / `submit_selector`.
- Post-login wait defaults to "URL no longer contains `login`"; override with `wait_for_url` (glob/regex) or `wait_for_selector`.

## Architecture

```
src/
├── main.py                        # FastAPI app, middleware, CORS, lifespan, OpenAPI
├── config.py                      # Pydantic settings (host, port, Playwright, scraping)
├── auth/
│   ├── __init__.py                # Auth module exports (KeycloakAuthMiddleware alias)
│   └── middleware.py              # License token auth middleware with TTL cache
├── routers/
│   ├── __init__.py                # Router module exports
│   ├── health.py                  # Health, liveness, readiness probes
│   ├── metrics.py                 # Prometheus /metrics endpoint
│   └── scrape.py                  # Single URL and bulk scrape endpoints
├── services/
│   ├── __init__.py                # Service module exports
│   ├── base_scraper.py            # Abstract BaseScraper, ScrapeResult, ScrapeConfig, enums
│   ├── scraper_manager.py         # Singleton manager with auto-selection and fallback
│   ├── playwright_scraper.py      # Chromium-based JS rendering scraper
│   ├── httpx_scraper.py           # httpx + BeautifulSoup fast static scraper
│   ├── curl_cffi_scraper.py       # Chrome TLS/JA3 impersonation scraper (anti-bot reach)
│   ├── scrapy_scraper.py          # Scrapy CrawlerRunner for link-following crawls
│   ├── content_extraction.py      # trafilatura text + Markdown extraction (shared)
│   ├── structured_data.py         # extruct JSON-LD/microdata/OpenGraph/microformat extraction
│   ├── link_scoring.py            # relevance scoring for discovered links (+ crawl frontier order)
│   ├── extraction_schema.py       # declarative CSS extraction → extracted_data
│   ├── virtual_scroll.py          # infinite-scroll + virtualized-list capture (Playwright)
│   ├── cache.py                   # In-memory TTL result cache
│   ├── metrics.py                 # Prometheus metric objects + recorders
│   ├── url_guard.py               # SSRF guard (per-hop allowlist enforcement)
│   └── image_utils.py             # Shared image download and enrichment utilities
├── schemas/
│   ├── __init__.py                # Schema module exports
│   ├── scrape.py                  # ScrapeRequest, ScrapeResponse, BulkScrapeRequest DTOs
│   └── health.py                  # HealthResponse DTO
└── custom_types/
    ├── __init__.py                # Custom types exports
    ├── responses.py               # Reusable OpenAPI response definitions
    └── constraint_types.py        # SafeString validated type annotations
```

## Scrapers

| Scraper | What It Does |
|---------|-------------|
| PlaywrightScraper | Renders JavaScript via headless Chromium, captures screenshots, handles SPAs and dynamic content |
| HttpxScraper | Fetches static HTML via httpx, parses with BeautifulSoup+lxml, integrates trafilatura for article text + Markdown extraction |
| CurlCffiScraper | Static-HTML fetch impersonating a Chrome TLS/JA3 fingerprint to clear network-layer anti-bot blocks; reuses HttpxScraper's parsing; no JavaScript |
| ScrapyScraper | Crawls multiple pages via Scrapy CrawlerRunner with link-following, depth control, and robots.txt compliance |
| ScraperManager | Singleton orchestrator that selects backends via HEAD probing, manages lifecycle, and handles fallback chains |

## Design Principles

1. **Auto-selection with fallback** — HEAD-probe routing picks the lightest backend that works, with automatic Playwright fallback when simpler scrapers fail
2. **Stateless** — no database; results are returned inline in the response. The optional result cache is in-memory and ephemeral (lost on restart), so the service stays horizontally restartable
3. **Sequential bulk processing** — bulk scrapes run URLs one at a time to avoid overwhelming target servers and triggering rate limits
4. **Backend isolation** — each scraper implements a common interface (BaseScraper) so new backends can be added without touching routing logic

## Testing

```bash
uv run pytest tests/ -v --cov=src          # All tests with coverage
uv run ruff check src/ tests/              # Lint
uv run ruff format --check src/ tests/     # Format check
```

Coverage threshold: 85% (enforced in pyproject.toml).

The `tests/` directory holds the suite (pytest, pytest-asyncio, pytest-cov, configured in pyproject.toml); run it with `uv run pytest`.

## CI/CD

GitHub Actions workflow: `ruff + unit tests -> build amd64 -> smoke test -> push`

Image pushed to `registry.example.com/webscraper_service:sha-<commit>`, promoted to `:latest` on success. Flux auto-deploys to K3s.

## Docker

```bash
docker build -t webscraper-service .
docker run -p 9119:9119 webscraper-service
docker run -p 9119:9119 -e LOG_LEVEL=DEBUG -e PLAYWRIGHT_HEADLESS=true webscraper-service
```

## Dependencies

- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **Pydantic** + **pydantic-settings** — settings and schema validation
- **httpx** — async HTTP client for httpx scraper and license introspection
- **Playwright** — headless Chromium browser automation
- **curl_cffi** — Chrome TLS/JA3 impersonation HTTP client (anti-bot reach)
- **Scrapy** — crawling framework for bulk/deep scraping
- **BeautifulSoup4** + **lxml** — HTML parsing for httpx scraper
- **trafilatura** — article text + Markdown extraction
- **markdownify** — HTML→Markdown backstop when trafilatura returns thin
- **extruct** — JSON-LD / microdata / OpenGraph / microformat structured-data extraction
- **playwright-stealth** — browser-level fingerprint patches for the stealth tier
- **Pillow** — image dimension extraction and processing
- **cachetools** — TTL cache for license-token introspection and scrape results
- **prometheus-client** — `/metrics` instrumentation

## Tests

`uv run pytest tests/` runs the unit tests AND the `e2e` suite (in-process
full stack against a local HTTP server, no external network) — this is what CI
runs. Only `integration` tests are deselected: they need a live deployment
(`LICENSE_TOKEN`) or a real Chromium; opt in with `-m integration`. The archetype `Lint, types, tests` job runs ruff + ruff
format + ty + pytest + pip-audit + detect-secrets; the build job pushes to
`registry.example.com` with a 3-attempt retry. CI is green on `main`.
