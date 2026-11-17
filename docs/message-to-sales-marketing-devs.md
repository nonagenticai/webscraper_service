# Message for the sales / marketing developers

> **Subject: webscraper_service already returns clean Markdown + structured data — and three new capabilities are landing**
>
> Hi — you both call `webscraper.example.com:9119/api/scrape/url`
> directly (not via the kag gateway), so you're the two consumers I can't see
> from the rest of the tree. I swept the response contract and grepped both your
> repos, so this is specific rather than a broadcast.
>
> **marketing:** you already mine `structured_data` —
> `src/services/recon/understanding/owned_cadence.py` walks the webscraper's
> JSON-LD/OpenGraph for publish dates to
> measure a prospect's blog cadence. Good: that's exactly what it's for, and it
> stays. **Neither of you reads `markdown_content`**, and I think you want it.
>
> ## The field you're probably re-deriving by hand
>
> **`markdown_content`** — the page's main content as Markdown, with headings,
> lists and tables preserved (trafilatura extraction behind a recall guard, so
> it doesn't over-prune directory/table pages). If you feed `text_content` into
> an LLM or a chunker, this is strictly better input: flat text throws away the
> structure a model uses to tell a heading from a table cell. It's `null` only
> when extraction genuinely found nothing, so `markdown_content or text_content`
> is the right read.
>
> **`structured_data`** (sales_app — marketing already uses this) — the page's own
> embedded JSON-LD / microdata / OpenGraph / microformats, parsed out: schema.org
> `Organization`, `Product`, `Article` — author, publish date, price, address,
> social handles, company details. For prospecting and technographic signals this
> is publisher-declared first-party data, far more reliable than regexing the
> rendered text, and it costs nothing extra — it's already in the response you're
> getting. marketing's `_iter_structured_dates` is a good worked example.
>
> ## Three new capabilities (PR #49 — merged and deployed 2026-07-19)
>
> These were written up before PR #49 landed, so an earlier copy of this note
> said "just merged" while the image still had none of them. They are live now:
> `https://webscraper.example.com/openapi.json` is the check — if `ScrapeRequest`
> lists `link_query` and `LinkInfo` lists `score`, the rolled image has them.
>
> **1. Relevance-scored links.** Every entry in `links[]` now carries a
> `score` in `[0,1]`. Pass `link_query` on the request and the score blends in
> topical relevance from the anchor text and URL slug; add
> `sort_links_by_score: true` to get them ranked.
>
> ```json
> { "url": "https://acme.com", "link_query": "pricing careers", "sort_links_by_score": true }
> ```
>
> So "find this company's pricing/careers/about pages" becomes one call instead
> of a scrape-then-guess loop. With `follow_links` on `/bulk` the same score
> orders the crawl frontier best-first, so a depth-limited crawl spends its
> budget on relevant pages instead of nav and footer chrome.
>
> **2. Declarative CSS extraction.** Send an `extraction_schema` and get
> `extracted_data` — structured rows, no HTML post-parsing on your side:
>
> ```json
> {
>   "url": "https://directory.example/vendors",
>   "extraction_schema": {
>     "baseSelector": "article.card",
>     "fields": [
>       { "name": "company", "selector": "h2.name", "type": "text" },
>       { "name": "city",    "selector": ".city",   "type": "text", "default": "unknown" },
>       { "name": "website", "selector": "a.site",  "type": "attribute", "attribute": "href" },
>       { "name": "tags",    "selector": "ul.tags li", "type": "list",
>         "fields": [{ "name": "label", "type": "text" }] }
>     ]
>   }
> }
> ```
>
> Nested objects and lists are supported. It runs against the page HTML, so it
> works with every backend. It's wire-compatible with crawl4ai's
> `JsonCssExtractionStrategy` dialect, so an existing crawl4ai schema pastes in
> verbatim — except `computed` and `regex` fields, which we deliberately don't
> support (they'd mean eval'ing or regexing caller input server-side). Those
> return a 422 rather than silently doing nothing.
>
> **3. Infinite scroll + stealth.** `virtual_scroll` scrolls the page (or a
> container) before extraction, so infinite feeds return more than the first
> viewport — including feeds that recycle DOM nodes, which are accumulated and
> merged so nothing is lost. `stealth` applies browser fingerprint patches; it's
> also applied automatically when a site has already blocked our Chrome-TLS
> impersonation.
>
> ## Compatibility note
>
> All of the above is **additive**. `links[]` keeps its exact
> `{text, url, rel, type}` shape and only gains a `score` key; `extracted_data`
> is a new optional field. I checked both your clients parse the response with
> `.get()` rather than a strict model, so nothing here can break you. If you ever
> add `extra="forbid"` to a webscraper response model, tell me first.
>
> ## What I need from you
>
> 1. **Would `markdown_content` improve anything you feed an LLM?** It's a
>    one-line change (`markdown_content or text_content`) and I'd like to know if
>    it's worth me hardening further.
> 2. **Which of the three new capabilities would you actually use?** That ranks
>    what I do next — particularly whether `link_query` beats whatever page-picking
>    heuristic you have now. If none of them fit, tell me that too; I'd rather know
>    than keep building.
> 3. **sales_app:** anything you'd want that isn't listed here? You're the consumer
>    I know least about.
>
> Full docs are in the service README; the response contract is in
> `https://webscraper.example.com/openapi.json`.
