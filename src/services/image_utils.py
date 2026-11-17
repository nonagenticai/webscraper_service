"""Shared image download, enrichment, and utility functions.

Used by all scraper backends (Playwright, httpx, Scrapy) to avoid
code duplication for image downloading, data URI decoding, and
dimension extraction.
"""

import asyncio
import base64
import logging
import mimetypes
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from config import get_settings
from services.url_guard import make_outbound_guard_hook

logger = logging.getLogger(__name__)

# Limits for image downloading
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB per image
MAX_IMAGES_TO_DOWNLOAD = 100
IMAGE_DOWNLOAD_TIMEOUT = 30.0

# Hard ceiling on the WHOLE enrichment phase. With 100 images, a 30s per-image
# timeout and max_connections=10, the worst case is ~300s — far past any
# caller's request timeout. Callers (httpx/scrapy paths) wrap enrichment in
# asyncio.wait_for with this cap and fall back to un-enriched images on timeout
# so a slow image host can't hang an otherwise-good scrape. (The Playwright path
# is already bounded by its own overall_budget.)
IMAGE_ENRICH_OVERALL_TIMEOUT = 90.0


def decode_data_uri(src: str) -> tuple[bytes, str] | None:
    """Decode a data: URI into raw bytes and MIME type.

    Args:
        src: A data URI string like ``data:image/png;base64,iVBOR...``

    Returns:
        Tuple of (image_bytes, mime_type), or None on failure.
    """
    if not src.startswith("data:"):
        return None
    try:
        header, b64_data = src.split(",", 1)
        mime = header.split(":")[1].split(";")[0]  # e.g. "image/png"
        if not mime.startswith("image/"):
            return None
        img_bytes = base64.b64decode(b64_data)
        if len(img_bytes) > MAX_IMAGE_BYTES:
            return None
        return img_bytes, mime
    except Exception:
        return None


def extract_dimensions(img_bytes: bytes) -> tuple[int, int] | None:
    """Extract pixel dimensions from raw image bytes using PIL.

    Args:
        img_bytes: Raw image file bytes.

    Returns:
        Tuple of (width, height) in pixels, or None on failure.
    """
    try:
        with Image.open(BytesIO(img_bytes)) as im:
            return im.size  # (width, height)
    except Exception:
        return None


async def download_image(
    client: httpx.AsyncClient,
    src: str,
    max_bytes: int = MAX_IMAGE_BYTES,
    timeout: float = IMAGE_DOWNLOAD_TIMEOUT,
) -> tuple[bytes, str, str] | None:
    """Download a single image by URL or decode a data URI.

    Args:
        client: httpx async client for HTTP downloads.
        src: Image URL (http/https) or data URI.
        max_bytes: Maximum allowed image size.
        timeout: Download timeout in seconds.

    Returns:
        Tuple of (bytes, mime_type, src) or None on failure.
    """
    if not src:
        return None

    # Handle data: URIs inline
    if src.startswith("data:"):
        result = decode_data_uri(src)
        if result:
            img_bytes, mime = result
            return img_bytes, mime, src
        return None

    # Skip non-HTTP URLs
    if not src.startswith(("http://", "https://")):
        return None

    try:
        # Stream so an oversized/malicious image is bounded — never buffer the
        # whole body just to discover it's over max_bytes. Stop reading once we
        # exceed the cap (read one extra byte to detect "> max_bytes").
        async with client.stream("GET", src, timeout=timeout) as resp:
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if not ct.startswith("image/"):
                return None
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    return None
                chunks.append(chunk)
        body = b"".join(chunks)
        mime = mimetypes.guess_type(src)[0] or ct.split(";")[0]
        return body, mime, src
    except Exception:
        return None


async def enrich_images_with_content(
    images: list[dict[str, Any]],
    max_count: int = MAX_IMAGES_TO_DOWNLOAD,
    max_bytes: int = MAX_IMAGE_BYTES,
    timeout: float = IMAGE_DOWNLOAD_TIMEOUT,
) -> list[dict[str, Any]]:
    """Download image bytes for discovered images and add content_base64.

    Also fills in width/height via PIL when not already set.

    Args:
        images: List of image dicts with at least a ``src`` key.
        max_count: Maximum number of images to download.
        max_bytes: Maximum size per image.
        timeout: Per-image download timeout.

    Returns:
        Enriched copy of the images list with content_base64, mime_type,
        byte_size, and (if missing) width/height fields added.
    """
    # Separate downloadable (http/https) and data URI images
    downloadable: list[dict[str, Any]] = []
    data_uri_images: list[dict[str, Any]] = []
    for img in images:
        src = img.get("src", "")
        if src.startswith(("http://", "https://")):
            downloadable.append(img)
        elif src.startswith("data:"):
            data_uri_images.append(img)

    # ``max_count`` is a hard ceiling on the COMBINED set — cap downloadable
    # first, then fill the remaining budget with data URIs. (A page with
    # thousands of inline `data:` images would otherwise decode up to 5MB each
    # without bound, and capping the two lists independently let a page enrich
    # up to 2x ``max_count`` total.)
    downloadable = downloadable[:max_count]
    data_uri_images = data_uri_images[: max(0, max_count - len(downloadable))]

    # Download HTTP images concurrently
    src_to_data: dict[str, tuple[bytes, str]] = {}
    if downloadable:
        # SSRF: image src comes from the (untrusted) scraped page and the client
        # follows redirects — guard every hop so a `<img src=internal>` (or one
        # that 302s there) can't exfiltrate internal content into the response.
        settings = get_settings()
        event_hooks = (
            {"request": [make_outbound_guard_hook(settings.ssrf_allowed_hosts)]}
            if settings.ssrf_protection_enabled
            else {}
        )
        async with httpx.AsyncClient(
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10),
            verify=True,
            event_hooks=event_hooks,
            # Optional ops-level egress proxy (None → direct), matching every other
            # user-content fetch in the service (httpx/curl_cffi/playwright/scrapy/
            # scraper_manager). Image src is scraped, untrusted content, so it belongs
            # on the same egress path. The SSRF hook still validates the target host
            # regardless of the proxy.
            proxy=settings.outbound_proxy or None,
        ) as client:
            tasks = [download_image(client, img["src"], max_bytes, timeout) for img in downloadable]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple):
                img_bytes, mime, src = result
                src_to_data[src] = (img_bytes, mime)
            elif isinstance(result, Exception):
                logger.debug(f"Image download failed: {result}")

    # Decode data URI images. base64 decode of up to 5MB is CPU work — offload
    # it so a batch of inline images can't block the event loop.
    for img in data_uri_images:
        decoded = await asyncio.to_thread(decode_data_uri, img["src"])
        if decoded:
            img_bytes, mime = decoded
            src_to_data[img["src"]] = (img_bytes, mime)

    # Build enriched list
    enriched: list[dict[str, Any]] = []
    for img in images:
        img_copy = dict(img)
        data = src_to_data.get(img.get("src", ""))
        if data:
            img_bytes, mime = data
            img_copy["content_base64"] = base64.b64encode(img_bytes).decode("utf-8")
            img_copy["mime_type"] = mime
            img_copy["byte_size"] = len(img_bytes)
            # Fill in dimensions from PIL only when genuinely missing. Use an
            # explicit None check, not truthiness, so a legitimate 0 dimension
            # isn't treated as missing and clobbered.
            if img_copy.get("width") is None or img_copy.get("height") is None:
                dims = extract_dimensions(img_bytes)
                if dims:
                    img_copy["width"] = dims[0]
                    img_copy["height"] = dims[1]
        enriched.append(img_copy)
    return enriched


__all__ = [
    "MAX_IMAGE_BYTES",
    "MAX_IMAGES_TO_DOWNLOAD",
    "IMAGE_DOWNLOAD_TIMEOUT",
    "decode_data_uri",
    "extract_dimensions",
    "download_image",
    "enrich_images_with_content",
]
