"""Tests for shared image_utils functions."""

import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from PIL import Image

from services.image_utils import (
    decode_data_uri,
    download_image,
    enrich_images_with_content,
    extract_dimensions,
)

# 1x1 red PNG pixel (valid image)
RED_PIXEL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _make_png(width: int, height: int) -> bytes:
    """Create a minimal PNG in memory."""
    img = Image.new("RGB", (width, height), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestDecodeDataUri:
    def test_non_data_uri_returns_none(self):
        assert decode_data_uri("https://example.com/img.png") is None

    def test_non_image_mime_returns_none(self):
        b64 = base64.b64encode(b"hello").decode()
        assert decode_data_uri(f"data:text/plain;base64,{b64}") is None

    def test_too_large_returns_none(self):
        big = base64.b64encode(b"\x00" * (5 * 1024 * 1024 + 1)).decode()
        assert decode_data_uri(f"data:image/png;base64,{big}") is None

    def test_invalid_base64_returns_none(self):
        assert decode_data_uri("data:image/png;base64,!!!invalid!!!") is None

    def test_valid_data_uri(self):
        b64 = base64.b64encode(RED_PIXEL_PNG).decode()
        result = decode_data_uri(f"data:image/png;base64,{b64}")
        assert result is not None
        img_bytes, mime = result
        assert img_bytes == RED_PIXEL_PNG
        assert mime == "image/png"

    def test_empty_string(self):
        assert decode_data_uri("") is None

    def test_malformed_no_comma(self):
        assert decode_data_uri("data:image/png;base64") is None

    def test_jpeg_mime(self):
        raw = b"\xff\xd8\xff\xe0" + b"\x00" * 10
        b64 = base64.b64encode(raw).decode()
        result = decode_data_uri(f"data:image/jpeg;base64,{b64}")
        assert result is not None
        assert result[1] == "image/jpeg"


class TestExtractDimensions:
    def test_valid_png(self):
        dims = extract_dimensions(RED_PIXEL_PNG)
        assert dims == (1, 1)

    def test_larger_png(self):
        png_bytes = _make_png(640, 480)
        dims = extract_dimensions(png_bytes)
        assert dims == (640, 480)

    def test_invalid_bytes(self):
        assert extract_dimensions(b"not an image") is None

    def test_empty_bytes(self):
        assert extract_dimensions(b"") is None


def _stream_client(*, content_type, body=b"", raise_exc=None):
    """Build an AsyncMock httpx client whose ``.stream()`` yields a streaming
    response — mirrors how ``download_image`` now reads bounded chunks instead
    of buffering ``resp.content`` whole."""
    resp = MagicMock()
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock()

    async def _aiter():
        for i in range(0, len(body), 32):
            yield body[i : i + 32]

    resp.aiter_bytes = lambda: _aiter()

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)

    client = AsyncMock(spec=httpx.AsyncClient)
    if raise_exc is not None:
        client.stream = MagicMock(side_effect=raise_exc)
    else:
        client.stream = MagicMock(return_value=cm)
    return client


class TestDownloadImage:
    @pytest.mark.asyncio
    async def test_http_success(self):
        fake_bytes = b"\x89PNG" + b"\x00" * 50
        client = _stream_client(content_type="image/png", body=fake_bytes)

        result = await download_image(client, "https://example.com/img.png")
        assert result is not None
        body, mime, src = result
        assert body == fake_bytes
        assert src == "https://example.com/img.png"

    @pytest.mark.asyncio
    async def test_data_uri_decoded_no_http_call(self):
        raw = b"\x89PNG" + b"\x00" * 20
        b64 = base64.b64encode(raw).decode()
        data_uri = f"data:image/png;base64,{b64}"

        client = AsyncMock(spec=httpx.AsyncClient)
        result = await download_image(client, data_uri)
        assert result is not None
        body, mime, src = result
        assert body == raw
        assert mime == "image/png"
        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_http_url_returns_none(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        assert await download_image(client, "ftp://example.com/img.png") is None

    @pytest.mark.asyncio
    async def test_relative_url_returns_none(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        assert await download_image(client, "/path/img.png") is None

    @pytest.mark.asyncio
    async def test_empty_src_returns_none(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        assert await download_image(client, "") is None

    @pytest.mark.asyncio
    async def test_data_uri_non_image_returns_none(self):
        client = AsyncMock(spec=httpx.AsyncClient)
        result = await download_image(client, "data:text/plain;base64,aGVsbG8=")
        assert result is None

    @pytest.mark.asyncio
    async def test_non_image_content_type_returns_none(self):
        """HTTP response with non-image content-type returns None."""
        client = _stream_client(content_type="text/html", body=b"<html>not an image</html>")

        result = await download_image(client, "https://example.com/page.html")
        assert result is None

    @pytest.mark.asyncio
    async def test_body_too_large_returns_none(self):
        """A body larger than max_bytes is rejected mid-stream (never fully buffered)."""
        client = _stream_client(content_type="image/png", body=b"\x89PNG" + b"\x00" * 200)

        # Set max_bytes very low to trigger the streaming cap
        result = await download_image(client, "https://example.com/big.png", max_bytes=100)
        assert result is None

    @pytest.mark.asyncio
    async def test_http_exception_returns_none(self):
        """HTTP exception during download returns None."""
        client = _stream_client(
            content_type="image/png", raise_exc=httpx.ConnectError("Connection refused")
        )

        result = await download_image(client, "https://example.com/img.png")
        assert result is None


class TestEnrichImagesWithContent:
    @pytest.mark.asyncio
    async def test_data_uri_images_enriched(self):
        """Data URI images are decoded and enriched without HTTP calls."""
        b64 = base64.b64encode(RED_PIXEL_PNG).decode()
        data_uri = f"data:image/png;base64,{b64}"
        images = [{"src": data_uri, "alt": "icon"}]

        result = await enrich_images_with_content(images)

        assert len(result) == 1
        assert result[0]["content_base64"] == b64
        assert result[0]["mime_type"] == "image/png"
        assert result[0]["byte_size"] == len(RED_PIXEL_PNG)
        # PIL should fill dimensions from the 1x1 PNG
        assert result[0]["width"] == 1
        assert result[0]["height"] == 1

    @pytest.mark.asyncio
    async def test_pil_fills_missing_dimensions(self):
        """Width/height filled from PIL when not already set."""
        png_bytes = _make_png(100, 50)
        images = [{"src": "https://example.com/img.png", "alt": "test"}]

        async def fake_download(client, src, max_bytes=None, timeout=None):
            return png_bytes, "image/png", src

        with patch("services.image_utils.download_image", side_effect=fake_download):
            result = await enrich_images_with_content(images)

        assert result[0]["width"] == 100
        assert result[0]["height"] == 50

    @pytest.mark.asyncio
    async def test_pil_does_not_overwrite_existing_dimensions(self):
        """Existing width/height should NOT be replaced by PIL."""
        png_bytes = _make_png(100, 50)
        images = [
            {"src": "https://example.com/img.png", "alt": "test", "width": 200, "height": 100}
        ]

        async def fake_download(client, src, max_bytes=None, timeout=None):
            return png_bytes, "image/png", src

        with patch("services.image_utils.download_image", side_effect=fake_download):
            result = await enrich_images_with_content(images)

        assert result[0]["width"] == 200
        assert result[0]["height"] == 100

    @pytest.mark.asyncio
    async def test_zero_dimensions_not_clobbered(self):
        """A legitimate 0 width/height is preserved (was treated as missing)."""
        png_bytes = _make_png(100, 50)
        images = [{"src": "https://example.com/img.png", "alt": "test", "width": 0, "height": 0}]

        async def fake_download(client, src, max_bytes=None, timeout=None):
            return png_bytes, "image/png", src

        with patch("services.image_utils.download_image", side_effect=fake_download):
            result = await enrich_images_with_content(images)

        assert result[0]["width"] == 0
        assert result[0]["height"] == 0

    @pytest.mark.asyncio
    async def test_mixed_http_data_uri_and_relative(self):
        """All source types handled correctly in one batch."""
        http_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20
        b64 = base64.b64encode(RED_PIXEL_PNG).decode()
        data_uri = f"data:image/png;base64,{b64}"

        images = [
            {"src": "https://example.com/photo.jpg", "alt": "photo"},
            {"src": data_uri, "alt": "icon"},
            {"src": "/relative.png", "alt": "skipped"},
        ]

        async def fake_download(client, src, max_bytes=None, timeout=None):
            if src.startswith("https://"):
                return http_bytes, "image/jpeg", src
            return None

        with patch("services.image_utils.download_image", side_effect=fake_download):
            result = await enrich_images_with_content(images)

        assert len(result) == 3
        assert "content_base64" in result[0]
        assert result[0]["mime_type"] == "image/jpeg"
        assert "content_base64" in result[1]
        assert result[1]["mime_type"] == "image/png"
        assert "content_base64" not in result[2]

    @pytest.mark.asyncio
    async def test_download_exception_handled_gracefully(self):
        """Download exceptions are handled per-image without crashing."""
        images = [
            {"src": "https://example.com/fail.jpg", "alt": "fail"},
            {"src": "https://example.com/ok.jpg", "alt": "ok"},
        ]

        call_count = 0

        async def fake_download(client, src, max_bytes=None, timeout=None):
            nonlocal call_count
            call_count += 1
            if "fail" in src:
                raise Exception("Download failed")
            return b"\x89PNG" + b"\x00" * 20, "image/png", src

        with patch("services.image_utils.download_image", side_effect=fake_download):
            result = await enrich_images_with_content(images)

        assert len(result) == 2
        # The failed image should not have content
        assert "content_base64" not in result[0]
        # The successful one should
        assert "content_base64" in result[1]

    @pytest.mark.asyncio
    async def test_max_count_limits_downloads(self):
        """Only max_count images are downloaded."""
        images = [{"src": f"https://example.com/img{i}.jpg", "alt": f"img{i}"} for i in range(10)]

        download_count = 0

        async def fake_download(client, src, max_bytes=None, timeout=None):
            nonlocal download_count
            download_count += 1
            return b"\x89PNG" + b"\x00" * 20, "image/png", src

        with patch("services.image_utils.download_image", side_effect=fake_download):
            result = await enrich_images_with_content(images, max_count=3)

        assert len(result) == 10
        # Only 3 should have been downloaded
        enriched = [r for r in result if "content_base64" in r]
        assert len(enriched) == 3


class TestDataUriCountCap:
    @pytest.mark.asyncio
    async def test_data_uri_images_count_capped(self):
        """Only the first max_count data-URI images are decoded (parity with http)."""
        uris = [
            "data:image/png;base64," + base64.b64encode(_make_png(2 + i, 2 + i)).decode()
            for i in range(5)
        ]
        images = [{"src": u, "alt": "", "title": ""} for u in uris]
        out = await enrich_images_with_content(images, max_count=2)
        enriched = [i for i in out if i.get("content_base64")]
        assert len(enriched) == 2
        # All 5 still returned (un-enriched ones just lack content_base64).
        assert len(out) == 5

    @pytest.mark.asyncio
    async def test_combined_http_and_data_uri_capped_to_max_count(self):
        """max_count is a HARD ceiling on the COMBINED set: data URIs fill only
        the budget left after downloadable images (previously each list was
        capped independently, letting a page enrich up to 2x max_count)."""
        http_images = [{"src": f"https://example.com/img{i}.jpg", "alt": ""} for i in range(4)]
        data_uris = [
            {
                "src": "data:image/png;base64,"
                + base64.b64encode(_make_png(2 + i, 2 + i)).decode(),
                "alt": "",
            }
            for i in range(4)
        ]
        images = http_images + data_uris

        async def fake_download(client, src, max_bytes=None, timeout=None):
            return _make_png(3, 3), "image/png", src

        with patch("services.image_utils.download_image", side_effect=fake_download):
            out = await enrich_images_with_content(images, max_count=5)

        enriched = [i for i in out if i.get("content_base64")]
        # 4 downloadable + only 1 data-URI (budget 5 - 4) = 5 total, NOT 4 + 4.
        assert len(enriched) == 5
        assert len(out) == 8


class TestOutboundProxy:
    """Image download is user-content egress and must honor settings.outbound_proxy.

    This is the sixth outbound path in the service; the other five
    (curl_cffi/httpx/playwright/scraper_manager/scrapy) already thread the proxy.
    """

    @staticmethod
    def _spy_client(monkeypatch: pytest.MonkeyPatch, captured: dict[str, str | None]) -> None:
        real_client = httpx.AsyncClient

        def _spy(*args, **kwargs):
            proxy = kwargs.get("proxy")
            captured["proxy"] = proxy if isinstance(proxy, str) else None
            return real_client(*args, **kwargs)

        monkeypatch.setattr("services.image_utils.httpx.AsyncClient", _spy)

    @staticmethod
    async def _fake_download(
        client: object, src: str, max_bytes: int | None = None, timeout: float | None = None
    ) -> tuple[bytes, str, str]:
        return RED_PIXEL_PNG, "image/png", src

    @pytest.mark.asyncio
    async def test_image_client_passes_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from config import get_settings

        monkeypatch.setattr(
            get_settings(), "outbound_proxy", "http://proxy.test:3128", raising=False
        )
        captured: dict[str, str | None] = {}
        self._spy_client(monkeypatch, captured)

        with patch("services.image_utils.download_image", side_effect=self._fake_download):
            await enrich_images_with_content([{"src": "https://example.com/img.png", "alt": ""}])

        assert captured["proxy"] == "http://proxy.test:3128"

    @pytest.mark.asyncio
    async def test_image_client_no_proxy_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from config import get_settings

        monkeypatch.setattr(get_settings(), "outbound_proxy", None, raising=False)
        captured: dict[str, str | None] = {}
        self._spy_client(monkeypatch, captured)

        with patch("services.image_utils.download_image", side_effect=self._fake_download):
            await enrich_images_with_content([{"src": "https://example.com/img.png", "alt": ""}])

        assert captured["proxy"] is None
