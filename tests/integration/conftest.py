import os

import httpx
import pytest

pytestmark = pytest.mark.integration

WEBSCRAPER_URL = os.environ.get("WEBSCRAPER_URL", "https://webscraper.example.com")
# No default. A live production licence token used to sit here as the fallback,
# which put a real credential in source control AND made the suite look
# authenticated where it was not. Supply LICENSE_TOKEN in the environment.
LICENSE_TOKEN = os.environ.get("LICENSE_TOKEN", "")


@pytest.fixture(scope="session")
def scraper_client():
    if not LICENSE_TOKEN:
        pytest.skip("LICENSE_TOKEN is not set; integration tests need a real licence token")

    with httpx.Client(
        base_url=WEBSCRAPER_URL,
        # The fleet carries the licence token in X-Tenant-Id, NOT Authorization
        # (see src/auth/middleware.py::_extract_license_token). Sent as
        # Authorization the service never saw a token at all, and these tests
        # passed purely because the caller sits on the trusted network — so they
        # proved nothing whatsoever about authentication.
        headers={"X-Tenant-Id": f"Bearer {LICENSE_TOKEN}"},
        verify=False,
        timeout=30.0,
    ) as client:
        yield client
