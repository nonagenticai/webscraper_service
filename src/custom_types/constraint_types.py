"""Constraint types for string validation.

This module provides reusable type annotations with validation patterns
for safe string handling across the application.
"""

import re
from typing import Annotated

from pydantic import Field

SAFE_STRING_PATTERN = r"^[^\x00-\x08\x0B\x0C\x0E-\x1F\x7F]*$"

# The complement of SAFE_STRING_PATTERN: the C0 control chars (except tab/LF/CR)
# and DEL that the pattern forbids. Scraped content (anchor text, alt text,
# titles, URLs) comes from arbitrary pages and can legitimately contain these;
# feeding them straight into a pattern-validated response model raises
# ValidationError → HTTP 500. Sanitize at the boundary instead of rejecting.
_UNSAFE_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def sanitize_safe_string(value: str) -> str:
    """Strip control chars disallowed by ``SAFE_STRING_PATTERN`` from ``value``.

    Keeps tab/newline/carriage-return; removes other C0 controls and DEL so the
    result always satisfies ``SAFE_STRING_PATTERN`` (and never log-injects into
    consumers). Non-str input is returned unchanged.
    """
    if not isinstance(value, str):
        return value
    return _UNSAFE_CONTROL_RE.sub("", value)


SafeString = Annotated[str, Field(pattern=SAFE_STRING_PATTERN)]
SafeStringShort = Annotated[str, Field(min_length=1, max_length=256, pattern=SAFE_STRING_PATTERN)]
SafeStringMedium = Annotated[str, Field(min_length=1, max_length=1000, pattern=SAFE_STRING_PATTERN)]
SafeStringLong = Annotated[str, Field(min_length=1, max_length=10000, pattern=SAFE_STRING_PATTERN)]
SafeCodeString = Annotated[str, Field(max_length=50000, pattern=SAFE_STRING_PATTERN)]
