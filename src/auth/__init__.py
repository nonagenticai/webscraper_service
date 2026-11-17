"""Authentication module - License-token auth."""

from .dependencies import get_current_user
from .middleware import AuthMiddleware

__all__ = ["AuthMiddleware", "get_current_user"]
