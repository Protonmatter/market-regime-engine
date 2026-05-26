# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from fastapi import Request

log = logging.getLogger(__name__)


class _InProcessRateLimiter:
    """Small slowapi-compatible fallback for local/test deployments.

    It implements only the ``limit(spec)(handler)`` decorator surface used by
    this project. Production deployments should install the ``[security]``
    extra and set ``MRE_FI_RATE_LIMIT_ENABLED=1`` to fail closed if slowapi is
    unavailable.
    """

    uses_slowapi = False

    def __init__(self, spec: str) -> None:
        self.spec = spec
        self.max_calls, self.window_seconds = self._parse_spec(spec)
        self._buckets: dict[str, deque[float]] = {}

    @staticmethod
    def _parse_spec(spec: str) -> tuple[int, float]:
        raw = (spec or _DEFAULT_RATE_LIMIT).strip().lower()
        try:
            count_raw, unit = raw.split('/', 1)
            count = max(1, int(count_raw.strip()))
        except Exception:
            return 100, 1.0
        unit = unit.strip()
        if unit.startswith('sec') or unit.startswith('s'):
            return count, 1.0
        if unit.startswith('min') or unit.startswith('m'):
            return count, 60.0
        if unit.startswith('hour') or unit.startswith('h'):
            return count, 3600.0
        return count, 1.0

    def _key(self, request: Request | None) -> str:
        if request is None:
            return 'anonymous'
        return request.headers.get('X-API-Key') or 'anonymous'

    def _allow(self, key: str) -> bool:
        now = time.monotonic()
        bucket = self._buckets.setdefault(key, deque())
        cutoff = now - self.window_seconds
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= self.max_calls:
            return False
        bucket.append(now)
        return True

    def limit(self, spec: str) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        # Keep the decorator API compatible with slowapi; use the instance spec
        # parsed at construction so api_v1 can pass the same limiter to multiple
        # handlers if needed.
        def _decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            @wraps(func)
            async def _wrapped(*args: Any, **kwargs: Any) -> Any:
                request = kwargs.get('request')
                if request is None:
                    for arg in args:
                        if isinstance(arg, Request):
                            request = arg
                            break
                if not self._allow(self._key(request)):
                    from fastapi.responses import JSONResponse

                    return JSONResponse(
                        {'detail': f'rate limit exceeded: {self.spec}'},
                        status_code=429,
                        headers={'Retry-After': '1'},
                    )
                return await func(*args, **kwargs)

            return _wrapped

        return _decorator

_DEFAULT_RATE_LIMIT: str = "100/second"
_RATE_LIMIT_ENV: str = "MRE_FI_EXEC_CONF_RATE_LIMIT"
_RATE_LIMIT_ENABLED_ENV: str = "MRE_FI_RATE_LIMIT_ENABLED"
_BODY_SIZE_CAP_BYTES: int = 32 * 1024

def rate_limit_enabled() -> bool:
    """Return True iff the operator opted-in to the slowapi rate limiter.

    v1.5.1 (PR-9 FIX 1): the rate limiter is gated on the
    ``MRE_FI_RATE_LIMIT_ENABLED`` env var. Accepted truthy values are
    ``"1"``, ``"true"``, ``"yes"`` (case-insensitive); any other value
    (including unset) leaves the limiter off. This is intentionally
    distinct from :data:`_RATE_LIMIT_ENV` which sets the limit *spec*
    (e.g. ``"100/second"``).
    """
    raw = os.getenv(_RATE_LIMIT_ENABLED_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def assert_slowapi_available() -> None:
    """Raise ``RuntimeError`` when the limiter is opted-in but slowapi is missing.

    v1.5.1 (PR-9 FIX 1): production callers set
    ``MRE_FI_RATE_LIMIT_ENABLED=1`` to require rate limiting on
    POST /v1/execution_confidence. If the import fails (e.g. the
    deployment forgot the ``[security]`` extra) we MUST raise at
    startup rather than silently mount an unlimited handler. Tests
    can monkeypatch ``sys.modules["slowapi"] = None`` to simulate the
    missing-dependency path.

    No-op when ``MRE_FI_RATE_LIMIT_ENABLED`` is unset / false.
    """
    if not rate_limit_enabled():
        return
    import importlib
    import sys

    cached = sys.modules.get("slowapi")
    if cached is None and "slowapi" in sys.modules:
        # Tests inject ``sys.modules["slowapi"] = None`` to force the
        # missing-dependency branch without uninstalling slowapi.
        raise RuntimeError(
            "slowapi required when MRE_FI_RATE_LIMIT_ENABLED=1; "
            "install with: pip install market-regime-engine[security]"
        )
    try:
        importlib.import_module("slowapi")
    except Exception as exc:
        raise RuntimeError(
            "slowapi required when MRE_FI_RATE_LIMIT_ENABLED=1; "
            "install with: pip install market-regime-engine[security]"
        ) from exc


def _resolve_rate_limit_spec() -> str:
    raw = os.getenv(_RATE_LIMIT_ENV, "").strip()
    return raw if raw else _DEFAULT_RATE_LIMIT


def _build_rate_limiter() -> Any | None:
    """Construct a slowapi :class:`Limiter` keyed by API key.

    Returns ``None`` when slowapi is not installed; the POST handler
    then runs without rate limiting (the body cap + Pydantic validation
    still apply). This keeps the FI router optional-dependency-light
    so a vanilla install does not have to install slowapi.
    """
    try:
        from slowapi import Limiter
    except Exception:
        log.warning(
            "slowapi not installed; using in-process fallback rate limiter for "
            "POST /v1/execution_confidence. Install with `pip install "
            "market-regime-engine[security]` and set MRE_FI_RATE_LIMIT_ENABLED=1 "
            "for production fail-closed startup semantics."
        )
        return _InProcessRateLimiter(_resolve_rate_limit_spec())

    def _key_func(request: Request) -> str:
        # API-key-scoped: callers without an API key get one shared bucket.
        return request.headers.get("X-API-Key") or "anonymous"

    return Limiter(key_func=_key_func, default_limits=[_resolve_rate_limit_spec()])


__all__ = [
    "_BODY_SIZE_CAP_BYTES",
    "_InProcessRateLimiter",
    "_build_rate_limiter",
    "_resolve_rate_limit_spec",
    "assert_slowapi_available",
    "rate_limit_enabled",
]
