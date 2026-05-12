# SPDX-License-Identifier: Apache-2.0
"""Starlette ASGI middleware shared by the FI POST endpoints.

v1.5 PR-8 (Tier-2 fix B-Ask-1, REVIEW.md):
:class:`MaxBodySizeMiddleware` accumulates request body bytes inside
the ASGI receive callable and rejects with HTTP 413 BEFORE the route
handler runs. Pre-fix the body-size cap was checked in the FI POST
handler after FastAPI had already parsed the body — chunked
``Transfer-Encoding`` requests carry no ``Content-Length`` header, so
an attacker streaming an unbounded body bypassed the cap entirely.

The middleware applies to a configurable URL path prefix so the global
``api_v1.app`` can install it for ``/v1/execution_confidence`` without
affecting macro routes that don't need a body cap.

Reference: Starlette middleware patterns —
https://www.starlette.io/middleware/
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger(__name__)

# v1.5 PR-8 keep parity with the in-handler historical default. Operators
# can override via ``MRE_FI_BODY_SIZE_CAP_BYTES``; the middleware reads
# the env var lazily at request time so a test fixture can patch it
# per-test.
DEFAULT_BODY_SIZE_CAP_BYTES: int = 32 * 1024
_BODY_SIZE_ENV: str = "MRE_FI_BODY_SIZE_CAP_BYTES"

# Production mode flag — when MRE_ENV=production (or
# MRE_FI_REJECT_CHUNKED=1), chunked-without-Content-Length requests on
# protected paths are rejected at the middleware so the cap can be
# enforced. Dev / test profiles accept chunked input as long as the
# accumulated body stays under the cap.
_ENV_NAME_ENV: str = "MRE_ENV"
_REJECT_CHUNKED_ENV: str = "MRE_FI_REJECT_CHUNKED"


def _body_size_cap() -> int:
    raw = os.environ.get(_BODY_SIZE_ENV, "").strip()
    if not raw:
        return DEFAULT_BODY_SIZE_CAP_BYTES
    try:
        return max(1, int(raw))
    except ValueError:
        log.warning(
            "%s=%r is not an integer; falling back to default %d",
            _BODY_SIZE_ENV,
            raw,
            DEFAULT_BODY_SIZE_CAP_BYTES,
        )
        return DEFAULT_BODY_SIZE_CAP_BYTES


def _should_reject_chunked() -> bool:
    if os.environ.get(_REJECT_CHUNKED_ENV) == "1":
        return True
    return os.environ.get(_ENV_NAME_ENV, "").lower() == "production"


def _path_matches(scope: Scope, prefixes: Iterable[str]) -> bool:
    path = scope.get("path") or ""
    return any(path.startswith(p) for p in prefixes)


def _headers_lookup(scope: Scope, name: bytes) -> bytes | None:
    for header_name, header_value in scope.get("headers", []) or []:
        if header_name.lower() == name:
            return header_value
    return None


def _is_chunked_without_content_length(scope: Scope) -> bool:
    cl = _headers_lookup(scope, b"content-length")
    if cl is not None:
        return False
    te = _headers_lookup(scope, b"transfer-encoding") or b""
    return b"chunked" in te.lower()


class MaxBodySizeMiddleware:
    """ASGI middleware that caps request body size BEFORE the route runs.

    Wraps the ``receive`` callable and accumulates body bytes; when the
    running total exceeds ``max_size`` the middleware sends an HTTP 413
    response directly without invoking the downstream app.

    Parameters
    ----------
    app
        Downstream ASGI app (FastAPI / Starlette).
    max_size
        Per-request body byte cap. Defaults to
        :data:`DEFAULT_BODY_SIZE_CAP_BYTES` (32 KB). When ``None`` the
        cap is read from ``MRE_FI_BODY_SIZE_CAP_BYTES`` per request so
        tests can override via env var.
    protected_path_prefixes
        URL path prefixes that the middleware acts on. Defaults to
        ``("/v1/execution_confidence",)`` so macro endpoints are
        untouched.
    reject_chunked_when_unsafe
        When ``True`` (default), requests with chunked
        ``Transfer-Encoding`` and no ``Content-Length`` are rejected
        with HTTP 413 in production mode (``MRE_ENV=production`` or
        ``MRE_FI_REJECT_CHUNKED=1``) so the cap can be enforced. When
        ``False`` (override for tests), the middleware still
        accumulates bytes and rejects once the cap is exceeded but
        does not pre-reject chunked-without-CL.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_size: int | None = None,
        protected_path_prefixes: Iterable[str] = ("/v1/execution_confidence",),
        reject_chunked_when_unsafe: bool = True,
    ) -> None:
        self.app = app
        self._explicit_max = max_size
        self._protected = tuple(protected_path_prefixes)
        self._reject_chunked_when_unsafe = bool(reject_chunked_when_unsafe)

    @property
    def max_size(self) -> int:
        if self._explicit_max is not None:
            return self._explicit_max
        return _body_size_cap()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if not _path_matches(scope, self._protected):
            await self.app(scope, receive, send)
            return

        max_size = self.max_size

        # v1.5 PR-8 Tier-2 (B-Ask-1): production mode rejects chunked
        # bodies that omit Content-Length because an attacker can
        # otherwise stream an unbounded body. We still cap-by-accumulation
        # below as defence in depth.
        if (
            self._reject_chunked_when_unsafe
            and _should_reject_chunked()
            and _is_chunked_without_content_length(scope)
        ):
            await _send_413(
                send,
                detail="chunked transfer-encoding without Content-Length is "
                "rejected in production",
                limit_bytes=max_size,
            )
            return

        # Pre-screen Content-Length: bail before consuming the body if
        # the declared length already exceeds the cap.
        cl_header = _headers_lookup(scope, b"content-length")
        if cl_header is not None:
            try:
                declared = int(cl_header.decode("ascii"))
            except (ValueError, UnicodeDecodeError):
                declared = -1
            if declared > max_size:
                await _send_413(
                    send,
                    detail=_cap_exceeded_detail(max_size, declared=declared),
                    limit_bytes=max_size,
                )
                return

        # Wrap ``receive`` so we count bytes as the body streams in.
        # ``state["exceeded"]`` mutates to True once the running total
        # exceeds the cap; the wrapped receive then yields a synthetic
        # ``http.disconnect`` and the middleware short-circuits the
        # downstream response with HTTP 413.
        wrapped, state = _wrap_receive_with_size_check(receive, max_size)

        # Capture whether the downstream app sent a response so the
        # middleware can emit a 413 itself when the cap fires.
        response_started: dict[str, bool] = {"value": False}

        async def _send_filtered(message: Message) -> None:
            if state["exceeded"]:
                # Downstream app saw the disconnect and is trying to
                # respond anyway — swallow its messages, we'll send the
                # 413 ourselves once the app returns.
                return
            if message["type"] == "http.response.start":
                response_started["value"] = True
            await send(message)

        await self.app(scope, wrapped, _send_filtered)

        if state["exceeded"] and not response_started["value"]:
            # The downstream app never started a response (it saw the
            # disconnect from our wrapped receive). Emit the 413 now.
            await _send_413(
                send,
                detail=_cap_exceeded_detail(max_size),
                limit_bytes=max_size,
            )


def _wrap_receive_with_size_check(
    receive: Receive, max_size: int
) -> tuple[Receive, dict[str, Any]]:
    """Return a ``(wrapped_receive, state)`` pair.

    ``state["exceeded"]`` is mutated to ``True`` once the running total
    exceeds ``max_size``; the wrapped receive then yields a synthetic
    ``http.disconnect`` so the downstream app sees the body stop. The
    middleware itself emits the 413 response after the downstream
    app returns.
    """
    state: dict[str, Any] = {"exceeded": False, "total": 0}

    async def wrapped() -> Message:
        if state["exceeded"]:
            return {"type": "http.disconnect"}
        message = await receive()
        if message["type"] != "http.request":
            return message
        body = message.get("body", b"") or b""
        state["total"] += len(body)
        if state["total"] > max_size:
            state["exceeded"] = True
            return {"type": "http.disconnect"}
        return message

    return wrapped, state


def _cap_exceeded_detail(max_size: int, *, declared: int | None = None) -> str:
    """Return a human-readable 413 detail string.

    Mirrors the historical in-handler ``"request body exceeds 32 KB cap"``
    phrasing when the cap matches the canonical 32 KB so existing
    runbook / dashboard / test assertions keep working; falls back to
    a generic byte-count for non-default caps.
    """
    if max_size == DEFAULT_BODY_SIZE_CAP_BYTES:
        cap_human = "32 KB cap"
    else:
        cap_human = f"{max_size} byte cap"
    if declared is not None:
        return f"request body (Content-Length {declared}) exceeds {cap_human}"
    return f"request body exceeds {cap_human}"


async def _send_413(send: Send, *, detail: str, limit_bytes: int) -> None:
    """Emit a 413 response directly from the middleware."""
    body = json.dumps(
        {
            "detail": detail,
            "limit_bytes": limit_bytes,
        }
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        }
    )


def install_max_body_size_middleware(
    app: Any,
    *,
    max_size: int | None = None,
    protected_path_prefixes: Iterable[str] = ("/v1/execution_confidence",),
    reject_chunked_when_unsafe: bool = True,
) -> None:
    """Register :class:`MaxBodySizeMiddleware` on ``app``.

    Convenience for ``api_v1.app`` so the registration call site is one
    line instead of constructing the class manually.
    """
    app.add_middleware(
        MaxBodySizeMiddleware,
        max_size=max_size,
        protected_path_prefixes=tuple(protected_path_prefixes),
        reject_chunked_when_unsafe=reject_chunked_when_unsafe,
    )


__all__ = [
    "DEFAULT_BODY_SIZE_CAP_BYTES",
    "MaxBodySizeMiddleware",
    "install_max_body_size_middleware",
]
