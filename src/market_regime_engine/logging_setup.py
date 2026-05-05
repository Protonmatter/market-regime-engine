# SPDX-License-Identifier: Apache-2.0
"""Structured logging for the Market Regime Engine.

Two formats are supported:

- ``human`` (default): readable single-line records aimed at interactive CLI use.
- ``json``: one JSON object per record, suitable for log shippers, Loki, ELK,
  etc. Activate via ``MRE_LOG_FORMAT=json`` or ``--json-logs`` on the CLI.

The configuration is idempotent: repeated calls to :func:`configure_logging`
replace the root handlers rather than stacking them, so re-importing the engine
inside a notebook or a long-running orchestrator does not duplicate output.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from logging import Handler, LogRecord
from typing import Any

_DEFAULT_FIELDS = (
    "timestamp",
    "level",
    "logger",
    "message",
    "module",
    "function",
    "line",
)


class _JSONFormatter(logging.Formatter):
    """Emit one JSON object per log record. Keeps key order stable."""

    def format(self, record: LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Attach any user-supplied "extra" key/value pairs that are not part of
        # the standard ``LogRecord`` attribute set.
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _STANDARD_ATTRS:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = repr(value)
        return json.dumps(payload, sort_keys=False, default=str)


class _HumanFormatter(logging.Formatter):
    """Compact single-line human-readable formatter."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
            datefmt="%H:%M:%S",
        )


_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {
    "asctime",
    "message",
}


def configure_logging(
    level: str | int = "INFO",
    *,
    fmt: str | None = None,
    stream: Any = None,
) -> None:
    """Configure the root logger for the engine.

    Parameters
    ----------
    level:
        Python logging level name or numeric level. Honors ``MRE_LOG_LEVEL``.
    fmt:
        Either ``"human"`` or ``"json"``. Honors ``MRE_LOG_FORMAT``.
    stream:
        Output stream (default :data:`sys.stderr`). Used for tests.
    """
    resolved_level = os.getenv("MRE_LOG_LEVEL", str(level)).upper()
    resolved_fmt = (fmt or os.getenv("MRE_LOG_FORMAT") or "human").lower()

    formatter: logging.Formatter
    if resolved_fmt == "json":
        formatter = _JSONFormatter()
    else:
        formatter = _HumanFormatter()

    handler: Handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    try:
        root.setLevel(resolved_level)
    except (TypeError, ValueError):
        root.setLevel(logging.INFO)

    # Quieten noisy third-party loggers below WARNING; they can be re-enabled
    # by callers explicitly.
    for noisy in ("urllib3", "requests", "asyncio", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor; ensures default config is applied lazily."""
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
