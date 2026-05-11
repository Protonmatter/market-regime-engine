# SPDX-License-Identifier: Apache-2.0
"""UTC timestamp enforcement for the Fixed-Income boundary (REVIEW.md §3.4 Q-7).

The macro side operates on monthly cadence, so timezone-naive
:class:`pandas.Timestamp` values are unambiguous in practice. FI is
intraday: TRACE / RFQ / dealer-quote timestamps arrive in NY Eastern
(venue clock), at the backend in UTC, and at the API boundary in
client-local-time. Mixing the three silently produces ~5-hour lookahead
leaks (an ET trade at 14:00 looks "before" a UTC decision at 18:00
when both are interpreted as naive on the same axis).

This module ships the FI ingest / API boundary policy::

    1. naive datetimes are rejected (raise ``ValueError``);
    2. aware datetimes are converted to UTC;
    3. strings are parsed (and required to carry tz info);
    4. ``None`` passes through.

The companion :func:`iso8601_z` serialiser writes ISO-8601 with the
explicit ``Z`` suffix so the warehouse storage convention is one
canonical bytestring per instant regardless of the source tz.
"""

from __future__ import annotations

from datetime import datetime
from typing import overload

import pandas as pd

__all__ = ["assert_utc", "iso8601_z", "to_utc"]


@overload
def to_utc(ts: None) -> None: ...


@overload
def to_utc(ts: pd.Timestamp | datetime | str) -> pd.Timestamp: ...


def to_utc(ts: pd.Timestamp | datetime | str | None) -> pd.Timestamp | None:
    """Convert ``ts`` to UTC. Naive inputs raise ``ValueError``.

    Per REVIEW.md §3.4 Q-7: every FI ingest call site should pipe its
    incoming timestamp through this helper exactly once. The naive
    rejection is the operational backstop — naïve "5pm" could be ET
    or UTC and the difference matters for PIT.

    Parameters
    ----------
    ts:
        Input timestamp. ``None`` is passed through (so optional
        timestamp fields don't need a special-case at the call site).
        Strings are parsed; the parsed value must carry tz info, else
        ``ValueError`` is raised.

    Returns
    -------
    Aware ``pd.Timestamp`` in UTC, or ``None`` when ``ts is None``.
    """
    if ts is None:
        return None
    if isinstance(ts, str):
        parsed = pd.Timestamp(ts)
        if parsed.tzinfo is None:
            raise ValueError(
                f"naive timestamp string at FI boundary: {ts!r} (must carry tz info, e.g. ISO-8601 'Z' suffix)"
            )
        return parsed.tz_convert("UTC")
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        raise ValueError(f"naive datetime at FI boundary: {parsed.isoformat()!r} (must be tz-aware; UTC convention)")
    return parsed.tz_convert("UTC")


def assert_utc(ts: pd.Timestamp, *, label: str = "timestamp") -> None:
    """Raise ``ValueError`` if ``ts`` is naive or not in UTC.

    Stricter than :func:`to_utc`: rejects aware non-UTC timestamps too
    so storage code can use this as a write-path invariant after
    `to_utc` has run elsewhere in the pipeline.
    """
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"{label} is naive: {ts!r} (must be UTC-aware)")
    # ``utcoffset()`` returns a ``timedelta``; UTC offsets compare equal to zero.
    if ts.utcoffset() != pd.Timedelta(0):
        raise ValueError(
            f"{label} is not UTC: {ts.isoformat()} (utcoffset={ts.utcoffset()!r}); call to_utc(...) at the boundary"
        )


def iso8601_z(ts: pd.Timestamp) -> str:
    """Return ISO-8601 with the explicit ``Z`` suffix.

    ``pd.Timestamp.isoformat()`` writes ``+00:00`` for UTC; FI storage
    convention requires the ``Z`` suffix per AGENT.md §"Storage". This
    helper accepts only UTC timestamps (raises otherwise) so callers
    cannot accidentally write an ET timestamp with a ``Z`` suffix.
    """
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    assert_utc(ts, label="iso8601_z input")
    # ``isoformat`` on UTC ends with ``+00:00`` per ISO 8601; swap to ``Z``.
    iso = ts.isoformat()
    if iso.endswith("+00:00"):
        return iso[:-6] + "Z"
    if iso.endswith("+0000"):
        return iso[:-5] + "Z"
    return iso + "Z"
