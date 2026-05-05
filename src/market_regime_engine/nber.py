# SPDX-License-Identifier: Apache-2.0
"""NBER recession windows + staleness-aware label provisioning.

Two label sources are supported:

- The built-in :data:`NBER_RECESSIONS` table — frozen at the dates listed below.
  Used for offline tests and as a fallback when FRED is unreachable.
- Live FRED ``USREC`` series via :func:`market_regime_engine.fred_recession.fetch_fred_recession_indicator`.

The historical contract (``label_recession_months``) is preserved so existing
callers continue to work, but :func:`label_recessions_with_fallback` is the new
default. It transparently prefers FRED when ``FRED_API_KEY`` is set,
falls back to the built-in window list otherwise, and emits a structured
staleness report so downstream confidence/release-gate components can register
when the labels are out of date relative to the panel.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd

# NBER monthly U.S. recession windows. Authoritative through 2020-04. New
# additions should be inserted here in commit messages so the audit log shows
# when the table moved.
NBER_RECESSIONS = [
    ("1948-11-01", "1949-10-01"),
    ("1953-07-01", "1954-05-01"),
    ("1957-08-01", "1958-04-01"),
    ("1960-04-01", "1961-02-01"),
    ("1969-12-01", "1970-11-01"),
    ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]


@dataclass(frozen=True)
class RecessionWindow:
    peak_month: str
    trough_month: str

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.peak_month)

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.trough_month)


WINDOWS = [RecessionWindow(a, b) for a, b in NBER_RECESSIONS]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def recession_indicator(dates: pd.DatetimeIndex | pd.Series) -> pd.Series:
    idx = pd.to_datetime(dates)
    out = pd.Series(0.0, index=idx)
    for w in WINDOWS:
        out[(idx >= w.start) & (idx <= w.end)] = 1.0
    return out


def label_recession_months(dates: pd.DatetimeIndex | pd.Series) -> pd.DataFrame:
    """Return a per-month label frame from the built-in NBER window list."""
    idx = pd.to_datetime(dates)
    rec = recession_indicator(idx)
    rows = []
    for d, v in rec.items():
        active = [w for w in WINDOWS if w.start <= d <= w.end]
        meta = {}
        if active:
            w = active[0]
            meta = {"peak_month": w.peak_month, "trough_month": w.trough_month}
        rows.append(
            {
                "date": d,
                "recession": float(v),
                "source": "built_in_nber_windows",
                "metadata_json": json.dumps(meta, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def add_forward_recession_targets(labels: pd.DataFrame, horizons: tuple[int, ...] = (3, 6, 12)) -> pd.DataFrame:
    if labels.empty:
        return labels.copy()
    frame = labels.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").set_index("date")
    rec = frame["recession"].astype(float)
    for h in horizons:
        vals = []
        for i in range(len(rec)):
            window = rec.iloc[i + 1 : i + h + 1]
            vals.append(float(window.max()) if len(window) else float("nan"))
        frame[f"recession_next_{h}m"] = vals
    return frame.reset_index()


# ---------------------------------------------------------------------------
# new: staleness-aware default labeller
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelStaleness:
    source: str  # "fred_usrec" | "built_in_nber_windows"
    last_label_date: str
    panel_last_date: str
    months_stale: int  # nonnegative; 0 means up-to-date
    fetch_error: str = ""

    def to_metadata(self) -> dict:
        return {
            "source": self.source,
            "last_label_date": self.last_label_date,
            "panel_last_date": self.panel_last_date,
            "months_stale": int(self.months_stale),
            "fetch_error": self.fetch_error,
        }


def _staleness(
    labels: pd.DataFrame, panel_dates: pd.DatetimeIndex, *, source: str, fetch_error: str = ""
) -> LabelStaleness:
    if labels is None or labels.empty:
        return LabelStaleness(
            source=source,
            last_label_date="unknown",
            panel_last_date="unknown",
            months_stale=99999,
            fetch_error=fetch_error,
        )
    last_label = pd.to_datetime(labels["date"]).max()
    last_panel = pd.to_datetime(panel_dates).max() if len(panel_dates) else last_label
    months = max(0, (last_panel.year - last_label.year) * 12 + (last_panel.month - last_label.month))
    return LabelStaleness(
        source=source,
        last_label_date=last_label.strftime("%Y-%m-%d"),
        panel_last_date=last_panel.strftime("%Y-%m-%d"),
        months_stale=int(months),
        fetch_error=fetch_error,
    )


def label_recessions_with_fallback(
    panel_dates: pd.DatetimeIndex | pd.Series,
    *,
    api_key: str | None = None,
    prefer: str = "fred",
    fred_series: str = "USREC",
) -> tuple[pd.DataFrame, LabelStaleness]:
    """Build recession labels for ``panel_dates`` with explicit staleness.

    Resolution order
    ----------------
    1. If ``prefer == "fred"`` and an API key is reachable, fetch ``USREC``
       (or whichever ``fred_series`` is requested) and use that as the source.
    2. Otherwise (or on any FRED failure) fall back to the built-in NBER
       window table. The returned :class:`LabelStaleness` captures any error
       message so downstream components can surface it.

    The output frame matches the shape of :func:`label_recession_months` so
    existing callers do not need to change.
    """
    panel_idx = pd.to_datetime(panel_dates)
    panel_idx = pd.DatetimeIndex(panel_idx)
    fetch_err = ""
    if prefer == "fred":
        api_key = api_key or os.getenv("FRED_API_KEY")
        if api_key:
            try:
                from market_regime_engine.fred_recession import fetch_fred_recession_indicator

                live = fetch_fred_recession_indicator(series_id=fred_series, api_key=api_key)
                if not live.empty:
                    return live, _staleness(live, panel_idx, source=f"fred:{fred_series}")
            except Exception as exc:  # pragma: no cover - network branch
                fetch_err = str(exc)
    builtin = label_recession_months(panel_idx)
    return builtin, _staleness(builtin, panel_idx, source="built_in_nber_windows", fetch_error=fetch_err)


__all__ = [
    "NBER_RECESSIONS",
    "WINDOWS",
    "LabelStaleness",
    "RecessionWindow",
    "add_forward_recession_targets",
    "label_recession_months",
    "label_recessions_with_fallback",
    "recession_indicator",
]
