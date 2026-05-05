# SPDX-License-Identifier: Apache-2.0
"""Live release calendar fetchers + on-disk YAML cache.

The v1.3 ``release_calendar_exact.build_exact_release_calendar`` falls
back to a hand-coded ``DEFAULT_LAGS`` table for every series. That keeps
the engine dependency-free, but it can't reflect the actual BLS / BEA /
Census / Fed schedule. v1.4 ships a hybrid:

1. Hand-curated YAML files under ``config/release_calendars/{bls,bea,
   census,fed}.yaml`` so ``build_exact_release_calendar`` produces real
   release timestamps for the 16 catalog series out-of-the-box.
2. A ``CalendarFetcher`` Protocol + four concrete fetchers for live
   refreshes. Each fetcher pulls the public agency calendar HTML, parses
   it, and rewrites the on-disk YAML deterministically (sorted on
   ``series_id``+``observation_date``).

Soft-degrades: ``beautifulsoup4`` / ``lxml`` live behind the new
``[scraping]`` extra. Importing this module is *cheap* (only stdlib +
``yaml`` + ``requests``); the bs4 import is deferred to the fetchers.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Live release-calendar fetchers require the optional [scraping] "
    "extra. Install with `pip install market-regime-engine[scraping]`."
)


def _require_bs4() -> Any:
    try:
        import bs4
    except ImportError as exc:  # pragma: no cover - import path
        raise ImportError(_INSTALL_HINT) from exc
    return bs4


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalendarEntry:
    """One observation_date → release_timestamp_utc binding.

    The ``raw_payload_hash`` is the sha256 of the source HTML the entry
    was parsed from, so a refresh that produces identical output can be
    detected without re-running the fetcher logic.
    """

    agency: str
    series_id: str
    observation_date: str  # YYYY-MM-DD
    release_timestamp_utc: str  # YYYY-MM-DDTHH:MM:SSZ
    source_url: str
    fetched_at_utc: str
    raw_payload_hash: str


def _normalise_iso(date_str: str, *, hour: int = 13, default_minute: int = 30) -> str:
    """Round-trip a release date string into the canonical ISO Z form."""
    cleaned = (date_str or "").strip()
    if not cleaned:
        raise ValueError("empty release date")
    fmts = ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%dT%H:%M:%SZ")
    parsed: datetime | None = None
    for fmt in fmts:
        with contextlib.suppress(ValueError):
            parsed = datetime.strptime(cleaned, fmt)
            break
    if parsed is None:
        raise ValueError(f"unparseable release date {date_str!r}")
    if parsed.hour == 0 and parsed.minute == 0:
        parsed = parsed.replace(hour=hour, minute=default_minute)
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Fetcher protocol + concrete implementations
# ---------------------------------------------------------------------------


class CalendarFetcher(Protocol):
    """Implementations return a list of ``CalendarEntry`` instances."""

    agency: str
    source_url: str

    def fetch(self) -> list[CalendarEntry]: ...


def _http_get(url: str, *, timeout: float = 15.0) -> tuple[str, str]:
    """Fetch ``url`` and return (text, sha256(text))."""
    import requests  # local import — requests is already a hard dep

    headers = {"User-Agent": "market-regime-engine/1.4 release-calendar refresher"}
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    text = resp.text
    return text, hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class BLSCalendarFetcher:
    """Bureau of Labor Statistics — Schedule of releases page.

    Targets ``https://www.bls.gov/schedule/news_release/`` which lists
    the upcoming and recent releases for the major series families
    (employment situation, CPI, PPI, ECI, etc.). The parser is deliberately
    forgiving: it extracts ``YYYY-MM-DD`` plus a release time when present
    and is tolerant of layout changes (returns an empty list + warns
    rather than crashing).
    """

    agency: str = "bls"
    source_url: str = "https://www.bls.gov/schedule/news_release/empsit.htm"

    def fetch(self) -> list[CalendarEntry]:
        _require_bs4()
        try:
            text, payload_hash = _http_get(self.source_url)
        except Exception as exc:
            log.warning("bls_fetch_failed: %s", exc)
            return []
        return self._parse(text, payload_hash)

    def _parse(self, text: str, payload_hash: str) -> list[CalendarEntry]:
        bs4 = _require_bs4()
        soup = bs4.BeautifulSoup(text, "html.parser")
        entries: list[CalendarEntry] = []
        fetched = _now_iso()
        # The release page lays out a sequence of (Reference Period,
        # Release Date) pairs. We pick the first table whose header
        # contains "Reference Month" or "Reference Period".
        candidate_tables = soup.find_all("table") or []
        for table in candidate_tables:
            header_cells = [c.get_text(strip=True).lower() for c in table.find_all("th")]
            if not any("reference" in h or "release" in h for h in header_cells):
                continue
            for row in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
                if len(cells) < 2 or cells[0].lower().startswith("reference"):
                    continue
                ref_period = cells[0]
                release_str = cells[1]
                try:
                    release_iso = _normalise_iso(release_str, hour=12, default_minute=30)
                    obs_date = _ref_to_first_of_month(ref_period)
                except Exception:
                    continue
                entries.append(
                    CalendarEntry(
                        agency=self.agency,
                        series_id="PAYEMS",  # Employment Situation headline
                        observation_date=obs_date,
                        release_timestamp_utc=release_iso,
                        source_url=self.source_url,
                        fetched_at_utc=fetched,
                        raw_payload_hash=payload_hash,
                    )
                )
            if entries:
                break
        return entries


@dataclass
class BEACalendarFetcher:
    """Bureau of Economic Analysis — News Release Schedule."""

    agency: str = "bea"
    source_url: str = "https://www.bea.gov/news/schedule"

    def fetch(self) -> list[CalendarEntry]:
        try:
            text, payload_hash = _http_get(self.source_url)
        except Exception as exc:
            log.warning("bea_fetch_failed: %s", exc)
            return []
        return self._parse(text, payload_hash)

    def _parse(self, text: str, payload_hash: str) -> list[CalendarEntry]:
        bs4 = _require_bs4()
        soup = bs4.BeautifulSoup(text, "html.parser")
        entries: list[CalendarEntry] = []
        fetched = _now_iso()
        # BEA renders schedule rows inside a div.view-content-news-release.
        rows = soup.select(".view-news-release-schedule .views-row, .views-row")
        for r in rows:
            text = r.get_text(" ", strip=True)
            # Try to extract "Month DD, YYYY" + an indicator name.
            m = re.search(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
            if not m:
                continue
            try:
                release_iso = _normalise_iso(m.group(1), hour=12, default_minute=30)
            except Exception:
                continue
            indicator = (
                "GDP"
                if "gross domestic product" in text.lower()
                else ("PCE" if "personal" in text.lower() and "income" in text.lower() else "BEA")
            )
            entries.append(
                CalendarEntry(
                    agency=self.agency,
                    series_id=indicator,
                    observation_date=release_iso[:10],
                    release_timestamp_utc=release_iso,
                    source_url=self.source_url,
                    fetched_at_utc=fetched,
                    raw_payload_hash=payload_hash,
                )
            )
        return entries


@dataclass
class CensusCalendarFetcher:
    """U.S. Census Bureau — Economic Indicators schedule."""

    agency: str = "census"
    source_url: str = "https://www.census.gov/economic-indicators/calendar.html"

    def fetch(self) -> list[CalendarEntry]:
        try:
            text, payload_hash = _http_get(self.source_url)
        except Exception as exc:
            log.warning("census_fetch_failed: %s", exc)
            return []
        return self._parse(text, payload_hash)

    def _parse(self, text: str, payload_hash: str) -> list[CalendarEntry]:
        bs4 = _require_bs4()
        soup = bs4.BeautifulSoup(text, "html.parser")
        entries: list[CalendarEntry] = []
        fetched = _now_iso()
        # The Census calendar is a dl/dt/dd cascade.
        for dt in soup.find_all("dt"):
            dd = dt.find_next("dd")
            if dd is None:
                continue
            for li in dd.find_all("li") or [dd]:
                line = li.get_text(" ", strip=True)
                m = re.search(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})", line)
                if not m:
                    continue
                try:
                    release_iso = _normalise_iso(m.group(1), hour=14, default_minute=0)
                except Exception:
                    continue
                series = "PERMIT" if "permits" in line.lower() else ("HOUST" if "starts" in line.lower() else "CENSUS")
                entries.append(
                    CalendarEntry(
                        agency=self.agency,
                        series_id=series,
                        observation_date=release_iso[:10],
                        release_timestamp_utc=release_iso,
                        source_url=self.source_url,
                        fetched_at_utc=fetched,
                        raw_payload_hash=payload_hash,
                    )
                )
        return entries


@dataclass
class FedH15Fetcher:
    """Federal Reserve H.15 Selected Interest Rates page."""

    agency: str = "fed"
    source_url: str = "https://www.federalreserve.gov/releases/h15/"

    def fetch(self) -> list[CalendarEntry]:
        try:
            text, payload_hash = _http_get(self.source_url)
        except Exception as exc:
            log.warning("fed_fetch_failed: %s", exc)
            return []
        return self._parse(text, payload_hash)

    def _parse(self, text: str, payload_hash: str) -> list[CalendarEntry]:
        # bs4 is required only for a uniform soft-degrade on missing
        # extras; the H.15 page is regex-friendly enough that we don't
        # need a full DOM walk.
        _require_bs4()
        entries: list[CalendarEntry] = []
        fetched = _now_iso()
        # The H.15 release lists "Release Date: YYYY-MM-DD" near the top.
        m = re.search(r"Release Date[:\s]+([A-Za-z0-9,\s\-/]+)", text)
        if m:
            try:
                release_iso = _normalise_iso(m.group(1).strip(), hour=20, default_minute=15)
            except Exception:
                return []
            for series in ("DGS10", "FEDFUNDS", "T10Y3M", "MORTGAGE30US"):
                entries.append(
                    CalendarEntry(
                        agency=self.agency,
                        series_id=series,
                        observation_date=release_iso[:10],
                        release_timestamp_utc=release_iso,
                        source_url=self.source_url,
                        fetched_at_utc=fetched,
                        raw_payload_hash=payload_hash,
                    )
                )
        return entries


def _ref_to_first_of_month(text: str) -> str:
    """Extract a month-year and return ``YYYY-MM-01``."""
    cleaned = text.strip()
    fmts = ("%B %Y", "%b %Y", "%B, %Y", "%Y-%m", "%m/%Y")
    for fmt in fmts:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            return parsed.strftime("%Y-%m-01")
        except ValueError:
            continue
    raise ValueError(f"unparseable reference period {text!r}")


# ---------------------------------------------------------------------------
# YAML cache + audit hooks
# ---------------------------------------------------------------------------


def _calendar_dir() -> Path:
    """Return the ``config/release_calendars/`` directory.

    Resolution order:

    1. ``config/release_calendars/`` relative to the current working
       directory (the install path of the engine repo).
    2. The package-relative copy under
       ``site-packages/market_regime_engine/_release_calendars/``
       (shipped via setuptools data files when the wheel is installed).
    """
    cwd_dir = Path("config") / "release_calendars"
    if cwd_dir.exists():
        return cwd_dir
    # Walk up parents (3 levels: this file → frontier → market_regime_engine
    # → src → repo root). If a sibling ``config`` exists, prefer it.
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "config" / "release_calendars"
        if candidate.exists():
            return candidate
    return cwd_dir


def load_yaml_calendar(agency: str) -> list[dict[str, Any]]:
    """Load the on-disk calendar YAML for an agency (empty if missing)."""
    import yaml  # local import — yaml is already a hard dep

    path = _calendar_dir() / f"{agency}.yaml"
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        log.warning("calendar_yaml_parse_failed for %s: %s", agency, exc)
        return []
    entries = data.get("entries", []) if isinstance(data, dict) else []
    return [e for e in entries if isinstance(e, dict)]


def write_yaml_calendar(agency: str, entries: Iterable[CalendarEntry], out_dir: Path | None = None) -> Path:
    """Write the calendar YAML deterministically (sorted entries).

    Sorted by ``(series_id, observation_date)`` so the file diff is
    stable across refreshes.
    """
    import yaml  # local import

    out_root = out_dir or _calendar_dir()
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / f"{agency}.yaml"
    sorted_entries = sorted(
        (dataclasses.asdict(e) for e in entries),
        key=lambda x: (x.get("series_id", ""), x.get("observation_date", "")),
    )
    payload = {
        "agency": agency,
        "generated_at_utc": _now_iso(),
        "entries": sorted_entries,
    }
    text = yaml.safe_dump(payload, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    return path


def refresh_release_calendars(
    *,
    agencies: Iterable[str] | None = None,
    out_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run every fetcher and write their YAML outputs.

    Returns a status dict per agency that the CLI persists into the
    ``release_calendar_refreshes`` warehouse table.
    """
    fetchers: dict[str, CalendarFetcher] = {
        "bls": BLSCalendarFetcher(),
        "bea": BEACalendarFetcher(),
        "census": CensusCalendarFetcher(),
        "fed": FedH15Fetcher(),
    }
    selected = list(agencies) if agencies else list(fetchers)
    out_root = out_dir or _calendar_dir()
    out_root.mkdir(parents=True, exist_ok=True)
    status: dict[str, dict[str, Any]] = {}
    for agency in selected:
        fetcher = fetchers.get(agency)
        if fetcher is None:
            status[agency] = {
                "status": "skipped",
                "error": f"unknown agency {agency!r}",
                "entries_count": 0,
                "fetched_at_utc": _now_iso(),
            }
            continue
        try:
            entries = fetcher.fetch()
        except ImportError:
            # Soft-degrade: scraping extra missing — emit empty YAML so
            # the warehouse status row reflects the skip cleanly.
            status[agency] = {
                "status": "skipped_missing_extra",
                "error": _INSTALL_HINT,
                "entries_count": 0,
                "fetched_at_utc": _now_iso(),
            }
            continue
        except Exception as exc:  # pragma: no cover - network shape
            status[agency] = {
                "status": "error",
                "error": str(exc),
                "entries_count": 0,
                "fetched_at_utc": _now_iso(),
            }
            continue
        path = write_yaml_calendar(agency, entries, out_dir=out_root)
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        status[agency] = {
            "status": "ok" if entries else "empty",
            "error": None,
            "entries_count": len(entries),
            "fetched_at_utc": _now_iso(),
            "source_hash": source_hash,
            "source_url": fetcher.source_url,
            "out_path": str(path),
        }
    return status


# ---------------------------------------------------------------------------
# Reconciliation helper
# ---------------------------------------------------------------------------


def reconcile_against_vintages(
    vintage_observations: Any,  # pd.DataFrame
    *,
    tolerance_days: int = 3,
) -> Any:
    """Compare ``vintage_observations`` to the loaded YAML calendar.

    Returns a frame with one row per series_id × observation_date that
    differs by more than ``tolerance_days`` from the calendar's
    ``release_timestamp_utc``. The CLI ``mre audit-release-calendar
    --enforce`` surfaces the result.
    """
    import pandas as pd

    if vintage_observations is None or len(vintage_observations) == 0:
        return pd.DataFrame(
            columns=[
                "series_id",
                "observation_date",
                "agency",
                "calendar_release",
                "vintage_realtime_start",
                "delta_days",
            ]
        )
    cal_rows: list[dict[str, Any]] = []
    for agency in ("bls", "bea", "census", "fed"):
        for entry in load_yaml_calendar(agency):
            cal_rows.append(
                {
                    "series_id": entry.get("series_id"),
                    "observation_date": entry.get("observation_date"),
                    "agency": agency,
                    "calendar_release": entry.get("release_timestamp_utc"),
                }
            )
    if not cal_rows:
        return pd.DataFrame(
            columns=[
                "series_id",
                "observation_date",
                "agency",
                "calendar_release",
                "vintage_realtime_start",
                "delta_days",
            ]
        )
    cal = pd.DataFrame(cal_rows).drop_duplicates(["series_id", "observation_date"], keep="first")
    obs = vintage_observations.copy()
    if "observation_date" not in obs.columns:
        return pd.DataFrame()
    obs["observation_date"] = pd.to_datetime(obs["observation_date"]).dt.strftime("%Y-%m-%d")
    if "realtime_start" not in obs.columns:
        return pd.DataFrame()
    merged = obs.merge(
        cal,
        on=["series_id", "observation_date"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "series_id",
                "observation_date",
                "agency",
                "calendar_release",
                "vintage_realtime_start",
                "delta_days",
            ]
        )
    merged["calendar_release_dt"] = pd.to_datetime(merged["calendar_release"], utc=True, errors="coerce")
    merged["vintage_realtime_dt"] = pd.to_datetime(merged["realtime_start"], utc=True, errors="coerce")
    merged["delta_days"] = (merged["vintage_realtime_dt"] - merged["calendar_release_dt"]).dt.total_seconds() / 86400.0
    flagged = merged[merged["delta_days"].abs() > float(tolerance_days)].copy()
    flagged = flagged.rename(columns={"realtime_start": "vintage_realtime_start"})
    return flagged[
        [
            "series_id",
            "observation_date",
            "agency",
            "calendar_release",
            "vintage_realtime_start",
            "delta_days",
        ]
    ].reset_index(drop=True)


def write_status_to_warehouse(status: dict[str, dict[str, Any]], db_path: str) -> int:
    """Persist a refresh status dict into ``release_calendar_refreshes``."""
    import pandas as pd

    from market_regime_engine.storage import Warehouse

    if not status:
        return 0
    rows = []
    for agency, payload in status.items():
        rows.append(
            {
                "agency": agency,
                "fetched_at_utc": payload.get("fetched_at_utc", _now_iso()),
                "entries_count": int(payload.get("entries_count", 0)),
                "status": str(payload.get("status", "unknown")),
                "error": payload.get("error"),
                "source_hash": payload.get("source_hash"),
                "metadata_json": json.dumps(
                    {
                        k: v
                        for k, v in payload.items()
                        if k not in {"fetched_at_utc", "entries_count", "status", "error", "source_hash"}
                    },
                    sort_keys=True,
                    default=str,
                ),
            }
        )
    df = pd.DataFrame(rows)
    db = Warehouse(db_path)
    try:
        return db.write_release_calendar_refreshes(df)
    finally:
        db.close()


__all__ = [
    "BEACalendarFetcher",
    "BLSCalendarFetcher",
    "CalendarEntry",
    "CalendarFetcher",
    "CensusCalendarFetcher",
    "FedH15Fetcher",
    "load_yaml_calendar",
    "reconcile_against_vintages",
    "refresh_release_calendars",
    "write_status_to_warehouse",
    "write_yaml_calendar",
]
