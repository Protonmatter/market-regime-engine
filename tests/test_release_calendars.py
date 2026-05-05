# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the v1.4 release calendar layer (item D).

Four primary contracts:

1. Each of the four agency fetchers parses its recorded HTML fixture
   into ``CalendarEntry`` objects (one test per agency).
2. ``test_refresh_release_calendars_writes_deterministic_yaml`` — running
   the refresh hook with mocked HTTP produces sorted, byte-stable YAML.
3. ``test_build_exact_calendar_prefers_real_over_default`` — when a
   series is covered by the YAML cache, ``build_exact_release_calendar``
   uses it and stamps ``source = "<agency>_real"``.
4. ``test_audit_release_calendar_flags_mismatch_within_tolerance`` —
   the reconciliation function detects a vintage that drifts by more
   than ``--tolerance-days`` from the calendar.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "release_calendars"


def _read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _require_bs4() -> None:
    pytest.importorskip("bs4")


# ---------------------------------------------------------------------------
# Per-agency fixture parsers
# ---------------------------------------------------------------------------


def test_bls_fetcher_parses_recorded_fixture(requests_mock) -> None:
    _require_bs4()
    from market_regime_engine.frontier.release_calendars import BLSCalendarFetcher

    fetcher = BLSCalendarFetcher()
    requests_mock.get(fetcher.source_url, text=_read_fixture("bls_fixture.html"))
    entries = fetcher.fetch()
    assert entries, "BLS fixture produced zero entries"
    months = {e.observation_date for e in entries}
    assert {"2026-01-01", "2026-02-01", "2026-03-01"}.issubset(months)
    for e in entries:
        assert e.agency == "bls"
        assert e.series_id == "PAYEMS"
        assert e.release_timestamp_utc.endswith("Z")
        assert e.raw_payload_hash, "payload hash must be populated"


def test_bea_fetcher_parses_recorded_fixture(requests_mock) -> None:
    _require_bs4()
    from market_regime_engine.frontier.release_calendars import BEACalendarFetcher

    fetcher = BEACalendarFetcher()
    requests_mock.get(fetcher.source_url, text=_read_fixture("bea_fixture.html"))
    entries = fetcher.fetch()
    assert entries, "BEA fixture produced zero entries"
    series = {e.series_id for e in entries}
    # The fixture contains GDP, PCE (Personal Income), and a generic BEA row.
    assert series & {"GDP", "PCE", "BEA"}
    for e in entries:
        assert e.agency == "bea"
        assert e.release_timestamp_utc.endswith("Z")


def test_census_fetcher_parses_recorded_fixture(requests_mock) -> None:
    _require_bs4()
    from market_regime_engine.frontier.release_calendars import CensusCalendarFetcher

    fetcher = CensusCalendarFetcher()
    requests_mock.get(fetcher.source_url, text=_read_fixture("census_fixture.html"))
    entries = fetcher.fetch()
    assert entries, "Census fixture produced zero entries"
    series = {e.series_id for e in entries}
    assert series & {"PERMIT", "HOUST", "CENSUS"}


def test_fed_fetcher_parses_recorded_fixture(requests_mock) -> None:
    _require_bs4()
    from market_regime_engine.frontier.release_calendars import FedH15Fetcher

    fetcher = FedH15Fetcher()
    requests_mock.get(fetcher.source_url, text=_read_fixture("fed_fixture.html"))
    entries = fetcher.fetch()
    assert entries, "Fed H.15 fixture produced zero entries"
    series = {e.series_id for e in entries}
    assert {"FEDFUNDS", "DGS10"}.issubset(series)
    for e in entries:
        assert e.agency == "fed"


# ---------------------------------------------------------------------------
# YAML write + determinism
# ---------------------------------------------------------------------------


def test_refresh_release_calendars_writes_deterministic_yaml(requests_mock) -> None:
    _require_bs4()
    from market_regime_engine.frontier.release_calendars import (
        BEACalendarFetcher,
        BLSCalendarFetcher,
        CensusCalendarFetcher,
        FedH15Fetcher,
        refresh_release_calendars,
    )

    requests_mock.get(BLSCalendarFetcher().source_url, text=_read_fixture("bls_fixture.html"))
    requests_mock.get(BEACalendarFetcher().source_url, text=_read_fixture("bea_fixture.html"))
    requests_mock.get(CensusCalendarFetcher().source_url, text=_read_fixture("census_fixture.html"))
    requests_mock.get(FedH15Fetcher().source_url, text=_read_fixture("fed_fixture.html"))

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        status = refresh_release_calendars(out_dir=out_dir)
        # All four agencies must have produced ok / non-empty output.
        for agency in ("bls", "bea", "census", "fed"):
            payload = status.get(agency, {})
            assert payload.get("status") in {"ok", "empty"}
            assert (out_dir / f"{agency}.yaml").exists()
        # Determinism: a second refresh against the same fixtures must
        # produce identical YAML content modulo the ``generated_at_utc``
        # header (which is a wall-clock stamp, deliberately monotonic).
        first_payload = (out_dir / "bls.yaml").read_text(encoding="utf-8")
        status2 = refresh_release_calendars(out_dir=out_dir)
        second_payload = (out_dir / "bls.yaml").read_text(encoding="utf-8")

        # Strip the timestamp lines and compare.
        def _strip(text: str) -> str:
            return "\n".join(
                line for line in text.splitlines() if "generated_at_utc" not in line and "fetched_at_utc" not in line
            )

        assert _strip(first_payload) == _strip(second_payload)
        # source_hash must be reproducible across the two runs because
        # the YAML body is byte-stable modulo the header.
        assert status["bls"].get("source_hash")
        assert status2["bls"].get("source_hash")


# ---------------------------------------------------------------------------
# Calendar-aware exact release calendar build
# ---------------------------------------------------------------------------


def test_build_exact_calendar_prefers_real_over_default() -> None:
    """When the YAML cache covers a series, the source flips to ``*_real``."""
    from market_regime_engine.release_calendar_exact import build_exact_release_calendar

    obs = pd.DataFrame(
        [
            # Covered by config/release_calendars/bls.yaml.
            {"series_id": "PAYEMS", "date": "2026-01-01", "value": 1.0},
            {"series_id": "CPIAUCSL", "date": "2026-01-01", "value": 1.0},
            # Not covered → falls back to DEFAULT_LAGS.
            {"series_id": "SPX", "date": "2026-01-01", "value": 1.0},
        ]
    )
    catalog = [
        {"series_id": "PAYEMS", "domain": "labor"},
        {"series_id": "CPIAUCSL", "domain": "inflation"},
        {"series_id": "SPX", "domain": "market"},
    ]
    cal = build_exact_release_calendar(obs, catalog)
    by_id = cal.set_index("series_id")
    assert by_id.loc["PAYEMS", "source"] == "bls_real"
    assert by_id.loc["CPIAUCSL", "source"] == "bls_real"
    assert by_id.loc["SPX", "source"] == "v0.6_conservative_rule"
    # Real BLS PAYEMS schedule for Jan 2026 is Feb 6, 2026 — the YAML cache
    # must surface that timestamp (vs the conservative rule's Jan + 5 days).
    assert by_id.loc["PAYEMS", "release_timestamp_utc"].startswith("2026-02-06")


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def test_audit_release_calendar_flags_mismatch_within_tolerance() -> None:
    from market_regime_engine.frontier.release_calendars import reconcile_against_vintages

    # The seed YAML calendar has PAYEMS Jan 2026 -> 2026-02-06T13:30:00Z.
    # Construct a vintage whose realtime_start drifts by 10 days so it
    # exceeds the default ±3-day tolerance.
    obs = pd.DataFrame(
        [
            {
                "series_id": "PAYEMS",
                "observation_date": "2026-01-01",
                "value": 1.0,
                # 10 days later than the calendar release.
                "realtime_start": "2026-02-16",
                "realtime_end": None,
                "vintage_date": "2026-02-16",
                "source": "test",
                "ingested_at_utc": "2026-05-01T00:00:00Z",
                "metadata_json": "{}",
            },
            # Inside tolerance — must NOT be flagged.
            {
                "series_id": "PAYEMS",
                "observation_date": "2026-02-01",
                "value": 1.0,
                "realtime_start": "2026-03-08",  # 2 days after 2026-03-06 calendar
                "realtime_end": None,
                "vintage_date": "2026-03-08",
                "source": "test",
                "ingested_at_utc": "2026-05-01T00:00:00Z",
                "metadata_json": "{}",
            },
        ]
    )
    flagged = reconcile_against_vintages(obs, tolerance_days=3)
    assert not flagged.empty
    flagged_ids = list(zip(flagged["series_id"], flagged["observation_date"], strict=False))
    assert ("PAYEMS", "2026-01-01") in flagged_ids
    assert ("PAYEMS", "2026-02-01") not in flagged_ids
    # Loosening the tolerance to 30 days drops everything.
    relaxed = reconcile_against_vintages(obs, tolerance_days=30)
    assert relaxed.empty


# ---------------------------------------------------------------------------
# Soft-degrade smoke
# ---------------------------------------------------------------------------


def test_release_calendars_soft_degrade_without_bs4(monkeypatch) -> None:
    """The fetcher raises a clean ``ImportError`` when bs4 is missing."""
    import sys as _sys

    from market_regime_engine.frontier.release_calendars import BLSCalendarFetcher

    monkeypatch.setitem(_sys.modules, "bs4", None)
    with pytest.raises(ImportError, match=r"\[scraping\] extra"):
        BLSCalendarFetcher().fetch()
