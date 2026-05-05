# SPDX-License-Identifier: Apache-2.0
"""Recorded-fixture ALFRED ingestion tests (v1.3 item I).

The historical ``tests/test_phase6_phase7.py`` only exercised the
``alfred_real`` planner; the actual ALFRED HTTP path was effectively
untested in CI because it requires a FRED API key. v1.3 plugs the gap
with vcrpy-style cassette playback. The cassettes are committed under
``tests/fixtures/alfred/`` and contain a synthetic but
schema-faithful response set (see ``tests/fixtures/alfred/README.md``
for the rationale and re-record procedure).

The test asserts the lineage invariants the ALFRED ingestion path
must preserve regardless of upstream:

- ``observation_date`` is non-decreasing within each vintage.
- ``realtime_start == vintage_date`` for every row.
- ``value`` is never null.
- The total row count matches the deterministic fixture row count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "alfred"
SERIES = ("UNRATE", "CPIAUCSL", "FEDFUNDS")
# 5 vintages per series. The dates are picked to look like real ALFRED
# vintage dates (the first business day of a month, post-2020).
VINTAGES = (
    "2024-08-02",
    "2024-09-06",
    "2024-10-04",
    "2024-11-01",
    "2024-12-06",
)
# Each vintage carries 3 observations (Jun/Jul/Aug 2024). 3 series x 5
# vintages x 3 observations per vintage = 45 expected rows.
OBSERVATION_DATES = ("2024-06-01", "2024-07-01", "2024-08-01")


def _synthesize_responses() -> dict[str, dict]:
    """Build a deterministic dict of (url-path, query-key) → JSON body."""
    payloads: dict[str, dict] = {}
    payloads["__vintage_dates__"] = {sid: list(VINTAGES) for sid in SERIES}
    obs_payloads: dict[str, dict] = {}
    for sid in SERIES:
        for v_idx, vintage in enumerate(VINTAGES):
            obs = []
            for o_idx, obs_date in enumerate(OBSERVATION_DATES):
                # Deterministic synthetic value: distinguishable per
                # series / vintage / observation_date.
                base = {"UNRATE": 3.5, "CPIAUCSL": 305.0, "FEDFUNDS": 5.25}[sid]
                value = base + 0.1 * v_idx + 0.05 * o_idx
                obs.append(
                    {
                        "date": obs_date,
                        "value": f"{value:.3f}",
                        "realtime_start": vintage,
                        "realtime_end": vintage,
                    }
                )
            obs_payloads[(sid, vintage)] = {"observations": obs}
    payloads["__observations__"] = obs_payloads
    return payloads


_PAYLOADS = _synthesize_responses()


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` used by the ALFRED path."""

    def __init__(self, body: dict, status: int = 200) -> None:
        self._body = body
        self.status_code = status
        self.text = json.dumps(body)
        self.content = self.text.encode("utf-8")

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _fake_get(url: str, params: dict | None = None, **_kwargs):
    """Replay a deterministic synthetic FRED response.

    The cassette layer is conceptually equivalent to vcrpy
    ``record_mode="none"``: every request the production code makes
    must match a known endpoint, otherwise the test fails. The
    advantage of doing this in-test (vs. shipping vcrpy YAML cassettes)
    is that the synthetic responses are 100% deterministic and don't
    need re-recording when the underlying HTTPS chain rotates a cert.
    """
    params = params or {}
    if url.endswith("/series/vintagedates"):
        sid = params["series_id"]
        return _FakeResponse({"vintage_dates": _PAYLOADS["__vintage_dates__"][sid]})
    if url.endswith("/series/observations"):
        sid = params["series_id"]
        vintage = params["realtime_start"]
        body = _PAYLOADS["__observations__"].get((sid, vintage))
        if body is None:
            return _FakeResponse({"observations": []})
        return _FakeResponse(body)
    return _FakeResponse({}, status=404)


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "synthetic-key-not-used-by-cassette")


def test_recorded_alfred_replay_yields_lineage_invariant_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay the cassettes through ``fetch_real_alfred_vintage_observations``.

    Asserts the four contract invariants from item I:

    1. ``observation_date`` ordering per vintage (non-decreasing).
    2. ``realtime_start == vintage_date`` for every row.
    3. No null ``value``s.
    4. Total row count matches the synthetic fixture
       (3 series × 5 vintages × 3 observations = 45 rows).
    """
    from market_regime_engine import alfred_real

    monkeypatch.setattr(alfred_real.requests, "get", _fake_get)

    vintages, observations, manifest = alfred_real.fetch_real_alfred_vintage_observations(
        SERIES,
        api_key="synthetic-key-not-used-by-cassette",
        observation_start="2024-01-01",
        vintage_start="2024-01-01",
    )

    # ----- shape -----
    assert not observations.empty, "expected non-empty observations frame"
    assert len(observations) == len(SERIES) * len(VINTAGES) * len(OBSERVATION_DATES)

    # ----- invariant 4: total row count -----
    expected_rows = 3 * 5 * 3
    assert len(observations) == expected_rows

    # ----- invariant 3: no null values -----
    assert observations["value"].notna().all(), "value column must have no nulls"

    # ----- invariant 2: realtime_start == vintage_date per row -----
    rs = pd.to_datetime(observations["realtime_start"]).dt.strftime("%Y-%m-%d")
    vd = pd.to_datetime(observations["vintage_date"]).dt.strftime("%Y-%m-%d")
    assert (rs == vd).all(), "realtime_start must equal vintage_date"

    # ----- invariant 1: observation_date ordering per vintage -----
    for (_sid, _vintage), grp in observations.groupby(["series_id", "vintage_date"]):
        dates = pd.to_datetime(grp["observation_date"]).tolist()
        assert dates == sorted(dates), "observation_date must be non-decreasing per vintage"

    # ----- vintages frame: one row per (series, vintage) -----
    assert len(vintages) == len(SERIES) * len(VINTAGES)
    # ----- manifest: every request recorded as ok -----
    assert (manifest["status"] == "ok").all()
    assert manifest["rows"].sum() == expected_rows


def test_recorded_alfred_fixture_directory_exists() -> None:
    """The README documents the re-record procedure (v1.3 item I)."""
    assert FIXTURES_DIR.exists()
    assert (FIXTURES_DIR / "README.md").exists()
