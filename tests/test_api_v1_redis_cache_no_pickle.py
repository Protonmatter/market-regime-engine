# SPDX-License-Identifier: Apache-2.0
"""PR-5 AF-5 / ASK-10: Redis cache defaults to JSON, pickle is opt-in.

The legacy ``_RedisTTLCache`` used ``pickle.dumps`` / ``pickle.loads`` on
every entry. An attacker with write access to the shared Redis instance
(or who could inject keys via another service) could land arbitrary-code
execution on the FastAPI worker. PR-5 makes JSON the default; pickle is
gated behind ``MRE_CACHE_ALLOW_PICKLE=1``.
"""

from __future__ import annotations

import json
import pickle

from market_regime_engine.api_v1 import (
    _deserialize_cache_value,
    _serialize_cache_value,
)


def test_default_serializer_is_json(monkeypatch) -> None:
    monkeypatch.delenv("MRE_CACHE_ALLOW_PICKLE", raising=False)
    blob = _serialize_cache_value({"date": "2026-05-01", "score": 75.2})
    assert json.loads(blob.decode("utf-8")) == {"date": "2026-05-01", "score": 75.2}


def test_default_deserializer_rejects_pickle_payload(monkeypatch) -> None:
    monkeypatch.delenv("MRE_CACHE_ALLOW_PICKLE", raising=False)
    pickled = pickle.dumps({"x": 1})
    # JSON path tries ``json.loads(blob.decode('utf-8'))``; pickle's
    # binary header (\x80) is not valid UTF-8 so the deserialiser must
    # fail closed (raise, not load). Either UnicodeDecodeError or
    # JSONDecodeError is acceptable — both surface the rejection.
    import pytest

    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
        _deserialize_cache_value(pickled)


def test_pickle_allowed_when_opted_in(monkeypatch) -> None:
    monkeypatch.setenv("MRE_CACHE_ALLOW_PICKLE", "1")
    obj = {"score": 0.95}
    blob = _serialize_cache_value(obj)
    assert _deserialize_cache_value(blob) == obj


def test_pickle_opt_in_falls_back_to_json_on_non_pickle_payload(monkeypatch) -> None:
    """An in-place toggle from JSON → pickle must not invalidate previously
    stored JSON entries. The pickle-opt-in deserialiser falls back to JSON
    when pickle parsing fails."""
    monkeypatch.delenv("MRE_CACHE_ALLOW_PICKLE", raising=False)
    json_blob = _serialize_cache_value({"x": 1})
    monkeypatch.setenv("MRE_CACHE_ALLOW_PICKLE", "1")
    assert _deserialize_cache_value(json_blob) == {"x": 1}


def test_json_round_trip_includes_pandas_records(monkeypatch) -> None:
    """The default ``default=str`` falls back to ``str(value)`` for non-JSON
    types like pandas Timestamp / numpy scalars so the encoder never
    crashes on a typical model_output records payload."""
    import pandas as pd

    monkeypatch.delenv("MRE_CACHE_ALLOW_PICKLE", raising=False)
    payload = {
        "as_of_date": pd.Timestamp("2026-05-01"),
        "score": 75.2,
        "records": [
            {"date": pd.Timestamp("2026-05-01"), "value": 1.5},
        ],
    }
    blob = _serialize_cache_value(payload)
    decoded = _deserialize_cache_value(blob)
    assert decoded["score"] == 75.2
    # Timestamp coerces to its str() form on serialise.
    assert "2026-05-01" in decoded["as_of_date"]
