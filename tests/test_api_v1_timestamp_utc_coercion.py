# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 — A9 / Finding §3.4 regression tests.

Pin the contract that the Pydantic timestamp validator coerces any
tz-aware ISO-8601 timestamp to canonical UTC ``...Z`` form BEFORE the
value flows into canonical-JSON hashing. Two requests for the same
logical instant from different operator timezones must therefore
produce identical ``artifact_hash`` values.

Before this fix, the validator accepted ``+05:30`` and ``Z`` strings
verbatim, so two operators submitting the same instant under different
local offsets produced different canonical bytes and different
artifact hashes.
"""

from __future__ import annotations

import pytest

from market_regime_engine.fixed_income.api import ExecutionConfidenceRequestModel


def _base_payload(timestamp: str) -> dict:
    return {
        "timestamp": timestamp,
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000.0,
        "protocol": "Auto-X",
        "urgency": "normal",
        "request_id": "req-a9",
    }


def test_z_suffixed_timestamp_is_returned_canonical_unchanged() -> None:
    payload = _base_payload("2026-05-08T14:30:00Z")
    model = ExecutionConfidenceRequestModel(**payload)
    assert model.timestamp == "2026-05-08T14:30:00Z"


def test_offset_timestamp_is_coerced_to_utc_z_form() -> None:
    """A9 core contract: a ``-04:00`` offset produces the SAME canonical
    string as the equivalent ``Z`` form."""
    payload_offset = _base_payload("2026-05-08T10:30:00-04:00")
    payload_utc = _base_payload("2026-05-08T14:30:00Z")
    model_offset = ExecutionConfidenceRequestModel(**payload_offset)
    model_utc = ExecutionConfidenceRequestModel(**payload_utc)
    assert model_offset.timestamp == "2026-05-08T14:30:00Z"
    assert model_offset.timestamp == model_utc.timestamp


def test_microsecond_precision_is_preserved_in_canonical_form() -> None:
    payload = _base_payload("2026-05-08T14:30:00.123456+00:00")
    model = ExecutionConfidenceRequestModel(**payload)
    assert model.timestamp == "2026-05-08T14:30:00.123456Z"


def test_offset_with_microseconds_is_coerced_to_utc_with_microseconds() -> None:
    payload = _base_payload("2026-05-08T10:30:00.987654-04:00")
    model = ExecutionConfidenceRequestModel(**payload)
    assert model.timestamp == "2026-05-08T14:30:00.987654Z"


def test_naive_timestamp_still_rejected() -> None:
    payload = _base_payload("2026-05-08T14:30:00")
    with pytest.raises(ValueError, match="explicit tz info"):
        ExecutionConfidenceRequestModel(**payload)


def test_to_dataclass_propagates_canonical_timestamp() -> None:
    """A9: the canonical Z-suffixed form flows into the internal
    dataclass so downstream hashing sees the normalised string."""
    payload = _base_payload("2026-05-08T10:30:00-04:00")
    model = ExecutionConfidenceRequestModel(**payload)
    dc = model.to_dataclass()
    assert dc.timestamp == "2026-05-08T14:30:00Z"


def test_artifact_hash_identical_for_equivalent_offset_and_utc_inputs() -> None:
    """A9 end-to-end: two semantically-identical inbound timestamps
    produce the same canonical artifact bytes therefore the same
    SHA-256."""
    from market_regime_engine.fixed_income.hashing import canonical_sha256

    payload_offset = _base_payload("2026-05-08T10:30:00-04:00")
    payload_utc = _base_payload("2026-05-08T14:30:00Z")
    model_offset = ExecutionConfidenceRequestModel(**payload_offset)
    model_utc = ExecutionConfidenceRequestModel(**payload_utc)

    # The artifact payload that downstream code hashes always uses the
    # normalised timestamp coming out of the validator; mirroring the
    # production score_execution_confidence wrapper:
    artifact_offset = {
        "timestamp": model_offset.timestamp,
        "cusip": model_offset.cusip,
        "side": model_offset.side,
    }
    artifact_utc = {
        "timestamp": model_utc.timestamp,
        "cusip": model_utc.cusip,
        "side": model_utc.side,
    }
    assert canonical_sha256(artifact_offset) == canonical_sha256(artifact_utc)
