# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from market_regime_engine.fixed_income.numeric_contracts import (
    DEFAULT_NUMERIC_POLICY,
    assert_no_float_artifact,
    bps_to_q4,
    money_to_cents,
    price_to_q6,
    prob_to_ppm,
    timestamp_to_epoch_ns_str,
)


def test_fixed_point_quantizers_use_half_even_rounding() -> None:
    assert prob_to_ppm(0.1234565) == 123456
    assert prob_to_ppm("0.1234575") == 123458
    assert bps_to_q4("2.12505") == 21250
    assert price_to_q6("101.2500005") == 101250000
    assert money_to_cents("10.005") == 1000
    assert DEFAULT_NUMERIC_POLICY.rounding == "half_even"


def test_timestamp_epoch_ns_is_string_token() -> None:
    out = timestamp_to_epoch_ns_str("2026-05-26T12:31:00Z")
    assert out == "1779798660000000000"
    assert isinstance(out, str)


def test_quantizers_reject_nonfinite_and_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="finite"):
        prob_to_ppm(float("nan"))
    with pytest.raises(ValueError, match="tz-aware"):
        timestamp_to_epoch_ns_str("2026-05-26T12:31:00")


def test_assert_no_float_artifact_reports_path() -> None:
    assert_no_float_artifact({"ok": {"score_ppm": 123456}, "items": [1, "x", None]})
    with pytest.raises(TypeError, match="root\\.model_outputs\\.score"):
        assert_no_float_artifact({"model_outputs": {"score": 0.1}})
