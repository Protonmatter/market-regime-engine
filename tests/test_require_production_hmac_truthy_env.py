# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 — F5 / Finding §3.11 regression tests.

Pin the contract that ``require_production_hmac`` recognises every
``{1, true, yes, on}`` truthy string (case-insensitive, whitespace
stripped) on ``MRE_FI_REQUIRE_HMAC`` so the FI HMAC enforcement is
consistent with ``rate_limit_enabled`` (api.py) and other env-var
helpers in this repo.

Before this fix the validator used exact ``=='1'`` matching, so an
operator setting ``MRE_FI_REQUIRE_HMAC=true`` would silently disable
the enforcement.
"""

from __future__ import annotations

import pytest

from market_regime_engine.fixed_income.evidence_pack import require_production_hmac


@pytest.mark.parametrize(
    "value",
    ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON", " 1 ", "  true\t"],
)
def test_require_production_hmac_truthy_values_enforce(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.delenv("MRE_ENV", raising=False)
    monkeypatch.setenv("MRE_FI_REQUIRE_HMAC", value)
    assert require_production_hmac() is True


@pytest.mark.parametrize(
    "value",
    ["0", "false", "no", "off", "", "  ", "maybe", "2"],
)
def test_require_production_hmac_falsy_values_do_not_enforce(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.delenv("MRE_ENV", raising=False)
    monkeypatch.setenv("MRE_FI_REQUIRE_HMAC", value)
    assert require_production_hmac() is False


def test_require_production_hmac_mre_env_production_still_enforces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-compat: ``MRE_ENV=production`` alone (no MRE_FI_REQUIRE_HMAC)
    still triggers enforcement."""
    monkeypatch.delenv("MRE_FI_REQUIRE_HMAC", raising=False)
    monkeypatch.setenv("MRE_ENV", "production")
    assert require_production_hmac() is True


def test_require_production_hmac_returns_false_when_both_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MRE_FI_REQUIRE_HMAC", raising=False)
    monkeypatch.delenv("MRE_ENV", raising=False)
    assert require_production_hmac() is False
