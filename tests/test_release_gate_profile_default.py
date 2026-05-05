# SPDX-License-Identifier: Apache-2.0
"""v1.4.1 (item F) release-gate default profile flip.

Pre-v1.4.1, ``evaluate_release_gate(...)`` with no ``profile=`` kwarg
applied the v1.2.1 looser baseline (``min_confidence=0.55``,
``require_mcs_membership=False``, ``min_coverage=None``). The CLI
``mre release-gate`` had ``--profile`` defaulting to ``None`` so a
production operator running the command with no flags got the loose
defaults.

v1.4.1 flips the default to ``production`` with the following
resolution priority:

1. Explicit ``profile=`` kwarg wins.
2. Else ``MRE_ENV`` env var: ``MRE_ENV=production`` → ``"production"``;
   ``MRE_ENV=dev`` (or ``development`` / ``staging`` / ``test``) →
   ``"default"``.
3. Else fall back to ``"production"``.

Explicit per-rail kwargs (``min_confidence=...``,
``require_mcs_membership=...``, etc.) always win over the
profile-resolved defaults so a caller can relax a single rail without
tearing down the rest of the production posture.
"""

from __future__ import annotations

import pandas as pd
import pytest

from market_regime_engine.release_gates import default_profile, evaluate_release_gate


def _v121_permissive_inputs() -> dict[str, pd.DataFrame]:
    """Inputs that pass the v1.2.1 looser defaults but fail production.

    - ``confidence`` 0.65: above the v1.2.1 0.55 floor but below
      the production 0.75 floor.
    - ``promotion`` row has ``mcs_evidence="absent"``: production
      profile sets ``require_mcs_membership=True``, so this fires the
      ``mcs_evidence_absent`` rail.
    """
    return {
        "confidence": pd.DataFrame([{"date": "2026-05-01", "confidence": 0.65, "grade": "B"}]),
        "drift": pd.DataFrame(),
        "invalidation": pd.DataFrame(),
        "promotion": pd.DataFrame(
            [
                {
                    "target": "drawdown_gt_10pct",
                    "horizon": "3m",
                    "promoted": True,
                    "mcs_evidence": "absent",
                }
            ]
        ),
    }


def test_release_gate_no_profile_no_env_uses_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``profile=`` arg AND no ``MRE_ENV`` → strict production thresholds."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = _v121_permissive_inputs()
    gate = evaluate_release_gate(**inputs)
    row = gate.iloc[0]
    assert bool(row["approved"]) is False
    reasons = str(row["reasons"])
    # Production defaults: min_confidence=0.75 → confidence_below_0.75
    # AND require_mcs_membership=True → mcs_evidence_absent.
    assert "confidence_below_0.75" in reasons, reasons
    assert "mcs_evidence_absent" in reasons, reasons


def test_release_gate_explicit_profile_default_uses_v121_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``profile="default"`` opt-in restores the v1.2.1 looser baseline."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = _v121_permissive_inputs()
    gate = evaluate_release_gate(**inputs, profile="default")
    row = gate.iloc[0]
    assert bool(row["approved"]) is True, row.to_dict()
    assert "confidence_below_" not in str(row["reasons"])
    assert "mcs_evidence_absent" not in str(row["reasons"])


def test_release_gate_mre_env_dev_uses_default_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MRE_ENV=dev`` resolves to the looser default profile."""
    monkeypatch.setenv("MRE_ENV", "dev")
    inputs = _v121_permissive_inputs()
    gate = evaluate_release_gate(**inputs)
    row = gate.iloc[0]
    assert bool(row["approved"]) is True, row.to_dict()


@pytest.mark.parametrize("env_value", ["development", "staging", "test"])
def test_release_gate_mre_env_dev_synonyms_use_default_profile(monkeypatch: pytest.MonkeyPatch, env_value: str) -> None:
    monkeypatch.setenv("MRE_ENV", env_value)
    inputs = _v121_permissive_inputs()
    gate = evaluate_release_gate(**inputs)
    assert bool(gate.iloc[0]["approved"]) is True, env_value


def test_release_gate_mre_env_production_uses_production_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MRE_ENV=production`` resolves to the strict production profile."""
    monkeypatch.setenv("MRE_ENV", "production")
    inputs = _v121_permissive_inputs()
    gate = evaluate_release_gate(**inputs)
    row = gate.iloc[0]
    assert bool(row["approved"]) is False
    assert "mcs_evidence_absent" in str(row["reasons"])


def test_release_gate_explicit_profile_default_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``profile="default"`` wins over ``MRE_ENV=production``."""
    monkeypatch.setenv("MRE_ENV", "production")
    inputs = _v121_permissive_inputs()
    gate = evaluate_release_gate(**inputs, profile="default")
    row = gate.iloc[0]
    assert bool(row["approved"]) is True, row.to_dict()


def test_release_gate_explicit_profile_production_overrides_env_dev(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``profile="production"`` wins over ``MRE_ENV=dev``."""
    monkeypatch.setenv("MRE_ENV", "dev")
    inputs = _v121_permissive_inputs()
    gate = evaluate_release_gate(**inputs, profile="production")
    row = gate.iloc[0]
    assert bool(row["approved"]) is False
    assert "mcs_evidence_absent" in str(row["reasons"])


def test_release_gate_explicit_kwargs_override_profile_resolved_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller-supplied kwargs win over profile-resolved defaults.

    With production resolved by default and an explicit
    ``min_confidence=0.40``, the confidence rail is relaxed to 0.40
    (so confidence=0.65 passes the rail) but every other production
    rail still fires. ``mcs_evidence_absent`` still fires.
    """
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = _v121_permissive_inputs()
    gate = evaluate_release_gate(**inputs, min_confidence=0.40)
    row = gate.iloc[0]
    reasons = str(row["reasons"])
    assert "confidence_below_" not in reasons, reasons
    # Production require_mcs_membership=True still fires.
    assert "mcs_evidence_absent" in reasons, reasons


def test_release_gate_explicit_require_mcs_false_overrides_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ``require_mcs_membership=False`` wins over the
    production-default True. The confidence rail (0.75) still fires
    on confidence=0.65."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = _v121_permissive_inputs()
    gate = evaluate_release_gate(**inputs, require_mcs_membership=False)
    row = gate.iloc[0]
    reasons = str(row["reasons"])
    assert "mcs_evidence_absent" not in reasons, reasons
    assert "confidence_below_0.75" in reasons, reasons


def test_release_gate_explicit_min_coverage_none_overrides_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``min_coverage=None`` wins over the production-default 0.85."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = _v121_permissive_inputs()
    bad_coverage = pd.DataFrame([{"coverage": 0.10, "bucket": "x", "n": 10}])
    gate = evaluate_release_gate(
        **inputs,
        coverage_report=bad_coverage,
        min_coverage=None,
    )
    row = gate.iloc[0]
    # Coverage rail explicitly disabled; the conformal floor reason
    # should NOT fire.
    assert "conformal_coverage_below_floor" not in str(row["reasons"])


def test_default_profile_factory_returns_v121_kwargs() -> None:
    """The :func:`default_profile` factory returns the v1.2.1 looser kwargs."""
    p = default_profile()
    assert p["min_confidence"] == 0.55
    assert p["require_mcs_membership"] is False
    assert p["min_coverage"] is None
    assert p["coverage_drop_pp"] == 0.05
    assert p["promotion_method"] == "mcs"


def test_release_gate_unknown_profile_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typos surface as ValueError so a misconfigured operator fails fast."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = _v121_permissive_inputs()
    with pytest.raises(ValueError, match="Unknown release-gate profile"):
        evaluate_release_gate(**inputs, profile="prodcution")  # typo
