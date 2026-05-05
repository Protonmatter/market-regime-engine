"""Regression tests for the v1.2.1 ``verify_run`` training-audit checks.

Pre-v1.2.1, ``verify_run`` skipped the ``extra`` field of the stored
reproducibility envelope. That meant the ``training_audit`` dict (which
records whether the run trained on PIT data, on a silent legacy fallback,
or on an explicitly-authorized legacy fallback) was stored but never
verified. The verify-run gate could approve a run that had been quietly
trained on revised macro data — exactly the scenario the audit was
designed to catch.

v1.2.1 changes:

- ``extra`` is verified structurally; ``extra.training_audit.mode_used``
  must be ``"point_in_time"`` for the report to remain ``approved``.
- A ``training_mode_drift`` entry is appended to ``differences`` when the
  mode is anything other than ``"point_in_time"``.
- When ``training_audit.fallback_authorized`` is True the run is still
  approved (operator opted in) but ``"legacy_fallback_authorized"`` is
  appended to ``warnings`` so a change-management gate can see the
  conscious downgrade.
"""

from __future__ import annotations

import json

import pandas as pd

from market_regime_engine.model_runs import (
    ReproEnvelope,
    build_repro_envelope,
    create_model_run,
    verify_run,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature_name": "f1", "date": "2024-01-01", "value": 1.0},
            {"feature_name": "f2", "date": "2024-01-01", "value": 2.0},
        ]
    )


def _outputs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "m",
                "date": "2024-01-01",
                "horizon": "3m",
                "target": "t",
                "value": 0.5,
            }
        ]
    )


def _build_run(audit: dict) -> tuple[pd.Series, ReproEnvelope]:
    """Create a model run with the supplied training audit and return the
    row + a fresh envelope built from the same inputs (so all
    feature/output/vintage payload hashes match by construction).

    v1.4.1 (item D): the structural ``extra`` compare is strict, so
    the current envelope must carry forward the same auto-stamped
    ``engine_version`` + ``purpose`` keys that ``create_model_run``
    persists. ``training_audit`` is *deliberately* not forwarded so
    the existing v1.2.1 friendly-handling tests continue to surface
    the ``training_mode_drift`` / ``legacy_fallback_authorized``
    semantics unchanged.
    """
    features = _features()
    outputs = _outputs()
    run = create_model_run(
        engine_version="1.2.1-test",
        purpose="verify_run unit test",
        features=features,
        model_outputs=outputs,
        training_audit=audit,
    )
    row = pd.Series(
        {
            "run_id": run.run_id,
            "metadata_json": run.metadata_json,
        }
    )
    envelope = build_repro_envelope(
        features=features,
        model_outputs=outputs,
        extra={"engine_version": "1.2.1-test", "purpose": "verify_run unit test"},
    )
    return row, envelope


def test_verify_run_passes_when_training_mode_is_pit() -> None:
    audit = {
        "mode": "point_in_time",
        "mode_used": "point_in_time",
        "fallback_authorized": False,
        "rows": 100,
        "as_of_dates": 50,
    }
    row, envelope = _build_run(audit)
    report = verify_run(row["run_id"], row, current_envelope=envelope)
    assert report["approved"] is True, report
    assert "training_mode_drift" not in report["differences"]
    assert report["warnings"] == []


def test_verify_run_fails_when_training_mode_is_legacy() -> None:
    """Pre-v1.2.1 this approved silently because ``extra`` was skipped."""
    audit = {
        "mode": "point_in_time",
        "mode_used": "legacy",
        "fallback_authorized": False,
        "rows": 100,
        "fallback_reason": "feature_asof_values empty",
    }
    row, envelope = _build_run(audit)
    report = verify_run(row["run_id"], row, current_envelope=envelope)
    assert report["approved"] is False, report
    assert "training_mode_drift" in report["differences"]
    drift = report["differences"]["training_mode_drift"]
    assert drift["stored_mode"] == "legacy"
    assert drift["expected"] == "point_in_time"


def test_verify_run_fails_when_fail_closed_audit_is_stored() -> None:
    """A run whose audit was stamped ``mode_used == "fail_closed"`` (i.e. the
    operator's training step raised RuntimeError but somehow the run record
    was created anyway) must also fail verification."""
    audit = {
        "mode": "point_in_time",
        "mode_used": "fail_closed",
        "fallback_authorized": False,
        "fallback_reason": "feature_asof_values empty",
    }
    row, envelope = _build_run(audit)
    report = verify_run(row["run_id"], row, current_envelope=envelope)
    assert report["approved"] is False
    assert "training_mode_drift" in report["differences"]


def test_verify_run_warns_when_fallback_authorized() -> None:
    """``allow_legacy_fallback=True`` is approvable but must surface a
    non-fatal warning so operators see the conscious downgrade."""
    audit = {
        "mode": "point_in_time",
        "mode_used": "legacy_fallback_explicit",
        "fallback_authorized": True,
        "fallback_reason": "feature_asof_values empty",
        "rows": 100,
    }
    row, envelope = _build_run(audit)
    report = verify_run(row["run_id"], row, current_envelope=envelope)
    # Authorized fallback is still legacy mode, so it must NOT be
    # approved — the gate should require the operator to either run
    # materialize-asof-features or accept that the run is non-PIT.
    assert report["approved"] is False
    assert "training_mode_drift" in report["differences"]
    # AND the audit should be surfaced as an explicit advisory.
    assert "legacy_fallback_authorized" in report["warnings"]


def test_verify_run_warning_only_on_explicit_authorization() -> None:
    """A LEGACY-mode run that did not opt in to fallback should not surface
    the ``legacy_fallback_authorized`` warning (it's reserved for the
    explicit opt-in path)."""
    audit = {
        "mode": "legacy",
        "mode_used": "legacy",
        "fallback_authorized": False,
        "rows": 100,
    }
    row, envelope = _build_run(audit)
    report = verify_run(row["run_id"], row, current_envelope=envelope)
    assert "legacy_fallback_authorized" not in report["warnings"]


def test_verify_run_handles_missing_training_audit_cleanly() -> None:
    """A run created without a training_audit (older runs from v1.0/v1.1 or
    a manual call site) must still verify cleanly — the absence of an
    audit is not itself a drift signal."""
    features = _features()
    outputs = _outputs()
    run = create_model_run(
        engine_version="1.2.1-test",
        purpose="verify_run unit test (no audit)",
        features=features,
        model_outputs=outputs,
    )
    row = pd.Series({"run_id": run.run_id, "metadata_json": run.metadata_json})
    envelope = build_repro_envelope(
        features=features,
        model_outputs=outputs,
        extra={"engine_version": "1.2.1-test", "purpose": "verify_run unit test (no audit)"},
    )
    report = verify_run(row["run_id"], row, current_envelope=envelope)
    assert report["approved"] is True, report
    assert report["warnings"] == []


def test_verify_run_extra_is_no_longer_in_skip_set() -> None:
    """Static guard: the skip set must be exactly {"rng_seeds"} so the
    governance contract is enforced by the test, not by reading the
    source. Anything that re-introduces ``extra`` to the skip set should
    immediately trip this test."""
    import inspect

    from market_regime_engine import model_runs

    src = inspect.getsource(model_runs.verify_run)
    # Allow either the literal set or the "in {...}" form.
    assert '{"rng_seeds", "extra"}' not in src, (
        "verify_run skip set must not include 'extra'; the training audit lives there and must be verified."
    )
    # Sanity: ensure the new structural check is present.
    assert "training_mode_drift" in src
    assert "legacy_fallback_authorized" in src


def test_create_model_run_records_training_audit_in_metadata_v121() -> None:
    """Round-trip: ``create_model_run(training_audit=...)`` must embed the
    audit into ``metadata.training_audit`` AND
    ``repro_envelope.extra.training_audit`` so verify-run can read it."""
    audit = {
        "mode": "point_in_time",
        "mode_used": "point_in_time",
        "fallback_authorized": False,
        "rows": 200,
    }
    run = create_model_run(
        engine_version="1.2.1-test",
        purpose="round-trip",
        features=_features(),
        model_outputs=_outputs(),
        training_audit=audit,
    )
    meta = json.loads(run.metadata_json)
    assert meta["training_audit"] == audit
    assert meta["repro_envelope"]["extra"]["training_audit"] == audit
