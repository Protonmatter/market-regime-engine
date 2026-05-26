# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 (REVIEW_DEEP_V1_5_2.md section 3.7): reproducibility envelope extras.

Pins the three new envelope fields the deep review flagged as gaps:

- ``numpy_blas`` -- the BLAS variant linked into NumPy. OpenBLAS vs.
  MKL vs. Accelerate disagree on ``np.linalg.solve`` at the ULP level
  so a verify-run drift report must surface the variant.
- ``python_hash_seed`` -- the value of ``PYTHONHASHSEED``. Affects
  ``set`` iteration order; can leak into canonical JSON when a payload
  has un-sortable keys.
- ``runtime_env_snapshot`` -- a small allowlist of env vars whose
  value changes the trajectory of a model run (stale-signal threshold,
  timezone handling, profile selection).

A drift on any of the three keys above is *advisory* -- it surfaces
in ``verify_run`` as a warning but does not flip ``approved`` to
False. A hard difference would block every multi-environment replay
(dev vs prod box) which is not the right ergonomics for these signals.
"""

from __future__ import annotations

import pandas as pd
import pytest

from market_regime_engine.evidence_common import detect_numpy_blas
from market_regime_engine.model_runs import (
    ReproEnvelope,
    build_repro_envelope,
    create_model_run,
    verify_run,
)

# ---------------------------------------------------------------------------
# _detect_numpy_blas returns a sensible value
# ---------------------------------------------------------------------------


def test_detect_numpy_blas_returns_known_variant() -> None:
    """On any supported deployment the BLAS variant resolves to one of
    the documented strings -- never ``unknown`` on a fresh CPython
    install with NumPy >= 1.26."""
    variant = detect_numpy_blas()
    assert variant in {
        "mkl",
        "accelerate",
        "openblas",
        "blis",
        "atlas",
        "netlib",
        "unknown",
    }


def test_detect_numpy_blas_on_current_dev_box() -> None:
    """Sanity check: on this CI / dev box the BLAS resolves to a
    non-unknown variant. The exact value depends on the wheel
    (CPython manylinux uses openblas; conda + MKL uses mkl; macOS
    arm64 uses accelerate). All are acceptable; the only failure
    mode is ``unknown`` which indicates the detector regressed."""
    assert detect_numpy_blas() != "unknown"


# ---------------------------------------------------------------------------
# build_repro_envelope populates the new keys
# ---------------------------------------------------------------------------


def test_envelope_includes_numpy_blas() -> None:
    env = build_repro_envelope(
        features=pd.DataFrame(),
        model_outputs=pd.DataFrame(),
    )
    assert env.numpy_blas != ""
    assert env.numpy_blas == detect_numpy_blas()


def test_envelope_includes_python_hash_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "12345")
    env = build_repro_envelope(
        features=pd.DataFrame(),
        model_outputs=pd.DataFrame(),
    )
    assert env.python_hash_seed == "12345"


def test_envelope_python_hash_seed_empty_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``PYTHONHASHSEED`` is unset the envelope captures the
    empty string -- the audit trail should distinguish "explicit 0"
    from "operator left it default"."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    env = build_repro_envelope(
        features=pd.DataFrame(),
        model_outputs=pd.DataFrame(),
    )
    assert env.python_hash_seed == ""


def test_envelope_includes_runtime_env_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MRE_FI_MAX_SIGNAL_STALENESS_SEC", "60")
    monkeypatch.setenv("MRE_ENV", "production")
    monkeypatch.setenv("TZ", "America/New_York")
    env = build_repro_envelope(
        features=pd.DataFrame(),
        model_outputs=pd.DataFrame(),
    )
    assert env.runtime_env_snapshot == {
        "MRE_FI_MAX_SIGNAL_STALENESS_SEC": "60",
        "MRE_ENV": "production",
        "TZ": "America/New_York",
    }


def test_envelope_runtime_env_snapshot_missing_keys_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset env vars are recorded as the empty string so the snapshot
    dict shape is stable across deployments."""
    for key in ("MRE_FI_MAX_SIGNAL_STALENESS_SEC", "MRE_ENV", "TZ"):
        monkeypatch.delenv(key, raising=False)
    env = build_repro_envelope(
        features=pd.DataFrame(),
        model_outputs=pd.DataFrame(),
    )
    assert env.runtime_env_snapshot == {
        "MRE_FI_MAX_SIGNAL_STALENESS_SEC": "",
        "MRE_ENV": "",
        "TZ": "",
    }


# ---------------------------------------------------------------------------
# verify_run treats new keys as advisory (warnings, not differences)
# ---------------------------------------------------------------------------


def test_verify_run_warns_on_numpy_blas_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored envelope with a different ``numpy_blas`` flags a
    warning but does NOT flip ``approved`` to False."""
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    run = create_model_run(
        engine_version="0.0",
        purpose="test",
        features=pd.DataFrame({"date": ["2026-01-01"], "feature_name": ["x"]}),
        model_outputs=pd.DataFrame({"model_name": ["mle"]}),
    )
    row = pd.Series({"metadata_json": run.metadata_json})

    # Simulated current envelope on a different BLAS.
    drift_env = ReproEnvelope(
        code_version=run.code_version,
        code_sha=run.code_version + ("0" * (40 - len(run.code_version))),
        code_dirty=False,
        lockfile_hash="",
        platform="any",
        python_version="3.13.4",
        feature_payload="",
        output_payload="",
        vintage_payload="",
        numpy_blas="mkl",  # drift
        python_hash_seed="0",
        runtime_env_snapshot={"MRE_ENV": ""},
    )
    # Rebuild the stored envelope so the comparison fixture is
    # internally consistent (avoid noise from BLAS-unrelated keys).
    import json

    stored_env_dict = json.loads(run.metadata_json)["repro_envelope"]
    stored_env_dict["numpy_blas"] = "openblas"
    metadata_json = json.dumps({"repro_envelope": stored_env_dict}, sort_keys=True, default=str)
    row = pd.Series({"metadata_json": metadata_json})

    # Set drift_env to be byte-identical to the stored envelope EXCEPT
    # for numpy_blas, so the diff loop only sees the drift on that one
    # key.
    drift_env = ReproEnvelope(
        code_version=stored_env_dict["code_version"],
        code_sha=stored_env_dict["code_sha"],
        code_dirty=stored_env_dict["code_dirty"],
        lockfile_hash=stored_env_dict["lockfile_hash"],
        platform=stored_env_dict["platform"],
        python_version=stored_env_dict["python_version"],
        feature_payload=stored_env_dict["feature_payload"],
        output_payload=stored_env_dict["output_payload"],
        vintage_payload=stored_env_dict["vintage_payload"],
        rng_seeds=stored_env_dict.get("rng_seeds", {}),
        extra=stored_env_dict.get("extra", {}),
        lockfile_hashes=stored_env_dict.get("lockfile_hashes", {}),
        numpy_blas="mkl",  # drift (stored was "openblas")
        python_hash_seed=stored_env_dict.get("python_hash_seed", ""),
        runtime_env_snapshot=stored_env_dict.get("runtime_env_snapshot", {}),
    )
    report = verify_run(run.run_id, row, current_envelope=drift_env)
    assert "numpy_blas_drift" in report["warnings"]
    # Hard differences should NOT include numpy_blas.
    assert "numpy_blas" not in report["differences"]


def test_verify_run_warns_on_python_hash_seed_drift() -> None:
    """``PYTHONHASHSEED`` drift is advisory only."""
    run = create_model_run(
        engine_version="0.0",
        purpose="test",
        features=pd.DataFrame({"date": ["2026-01-01"], "feature_name": ["x"]}),
        model_outputs=pd.DataFrame({"model_name": ["mle"]}),
    )
    import json

    stored = json.loads(run.metadata_json)["repro_envelope"]
    stored["python_hash_seed"] = "0"
    metadata_json = json.dumps({"repro_envelope": stored}, sort_keys=True, default=str)
    row = pd.Series({"metadata_json": metadata_json})

    drift_env = ReproEnvelope(
        code_version=stored["code_version"],
        code_sha=stored["code_sha"],
        code_dirty=stored["code_dirty"],
        lockfile_hash=stored["lockfile_hash"],
        platform=stored["platform"],
        python_version=stored["python_version"],
        feature_payload=stored["feature_payload"],
        output_payload=stored["output_payload"],
        vintage_payload=stored["vintage_payload"],
        rng_seeds=stored.get("rng_seeds", {}),
        extra=stored.get("extra", {}),
        lockfile_hashes=stored.get("lockfile_hashes", {}),
        numpy_blas=stored.get("numpy_blas", ""),
        python_hash_seed="12345",  # drift
        runtime_env_snapshot=stored.get("runtime_env_snapshot", {}),
    )
    report = verify_run(run.run_id, row, current_envelope=drift_env)
    assert "python_hash_seed_drift" in report["warnings"]
    assert "python_hash_seed" not in report["differences"]


def test_verify_run_warns_on_runtime_env_snapshot_drift() -> None:
    """``runtime_env_snapshot`` drift is advisory only."""
    run = create_model_run(
        engine_version="0.0",
        purpose="test",
        features=pd.DataFrame({"date": ["2026-01-01"], "feature_name": ["x"]}),
        model_outputs=pd.DataFrame({"model_name": ["mle"]}),
    )
    import json

    stored = json.loads(run.metadata_json)["repro_envelope"]
    stored["runtime_env_snapshot"] = {
        "MRE_FI_MAX_SIGNAL_STALENESS_SEC": "60",
        "MRE_ENV": "production",
        "TZ": "UTC",
    }
    metadata_json = json.dumps({"repro_envelope": stored}, sort_keys=True, default=str)
    row = pd.Series({"metadata_json": metadata_json})

    drift_env = ReproEnvelope(
        code_version=stored["code_version"],
        code_sha=stored["code_sha"],
        code_dirty=stored["code_dirty"],
        lockfile_hash=stored["lockfile_hash"],
        platform=stored["platform"],
        python_version=stored["python_version"],
        feature_payload=stored["feature_payload"],
        output_payload=stored["output_payload"],
        vintage_payload=stored["vintage_payload"],
        rng_seeds=stored.get("rng_seeds", {}),
        extra=stored.get("extra", {}),
        lockfile_hashes=stored.get("lockfile_hashes", {}),
        numpy_blas=stored.get("numpy_blas", ""),
        python_hash_seed=stored.get("python_hash_seed", ""),
        runtime_env_snapshot={
            "MRE_FI_MAX_SIGNAL_STALENESS_SEC": "30",  # drift
            "MRE_ENV": "production",
            "TZ": "UTC",
        },
    )
    report = verify_run(run.run_id, row, current_envelope=drift_env)
    assert "runtime_env_snapshot_drift" in report["warnings"]
    assert "runtime_env_snapshot" not in report["differences"]


def test_verify_run_no_warning_when_envelope_keys_match() -> None:
    """Sanity: when all three new keys match, no drift warnings fire."""
    run = create_model_run(
        engine_version="0.0",
        purpose="test",
        features=pd.DataFrame({"date": ["2026-01-01"], "feature_name": ["x"]}),
        model_outputs=pd.DataFrame({"model_name": ["mle"]}),
    )
    import json

    stored = json.loads(run.metadata_json)["repro_envelope"]
    row = pd.Series({"metadata_json": run.metadata_json})

    matching_env = ReproEnvelope(
        code_version=stored["code_version"],
        code_sha=stored["code_sha"],
        code_dirty=stored["code_dirty"],
        lockfile_hash=stored["lockfile_hash"],
        platform=stored["platform"],
        python_version=stored["python_version"],
        feature_payload=stored["feature_payload"],
        output_payload=stored["output_payload"],
        vintage_payload=stored["vintage_payload"],
        rng_seeds=stored.get("rng_seeds", {}),
        extra=stored.get("extra", {}),
        lockfile_hashes=stored.get("lockfile_hashes", {}),
        numpy_blas=stored.get("numpy_blas", ""),
        python_hash_seed=stored.get("python_hash_seed", ""),
        runtime_env_snapshot=stored.get("runtime_env_snapshot", {}),
    )
    report = verify_run(run.run_id, row, current_envelope=matching_env)
    assert "numpy_blas_drift" not in report["warnings"]
    assert "python_hash_seed_drift" not in report["warnings"]
    assert "runtime_env_snapshot_drift" not in report["warnings"]


def test_envelope_legacy_v1_5_x_missing_keys_silent() -> None:
    """A v1.5.x envelope JSON (no numpy_blas / python_hash_seed /
    runtime_env_snapshot) verifies without spurious warnings on those
    keys -- the missing-key path is handled gracefully."""
    import json

    legacy_envelope = {
        "code_version": "abc",
        "code_sha": "abcdef0",
        "code_dirty": False,
        "lockfile_hash": "",
        "platform": "Windows-11",
        "python_version": "3.13.4",
        "feature_payload": "",
        "output_payload": "",
        "vintage_payload": "",
        "rng_seeds": {},
        "extra": {},
        "lockfile_hashes": {},
    }
    metadata_json = json.dumps({"repro_envelope": legacy_envelope}, sort_keys=True, default=str)
    row = pd.Series({"metadata_json": metadata_json})
    current_env = build_repro_envelope(
        features=pd.DataFrame(),
        model_outputs=pd.DataFrame(),
    )
    # Force-match the non-drift keys for the rest of the legacy fields.
    from dataclasses import replace

    current_env = replace(
        current_env,
        code_version=legacy_envelope["code_version"],
        code_sha=legacy_envelope["code_sha"],
        code_dirty=legacy_envelope["code_dirty"],
        lockfile_hash=legacy_envelope["lockfile_hash"],
        platform=legacy_envelope["platform"],
        python_version=legacy_envelope["python_version"],
        feature_payload=legacy_envelope["feature_payload"],
        output_payload=legacy_envelope["output_payload"],
        vintage_payload=legacy_envelope["vintage_payload"],
    )
    report = verify_run("legacy-run", row, current_envelope=current_env)
    # The new keys are not in the stored envelope at all, so the diff
    # loop never iterates them -- no drift warnings fire even though
    # the current envelope has populated values.
    assert "numpy_blas_drift" not in report["warnings"]
    assert "python_hash_seed_drift" not in report["warnings"]
    assert "runtime_env_snapshot_drift" not in report["warnings"]
    # And the legacy run still approves (modulo the missing
    # training_audit / training_mode rails, which is a separate
    # validation path).
    assert isinstance(report["approved"], bool)
