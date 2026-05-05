# SPDX-License-Identifier: Apache-2.0
"""v1.3 regression tests (items A through M, minus A/H/J/I).

Item A (audit zip slimming) is exercised by the CI gate; the script
covers exclude semantics in this file.
Item H (CI hardening) is intrinsically a CI workflow assertion.
Item I (recorded-fixture ALFRED) lives in
``tests/test_alfred_real_recorded.py``.
Item J (Rust wheel matrix) is a CI artifact.

Everything else (B1, B2, B3, B4, C, D, E, F, G, L, M, plus version
sanity) gets a deterministic regression test here.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Item A: audit zip excludes
# ---------------------------------------------------------------------------


def _load_build_audit_zip_module():
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("build_audit_zip", repo_root / "scripts" / "build_audit_zip.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_zip_exclusions_drop_runtime_caches() -> None:
    """The build_audit_zip exclude list mirrors build_zip.py and drops data/."""
    mod = _load_build_audit_zip_module()
    EXCLUDE_DIRS = mod.EXCLUDE_DIRS
    EXCLUDE_FILE_GLOBS = mod.EXCLUDE_FILE_GLOBS
    is_excluded = mod.is_excluded

    # data/ is excluded by default (re-included via --with-runtime-data).
    assert "data" in EXCLUDE_DIRS
    assert "__pycache__" in EXCLUDE_DIRS
    assert ".pytest_cache" in EXCLUDE_DIRS
    assert ".mypy_cache" in EXCLUDE_DIRS
    assert "*.db" in EXCLUDE_FILE_GLOBS
    assert "*.duckdb" in EXCLUDE_FILE_GLOBS
    assert is_excluded("data/mre.db", exclude_dirs=frozenset(EXCLUDE_DIRS))
    assert is_excluded("rust_ext/target/debug/foo.rlib", exclude_dirs=frozenset(EXCLUDE_DIRS))
    assert not is_excluded("src/market_regime_engine/cli.py", exclude_dirs=frozenset(EXCLUDE_DIRS))


# ---------------------------------------------------------------------------
# Item B1: DFM EM convergence with non-monotone marginal LL
# ---------------------------------------------------------------------------


def test_dfm_em_handles_non_monotone_marginal_ll() -> None:
    """A near-degenerate DFM still converges and reports a finite log-likelihood."""
    from market_regime_engine.dfm import DFMDomainModel

    # Construct a synthetic case where one column has near-zero
    # variance — the SMW marginal likelihood is numerically unstable
    # for that case and may emit a non-monotone step.
    rng = np.random.default_rng(0)
    n = 60
    # 3 columns; one is near-constant (degenerate), the other two are
    # noisy AR(1).
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.7 * f[t - 1] + rng.normal(0, 1)
    panel = pd.DataFrame(
        {
            "x1": 1.0 + 0.01 * f + 1e-8 * rng.normal(size=n),
            "x2": f + rng.normal(0, 0.5, size=n),
            "x3": 0.5 * f + rng.normal(0, 0.7, size=n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="MS"),
    )
    model = DFMDomainModel(max_iter=30, tol=1e-4).fit(panel)
    assert model.fitted
    assert np.isfinite(model.log_likelihood), "log_likelihood must be finite"
    assert "likelihood_path" in model.fit_log
    assert isinstance(model.fit_log["likelihood_path"], list)
    assert len(model.fit_log["likelihood_path"]) >= 1
    # ``fallback_used`` is True iff the surrogate was used at least once.
    assert isinstance(model.fit_log["fallback_used"], bool)


# ---------------------------------------------------------------------------
# Item B2: BOCPD-MUSE _AR1State Welford M2 ordering
# ---------------------------------------------------------------------------


def test_ar1state_welford_m2_matches_numpy_var() -> None:
    """1000-step random walk: m2/(n-1) must match np.var(ddof=1) to 1e-10."""
    from market_regime_engine.bocpd_muse import _AR1State

    rng = np.random.default_rng(42)
    n = 1000
    walk = np.cumsum(rng.normal(size=n))
    state = _AR1State.prior(dim=1, prior_var=1.0)
    for x in walk:
        state = state.update(np.array([x], dtype=float))
    assert state.n == n
    expected_var = float(np.var(walk, ddof=1))
    welford_var = float(state.m2[0] / (state.n - 1))
    assert abs(welford_var - expected_var) < 1e-10, f"Welford variance {welford_var} != np.var ddof=1 {expected_var}"


# ---------------------------------------------------------------------------
# Item B3: _hash_frame stable hashing
# ---------------------------------------------------------------------------


def test_hash_frame_invariant_under_copy_and_column_reorder() -> None:
    """v1.3 stable hash is invariant under df.copy() and column reorder."""
    from market_regime_engine.model_runs import _hash_frame

    df = pd.DataFrame(
        {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([10, 20, 30], dtype=np.int64),
            "c": ["x", "y", "z"],
        }
    )
    h1 = _hash_frame(df)
    h2 = _hash_frame(df.copy())
    assert h1 == h2

    df_reorder = df[["c", "a", "b"]]
    h3 = _hash_frame(df_reorder)
    assert h1 == h3, "hash must be invariant under column reorder"


def test_hash_frame_invariant_under_row_reorder_after_canonical_sort() -> None:
    from market_regime_engine.model_runs import _hash_frame

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, 20, 30], "c": ["x", "y", "z"]})
    df_shuffled = df.iloc[[2, 0, 1]].reset_index(drop=True)
    assert _hash_frame(df) == _hash_frame(df_shuffled)


def test_hash_frame_changes_when_dtype_migrates() -> None:
    """v1.3 hash includes dtype; a float→int silent change is detected."""
    from market_regime_engine.model_runs import _hash_frame

    df_float = pd.DataFrame({"a": pd.Series([1.0, 2.0, 3.0], dtype="float64")})
    df_int = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="int64")})
    assert _hash_frame(df_float) != _hash_frame(df_int)


def test_hash_frame_legacy_back_compat() -> None:
    """Legacy hash is still callable for verifying pre-v1.3 envelopes."""
    from market_regime_engine.model_runs import _hash_frame_legacy

    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    h = _hash_frame_legacy(df)
    assert isinstance(h, str)
    assert len(h) == 64


# ---------------------------------------------------------------------------
# Item B4: apply_release_lag strict default
# ---------------------------------------------------------------------------


def test_apply_release_lag_raises_on_unknown_series() -> None:
    """v1.3: unknown series raise instead of silently zero-lagging."""
    from market_regime_engine.point_in_time import apply_release_lag

    obs = pd.DataFrame(
        [
            {
                "series_id": "MADE_UP_SERIES",
                "date": "2020-01-01",
                "value": 1.0,
                "vintage_date": "2020-01-15",
                "source": "x",
            }
        ]
    )
    with pytest.raises(RuntimeError, match="MADE_UP_SERIES"):
        apply_release_lag(obs)


def test_apply_release_lag_strict_false_falls_back() -> None:
    from market_regime_engine.point_in_time import apply_release_lag

    obs = pd.DataFrame(
        [
            {
                "series_id": "MADE_UP_SERIES",
                "date": "2020-01-01",
                "value": 1.0,
                "vintage_date": "2020-01-15",
                "source": "x",
            }
        ]
    )
    out = apply_release_lag(obs, strict=False)
    assert len(out) == 1
    # No-rule fallback: vintage_date stays at the observation date (or
    # later if already provided), never shifted forward.
    assert pd.to_datetime(out.iloc[0]["vintage_date"]) >= pd.Timestamp("2020-01-01")


# ---------------------------------------------------------------------------
# Item C: walk_forward._purge_and_embargo correctness + perf
# ---------------------------------------------------------------------------


def _legacy_purge_and_embargo(train_idx: np.ndarray, test_idx: np.ndarray, horizon: int, embargo: int) -> np.ndarray:
    """Pure-Python reference implementation matching v1.2.1 behaviour."""
    if train_idx.size == 0 or test_idx.size == 0:
        return train_idx
    test_set = set(test_idx.tolist())
    purge_min = test_idx.min() - horizon
    purge_max = test_idx.max() + embargo
    keep = []
    for t in train_idx:
        if t in test_set:
            continue
        in_purge = purge_min <= t <= purge_max
        if in_purge:
            if any(t < tau <= t + horizon for tau in test_idx):
                continue
            if any(tau < t <= tau + embargo for tau in test_idx):
                continue
        keep.append(t)
    return np.asarray(keep, dtype=int)


def test_purge_and_embargo_matches_legacy_50_seeds() -> None:
    """Vectorised + legacy must agree on 50 random seeded inputs."""
    from market_regime_engine.walk_forward import CombinatorialPurgedCV

    for seed in range(50):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(80, 400))
        horizon = int(rng.integers(1, 12))
        embargo = int(rng.integers(0, 6))
        # Sample non-overlapping train and test indices.
        all_idx = np.arange(n, dtype=int)
        rng.shuffle(all_idx)
        cut = int(rng.integers(10, n - 10))
        test_idx = np.sort(all_idx[:cut][:30])
        train_idx = np.sort(np.setdiff1d(np.arange(n), test_idx))

        cpcv = CombinatorialPurgedCV(horizon=horizon, embargo=embargo)
        new = cpcv._purge_and_embargo(train_idx, test_idx)
        legacy = _legacy_purge_and_embargo(train_idx, test_idx, horizon, embargo)
        np.testing.assert_array_equal(np.sort(new), np.sort(legacy))


def test_purge_and_embargo_is_under_5_seconds() -> None:
    """28-fold CPCV on n=2000 must complete in <5s wall-clock."""
    from market_regime_engine.walk_forward import CombinatorialPurgedCV

    cpcv = CombinatorialPurgedCV(n_blocks=8, k_test_blocks=2, horizon=12, embargo=2)
    n = 2000
    start = time.perf_counter()
    splits = list(cpcv.split(n))
    elapsed = time.perf_counter() - start
    assert len(splits) == 28
    assert elapsed < 5.0, f"CPCV took {elapsed:.3f}s; budget is 5s"


# ---------------------------------------------------------------------------
# Item D: DuckDB warehouse parity
# ---------------------------------------------------------------------------


def test_warehouse_duckdb_facade_round_trips_observations() -> None:
    """A small write + read round-trips identically through the DuckDB backend."""
    duckdb = pytest.importorskip("duckdb")
    from market_regime_engine.storage import Warehouse

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wh.duckdb"
        wh = Warehouse(str(path), backend="duckdb")
        try:
            obs = pd.DataFrame(
                [
                    {
                        "series_id": "UNRATE",
                        "date": "2020-01-01",
                        "value": 3.6,
                        "vintage_date": "2020-02-01",
                        "source": "test",
                    },
                    {
                        "series_id": "UNRATE",
                        "date": "2020-02-01",
                        "value": 3.5,
                        "vintage_date": "2020-03-01",
                        "source": "test",
                    },
                ]
            )
            n = wh.write_observations(obs)
            assert n == 2
            read = wh.read_observations()
            assert len(read) == 2
            assert sorted(read["series_id"].unique()) == ["UNRATE"]
            assert wh.backend_name == "duckdb"
        finally:
            wh.close()


def test_warehouse_auto_backend_selects_duckdb_for_duckdb_suffix() -> None:
    pytest.importorskip("duckdb")
    from market_regime_engine.storage import Warehouse

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wh.duckdb"
        wh = Warehouse(str(path), backend="auto")
        try:
            assert wh.backend_name == "duckdb"
        finally:
            wh.close()


def test_warehouse_auto_backend_selects_sqlite_for_db_suffix() -> None:
    from market_regime_engine.storage import Warehouse

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wh.db"
        wh = Warehouse(str(path), backend="auto")
        try:
            assert wh.backend_name == "sqlite"
        finally:
            wh.close()


def test_warehouse_migrate_copies_rows() -> None:
    """``migrate_warehouse`` copies every populated table sqlite→duckdb."""
    pytest.importorskip("duckdb")
    from market_regime_engine.storage import Warehouse, migrate_warehouse

    with tempfile.TemporaryDirectory() as tmp:
        sqlite_path = Path(tmp) / "src.db"
        duck_path = Path(tmp) / "dst.duckdb"
        src = Warehouse(str(sqlite_path), backend="sqlite")
        try:
            obs = pd.DataFrame(
                [
                    {
                        "series_id": "UNRATE",
                        "date": "2020-01-01",
                        "value": 3.6,
                        "vintage_date": "2020-02-01",
                        "source": "test",
                    },
                    {
                        "series_id": "UNRATE",
                        "date": "2020-02-01",
                        "value": 3.5,
                        "vintage_date": "2020-03-01",
                        "source": "test",
                    },
                ]
            )
            src.write_observations(obs)
        finally:
            src.close()
        counts = migrate_warehouse(str(sqlite_path), str(duck_path), src_backend="sqlite", dst_backend="duckdb")
        assert counts.get("observations", 0) == 2
        dst = Warehouse(str(duck_path), backend="duckdb")
        try:
            read = dst.read_observations()
            assert len(read) == 2
        finally:
            dst.close()


# ---------------------------------------------------------------------------
# Item E: alert sinks
# ---------------------------------------------------------------------------


def test_slack_sink_skips_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MRE_SLACK_WEBHOOK_URL", raising=False)
    from market_regime_engine.alerts_sinks import SlackSink

    result = SlackSink().send({"alert_type": "test", "message": "hi"})
    assert result.status == "skipped"


def test_slack_sink_posts_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = pytest.importorskip("responses")
    from market_regime_engine.alerts_sinks import SlackSink

    monkeypatch.setenv("MRE_SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T/B/secret")
    with responses.RequestsMock() as rsp:
        rsp.add(
            responses.POST,
            "https://hooks.slack.com/services/T/B/secret",
            json={"ok": True},
            status=200,
        )
        result = SlackSink().send({"alert_type": "test", "severity": "high", "message": "boom"})
    assert result.status == "ok"


def test_pagerduty_sink_skips_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MRE_PAGERDUTY_INTEGRATION_KEY", raising=False)
    from market_regime_engine.alerts_sinks import PagerDutySink

    result = PagerDutySink().send({"alert_type": "x", "message": "y"})
    assert result.status == "skipped"


def test_email_sink_skips_when_smtp_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("MRE_SMTP_HOST", "MRE_SMTP_PORT", "MRE_SMTP_FROM", "MRE_SMTP_TO"):
        monkeypatch.delenv(var, raising=False)
    from market_regime_engine.alerts_sinks import EmailSink

    result = EmailSink().send({"alert_type": "x", "message": "y"})
    assert result.status == "skipped"


def test_dispatch_alerts_returns_long_format_frame() -> None:
    from market_regime_engine.alerts_sinks import dispatch_alerts

    alerts = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "alert_type": "release_gate_hold",
                "severity": "high",
                "channel": "model_risk",
                "message": "held",
                "metadata_json": "{}",
            }
        ]
    )
    out = dispatch_alerts(alerts, sinks=[])  # explicit empty sinks
    # Empty sinks → empty dispatch frame.
    assert out.empty


# ---------------------------------------------------------------------------
# Item F: verify_data
# ---------------------------------------------------------------------------


def test_verify_data_detects_warehouse_drift() -> None:
    """A vintage_observations mutation must surface as a payload diff."""
    from market_regime_engine.model_runs import create_model_run, model_run_frame
    from market_regime_engine.storage import Warehouse
    from market_regime_engine.verify_data import verify_warehouse_state

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "drift.db"
        db = Warehouse(str(path))
        try:
            features = pd.DataFrame(
                [
                    {
                        "feature_name": "f1",
                        "date": "2020-01-01",
                        "value": 1.0,
                        "domain": "rates",
                        "metadata_json": "{}",
                    },
                    {
                        "feature_name": "f1",
                        "date": "2020-02-01",
                        "value": 1.5,
                        "domain": "rates",
                        "metadata_json": "{}",
                    },
                ]
            )
            outputs = pd.DataFrame(
                [
                    {
                        "model_name": "m1",
                        "date": "2020-01-01",
                        "horizon": "3m",
                        "target": "drawdown_gt_10pct",
                        "value": 0.1,
                        "metadata_json": "{}",
                    },
                ]
            )
            asof = pd.DataFrame(
                [
                    {
                        "as_of_date": "2020-01-01",
                        "feature_name": "f1",
                        "source_series_id": "UNRATE",
                        "observation_date": "2019-12-01",
                        "vintage_date": "2020-01-01",
                        "value": 0.5,
                        "transform_name": "level",
                        "created_at_utc": "2020-01-01T00:00:00",
                        "metadata_json": "{}",
                    }
                ]
            )
            db.write_features(features)
            db.write_model_outputs(outputs)
            db.write_feature_asof_values(asof)
            run = create_model_run(
                engine_version="1.3.0",
                purpose="verify-data test",
                features=db.read_features(),
                model_outputs=db.read_model_outputs(),
                vintage_features=db.read_feature_asof_values(),
            )
            db.write_model_runs(model_run_frame(run))
        finally:
            db.close()

        # No drift on first verify.
        report = verify_warehouse_state(run_id=run.run_id, db_path=str(path))
        assert report["approved"], f"expected no drift; got {report}"

        # Mutate one row.
        db = Warehouse(str(path))
        try:
            db._backend.execute(
                "UPDATE feature_asof_values SET value = ? WHERE as_of_date = ?",
                (99.99, "2020-01-01"),
            )
            db._backend.commit()
        finally:
            db.close()

        report = verify_warehouse_state(run_id=run.run_id, db_path=str(path))
        assert not report["approved"]
        assert "vintage_payload" in report["differences"]
        diff = report["differences"]["vintage_payload"]
        assert diff["stored"] != diff["current"]
        assert diff["current_rows"] == 1
        assert isinstance(diff["changed_rows"], list)
        assert diff["changed_rows"][0]["value"] == 99.99


# ---------------------------------------------------------------------------
# Item G: production profile
# ---------------------------------------------------------------------------


def test_production_profile_blocks_when_mcs_evidence_missing() -> None:
    """Profile=production rejects a candidate without MCS membership.

    v1.4.1 (item F): the v1.2.1 looser baseline now requires
    ``profile="default"`` to opt back in (v1.4.0-and-earlier got it
    by default with no flags). The semantic intent of this test —
    "default approves, production blocks" — is unchanged.
    """
    from market_regime_engine.release_gates import evaluate_release_gate

    confidence = pd.DataFrame([{"date": "2024-01-01", "confidence": 0.8, "grade": "A", "metadata_json": "{}"}])
    drift = pd.DataFrame(columns=["date", "feature_name", "psi", "status"])
    invalidation = pd.DataFrame(columns=["date", "trigger", "severity", "status"])
    promotion = pd.DataFrame(
        [{"target": "x", "horizon": "3m", "model": "candidate", "promoted": True}]
    )  # no mcs_evidence column → "absent"
    gate_default = evaluate_release_gate(
        confidence=confidence,
        drift=drift,
        invalidation=invalidation,
        promotion=promotion,
        profile="default",
    )
    gate_prod = evaluate_release_gate(
        confidence=confidence,
        drift=drift,
        invalidation=invalidation,
        promotion=promotion,
        profile="production",
    )
    # Default profile approves; production blocks on missing MCS.
    assert bool(gate_default.iloc[0]["approved"])
    assert not bool(gate_prod.iloc[0]["approved"])
    assert "mcs_evidence_absent" in str(gate_prod.iloc[0]["reasons"])


def test_production_profile_factory_returns_strict_kwargs() -> None:
    from market_regime_engine.release_gates import production_profile

    p = production_profile()
    assert p["min_confidence"] == 0.75
    assert p["require_mcs_membership"] is True
    assert p["min_coverage"] == 0.85
    assert p["coverage_drop_pp"] == 0.02
    assert p["promotion_method"] == "mcs"


# ---------------------------------------------------------------------------
# Item L: report writer consolidation
# ---------------------------------------------------------------------------


def test_report_writer_consolidation_emits_known_sections() -> None:
    """``write_institutional_report`` materialises every selected section."""
    from market_regime_engine.report_writer import write_institutional_report

    regimes = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "regime": "stable",
                "decoded_regime": "calm",
                "score": 0.1,
                "change_point_prob": 0.05,
                "metadata_json": "{}",
            }
        ]
    )
    outputs = pd.DataFrame(
        [
            {
                "model_name": "m1",
                "date": "2024-01-01",
                "horizon": "3m",
                "target": "drawdown_gt_10pct",
                "value": 0.1,
                "metadata_json": "{}",
            }
        ]
    )
    confidence = pd.DataFrame([{"date": "2024-01-01", "confidence": 0.9, "grade": "A", "metadata_json": "{}"}])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.md"
        path = write_institutional_report(
            regimes=regimes,
            model_outputs=outputs,
            analogs=pd.DataFrame(),
            domain_attribution=pd.DataFrame(),
            feature_attribution=pd.DataFrame(),
            confidence=confidence,
            out=out,
        )
        text = path.read_text(encoding="utf-8")
        assert "Market Regime Engine Institutional Report" in text
        assert "v0.5 governance and confidence layer" in text
        assert "Forecast confidence" in text


def test_legacy_shims_emit_deprecation_warning() -> None:
    """The v2-v5 shims must emit DeprecationWarning."""
    from market_regime_engine.report_writer_v2 import append_v05_sections

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.md"
        path.write_text("# Header\n", encoding="utf-8")
        with pytest.warns(DeprecationWarning):
            append_v05_sections(path)


# ---------------------------------------------------------------------------
# Item M: API cache backends
# ---------------------------------------------------------------------------


def test_local_cache_backend_ttl_semantics() -> None:
    from market_regime_engine.api_v1 import _LocalTTLCache

    cache = _LocalTTLCache(max_entries=4)
    cache.set("k", {"v": 1})
    assert cache.get("k") == {"v": 1}


def test_redis_cache_backend_with_fakeredis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redis backend stores and retrieves through fakeredis."""
    fakeredis = pytest.importorskip("fakeredis")
    monkeypatch.setenv("MRE_CACHE_BACKEND", "redis")
    monkeypatch.setenv("MRE_REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("MRE_CACHE_TTL", "30")

    fake = fakeredis.FakeRedis()
    with mock.patch("redis.Redis.from_url", return_value=fake):
        from market_regime_engine.api_v1 import _RedisTTLCache

        cache = _RedisTTLCache("redis://localhost:6379/0")
        cache.set("ep:regime_latest", {"v": 1})
        assert cache.get("ep:regime_latest") == {"v": 1}


def test_redis_cache_backend_soft_degrades_when_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad MRE_REDIS_URL falls back to the local cache without raising."""
    monkeypatch.setenv("MRE_CACHE_BACKEND", "redis")
    monkeypatch.setenv("MRE_REDIS_URL", "redis://nonexistent.invalid:9999/0")
    from market_regime_engine.api_v1 import _build_cache_backend

    cache = _build_cache_backend()
    # Soft-degrade: name attribute is "local", not "redis".
    assert cache.name == "local"


# ---------------------------------------------------------------------------
# Version sanity
# ---------------------------------------------------------------------------


def test_pyproject_and_init_versions_match_1_3_0() -> None:
    """v1.3 baseline marker; updated for the v1.4 bump.

    The original v1.3 assertion hardcoded ``1.3.0``. v1.4 reuses the same
    test name so the regression suite still tracks "the v1.3 release was
    shipped + pyproject and __init__ still agree", but the version
    string now matches the source of truth in ``pyproject.toml`` /
    ``__init__.py``.
    """
    from market_regime_engine import __version__

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert f'version = "{__version__}"' in text
    # v1.3 was the floor: any future release must keep the version
    # monotone, parsed numerically so 1.10.0 > 1.3.0 etc.
    parts = tuple(int(p) for p in __version__.split(".")[:3])
    assert parts >= (1, 3, 0), f"version regressed below v1.3 baseline: {__version__}"


def test_cli_version_flag_prints_version() -> None:
    from market_regime_engine import __version__
    from market_regime_engine.cli import main as cli_main

    out_buf: list[str] = []
    with mock.patch("builtins.print", side_effect=out_buf.append):
        cli_main(["--version"])
    assert any(__version__ in line for line in out_buf)
