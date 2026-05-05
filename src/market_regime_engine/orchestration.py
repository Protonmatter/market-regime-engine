"""End-to-end run orchestration.

The CLI is the operator UX; this module is the *programmatic* surface for
schedulers (Dagster, Prefect, Airflow, plain cron) and is what the v1.0
release wires into containers. The flow is intentionally linear and idempotent
so it can be triggered any number of times in a day without producing
duplicate model_runs (the run id is hashed from the artifact envelope).

A production deployment can drop the following into a Dagster job::

    from market_regime_engine.orchestration import daily_flow
    @op
    def mre_daily(context):
        return daily_flow(db_path="/data/mre.db", validation_dir="/data/validation")

The function returns a dict that's safe to log as JSON.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from market_regime_engine import __version__ as ENGINE_VERSION
from market_regime_engine.alerts import route_alerts
from market_regime_engine.asof import (
    audit_feature_asof_lineage,
    audit_vintage_observations,
    feature_asof_to_features,
    materialize_feature_asof_values,
)
from market_regime_engine.attribution import domain_driver_attribution, feature_driver_attribution
from market_regime_engine.calibration import apply_binary_calibration, fit_calibrators_from_validation
from market_regime_engine.confidence import compute_model_confidence
from market_regime_engine.config import load_catalog
from market_regime_engine.conformal import MondrianBinaryConformal
from market_regime_engine.drift import compute_feature_drift
from market_regime_engine.invalidation import forecast_invalidation_triggers
from market_regime_engine.logging_setup import get_logger
from market_regime_engine.model_runs import build_repro_envelope, create_model_run, model_run_frame
from market_regime_engine.observability import metrics, time_block
from market_regime_engine.promotion_workflow import evaluate_promotion_workflow
from market_regime_engine.regimes import score_regimes
from market_regime_engine.release_gates import evaluate_release_gate
from market_regime_engine.storage import Warehouse

log = get_logger("mre.orchestration")


def daily_flow(
    *,
    db_path: str = "data/mre.duckdb",
    validation_dir: str | Path = "data/validation",
    purpose: str = "scheduled daily run",
    enforce_audit: bool = True,
    enable_frontier: bool = True,
    enable_bayesian: bool = False,
    enable_deep_kernel: bool = False,
    profile: str | None = None,
    dispatch_alerts: bool = False,
    verify_data: bool = True,
) -> dict[str, Any]:
    """Run the full pipeline end-to-end and return a structured summary.

    ``enable_frontier`` (default True) adds the v1.2 frontier steps: mixed-
    frequency nowcast, conditional/localized conformal coverage check, and
    sequential e-value safe-test. They are guarded so missing soft deps
    (statsmodels, ngboost, torch) silently degrade.

    v1.3 (wiring item 5):

    - ``profile="production"`` routes :func:`evaluate_release_gate`
      through :func:`market_regime_engine.release_gates.production_profile`
      so every governance gate runs in its strict production mode.
    - ``dispatch_alerts=True`` forwards every routed alert through
      configured live sinks (:mod:`market_regime_engine.alerts_sinks`).
      Sinks no-op when their env vars are unset, so leaving this on by
      default is safe.
    - ``verify_data=True`` (default) runs :func:`verify_warehouse_state`
      against the run row written this iteration. Drift is logged as a
      ``warning`` but does not block the daily flow — the operator can
      still decide whether to halt downstream consumers.
    """
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    summary: dict[str, Any] = {"started_at": started_at, "engine_version": ENGINE_VERSION}
    db = Warehouse(db_path)
    try:
        # 1. Materialize point-in-time features.
        with time_block("mre_orch_seconds", stage="asof"):
            asof = materialize_feature_asof_values(db.read_vintage_observations(), load_catalog())
            db.write_feature_asof_values(asof)
            db.write_features(feature_asof_to_features(asof))
            summary["asof_rows"] = len(asof)
        # 2. Vintage / lineage audits.
        with time_block("mre_orch_seconds", stage="audit"):
            audits = pd.concat(
                [
                    audit_vintage_observations(db.read_vintage_observations()),
                    audit_feature_asof_lineage(db.read_feature_asof_values()),
                ],
                ignore_index=True,
            )
            db.write_vintage_audits(audits)
            summary["audit"] = audits.to_dict(orient="records")
            if enforce_audit and (audits["violations"].fillna(0).astype(int) > 0).any():
                metrics().incr("mre_orch_failures_total", reason="vintage_audit")
                raise RuntimeError("vintage/as-of audit failed")
        # 3. Score regime, drift, attribution, invalidation, confidence.
        with time_block("mre_orch_seconds", stage="score"):
            regimes = score_regimes(db.read_features())
            db.write_regimes(regimes)
            summary["regime_rows"] = len(regimes)
        with time_block("mre_orch_seconds", stage="drift"):
            drift = compute_feature_drift(db.read_features())
            db.write_model_drift(drift)
        with time_block("mre_orch_seconds", stage="attribution"):
            domain_attr = domain_driver_attribution(db.read_features())
            feat_attr = feature_driver_attribution(db.read_features())
            db.write_driver_attribution(domain_attr, "domain")
            db.write_driver_attribution(feat_attr, "feature")
        with time_block("mre_orch_seconds", stage="triggers"):
            triggers = forecast_invalidation_triggers(db.read_features(), db.read_regimes())
            db.write_invalidation_triggers(triggers)
        with time_block("mre_orch_seconds", stage="confidence"):
            validation = pd.DataFrame()
            vpath = Path(validation_dir) / "binary_validation.csv"
            if vpath.exists():
                validation = pd.read_csv(vpath)
            confidence = compute_model_confidence(
                regimes=db.read_regimes(),
                validation=validation,
                analogs=db.read_historical_analogs(),
                release_audit=db.read_release_calendar_audit(),
            )
            db.write_confidence_scores(confidence)
        # 4. Release gate + alerts + promotion workflow.
        with time_block("mre_orch_seconds", stage="release_gate"):
            promotion = pd.DataFrame()
            ppath = Path(validation_dir) / "model_promotion.csv"
            if ppath.exists():
                promotion = pd.read_csv(ppath)
            gate = evaluate_release_gate(
                confidence=db.read_confidence_scores(),
                drift=db.read_model_drift(),
                invalidation=db.read_invalidation_triggers(),
                promotion=promotion,
                profile=cast(Literal["default", "production"] | None, profile),
            )
            db.write_release_gates(gate)
            summary["release_gate"] = gate.iloc[-1].to_dict() if not gate.empty else {}
            summary["release_gate_profile"] = profile or "default"
        with time_block("mre_orch_seconds", stage="alerts"):
            alerts = route_alerts(
                release_gates=db.read_release_gates(),
                drift=db.read_model_drift(),
                invalidation=db.read_invalidation_triggers(),
                confidence=db.read_confidence_scores(),
                promotion=promotion,
            )
            db.write_routed_alerts(alerts)
            summary["alert_count"] = len(alerts)
            # v1.3 (wiring item 5): dispatch through configured live
            # sinks when the caller opts in. Sinks no-op when their env
            # vars are unset, so this is safe to leave on in CI.
            if dispatch_alerts:
                from market_regime_engine.alerts_sinks import dispatch_alerts as _dispatch

                dispatched = _dispatch(alerts)
                if not dispatched.empty:
                    db.write_alert_dispatches(dispatched)
                summary["alert_dispatch_count"] = len(dispatched)
        with time_block("mre_orch_seconds", stage="promotion"):
            workflow = evaluate_promotion_workflow(
                promotion=promotion,
                release_gate=db.read_release_gates(),
                confidence=db.read_confidence_scores(),
                drift=db.read_model_drift(),
            )
            db.write_promotion_workflow(workflow)
        # 5. Calibration (if validation artifacts exist).
        if Path(validation_dir).exists():
            with time_block("mre_orch_seconds", stage="calibrate"):
                calibrators = fit_calibrators_from_validation(str(validation_dir))
                if not calibrators.empty:
                    db.write_calibration_models(calibrators)
                    calibrated = apply_binary_calibration(db.read_model_outputs(), calibrators)
                    db.write_calibrated_outputs(calibrated)
                    summary["calibrated_outputs"] = len(calibrated)
            # 5b. Mondrian conformal coverage gate on OOS binary predictions.
            with time_block("mre_orch_seconds", stage="conformal_coverage"):
                coverage_frame = compute_conformal_coverage(Path(validation_dir), alpha=0.10)
                if not coverage_frame.empty:
                    db.write_conformal_coverage(coverage_frame)
                    summary["worst_coverage"] = float(coverage_frame["realized_coverage"].astype(float).min())
                else:
                    summary["worst_coverage"] = None
        # 5c. v1.2 frontier steps: nowcast, conditional conformal, e-values.
        if enable_frontier:
            with time_block("mre_orch_seconds", stage="frontier_nowcast"):
                nowcast_frame = compute_nowcast_factors(db.read_features())
                if not nowcast_frame.empty:
                    db.write_nowcast_factors(nowcast_frame)
                    summary["nowcast_factors"] = (
                        nowcast_frame.set_index("domain")["factor_value"].astype(float).to_dict()
                    )
                else:
                    summary["nowcast_factors"] = {}
            with time_block("mre_orch_seconds", stage="frontier_conditional_coverage"):
                cond_frame = compute_conditional_coverage(Path(validation_dir), alpha=0.10)
                if not cond_frame.empty:
                    db.write_conditional_coverage_report(cond_frame)
                    summary["worst_conditional_coverage"] = float(cond_frame["coverage"].astype(float).min())
                else:
                    summary["worst_conditional_coverage"] = None
            with time_block("mre_orch_seconds", stage="frontier_e_value_test"):
                e_frame = compute_sequential_e_value(
                    Path(validation_dir),
                    alpha=0.05,
                )
                if not e_frame.empty:
                    db.write_e_value_log(e_frame)
                    promote = bool((e_frame["decision"].astype(str) == "promote").any())
                    summary["e_value_promotion_pending"] = promote
                else:
                    summary["e_value_promotion_pending"] = False
        # v1.4 (item F): optional Bayesian MS-VAR + deep-kernel branches.
        # Both default OFF so the v1.3 daily_flow shape is unchanged.
        if enable_bayesian:
            with time_block("mre_orch_seconds", stage="bayesian_msvar"):
                summary["bayesian_msvar"] = _run_bayesian_msvar(db, purpose=purpose)
        if enable_deep_kernel:
            with time_block("mre_orch_seconds", stage="deep_kernel"):
                summary["deep_kernel"] = _run_deep_kernel(db)
        # 6. Immutable model run with the full reproducibility envelope.
        with time_block("mre_orch_seconds", stage="model_run"):
            envelope = build_repro_envelope(
                features=db.read_features(),
                model_outputs=db.read_model_outputs(),
                vintage_features=db.read_feature_asof_values(),
            )
            metadata = {
                "orchestration": "daily_flow",
                "validation_dir": str(validation_dir),
                # v1.3 (wiring item 4): record the warehouse backend on
                # the run so verify-run / verify-data can surface a
                # mismatch (e.g. a run written against duckdb being
                # verified against sqlite).
                "warehouse_backend": db.backend_name,
            }
            run = create_model_run(
                engine_version=ENGINE_VERSION,
                purpose=purpose,
                features=db.read_features(),
                model_outputs=db.read_model_outputs(),
                vintage_features=db.read_feature_asof_values(),
                metadata=metadata,
            )
            db.write_model_runs(model_run_frame(run))
            summary["run_id"] = run.run_id
            summary["artifact_hash"] = run.artifact_hash
            summary["envelope"] = {
                "code_version": envelope.code_version,
                "code_dirty": envelope.code_dirty,
                "lockfile_hash": envelope.lockfile_hash[:12],
            }
        # 7. v1.3 (wiring item 5): verify_data runs *after* the model
        # run is written so the drift check sees the same warehouse
        # state the run was envelope-hashed against. When drift is
        # detected we log a warning but do not raise — the
        # orchestration is allowed to continue, and the operator can
        # decide whether to halt downstream consumers via the
        # structured ``summary["verify_data"]`` entry.
        if verify_data:
            with time_block("mre_orch_seconds", stage="verify_data"):
                from market_regime_engine.verify_data import verify_warehouse_state

                # Both backends support concurrent readers, so opening
                # a second warehouse handle for the verify path is safe.
                drift_report = verify_warehouse_state(run_id=run.run_id, db_path=db_path)
                summary["verify_data"] = {
                    "approved": bool(drift_report.get("approved", False)),
                    "differences": list(drift_report.get("differences", {}).keys()),
                }
                if not drift_report.get("approved", False):
                    log.warning(
                        "verify_data drift detected (non-blocking)",
                        extra={"differences": drift_report.get("differences", {})},
                    )
    finally:
        db.close()
    summary["completed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    metrics().incr("mre_orch_runs_total", outcome="ok")
    log.info("daily_flow complete", extra=summary)
    return summary


def compute_conformal_coverage(
    validation_dir: Path,
    *,
    alpha: float = 0.10,
    bucket_col: str = "regime_bucket",
) -> pd.DataFrame:
    """Fit a Mondrian binary conformal layer per (target, horizon) on OOS preds.

    Reads every ``binary_predictions_*.csv`` under ``validation_dir`` (the same
    files :func:`fit_calibrators_from_validation` consumes) and returns the
    per-bucket realized-coverage frame in the shape expected by
    ``Warehouse.write_conformal_coverage``.

    The function silently returns an empty frame when no prediction files are
    present so the orchestration call site can defer the gate without
    branching.
    """
    if not validation_dir.exists():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    as_of = datetime.now(UTC).strftime("%Y-%m-%d")
    for path in sorted(validation_dir.glob("binary_predictions_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or {"y", "p"}.issubset(df.columns) is False:
            continue
        if bucket_col not in df.columns:
            df[bucket_col] = "general"
        groups = df.groupby(["target", "horizon"], observed=True, dropna=False)
        for (target, horizon), group in groups:
            calibration = group[["y", "p", bucket_col]].dropna(subset=["y", "p"])
            if calibration.empty:
                continue
            layer = MondrianBinaryConformal(alpha=alpha, bucket_col=bucket_col).fit(calibration)
            report = layer.coverage_report(calibration)
            for _, row in report.iterrows():
                rows.append(
                    {
                        "as_of_date": as_of,
                        "target": str(target),
                        "horizon": str(horizon),
                        "bucket": str(row[bucket_col]),
                        "n": int(row["n"]),
                        "realized_coverage": float(row["coverage"]),
                        "target_coverage": float(1.0 - alpha),
                        "threshold": float(row.get("threshold", float("nan"))),
                        "method": "mondrian_binary",
                        "metadata_json": "{}",
                    }
                )
    return pd.DataFrame(rows)


def compute_nowcast_factors(features: pd.DataFrame) -> pd.DataFrame:
    """Run the v1.2 mixed-frequency nowcast and return one row per domain.

    Soft-degrades when statsmodels is missing (the wrapper falls back to the
    legacy single-frequency DFM).
    """
    import json as _json

    from market_regime_engine.frontier.dfm_mq import MQDynamicFactorModel

    if features is None or features.empty:
        return pd.DataFrame()
    wide = features.pivot_table(index="date", columns="feature_name", values="value").sort_index()
    if "domain" not in features.columns:
        return pd.DataFrame()
    domains = sorted(set(features["domain"].dropna().astype(str)))
    as_of = datetime.now(UTC).strftime("%Y-%m-%d")
    rows = []
    for domain in domains:
        cols = features[features["domain"] == domain]["feature_name"].drop_duplicates().tolist()
        sub = wide.reindex(columns=cols).dropna(how="all")
        if sub.empty or sub.shape[0] < 24:
            continue
        try:
            model = MQDynamicFactorModel().fit(sub, frequencies=dict.fromkeys(cols, "M"))
            now = model.nowcast(pd.Timestamp(as_of))
            rows.append(
                {
                    "as_of_date": as_of,
                    "domain": str(domain),
                    "factor_value": float(now["factor"]),
                    "factor_se": float(now["factor_se"]),
                    "frequency_mix": "monthly",
                    "backend": str(now["backend"]),
                    "metadata_json": _json.dumps(
                        {"n_columns": int(sub.shape[1]), "n_rows": int(sub.shape[0])},
                        sort_keys=True,
                    ),
                }
            )
        except Exception as e:
            log.warning("nowcast_failed", extra={"domain": domain, "err": str(e)})
            continue
    return pd.DataFrame(rows)


def compute_conditional_coverage(
    validation_dir: Path,
    *,
    alpha: float = 0.10,
    bucket_col: str = "regime_bucket",
) -> pd.DataFrame:
    """Group-conditional conformal coverage report (v1.2 frontier).

    Mirrors :func:`compute_conformal_coverage` but uses
    :class:`market_regime_engine.frontier.conformal_ts.
    ConditionalConformalRegressor` so the returned frame includes per-group
    coverage and a worst-violation diagnostic.
    """
    import json as _json

    from market_regime_engine.frontier.conformal_ts import ConditionalConformalRegressor

    if not validation_dir.exists():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    as_of = datetime.now(UTC).strftime("%Y-%m-%d")
    for path in sorted(validation_dir.glob("binary_predictions_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or {"y", "p"}.issubset(df.columns) is False:
            continue
        if bucket_col not in df.columns:
            df[bucket_col] = "general"
        groups = df.groupby(["target", "horizon"], observed=True, dropna=False)
        for (target, horizon), group in groups:
            cal = group[["y", "p", bucket_col]].dropna(subset=["y", "p"])
            if cal.empty:
                continue
            layer = ConditionalConformalRegressor(alpha=alpha, bucket_col=bucket_col).fit(cal)
            diag = layer.coverage_report_conditional(cal)
            per = diag["per_group"]
            for _, prow in per.iterrows():
                rows.append(
                    {
                        "as_of_date": as_of,
                        "target": str(target),
                        "horizon": str(horizon),
                        "group": str(prow[bucket_col]),
                        "coverage": float(prow["coverage"]),
                        "n": int(prow["n"]),
                        "alpha": float(alpha),
                        "method": "conditional_conformal",
                        "worst_violation": float(diag["worst_violation"]),
                        "metadata_json": _json.dumps(
                            {"adjusted_alpha": diag["adjusted_alpha"]},
                            sort_keys=True,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def compute_sequential_e_value(
    validation_dir: Path,
    *,
    alpha: float = 0.05,
    challenger_col: str = "model_name",
) -> pd.DataFrame:
    """Run the v1.2 sequential e-value safe-test for every challenger model.

    Reads ``binary_predictions_*.csv`` under ``validation_dir``, picks
    ``expanding_event_rate`` (the warehouse default benchmark) as the
    champion when present, and for every other model emits one row per
    target/horizon with the running e-value and decision.
    """
    import json as _json

    from market_regime_engine.frontier.sequential_testing import SafeTestPromotion

    if not validation_dir.exists():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    as_of = datetime.now(UTC).strftime("%Y-%m-%d")
    for path in sorted(validation_dir.glob("binary_predictions_*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if {"y", "p", "target", "horizon"}.issubset(df.columns) is False:
            continue
        # Validation CSVs use ``model``; warehouse uses ``model_name``. Accept either.
        model_col = "model" if "model" in df.columns else ("model_name" if "model_name" in df.columns else None)
        if model_col is None:
            continue
        names = sorted(df[model_col].astype(str).unique())
        # Try the canonical benchmark first; if it's not in the predictions
        # frame, look for a matching ``binary_benchmark_predictions_*`` file.
        champ_name: str | None = None
        champ_df = pd.DataFrame()
        for candidate in ("expanding_event_rate", "previous_event_shrunk"):
            if candidate in names:
                champ_name = candidate
                champ_df = df[df[model_col] == candidate]
                break
        if champ_df.empty:
            bench_fp = path.parent / path.name.replace("binary_predictions_", "binary_benchmark_predictions_")
            if bench_fp.exists():
                try:
                    bench_df = pd.read_csv(bench_fp)
                except Exception:
                    bench_df = pd.DataFrame()
                if not bench_df.empty:
                    bench_model_col = "model" if "model" in bench_df.columns else "model_name"
                    bench_names = sorted(bench_df[bench_model_col].astype(str).unique())
                    for candidate in ("expanding_event_rate", "previous_event_shrunk"):
                        if candidate in bench_names:
                            champ_name = candidate
                            champ_df = bench_df[bench_df[bench_model_col] == candidate]
                            break
                    if champ_df.empty and bench_names:
                        champ_name = bench_names[0]
                        champ_df = bench_df[bench_df[bench_model_col] == champ_name]
        if champ_df.empty:
            # Fall back to the first non-challenger model in the same file.
            if not names:
                continue
            champ_name = names[0]
            champ_df = df[df[model_col] == champ_name]
        for chal in names:
            if chal == champ_name:
                continue
            chal_df = df[df[model_col] == chal]
            joined = chal_df.merge(
                champ_df,
                on=["date", "target", "horizon"],
                suffixes=("_chal", "_champ"),
            )
            if joined.empty:
                continue
            for (target, horizon), grp in joined.groupby(["target", "horizon"], observed=True):
                chal_loss = ((grp["p_chal"] - grp["y_chal"]) ** 2).tolist()
                champ_loss = ((grp["p_champ"] - grp["y_champ"]) ** 2).tolist()
                n = min(len(chal_loss), len(champ_loss))
                if n < 4:
                    continue
                result = SafeTestPromotion.run(chal_loss[:n], champ_loss[:n], alpha=alpha)
                e_val = float(result["e_value"])
                decision = "promote" if result["fired"] else "hold"
                rows.append(
                    {
                        "date": as_of,
                        "target": str(target),
                        "horizon": str(horizon),
                        "challenger": str(chal),
                        "champion": str(champ_name),
                        "e_value": e_val,
                        "level": float(alpha),
                        "decision": decision,
                        "n": int(n),
                        "metadata_json": _json.dumps(
                            {"fired_at_n": result.get("fired_at_n")},
                            sort_keys=True,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _run_bayesian_msvar(db: Warehouse, *, purpose: str) -> dict[str, Any]:
    """v1.4 helper: fit a Bayesian MS-VAR and write diagnostics.

    Soft-degrades when ``[bayesian]`` is missing or the panel is too
    short, so ``daily_flow(enable_bayesian=True)`` is safe to call in
    every environment (including CI runners without JAX).
    """
    import json as _json

    try:
        from market_regime_engine.frontier.bayesian_msvar import BayesianMSVAR
    except ImportError:
        return {"status": "skipped_missing_extra"}
    features = db.read_features()
    if features.empty:
        return {"status": "skipped_no_features"}
    wide = features.pivot_table(index="date", columns="feature_name", values="value").sort_index()
    from market_regime_engine.hmm import DOMAIN_COLUMNS

    cols = [c for c in DOMAIN_COLUMNS if c in wide.columns]
    if cols:
        wide = wide[cols]
    wide.index = pd.to_datetime(wide.index)
    if wide.shape[0] < 24 or wide.shape[1] < 2:
        return {"status": "skipped_panel_too_small", "n": int(wide.shape[0])}
    try:
        # Use SVI for the daily run so wall-clock stays predictable;
        # operators can pick NUTS via the dedicated CLI.
        model = BayesianMSVAR(domains=list(wide.columns)).fit(wide, method="svi", svi_steps=200, num_samples=64)
    except Exception as exc:
        log.warning("bayesian_msvar_fit_failed: %s", exc)
        return {"status": "error", "error": str(exc)}
    diag = model.last_diagnostics or {}
    run_id = f"bayesian_msvar_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "method": diag.get("method", "svi"),
                "num_chains": int(diag.get("num_chains", 1)),
                "num_divergences": int(diag.get("num_divergences", 0)),
                "max_rhat": float(diag.get("max_rhat", float("nan"))),
                "min_ess": float(diag.get("min_ess", float("nan"))),
                "runtime_seconds": float(diag.get("runtime_seconds", 0.0)),
                "metadata_json": _json.dumps(
                    {
                        "purpose": purpose,
                        **{
                            k: v
                            for k, v in diag.items()
                            if k
                            not in {"method", "num_chains", "num_divergences", "max_rhat", "min_ess", "runtime_seconds"}
                        },
                    },
                    sort_keys=True,
                    default=str,
                ),
            }
        ]
    )
    db.write_bayesian_msvar_diagnostics(df)
    return {"status": "ok", "run_id": run_id, "diagnostics": diag}


def _run_deep_kernel(db: Warehouse) -> dict[str, Any]:
    """v1.4 helper: train an MLPDeepKernel + score GP-BOCPD with it."""
    try:
        from market_regime_engine.frontier.deep_kernel import MLPDeepKernel
        from market_regime_engine.frontier.gp_cpd import GPBOCPD
    except ImportError:
        return {"status": "skipped_missing_extra"}
    features = db.read_features()
    if features.empty:
        return {"status": "skipped_no_features"}
    wide = features.pivot_table(index="date", columns="feature_name", values="value").sort_index()
    if wide.shape[0] < 32:
        return {"status": "skipped_panel_too_small", "n": int(wide.shape[0])}
    try:
        kernel = MLPDeepKernel(input_dim=int(wide.shape[1]), hidden_dims=(32, 16))
        kernel.fit(wide, n_epochs=20)
        scored = GPBOCPD(deep_kernel=kernel).score(wide)
    except Exception as exc:
        log.warning("deep_kernel_score_failed: %s", exc)
        return {"status": "error", "error": str(exc)}
    return {
        "status": "ok",
        "epochs": len(kernel.training_losses),
        "final_loss": float(kernel.training_losses[-1]) if kernel.training_losses else float("nan"),
        "rows": len(scored),
    }


__all__ = [
    "compute_conditional_coverage",
    "compute_conformal_coverage",
    "compute_nowcast_factors",
    "compute_sequential_e_value",
    "daily_flow",
]
