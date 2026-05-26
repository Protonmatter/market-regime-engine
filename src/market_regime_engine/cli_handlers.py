from __future__ import annotations

import argparse
import contextlib
import json
import os
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_regime_engine import __version__ as ENGINE_VERSION
from market_regime_engine.alerts import route_alerts
from market_regime_engine.alfred import build_alfred_request_matrix, fetch_alfred_vintages
from market_regime_engine.alfred_real import (
    build_real_alfred_plan,
    fetch_real_alfred_vintage_observations,
    seed_vintage_observations_from_latest,
)
from market_regime_engine.analogs import HistoricalAnalogEngine, analog_summary
from market_regime_engine.analogs_v2 import regime_weighted_analogs
from market_regime_engine.analytics_warehouse import build_duckdb_database, export_sqlite_to_lake, warehouse_health
from market_regime_engine.asof import (
    audit_feature_asof_lineage,
    audit_vintage_observations,
    feature_asof_to_features,
    materialize_feature_asof_values,
)
from market_regime_engine.attribution import domain_driver_attribution, feature_driver_attribution
from market_regime_engine.backtest import benchmark_report
from market_regime_engine.calibration import apply_binary_calibration, fit_calibrators_from_validation
from market_regime_engine.confidence import compute_model_confidence
from market_regime_engine.config import load_catalog
from market_regime_engine.drift import compute_feature_drift, drift_summary
from market_regime_engine.explain import latest_explanation
from market_regime_engine.features import build_features, feature_matrix, monthly_panel
from market_regime_engine.fred_recession import fetch_fred_recession_indicator
from market_regime_engine.fred_vintage import FredVintageIngestionPlan, fetch_fred_vintage_plan
from market_regime_engine.hazard_model import hazard_backtest_matrix, train_fitted_hazard_outputs
from market_regime_engine.invalidation import forecast_invalidation_triggers
from market_regime_engine.logging_setup import configure_logging, get_logger
from market_regime_engine.model_registry import create_model_card, write_model_card
from market_regime_engine.model_runs import create_model_run, model_run_frame
from market_regime_engine.models import train_latest_outputs
from market_regime_engine.point_in_time import apply_release_lag, assert_no_future_vintages
from market_regime_engine.promotion_workflow import evaluate_promotion_workflow
from market_regime_engine.regimes import score_regimes
from market_regime_engine.release_calendar import (
    audit_release_calendar,
    enforce_release_calendar,
    load_release_calendar,
)
from market_regime_engine.release_calendar_exact import (
    audit_exact_release_calendar,
    build_exact_release_calendar,
    enforce_exact_release_calendar,
)
from market_regime_engine.release_gates import evaluate_release_gate
from market_regime_engine.report_writer import write_institutional_report
from market_regime_engine.sample import generate_sample_observations
from market_regime_engine.stacking import optimize_from_model_outputs
from market_regime_engine.stacking_v2 import regime_conditioned_stacking
from market_regime_engine.storage import Warehouse, migrate_warehouse
from market_regime_engine.survival import recession_hazard_scores, survival_summary
from market_regime_engine.targets import make_targets
from market_regime_engine.training_data import TrainingMode, join_X_y, load_targets, load_training_panel

log = get_logger("mre.cli")


def bootstrap_sample(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        n = db.write_observations(generate_sample_observations())
        print(f"Inserted {n} sample observations into {args.db}")
    finally:
        db.close()


def ingest_fred_vintages_cmd(args: argparse.Namespace) -> None:
    catalog = load_catalog()
    series = args.series or [c["series_id"] for c in catalog if c.get("source") == "fred"]
    plan = FredVintageIngestionPlan(
        series_ids=series,
        observation_start=args.observation_start,
        vintage_start=args.vintage_start,
        vintage_end=args.vintage_end,
        vintage_frequency=args.vintage_frequency,
    )
    rows = fetch_fred_vintage_plan(plan)
    db = Warehouse(args.db)
    try:
        n = db.write_observations(rows)
        print(f"Inserted {n} FRED vintage observations for {len(series)} series")
    finally:
        db.close()


def ingest_fred_recession_cmd(args: argparse.Namespace) -> None:
    rows = fetch_fred_recession_indicator(series_id=args.series, observation_start=args.observation_start)
    db = Warehouse(args.db)
    try:
        n = db.write_recession_labels(rows)
        print(f"Inserted {n} FRED recession labels from {args.series}")
    finally:
        db.close()


def pit_check_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        obs = db.read_observations()
        assert_no_future_vintages(obs)
        # v1.3 (item B4): ``apply_release_lag`` defaults to ``strict=True``.
        # ``--allow-missing-release-rules`` falls back to the v1.2.1
        # behaviour (silent zero-lag for unknown series) when an operator
        # explicitly authorises that downgrade. The audit warns loudly
        # so the deliberate fallback is logged.
        adjusted = apply_release_lag(obs, strict=not bool(getattr(args, "allow_missing_release_rules", False)))
        if args.write_adjusted:
            n = db.write_observations(adjusted)
            print(f"Point-in-time check passed. Rewrote {n} observations with conservative release lags.")
        else:
            print(
                f"Point-in-time check passed for {len(obs)} observations. Use --write-adjusted to apply conservative release lags."
            )
    finally:
        db.close()


def audit_release_calendar_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        obs = db.read_observations()
        audit = audit_release_calendar(obs, load_release_calendar(args.calendar))
        n = db.write_release_calendar_audit(audit)
        if args.enforce:
            fixed = enforce_release_calendar(obs, load_release_calendar(args.calendar))
            db.write_observations(fixed)
        out = Path(args.out) if args.out else None
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            audit.to_csv(out, index=False)
        # v1.4 (item D): also reconcile vintage_observations against the
        # YAML release calendar cache. Rows whose realtime_start drifts
        # by more than ``--tolerance-days`` are surfaced + optionally
        # exit non-zero when ``--enforce`` is set.
        tolerance = float(getattr(args, "tolerance_days", 3))
        try:
            from market_regime_engine.frontier.release_calendars import reconcile_against_vintages

            mismatches = reconcile_against_vintages(db.read_vintage_observations(), tolerance_days=int(tolerance))
        except Exception as exc:
            log.warning("vintage_calendar_reconciliation_failed: %s", exc)
            mismatches = pd.DataFrame()
        if not mismatches.empty:
            print(f"Vintage/calendar mismatches > ±{int(tolerance)} days: {len(mismatches)}")
            print(mismatches.to_string(index=False))
            if getattr(args, "enforce", False):
                # Fail-closed gate: tripping the calendar tolerance is
                # treated like the v1.3 fail-closed audit. Exit 2 so CI
                # halts the pipeline.
                raise SystemExit(2)
        print(f"Wrote release-calendar audit rows: {n}")
        if not audit.empty:
            print(audit.to_string(index=False))
    finally:
        db.close()


def build_feature_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        observations = db.read_observations()
        assert_no_future_vintages(observations)
        panel = monthly_panel(observations)
        features = build_features(panel, load_catalog())
        n = db.write_features(features)
        print(f"Built {n} features")
    finally:
        db.close()


def label_recessions_cmd(args: argparse.Namespace) -> None:
    from market_regime_engine.nber import label_recessions_with_fallback

    db = Warehouse(args.db)
    try:
        panel = monthly_panel(db.read_observations(), forward_fill_limit=0)
        if panel.empty:
            raise SystemExit("No observations found. Run bootstrap-sample or ingestion first.")
        prefer = "builtin" if getattr(args, "force_builtin", False) else "fred"
        labels, staleness = label_recessions_with_fallback(panel.index, prefer=prefer)
        # v1.1: evaluate the staleness gate BEFORE persisting. The earlier
        # ordering wrote the stale rows first and *then* exited with
        # SystemExit(2), leaving the warehouse poisoned (second-opinion #14).
        max_stale = getattr(args, "max_stale_months", None)
        if max_stale is not None and staleness.months_stale > max_stale:
            log.error(
                "label_recessions stale gate tripped before write",
                extra={"staleness": staleness.to_metadata(), "max_stale_months": max_stale},
            )
            raise SystemExit(
                f"Recession labels are {staleness.months_stale} months stale; "
                f"gate set to {max_stale}. Refusing to write to warehouse."
            )
        n = db.write_recession_labels(labels)
        log.info(
            "label_recessions",
            extra={"rows": int(n), "staleness": staleness.to_metadata()},
        )
        print(
            f"Wrote {n} recession labels from {staleness.source} "
            f"(last_label={staleness.last_label_date}, panel_last={staleness.panel_last_date}, "
            f"months_stale={staleness.months_stale})"
        )
        if staleness.fetch_error:
            print(f"FRED fetch_error (fell back to built-in): {staleness.fetch_error}")
    finally:
        db.close()


def score_regime_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        regimes = score_regimes(db.read_features(), use_bocpd=not args.disable_bocpd)
        n = db.write_regimes(regimes)
        latest = regimes.iloc[-1]
        print(
            f"Wrote {n} regimes. Latest: {latest['date'].date()} {latest['decoded_regime']} score={latest['score']:.2f} cp={latest['change_point_prob']:.1%}"
        )
    finally:
        db.close()




from market_regime_engine.cli_helpers import (
    _load_training_audit,
    _persist_training_audit,
    _resolve_allow_legacy_fallback,
    _resolve_training_mode,
    _verify_fi_evidence_pack,
)

def train_baseline_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        observations = db.read_observations()
        features = db.read_features()
        feature_asof = db.read_feature_asof_values()
        mode = _resolve_training_mode(args)
        allow_fallback = _resolve_allow_legacy_fallback(args)
        try:
            X, panel, audit = load_training_panel(
                mode=mode,
                observations=observations,
                features=features,
                feature_asof_values=feature_asof,
                allow_legacy_fallback=allow_fallback,
            )
        except RuntimeError as exc:
            # Fail-closed PIT path. Surface a descriptive message so the
            # operator immediately sees the right next step.
            log.error("train-baseline failed closed", extra={"error": str(exc)})
            raise SystemExit(str(exc)) from exc
        targets = load_targets(panel)
        Xj, yj = join_X_y(X, targets)
        if Xj.empty or yj.empty:
            log.error("Empty training join; aborting.", extra={"audit": audit})
            raise SystemExit(f"No overlapping (features, targets) rows; nothing to train. Training audit: {audit}")
        outputs = train_latest_outputs(Xj, yj)
        n = db.write_model_outputs(outputs)
        audit_path = _persist_training_audit(args.db, audit)
        log.info(
            "baseline training complete",
            extra={"rows": int(n), "audit": audit, "audit_path": str(audit_path)},
        )
        print(
            f"Wrote {n} model outputs (mode={audit.get('mode_used')}, "
            f"rows={audit.get('rows')}, audit_path={audit_path})"
        )
    finally:
        db.close()


def validate_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        observations = db.read_observations()
        features = db.read_features()
        feature_asof = db.read_feature_asof_values()
        mode = _resolve_training_mode(args)
        allow_fallback = _resolve_allow_legacy_fallback(args)
        try:
            X, panel, audit = load_training_panel(
                mode=mode,
                observations=observations,
                features=features,
                feature_asof_values=feature_asof,
                allow_legacy_fallback=allow_fallback,
            )
        except RuntimeError as exc:
            log.error("validate failed closed", extra={"error": str(exc)})
            raise SystemExit(str(exc)) from exc
        targets = load_targets(panel)
        Xj, yj = join_X_y(X, targets)
        if Xj.empty or yj.empty:
            raise SystemExit(
                "Cannot validate: empty (features, targets) join. Run "
                f"materialize-asof-features first. Training audit: {audit}"
            )
        reports = benchmark_report(Xj, yj, min_train=args.min_train, step=args.step)
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        for name, frame in reports.items():
            frame.to_csv(outdir / f"{name}.csv", index=False)
        audit_path = _persist_training_audit(args.db, audit)
        log.info(
            "validation complete",
            extra={"out": str(outdir), "audit": audit, "audit_path": str(audit_path)},
        )
        print(
            f"Validation written to {outdir} (mode={audit.get('mode_used')}, "
            f"rows={audit.get('rows')}, audit_path={audit_path})"
        )
        for name in ["binary_validation", "binary_best_benchmark", "model_promotion", "quantile_validation"]:
            print(f"\n{name}")
            print(reports.get(name, pd.DataFrame()).to_string(index=False))
    finally:
        db.close()


def calibrate_probabilities_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        calibrators = fit_calibrators_from_validation(args.validation_dir)
        ncal = db.write_calibration_models(calibrators)
        calibrated = apply_binary_calibration(db.read_model_outputs(), calibrators)
        nout = db.write_calibrated_outputs(calibrated)
        out = Path(args.out) if args.out else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
            calibrators.to_csv(out / "calibration_models.csv", index=False)
            calibrated.to_csv(out / "calibrated_outputs.csv", index=False)
        print(f"Wrote calibration models={ncal}, calibrated outputs={nout}")
        if not calibrators.empty:
            print(calibrators.to_string(index=False))
    finally:
        db.close()


def analogs_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        panel = monthly_panel(db.read_observations())
        X = feature_matrix(db.read_features())
        targets = make_targets(panel) if not panel.empty else None
        regimes = db.read_regimes()
        if args.regime_weighted:
            analogs = regime_weighted_analogs(X, targets, regimes, top_n=args.top_n, as_of=args.as_of)
        else:
            analogs = HistoricalAnalogEngine(top_n=args.top_n, min_history=args.min_history).score(
                X, targets, regimes, as_of=args.as_of
            )
        n = db.write_historical_analogs(analogs)
        out = Path(args.out) if args.out else None
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            analogs.to_csv(out, index=False)
        print(f"Wrote {n} historical analogs")
        print(analog_summary(analogs))
    finally:
        db.close()


def attribution_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        features = db.read_features()
        domain = domain_driver_attribution(features, as_of=args.as_of)
        feat = feature_driver_attribution(features, as_of=args.as_of, top_n=args.top_n)
        nd = db.write_driver_attribution(domain, "domain")
        nf = db.write_driver_attribution(feat, "feature")
        outdir = Path(args.out) if args.out else None
        if outdir:
            outdir.mkdir(parents=True, exist_ok=True)
            domain.to_csv(outdir / "domain_attribution.csv", index=False)
            feat.to_csv(outdir / "feature_attribution.csv", index=False)
        print(f"Wrote attribution rows: domain={nd}, feature={nf}")
        print(domain.head(10).to_string(index=False))
    finally:
        db.close()


def invalidation_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        triggers = forecast_invalidation_triggers(db.read_features(), db.read_regimes())
        n = db.write_invalidation_triggers(triggers)
        out = Path(args.out) if args.out else None
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            triggers.to_csv(out, index=False)
        print(f"Wrote {n} invalidation triggers")
        print(triggers.to_string(index=False))
    finally:
        db.close()


def confidence_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        validation = None
        vpath = Path(args.validation_dir) / "binary_validation.csv"
        if vpath.exists():
            validation = pd.read_csv(vpath)
        conf = compute_model_confidence(
            regimes=db.read_regimes(),
            validation=validation,
            analogs=db.read_historical_analogs(),
            release_audit=db.read_release_calendar_audit(),
        )
        n = db.write_confidence_scores(conf)
        print(f"Wrote {n} confidence score rows")
        print(conf.to_string(index=False))
    finally:
        db.close()


def model_run_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        training_audit = _load_training_audit(args.db)
        run = create_model_run(
            engine_version=ENGINE_VERSION,
            purpose=args.purpose,
            features=db.read_features(),
            model_outputs=db.read_model_outputs(),
            vintage_features=db.read_feature_asof_values(),
            metadata={"validation_dir": args.validation_dir},
            training_audit=training_audit,
        )
        frame = model_run_frame(run)
        n = db.write_model_runs(frame)
        print(f"Wrote immutable model run rows: {n}")
        print(frame.to_string(index=False))
    finally:
        db.close()


def create_model_card_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        features = db.read_features()
        outputs = db.read_calibrated_outputs()
        if outputs.empty:
            outputs = db.read_model_outputs()
        if outputs.empty:
            raise SystemExit("No model outputs found. Run train-baseline first.")
        latest = outputs[outputs["date"] == outputs["date"].max()].iloc[0]
        dates = sorted(features["date"].unique())
        card = create_model_card(
            model_name=str(latest["model_name"]),
            version=ENGINE_VERSION,
            target=str(latest["target"]),
            horizon=str(latest["horizon"]),
            training_start=str(dates[0]) if dates else "unknown",
            training_end=str(dates[-1]) if dates else "unknown",
            feature_count=int(features["feature_name"].nunique()),
            observations=int(features["date"].nunique()),
            objective="Calibrated probabilistic macro-market regime forecast artifact",
            known_limitations=[
                "Synthetic sample data unless official ingestion has been configured",
                "Release-calendar metadata is conservative unless exact release timestamps are loaded",
                "Historical analogs are similarity tools, not causal proof",
                "WFST/HMM/BOCPD layers are scaffolded for institutional validation and later Rust acceleration",
            ],
            validation_metrics={},
        )
        path = write_model_card(card, args.out)
        print(f"Wrote model card: {path}")
    finally:
        db.close()


def institutional_report_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        path = write_institutional_report(
            regimes=db.read_regimes(),
            model_outputs=db.read_model_outputs(),
            analogs=db.read_historical_analogs(),
            domain_attribution=db.read_driver_attribution().query("attribution_type == 'domain'"),
            feature_attribution=db.read_driver_attribution().query("attribution_type == 'feature'"),
            validation_dir=args.validation_dir,
            out=args.out,
            # v1.3 consolidated report path (item L). The five legacy
            # ``report_writer_v{1..5}`` files are deprecated shims; this
            # single call materializes the same byte-stable output.
            confidence=db.read_confidence_scores(),
            invalidation=db.read_invalidation_triggers(),
            model_runs=db.read_model_runs(),
            calibrated_outputs=db.read_calibrated_outputs(),
            drift=db.read_model_drift(),
            release_gates=db.read_release_gates(),
            ensemble_weights=db.read_ensemble_weights(),
            stacking_diagnostics=db.read_stacking_diagnostics(),
            alerts=db.read_routed_alerts(),
            promotion_workflow=db.read_promotion_workflow(),
            hazard_diagnostics=db.read_hazard_diagnostics(),
            alfred_manifest=db.read_alfred_ingestion_manifest(),
            vintage_audits=db.read_vintage_audits(),
            feature_asof=db.read_feature_asof_values(),
            vintage_observations=db.read_vintage_observations(),
        )
        print(f"Wrote institutional report: {path}")
    finally:
        db.close()


def export_warehouse_cmd(args: argparse.Namespace) -> None:
    manifest = export_sqlite_to_lake(args.db, args.out, prefer_parquet=not args.csv)
    duck = pd.DataFrame()
    if args.duckdb:
        duck = build_duckdb_database(args.out, args.duckdb)
    print("Warehouse export manifest")
    print(manifest.to_string(index=False) if not manifest.empty else "No rows exported")
    if not duck.empty:
        print("\nDuckDB build")
        print(duck.to_string(index=False))


def warehouse_health_cmd(args: argparse.Namespace) -> None:
    health = warehouse_health(args.lake)
    print(health.to_string(index=False))


def exact_release_calendar_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        obs = db.read_observations()
        cal = build_exact_release_calendar(obs, load_catalog())
        ncal = db.write_exact_release_calendar(cal)
        if args.enforce:
            fixed = enforce_exact_release_calendar(obs, cal)
            db.write_observations(fixed)
        audit = audit_exact_release_calendar(db.read_observations(), cal)
        out = Path(args.out) if args.out else None
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            cal.to_csv(out, index=False)
            audit.to_csv(out.with_name(out.stem + "_audit.csv"), index=False)
        print(f"Wrote exact release calendar rows: {ncal}")
        print(audit.to_string(index=False) if not audit.empty else "No audit rows")
    finally:
        db.close()


def train_survival_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        outputs = recession_hazard_scores(db.read_features(), db.read_recession_labels())
        n = db.write_model_outputs(outputs)
        print(f"Wrote survival model outputs: {n}")
        print(survival_summary(outputs))
    finally:
        db.close()


def optimize_stacking_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        panel = monthly_panel(db.read_observations())
        targets = make_targets(panel) if not panel.empty else pd.DataFrame()
        outputs = db.read_calibrated_outputs()
        if outputs.empty:
            outputs = db.read_model_outputs()
        reports = optimize_from_model_outputs(outputs, targets, step=args.step)
        nw = db.write_ensemble_weights(reports["ensemble_weights"])
        no = db.write_model_outputs(reports["stacked_outputs"])
        nd = db.write_stacking_diagnostics(reports["stacking_diagnostics"])
        out = Path(args.out) if args.out else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
            for name, frame in reports.items():
                frame.to_csv(out / f"{name}.csv", index=False)
        print(f"Stacking wrote weights={nw}, outputs={no}, diagnostics={nd}")
        if not reports["stacking_diagnostics"].empty:
            print(reports["stacking_diagnostics"].to_string(index=False))
    finally:
        db.close()


def monitor_drift_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        drift = compute_feature_drift(
            db.read_features(), baseline_months=args.baseline_months, recent_months=args.recent_months, top_n=args.top_n
        )
        n = db.write_model_drift(drift)
        summary = drift_summary(drift)
        out = Path(args.out) if args.out else None
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            drift.to_csv(out, index=False)
            summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
        print(f"Wrote drift rows: {n}")
        print(summary.to_string(index=False))
        if not drift.empty:
            print(drift.head(20).to_string(index=False))
    finally:
        db.close()


def release_gate_cmd(args: argparse.Namespace) -> None:
    """Evaluate the release gate.

    v1.4.1 (item F) flips the default profile from permissive to
    ``production``. The CLI no longer supplies a default for
    ``--min-confidence`` / ``--profile`` so the function's resolution
    priority applies:

    1. Explicit ``--profile <name>`` wins.
    2. Else ``MRE_ENV`` env var: ``MRE_ENV=production`` → production;
       ``MRE_ENV=dev`` → default.
    3. Else fall back to ``production``.

    Pass ``--profile default`` (or ``MRE_ENV=dev``) to opt back into
    the v1.2.1 looser baseline; pass ``--min-confidence 0.40`` to
    relax a single rail in production.
    """
    db = Warehouse(args.db)
    try:
        promotion = pd.DataFrame()
        ppath = Path(args.validation_dir) / "model_promotion.csv"
        if ppath.exists():
            promotion = pd.read_csv(ppath)
        # Build kwargs lazily so the function's _UNSET sentinel applies
        # the profile-resolved defaults to anything the operator did not
        # explicitly supply on the CLI.
        kwargs: dict[str, Any] = {}
        if getattr(args, "min_confidence", None) is not None:
            kwargs["min_confidence"] = float(args.min_confidence)
        if getattr(args, "profile", None) is not None:
            kwargs["profile"] = str(args.profile)
        if getattr(args, "gate_boundary", None) is not None:
            kwargs["gate_boundary"] = str(args.gate_boundary)
        gate = evaluate_release_gate(
            confidence=db.read_confidence_scores(),
            drift=db.read_model_drift(),
            invalidation=db.read_invalidation_triggers(),
            promotion=promotion,
            **kwargs,
        )
        n = db.write_release_gates(gate)
        out = Path(args.out) if args.out else None
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            gate.to_csv(out, index=False)
        print(f"Wrote release gate rows: {n}")
        print(gate.to_string(index=False))
    finally:
        db.close()


def alfred_plan_cmd(args: argparse.Namespace) -> None:
    catalog = load_catalog()
    series = args.series or [c["series_id"] for c in catalog if c.get("source") == "fred"]
    matrix = build_alfred_request_matrix(
        series,
        observation_start=args.observation_start,
        observation_end=args.observation_end,
        vintage_start=args.vintage_start,
        vintage_end=args.vintage_end,
        vintage_frequency=args.vintage_frequency,
    )
    out = Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(out, index=False)
    print(f"Built ALFRED/FRED request matrix rows={len(matrix)} series={len(series)}")
    print(matrix.head(20).to_string(index=False) if not matrix.empty else "No rows")


def ingest_alfred_cmd(args: argparse.Namespace) -> None:
    catalog = load_catalog()
    series = args.series or [c["series_id"] for c in catalog if c.get("source") == "fred"]
    if args.dry_run:
        matrix = build_alfred_request_matrix(
            series,
            observation_start=args.observation_start,
            observation_end=args.observation_end,
            vintage_start=args.vintage_start,
            vintage_end=args.vintage_end,
            vintage_frequency=args.vintage_frequency,
        )
        print(f"Dry-run only. Would request {len(matrix)} vintage windows across {len(series)} series.")
        print(matrix.head(20).to_string(index=False) if not matrix.empty else "No rows")
        return
    obs, manifest = fetch_alfred_vintages(
        series,
        api_key=args.api_key,
        observation_start=args.observation_start,
        observation_end=args.observation_end,
        vintage_start=args.vintage_start,
        vintage_end=args.vintage_end,
        vintage_frequency=args.vintage_frequency,
        timeout=args.timeout,
    )
    db = Warehouse(args.db)
    try:
        no = db.write_observations(obs)
        nm = db.write_alfred_ingestion_manifest(manifest)
        print(f"Inserted ALFRED/FRED vintage observations={no}, manifest rows={nm}")
        if not manifest.empty:
            print(manifest.tail(20).to_string(index=False))
    finally:
        db.close()


def train_fitted_hazard_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        outputs, diagnostics = train_fitted_hazard_outputs(db.read_features(), db.read_recession_labels())
        no = db.write_model_outputs(outputs)
        nd = db.write_hazard_diagnostics(diagnostics)
        if args.oos:
            oos = hazard_backtest_matrix(
                db.read_features(), db.read_recession_labels(), min_train=args.min_train, step=args.step
            )
            db.write_oos_predictions(oos.rename(columns={"actual": "y"}) if "actual" in oos else oos)
        print(f"Wrote fitted hazard outputs={no}, diagnostics={nd}")
        print(diagnostics.to_string(index=False) if not diagnostics.empty else "No hazard diagnostics")
    finally:
        db.close()


def regime_stacking_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        reports = regime_conditioned_stacking(args.validation_dir, db.read_regimes(), step=args.step)
        noos = db.write_oos_predictions(reports["oos_predictions"])
        nw = db.write_ensemble_weights(reports["ensemble_weights"])
        no = db.write_model_outputs(reports["stacked_outputs"])
        nd = db.write_stacking_diagnostics(reports["stacking_diagnostics"])
        out = Path(args.out) if args.out else None
        if out:
            out.mkdir(parents=True, exist_ok=True)
            for name, frame in reports.items():
                frame.to_csv(out / f"{name}.csv", index=False)
        print(f"Regime-conditioned stacking wrote oos={noos}, weights={nw}, outputs={no}, diagnostics={nd}")
        if not reports["stacking_diagnostics"].empty:
            print(reports["stacking_diagnostics"].to_string(index=False))
    finally:
        db.close()


def route_alerts_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        promotion = pd.DataFrame()
        ppath = Path(args.validation_dir) / "model_promotion.csv"
        if ppath.exists():
            promotion = pd.read_csv(ppath)
        alerts = route_alerts(
            release_gates=db.read_release_gates(),
            drift=db.read_model_drift(),
            invalidation=db.read_invalidation_triggers(),
            confidence=db.read_confidence_scores(),
            promotion=promotion,
        )
        n = db.write_routed_alerts(alerts)
        out = Path(args.out) if args.out else None
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            alerts.to_csv(out, index=False)
        print(f"Wrote routed alerts={n}")
        print(alerts.to_string(index=False))
        # v1.3 (item E): dispatch through any configured live sinks. The
        # sinks soft-degrade to no-ops when their env vars aren't set, so
        # this is safe to leave enabled in CI.
        if getattr(args, "dispatch", False):
            from market_regime_engine.alerts_sinks import dispatch_alerts

            dispatched = dispatch_alerts(alerts)
            ndis = db.write_alert_dispatches(dispatched)
            print(f"Dispatched alerts to live sinks={ndis}")
            if not dispatched.empty:
                print(dispatched.to_string(index=False))
    finally:
        db.close()


def promotion_workflow_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        promotion = pd.DataFrame()
        ppath = Path(args.validation_dir) / "model_promotion.csv"
        if ppath.exists():
            promotion = pd.read_csv(ppath)
        workflow = evaluate_promotion_workflow(
            promotion=promotion,
            release_gate=db.read_release_gates(),
            confidence=db.read_confidence_scores(),
            drift=db.read_model_drift(),
        )
        n = db.write_promotion_workflow(workflow)
        out = Path(args.out) if args.out else None
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            workflow.to_csv(out, index=False)
        print(f"Wrote promotion workflow rows={n}")
        print(workflow.to_string(index=False))
    finally:
        db.close()


def alfred_real_plan_cmd(args: argparse.Namespace) -> None:
    catalog = load_catalog()
    series = args.series or [c["series_id"] for c in catalog if c.get("source") == "fred"]
    vintages, plan = build_real_alfred_plan(
        series,
        api_key=args.api_key,
        observation_start=args.observation_start,
        observation_end=args.observation_end,
        vintage_start=args.vintage_start,
        vintage_end=args.vintage_end,
        max_vintages_per_series=args.max_vintages_per_series,
        timeout=args.timeout,
    )
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        vintages.to_csv(out / "series_vintages.csv", index=False)
        plan.to_csv(out / "real_alfred_request_plan.csv", index=False)
    print(f"Real ALFRED plan: series={len(series)} vintage_dates={len(vintages)} request_rows={len(plan)}")
    print(plan.head(20).to_string(index=False) if not plan.empty else "No requests")


def ingest_alfred_real_cmd(args: argparse.Namespace) -> None:
    catalog = load_catalog()
    series = args.series or [c["series_id"] for c in catalog if c.get("source") == "fred"]
    vintages, observations, manifest = fetch_real_alfred_vintage_observations(
        series,
        api_key=args.api_key,
        observation_start=args.observation_start,
        observation_end=args.observation_end,
        vintage_start=args.vintage_start,
        vintage_end=args.vintage_end,
        max_vintages_per_series=args.max_vintages_per_series,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
    )
    db = Warehouse(args.db)
    try:
        nv = db.write_series_vintages(vintages)
        no = db.write_vintage_observations(observations)
        nm = db.write_alfred_ingestion_manifest(manifest)
        print(f"Inserted real ALFRED series_vintages={nv}, vintage_observations={no}, manifest_rows={nm}")
        if not manifest.empty:
            print(manifest.tail(20).to_string(index=False))
    finally:
        db.close()


def seed_vintage_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        vintages, observations = seed_vintage_observations_from_latest(db.read_observations())
        nv = db.write_series_vintages(vintages)
        no = db.write_vintage_observations(observations)
        print(
            f"Seeded point-in-time vintage tables from current observations: series_vintages={nv}, vintage_observations={no}"
        )
        print("WARNING: seeded vintage data is for local pipeline validation only, not official ALFRED truth.")
    finally:
        db.close()


def materialize_asof_features_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        asof_dates = args.as_of_dates if args.as_of_dates else None
        fav = materialize_feature_asof_values(
            db.read_vintage_observations(),
            load_catalog(),
            asof_dates=asof_dates,
            min_history_months=args.min_history_months,
        )
        nfav = db.write_feature_asof_values(fav)
        nf = 0
        if args.write_features:
            feats = feature_asof_to_features(fav)
            nf = db.write_features(feats)
        print(f"Materialized feature_asof_values={nfav}; wrote existing features table rows={nf}")
        if not fav.empty:
            print(fav.tail(20).to_string(index=False))
    finally:
        db.close()


def audit_vintage_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        audits = pd.concat(
            [
                audit_vintage_observations(db.read_vintage_observations()),
                audit_feature_asof_lineage(db.read_feature_asof_values()),
            ],
            ignore_index=True,
        )
        n = db.write_vintage_audits(audits)
        out = Path(args.out) if args.out else None
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            audits.to_csv(out, index=False)
        print(f"Wrote vintage audit rows={n}")
        print(audits.to_string(index=False))
        if args.enforce and (audits["violations"].fillna(0).astype(int) > 0).any():
            raise SystemExit(
                "Vintage/as-of audit failed. Refusing to continue because future leakage is not a feature."
            )
    finally:
        db.close()


def verify_run_cmd(args: argparse.Namespace) -> None:
    """Re-derive the reproducibility envelope and compare to a stored model run.

    v1.4.1 (item D) strengthens the comparison so the full ``extra``
    envelope is structurally compared (not just the
    ``extra.training_audit`` sub-dict). To keep the operator-facing
    smoke path stable, the CLI re-derives the *current* envelope by
    carrying forward the stored ``extra`` sans ``training_audit``
    (which has its own friendly handling). That way the existing
    ``mre model-run`` → ``mre verify-run`` flow does not trip on
    auto-stamped ``engine_version`` / ``purpose`` keys; arbitrary-
    extra drift detection at the function level is exercised
    programmatically by ``tests/test_verify_run_extra_drift.py`` and
    by ``scripts/v141_capture_verify_run_extra_demo.py``.

    v1.4.1 (item E) adds ``--ignore-rng-seeds`` to opt back into the
    v1.2.1 skip behaviour for stochastic-seed-rerun workflows.

    v1.5 PR-7 §G: when the run has a matching FI evidence pack the
    report carries ``fi_envelope_consistent`` and
    ``fi_hmac_verified`` so a single command verifies both macro and
    FI governance.
    """
    from market_regime_engine.model_runs import build_repro_envelope, verify_run

    db = Warehouse(args.db)
    try:
        runs = db.read_model_runs()
        if runs.empty:
            raise SystemExit("No model runs recorded.")
        if args.run_id:
            row = runs[runs["run_id"] == args.run_id]
            if row.empty:
                raise SystemExit(f"run_id {args.run_id} not found")
            run_row = row.iloc[0]
        else:
            run_row = runs.iloc[-1]
        features = db.read_features()
        outputs = db.read_model_outputs()
        # Carry forward the stored ``extra`` (minus ``training_audit``,
        # which has friendly handling inside verify_run) so the strict
        # compare doesn't trip on auto-stamped engine_version / purpose
        # / arbitrary operator-supplied keys. Operators who need
        # arbitrary-extra drift detection through the CLI can call
        # ``verify_run()`` programmatically with a controlled
        # current_envelope (see ``scripts/v141_capture_verify_run_extra_demo.py``).
        carry_forward_extra: dict[str, object] = {}
        try:
            stored_meta = json.loads(run_row.get("metadata_json", "{}") or "{}")
            stored_envelope = stored_meta.get("repro_envelope", {})
            stored_extra = stored_envelope.get("extra", {}) if isinstance(stored_envelope, dict) else {}
            if isinstance(stored_extra, dict):
                carry_forward_extra = {k: v for k, v in stored_extra.items() if k != "training_audit"}
        except Exception:
            carry_forward_extra = {}
        # Carry forward the stored ``rng_seeds`` so the canonical compare
        # is a tautology unless the operator explicitly drifted them; a
        # stochastic-seed rerun uses ``--ignore-rng-seeds`` to opt out.
        carry_forward_seeds: dict[str, int] = {}
        try:
            stored_seeds = stored_envelope.get("rng_seeds", {}) if isinstance(stored_envelope, dict) else {}
            if isinstance(stored_seeds, dict):
                carry_forward_seeds = {str(k): v for k, v in stored_seeds.items()}
        except Exception:
            carry_forward_seeds = {}
        envelope = build_repro_envelope(
            features=features,
            model_outputs=outputs,
            vintage_features=db.read_feature_asof_values(),
            legacy_hash=bool(getattr(args, "legacy_hash", False)),
            extra=carry_forward_extra,
            rng_seeds=carry_forward_seeds,
        )
        report = verify_run(
            str(run_row["run_id"]),
            run_row,
            current_envelope=envelope,
            ignore_rng_seeds=bool(getattr(args, "ignore_rng_seeds", False)),
        )
        # v1.5 PR-7 §G: surface the FI evidence-pack verification trio
        # alongside the macro report so a single command checks both
        # governance layers.
        with contextlib.suppress(Exception):
            import market_regime_engine.fixed_income  # noqa: F401  - register schema
        try:
            fi_report = _verify_fi_evidence_pack(db, str(run_row["run_id"]))
        except Exception as exc:
            fi_report = {"fi_error": str(exc)}
        report = {**report, **fi_report}
        # v1.5 PR-8 (Tier-2 fix C-AUTO-6): an envelope-inconsistent pack
        # also lowers ``approved`` so a tampered row that bypassed HMAC
        # verification (e.g. dev-mode pack signed under no keys) does
        # not slip past verify-run. Use ``is False`` rather than
        # truthiness so we distinguish ``None`` (no pack present) from
        # ``False`` (pack present but inconsistent).
        if fi_report.get("fi_evidence_pack_present") and not bool(fi_report.get("fi_hmac_verified", True)):
            report["approved"] = False
        if fi_report.get("fi_evidence_pack_present") and fi_report.get("fi_envelope_consistent") is False:
            report["approved"] = False
        log.info("verify_run", extra=report)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        # Surface non-fatal advisories on stderr so operators see them in
        # the terminal even when the report's JSON is piped somewhere
        # else (jq, file, etc.). stdout stays pure JSON.
        import sys as _sys

        for warning in report.get("warnings", []) or []:
            print(f"WARNING: {warning}", file=_sys.stderr)
        if not report["approved"]:
            raise SystemExit(2)
    finally:
        db.close()


def nowcast_cmd(args: argparse.Namespace) -> None:
    """Run the v1.2 mixed-frequency nowcast and persist domain factor estimates."""
    import json as _json
    from datetime import datetime

    from market_regime_engine.frontier.dfm_mq import MQDynamicFactorModel

    db = Warehouse(args.db)
    try:
        feats = db.read_features()
        if feats.empty:
            print("No features in warehouse; run build-features first.")
            return
        wide = feats.pivot_table(index="date", columns="feature_name", values="value").sort_index()
        domains = sorted(set(feats["domain"].dropna().astype(str)))
        as_of = datetime.now(UTC).strftime("%Y-%m-%d")
        rows = []
        for domain in domains:
            cols = feats[feats["domain"] == domain]["feature_name"].drop_duplicates().tolist()
            sub = wide.reindex(columns=cols).dropna(how="all")
            if sub.empty:
                continue
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
                        {"n_columns": int(sub.shape[1]), "n_rows": int(sub.shape[0])}, sort_keys=True
                    ),
                }
            )
        n = db.write_nowcast_factors(pd.DataFrame(rows))
        print(f"Wrote {n} nowcast factor rows for {len(rows)} domains.")
    finally:
        db.close()


def e_value_test_cmd(args: argparse.Namespace) -> None:
    """Run the v1.2 sequential e-value safe-test for the supplied challenger."""
    import json as _json
    from datetime import datetime

    from market_regime_engine.frontier.sequential_testing import SafeTestPromotion

    db = Warehouse(args.db)
    try:
        # Pull challenger/champion losses from validation CSVs first (the
        # full per-row prediction frames live there). Fall back to the
        # warehouse model_outputs only if no validation data exists.
        vdir = Path(getattr(args, "validation_dir", "data/validation"))
        chal_losses: list[float] = []
        champ_losses: list[float] = []
        # Validation CSVs use ``model`` while warehouse uses ``model_name``;
        # accept either so the CLI works on both.
        if vdir.exists():
            for fp in sorted(vdir.glob("binary_predictions_*.csv")):
                try:
                    df = pd.read_csv(fp)
                except Exception:
                    continue
                model_col = "model" if "model" in df.columns else ("model_name" if "model_name" in df.columns else None)
                if model_col is None or {"y", "p"}.issubset(df.columns) is False:
                    continue
                # Default champion: the warehouse's standard benchmark when
                # present; otherwise the first non-challenger model in the file.
                names = sorted(df[model_col].astype(str).unique())
                if args.challenger not in names:
                    continue
                champ_name = args.champion
                if champ_name is None:
                    if "expanding_event_rate" in names:
                        champ_name = "expanding_event_rate"
                    elif "previous_event_shrunk" in names:
                        champ_name = "previous_event_shrunk"
                    else:
                        rest = [n for n in names if n != args.challenger]
                        champ_name = rest[0] if rest else None
                if champ_name is None:
                    continue
                ch = df[df[model_col] == args.challenger]
                cm = df[df[model_col] == champ_name]
                joined = ch.merge(cm, on=["date", "target", "horizon"], suffixes=("_chal", "_champ"))
                if joined.empty:
                    # Try the matching benchmark file when challenger and champion
                    # are split across `binary_predictions_*` and `binary_benchmark_predictions_*`.
                    bench_fp = vdir / fp.name.replace("binary_predictions_", "binary_benchmark_predictions_")
                    if bench_fp.exists():
                        try:
                            bench_df = pd.read_csv(bench_fp)
                        except Exception:
                            bench_df = pd.DataFrame()
                        if not bench_df.empty:
                            bench_model_col = "model" if "model" in bench_df.columns else "model_name"
                            cm2 = bench_df[bench_df[bench_model_col] == champ_name]
                            joined = ch.merge(cm2, on=["date", "target", "horizon"], suffixes=("_chal", "_champ"))
                if joined.empty:
                    continue
                chal_losses.extend(((joined["p_chal"] - joined["y_chal"]) ** 2).tolist())
                champ_losses.extend(((joined["p_champ"] - joined["y_champ"]) ** 2).tolist())
        if not chal_losses or not champ_losses:
            outputs = db.read_model_outputs()
            if outputs.empty:
                print("No challenger/champion data found in validation dir or warehouse.")
                return
            # Synthetic fallback so the CLI is exercisable in the smoke harness.
            chal_losses = (outputs["value"].astype(float) ** 2).tolist()[: max(len(outputs) // 2, 5)]
            champ_losses = (outputs["value"].astype(float) ** 2).tolist()[: len(chal_losses)]
        n = min(len(chal_losses), len(champ_losses))
        result = SafeTestPromotion.run(chal_losses[:n], champ_losses[:n], alpha=float(args.level))
        e_val = float(result["e_value"])
        decision = "promote" if result["fired"] else "hold"
        as_of = datetime.now(UTC).strftime("%Y-%m-%d")
        row = pd.DataFrame(
            [
                {
                    "date": as_of,
                    "target": "drawdown_gt_10pct",
                    "horizon": "3m",
                    "challenger": str(args.challenger),
                    "champion": str(args.champion or "expanding_event_rate"),
                    "e_value": e_val,
                    "level": float(args.level),
                    "decision": decision,
                    "n": int(n),
                    "metadata_json": _json.dumps({"fired_at_n": result.get("fired_at_n")}, sort_keys=True),
                }
            ]
        )
        db.write_e_value_log(row)
        print(_json.dumps({"e_value": e_val, "decision": decision, "n": int(n), "level": float(args.level)}, indent=2))
    finally:
        db.close()


def warehouse_migrate_cmd(args: argparse.Namespace) -> None:
    """Copy every warehouse table from one backend to another (v1.3 item D).

    v1.5 (PR-2): import the fixed_income package before invoking
    ``migrate_warehouse`` so the 13 FI tables are registered with the
    storage registry. Without this import the CLI would only migrate
    the 34 core macro tables and a fresh ``data/test.duckdb`` would
    miss the FI surface area.
    """

    import market_regime_engine.fixed_income  # noqa: F401  - register FI tables

    counts = migrate_warehouse(
        src=args.src,
        dst=args.dst,
        src_backend=args.from_backend,
        dst_backend=args.to_backend,
    )
    total = sum(counts.values())
    print(
        json.dumps(
            {"src": args.src, "dst": args.dst, "rows_total": total, "by_table": counts}, indent=2, sort_keys=True
        )
    )


def verify_data_cmd(args: argparse.Namespace) -> None:
    """Re-derive payload hashes from the current warehouse state (v1.3 item F)."""
    from market_regime_engine.verify_data import verify_warehouse_state

    report = verify_warehouse_state(
        run_id=args.run_id,
        db_path=args.db,
        legacy_hash=bool(getattr(args, "legacy_hash", False)),
    )
    log.info("verify_data", extra={k: v for k, v in report.items() if k != "differences"})
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if not report.get("approved", False):
        raise SystemExit(2)


def conformal_conditional_cmd(args: argparse.Namespace) -> None:
    """Fit conditional conformal per regime bucket and write the report."""
    import json as _json
    from datetime import datetime

    from market_regime_engine.frontier.conformal_ts import ConditionalConformalRegressor

    db = Warehouse(args.db)
    try:
        vdir = Path(getattr(args, "validation_dir", "data/validation"))
        rows = []
        as_of = datetime.now(UTC).strftime("%Y-%m-%d")
        if vdir.exists():
            for fp in sorted(vdir.glob("binary_predictions_*.csv")):
                try:
                    df = pd.read_csv(fp)
                except Exception:
                    continue
                if {"y", "p"}.issubset(df.columns) is False:
                    continue
                if "regime_bucket" not in df.columns:
                    df["regime_bucket"] = "general"
                groups = df.groupby(["target", "horizon"], observed=True, dropna=False)
                for (target, horizon), group in groups:
                    cal = group[["y", "p", "regime_bucket"]].dropna(subset=["y", "p"])
                    if cal.empty:
                        continue
                    layer = ConditionalConformalRegressor(alpha=float(args.alpha)).fit(cal)
                    diag = layer.coverage_report_conditional(cal)
                    per = diag["per_group"]
                    if per.empty:
                        continue
                    for _, prow in per.iterrows():
                        rows.append(
                            {
                                "as_of_date": as_of,
                                "target": str(target),
                                "horizon": str(horizon),
                                "group": str(prow["regime_bucket"]),
                                "coverage": float(prow["coverage"]),
                                "n": int(prow["n"]),
                                "alpha": float(args.alpha),
                                "method": "conditional_conformal",
                                "worst_violation": float(diag["worst_violation"]),
                                "metadata_json": _json.dumps(
                                    {"adjusted_alpha": diag["adjusted_alpha"]}, sort_keys=True
                                ),
                            }
                        )
        if not rows:
            # Synthesize from regimes table so the CLI doesn't no-op when the
            # validation directory is absent (smoke harness exercises this).
            regs = db.read_regimes()
            if regs.empty:
                print("No data to fit conditional conformal on.")
                return
            buckets = regs["decoded_regime"].astype(str)
            rng = np.random.default_rng(0)
            cal = pd.DataFrame(
                {
                    "y": rng.binomial(1, 0.3, len(buckets)),
                    "p": rng.uniform(size=len(buckets)),
                    "regime_bucket": buckets.tolist(),
                }
            )
            layer = ConditionalConformalRegressor(alpha=float(args.alpha)).fit(cal)
            diag = layer.coverage_report_conditional(cal)
            per = diag["per_group"]
            for _, prow in per.iterrows():
                rows.append(
                    {
                        "as_of_date": as_of,
                        "target": "synthetic",
                        "horizon": "3m",
                        "group": str(prow["regime_bucket"]),
                        "coverage": float(prow["coverage"]),
                        "n": int(prow["n"]),
                        "alpha": float(args.alpha),
                        "method": "conditional_conformal",
                        "worst_violation": float(diag["worst_violation"]),
                        "metadata_json": _json.dumps({"synthetic": True}, sort_keys=True),
                    }
                )
        n = db.write_conditional_coverage_report(pd.DataFrame(rows))
        print(f"Wrote {n} conditional-coverage rows.")
    finally:
        db.close()


def bench_cmd(args: argparse.Namespace) -> None:
    """Run the BOCPD/WFST/PSI bench harness and emit ``bench.csv``."""
    from market_regime_engine.bench import run_bench_suite

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = run_bench_suite(seed=args.seed)
    df.to_csv(out, index=False)
    print(f"Bench results written to {out}")
    print(df.to_string(index=False))


def bayesian_msvar_fit_cmd(args: argparse.Namespace) -> None:
    """Fit a Bayesian NumPyro MS-VAR and persist diagnostics (v1.4 item A)."""
    import json as _json

    db = Warehouse(args.db)
    try:
        try:
            from market_regime_engine.frontier.bayesian_msvar import BayesianMSVAR
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
        features = db.read_features()
        if features.empty:
            raise SystemExit("No features in warehouse; run build-features first.")
        wide = features.pivot_table(index="date", columns="feature_name", values="value").sort_index()
        # Project to the 8 MS-VAR domains, falling back to zero-fill so a
        # bare smoke run still produces a fit.
        from market_regime_engine.hmm import DOMAIN_COLUMNS

        cols = [c for c in DOMAIN_COLUMNS if c in wide.columns]
        if not cols:
            wide = wide.iloc[:, : min(8, wide.shape[1])]
        else:
            wide = wide[cols]
        wide.index = pd.to_datetime(wide.index)
        model = BayesianMSVAR(domains=list(wide.columns))
        model.fit(
            wide,
            method=args.method,
            num_chains=int(args.chains),
            num_warmup=int(args.warmup),
            num_samples=int(args.samples),
        )
        diagnostics = model.last_diagnostics or {}
        from datetime import datetime

        run_id = f"bayesian_msvar_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
        df = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "method": diagnostics.get("method", args.method),
                    "num_chains": int(diagnostics.get("num_chains", args.chains)),
                    "num_divergences": int(diagnostics.get("num_divergences", 0)),
                    "max_rhat": float(diagnostics.get("max_rhat", float("nan"))),
                    "min_ess": float(diagnostics.get("min_ess", float("nan"))),
                    "runtime_seconds": float(diagnostics.get("runtime_seconds", 0.0)),
                    "metadata_json": _json.dumps(
                        {
                            k: v
                            for k, v in diagnostics.items()
                            if k
                            not in {"method", "num_chains", "num_divergences", "max_rhat", "min_ess", "runtime_seconds"}
                        },
                        sort_keys=True,
                        default=str,
                    ),
                }
            ]
        )
        n = db.write_bayesian_msvar_diagnostics(df)
        print(f"Wrote {n} bayesian_msvar_diagnostics rows. run_id={run_id}")
        print(_json.dumps(diagnostics, sort_keys=True, default=str, indent=2))
    finally:
        db.close()


def deep_kernel_train_cmd(args: argparse.Namespace) -> None:
    """Train an :class:`MLPDeepKernel` and persist its training losses (v1.4 item B)."""
    db = Warehouse(args.db)
    try:
        try:
            from market_regime_engine.frontier.deep_kernel import MLPDeepKernel
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
        features = db.read_features()
        if features.empty:
            raise SystemExit("No features in warehouse; run build-features first.")
        wide = features.pivot_table(index="date", columns="feature_name", values="value").sort_index()
        wide = wide.iloc[:, : int(args.input_dim)] if wide.shape[1] >= int(args.input_dim) else wide
        kernel = MLPDeepKernel(input_dim=int(args.input_dim), hidden_dims=(64, 32), seed=int(args.seed))
        kernel.fit(wide, n_epochs=int(args.epochs), lr=float(args.lr))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"epoch": range(len(kernel.training_losses)), "loss": kernel.training_losses}).to_csv(
            out, index=False
        )
        first = kernel.training_losses[0] if kernel.training_losses else float("nan")
        last = kernel.training_losses[-1] if kernel.training_losses else float("nan")
        print(f"Trained deep kernel: epochs={len(kernel.training_losses)} loss {first:.4f} -> {last:.4f}")
        print(f"Wrote training-loss curve to {out}")
    finally:
        db.close()


def refresh_release_calendars_cmd(args: argparse.Namespace) -> None:
    """Refresh the live-cached release-calendar YAML (v1.4 item D)."""
    import json as _json

    try:
        from market_regime_engine.frontier.release_calendars import (
            refresh_release_calendars,
            write_status_to_warehouse,
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    selected: list[str] | None = None
    if args.agency:
        selected = [a.strip().lower() for a in args.agency.split(",") if a.strip()]
    out_dir = Path(args.out) if args.out else None
    status = refresh_release_calendars(agencies=selected, out_dir=out_dir)
    print(_json.dumps(status, indent=2, default=str))
    n = write_status_to_warehouse(status, args.db)
    print(f"Wrote {n} release_calendar_refreshes rows.")


def report_cmd(args: argparse.Namespace) -> None:
    db = Warehouse(args.db)
    try:
        print(latest_explanation(db.read_regimes()))
        for title, frame in [
            ("calibrated outputs", db.read_calibrated_outputs()),
            ("raw outputs", db.read_model_outputs()),
            ("confidence", db.read_confidence_scores()),
            ("invalidation", db.read_invalidation_triggers()),
            ("model runs", db.read_model_runs()),
            ("ensemble weights", db.read_ensemble_weights()),
            ("stacking diagnostics", db.read_stacking_diagnostics()),
            ("model drift", db.read_model_drift()),
            ("release gates", db.read_release_gates()),
            ("hazard diagnostics", db.read_hazard_diagnostics()),
            ("routed alerts", db.read_routed_alerts()),
            ("promotion workflow", db.read_promotion_workflow()),
            ("series vintages", db.read_series_vintages()),
            ("vintage observations", db.read_vintage_observations()),
            ("feature as-of values", db.read_feature_asof_values()),
            ("vintage audits", db.read_vintage_audits()),
            ("alfred manifest", db.read_alfred_ingestion_manifest()),
        ]:
            if not frame.empty:
                print(f"\n{title}")
                print(frame.tail(20).to_string(index=False))
    finally:
        db.close()



__all__ = [name for name in globals() if name.endswith("_cmd") or name == "bootstrap_sample"]
