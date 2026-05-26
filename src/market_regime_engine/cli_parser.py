# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

from market_regime_engine.cli_handlers import *  # noqa: F401,F403 - parser binds handler names

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mre")
    sub = p.add_subparsers(required=True, dest="cmd")

    s = sub.add_parser("bootstrap-sample")
    s.add_argument("--db", default="data/mre.duckdb")
    s.set_defaults(func=bootstrap_sample)

    s = sub.add_parser("ingest-fred-vintages")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--series", nargs="*")
    s.add_argument("--observation-start", default="1960-01-01")
    s.add_argument("--vintage-start", default="1990-01-01")
    s.add_argument("--vintage-end")
    s.add_argument("--vintage-frequency", default="MS")
    s.set_defaults(func=ingest_fred_vintages_cmd)

    s = sub.add_parser("ingest-fred-recession")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--series", default="USREC")
    s.add_argument("--observation-start", default="1960-01-01")
    s.set_defaults(func=ingest_fred_recession_cmd)

    s = sub.add_parser("pit-check")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--write-adjusted", action="store_true")
    s.add_argument(
        "--allow-missing-release-rules",
        action="store_true",
        help=(
            "Fall back to the v1.2.1 silent zero-lag behaviour for any "
            "series that lacks an entry in DEFAULT_RELEASE_RULES. v1.3 "
            "raises by default; this flag re-enables the legacy path."
        ),
    )
    s.set_defaults(func=pit_check_cmd)

    s = sub.add_parser("audit-release-calendar")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--calendar", default="config/release_calendar.yaml")
    s.add_argument("--enforce", action="store_true")
    s.add_argument("--out", default="data/release_calendar_audit.csv")
    s.add_argument(
        "--tolerance-days",
        type=int,
        default=3,
        help=(
            "v1.4 (item D): tolerance in days between vintage_observations.realtime_start "
            "and the YAML calendar's release_timestamp_utc; vintages outside this band "
            "are flagged. ``--enforce`` exits 2 when any row is flagged."
        ),
    )
    s.set_defaults(func=audit_release_calendar_cmd)

    s = sub.add_parser("build-features")
    s.add_argument("--db", default="data/mre.duckdb")
    s.set_defaults(func=build_feature_cmd)

    s = sub.add_parser("label-recessions")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument(
        "--force-builtin",
        action="store_true",
        help="Skip FRED USREC fetch and use the built-in NBER window list.",
    )
    s.add_argument(
        "--max-stale-months",
        type=int,
        default=None,
        help="Fail if labels are more than this many months behind the panel.",
    )
    s.set_defaults(func=label_recessions_cmd)

    s = sub.add_parser("score-regime")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--disable-bocpd", action="store_true")
    s.set_defaults(func=score_regime_cmd)

    s = sub.add_parser("train-baseline")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument(
        "--legacy-features",
        action="store_true",
        help="Train from the legacy features table instead of feature_asof_values.",
    )
    s.add_argument(
        "--allow-legacy-fallback",
        action="store_true",
        help=(
            "Allow PIT mode to fall back to LEGACY when feature_asof_values "
            "is empty. Without this flag, missing PIT features fail closed "
            "(v1.2.1+); with it, the fallback is recorded as authorized in "
            "the training audit so verify-run surfaces a non-fatal warning."
        ),
    )
    s.set_defaults(func=train_baseline_cmd)

    s = sub.add_parser("validate")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--out", default="data/validation")
    s.add_argument("--min-train", type=int, default=120)
    s.add_argument("--step", type=int, default=6)
    s.add_argument("--legacy-features", action="store_true")
    s.add_argument(
        "--allow-legacy-fallback",
        action="store_true",
        help=(
            "See `train-baseline --allow-legacy-fallback`. Allows PIT mode to "
            "fall back to LEGACY when feature_asof_values is empty."
        ),
    )
    s.set_defaults(func=validate_cmd)

    s = sub.add_parser("calibrate-probabilities")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--validation-dir", default="data/validation")
    s.add_argument("--out", default="data/calibration")
    s.set_defaults(func=calibrate_probabilities_cmd)

    s = sub.add_parser("analogs")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--as-of")
    s.add_argument("--top-n", type=int, default=10)
    s.add_argument("--min-history", type=int, default=60)
    s.add_argument("--out")
    s.add_argument("--regime-weighted", action="store_true")
    s.set_defaults(func=analogs_cmd)

    s = sub.add_parser("attribute")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--as-of")
    s.add_argument("--top-n", type=int, default=20)
    s.add_argument("--out")
    s.set_defaults(func=attribution_cmd)

    s = sub.add_parser("invalidation-triggers")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--out", default="data/invalidation_triggers.csv")
    s.set_defaults(func=invalidation_cmd)

    s = sub.add_parser("score-confidence")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--validation-dir", default="data/validation")
    s.set_defaults(func=confidence_cmd)

    s = sub.add_parser("model-run")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--purpose", default="v0.7 local model run")
    s.add_argument("--validation-dir", default="data/validation")
    s.set_defaults(func=model_run_cmd)

    s = sub.add_parser("model-card")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--out", default="data/model_cards")
    s.set_defaults(func=create_model_card_cmd)

    s = sub.add_parser("institutional-report")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--out", default="data/reports/institutional_report.md")
    s.add_argument("--validation-dir", default="data/validation")
    s.set_defaults(func=institutional_report_cmd)

    s = sub.add_parser("export-warehouse")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--out", default="data/lake")
    s.add_argument("--duckdb", default="data/mre.duckdb")
    s.add_argument("--csv", action="store_true")
    s.set_defaults(func=export_warehouse_cmd)

    s = sub.add_parser("warehouse-health")
    s.add_argument("--lake", default="data/lake")
    s.set_defaults(func=warehouse_health_cmd)

    s = sub.add_parser("build-exact-release-calendar")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--out", default="data/exact_release_calendar.csv")
    s.add_argument("--enforce", action="store_true")
    s.set_defaults(func=exact_release_calendar_cmd)

    s = sub.add_parser("train-survival")
    s.add_argument("--db", default="data/mre.duckdb")
    s.set_defaults(func=train_survival_cmd)

    s = sub.add_parser("optimize-stacking")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--out", default="data/stacking")
    s.add_argument("--step", type=float, default=0.1)
    s.set_defaults(func=optimize_stacking_cmd)

    s = sub.add_parser("monitor-drift")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--baseline-months", type=int, default=120)
    s.add_argument("--recent-months", type=int, default=12)
    s.add_argument("--top-n", type=int, default=50)
    s.add_argument("--out", default="data/model_drift.csv")
    s.set_defaults(func=monitor_drift_cmd)

    s = sub.add_parser("release-gate")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--validation-dir", default="data/validation")
    s.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help=(
            "v1.4.1 (item F): default is None so the resolved profile "
            "supplies the threshold (production: 0.75, default: 0.55). "
            "Pass an explicit value to override the profile-resolved "
            "default for this single rail."
        ),
    )
    s.add_argument("--out", default="data/release_gate.csv")
    s.add_argument(
        "--gate-boundary",
        default="stable_core",
        choices=["stable_core", "experimental_frontier"],
        help=(
            "Target the mature stable core or the opt-in experimental frontier boundary. "
            "experimental_frontier requires MRE_ENABLE_EXPERIMENTAL_FRONTIER=1."
        ),
    )
    s.add_argument(
        "--profile",
        default=None,
        choices=[None, "default", "production", "certification"],
        help=(
            "v1.3 (item G) introduced ``production``; v1.4.1 (item F) "
            "flipped the default profile to ``production`` when no "
            "``--profile`` is passed and ``MRE_ENV`` is unset. "
            "Resolution priority: explicit ``--profile`` > "
            "``MRE_ENV=production|dev`` > fallback ``production``. "
            "Use ``--profile default`` (or ``MRE_ENV=dev``) to opt "
            "back into the v1.2.1 looser baseline; use ``--profile certification`` "
            "for fail-closed audit-grade gates."
        ),
    )
    s.set_defaults(func=release_gate_cmd)

    s = sub.add_parser("alfred-plan")
    s.add_argument("--series", nargs="*")
    s.add_argument("--observation-start", default="1960-01-01")
    s.add_argument("--observation-end")
    s.add_argument("--vintage-start", default="1990-01-01")
    s.add_argument("--vintage-end")
    s.add_argument("--vintage-frequency", default="MS")
    s.add_argument("--out", default="data/alfred_request_matrix.csv")
    s.set_defaults(func=alfred_plan_cmd)

    s = sub.add_parser("ingest-alfred")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--series", nargs="*")
    s.add_argument("--api-key")
    s.add_argument("--observation-start", default="1960-01-01")
    s.add_argument("--observation-end")
    s.add_argument("--vintage-start", default="1990-01-01")
    s.add_argument("--vintage-end")
    s.add_argument("--vintage-frequency", default="MS")
    s.add_argument("--timeout", type=int, default=30)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=ingest_alfred_cmd)

    s = sub.add_parser("train-fitted-hazard")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--oos", action="store_true")
    s.add_argument("--min-train", type=int, default=120)
    s.add_argument("--step", type=int, default=3)
    s.set_defaults(func=train_fitted_hazard_cmd)

    s = sub.add_parser("optimize-regime-stacking")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--validation-dir", default="data/validation")
    s.add_argument("--out", default="data/regime_stacking")
    s.add_argument("--step", type=float, default=0.1)
    s.set_defaults(func=regime_stacking_cmd)

    s = sub.add_parser("route-alerts")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--validation-dir", default="data/validation")
    s.add_argument("--out", default="data/routed_alerts.csv")
    s.add_argument(
        "--dispatch",
        action="store_true",
        help=(
            "v1.3 (item E): after writing routed_alerts, dispatch every "
            "alert through configured live sinks (Slack/Email/PagerDuty) "
            "and persist the result to the new alert_dispatches table. "
            "Sinks are no-ops when their env vars are unset."
        ),
    )
    s.set_defaults(func=route_alerts_cmd)

    s = sub.add_parser("promotion-workflow")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--validation-dir", default="data/validation")
    s.add_argument("--out", default="data/promotion_workflow.csv")
    s.set_defaults(func=promotion_workflow_cmd)

    s = sub.add_parser("alfred-real-plan")
    s.add_argument("--series", nargs="*")
    s.add_argument("--api-key")
    s.add_argument("--observation-start", default="1960-01-01")
    s.add_argument("--observation-end")
    s.add_argument("--vintage-start", default="1990-01-01")
    s.add_argument("--vintage-end")
    s.add_argument("--max-vintages-per-series", type=int)
    s.add_argument("--timeout", type=int, default=30)
    s.add_argument("--out", default="data/alfred_real_plan")
    s.set_defaults(func=alfred_real_plan_cmd)

    s = sub.add_parser("ingest-alfred-real")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--series", nargs="*")
    s.add_argument("--api-key")
    s.add_argument("--observation-start", default="1960-01-01")
    s.add_argument("--observation-end")
    s.add_argument("--vintage-start", default="1990-01-01")
    s.add_argument("--vintage-end")
    s.add_argument("--max-vintages-per-series", type=int)
    s.add_argument("--sleep-seconds", type=float, default=0.0)
    s.add_argument("--timeout", type=int, default=30)
    s.set_defaults(func=ingest_alfred_real_cmd)

    s = sub.add_parser("seed-vintage-from-observations")
    s.add_argument("--db", default="data/mre.duckdb")
    s.set_defaults(func=seed_vintage_cmd)

    s = sub.add_parser("materialize-asof-features")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--as-of-dates", nargs="*")
    s.add_argument("--min-history-months", type=int, default=36)
    s.add_argument("--write-features", action="store_true")
    s.set_defaults(func=materialize_asof_features_cmd)

    s = sub.add_parser("audit-vintage")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--out", default="data/vintage_audit.csv")
    s.add_argument("--enforce", action="store_true")
    s.set_defaults(func=audit_vintage_cmd)

    s = sub.add_parser("report")
    s.add_argument("--db", default="data/mre.duckdb")
    s.set_defaults(func=report_cmd)

    s = sub.add_parser("verify-run")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--run-id", default=None, help="Run id to verify (default: latest).")
    s.add_argument(
        "--legacy-hash",
        action="store_true",
        help=(
            "Use the v1.2.1 _hash_frame implementation when re-deriving "
            "the envelope. Required when verifying runs stored before "
            "the v1.3 hash migration."
        ),
    )
    s.add_argument(
        "--ignore-rng-seeds",
        action="store_true",
        help=(
            "v1.4.1 (item E): opt back into the v1.2.1 skip behaviour "
            "for ``rng_seeds``. Use only for stochastic-seed-rerun "
            "workflows where the operator legitimately re-derives a "
            "model with different seeds and the rest of the envelope "
            "is expected to match."
        ),
    )
    s.set_defaults(func=verify_run_cmd)

    s = sub.add_parser("bench")
    s.add_argument("--out", default="data/bench.csv")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=bench_cmd)

    # ----- v1.2 frontier CLI commands -----

    s = sub.add_parser("nowcast", help="Mixed-frequency DFM-MQ nowcast.")
    s.add_argument("--db", default="data/mre.duckdb")
    s.set_defaults(func=nowcast_cmd)

    s = sub.add_parser("e-value-test", help="Sequential e-value safe-test of a challenger model.")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--challenger", required=True)
    s.add_argument("--champion", default=None)
    s.add_argument("--validation-dir", default="data/validation")
    s.add_argument("--level", type=float, default=0.05)
    s.set_defaults(func=e_value_test_cmd)

    s = sub.add_parser("conformal-conditional", help="Group-conditional conformal coverage report.")
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--validation-dir", default="data/validation")
    s.add_argument("--alpha", type=float, default=0.10)
    s.set_defaults(func=conformal_conditional_cmd)

    # ----- v1.3 commands -----

    s = sub.add_parser(
        "warehouse-migrate",
        help="Copy every warehouse table from one backend (sqlite|duckdb) to another.",
    )
    s.add_argument("--src", required=True, help="Source warehouse path (e.g. data/mre.db).")
    s.add_argument("--dst", required=True, help="Destination warehouse path (e.g. data/mre.duckdb).")
    s.add_argument(
        "--from",
        dest="from_backend",
        default="auto",
        choices=["sqlite", "duckdb", "auto"],
        help="Source backend; ``auto`` infers from the path suffix.",
    )
    s.add_argument(
        "--to",
        dest="to_backend",
        default="auto",
        choices=["sqlite", "duckdb", "auto"],
        help="Destination backend; ``auto`` infers from the path suffix.",
    )
    s.set_defaults(func=warehouse_migrate_cmd)

    s = sub.add_parser(
        "verify-data",
        help="Detect warehouse drift between a stored model_run envelope and the current state.",
    )
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument(
        "--run-id",
        default=None,
        help="Run id to verify (default: latest).",
    )
    s.add_argument(
        "--legacy-hash",
        action="store_true",
        help=(
            "Use the v1.2.1 _hash_frame implementation. Required when the "
            "stored run was written before the v1.3 hash migration."
        ),
    )
    s.set_defaults(func=verify_data_cmd)

    # ----- v1.4 commands -----

    s = sub.add_parser(
        "bayesian-msvar-fit",
        help="Fit a NumPyro Bayesian MS-VAR (NUTS or SVI); persist diagnostics. v1.4 item A.",
    )
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--method", choices=["nuts", "svi"], default="nuts")
    s.add_argument("--chains", type=int, default=2)
    s.add_argument("--warmup", type=int, default=500)
    s.add_argument("--samples", type=int, default=500)
    s.set_defaults(func=bayesian_msvar_fit_cmd)

    s = sub.add_parser(
        "deep-kernel-train",
        help="Train an MLPDeepKernel for GP-BOCPD; emit a training-loss curve. v1.4 item B.",
    )
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--input-dim", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/deep_kernel_losses.csv")
    s.set_defaults(func=deep_kernel_train_cmd)

    s = sub.add_parser(
        "refresh-release-calendars",
        help="Refresh BLS/BEA/Census/Fed release calendars to YAML. v1.4 item D.",
    )
    s.add_argument("--db", default="data/mre.duckdb")
    s.add_argument(
        "--agency",
        default=None,
        help="Comma-separated agency list (default: all of bls,bea,census,fed).",
    )
    s.add_argument(
        "--out",
        default=None,
        help="Override output directory (default: config/release_calendars/).",
    )
    s.set_defaults(func=refresh_release_calendars_cmd)

    return p



__all__ = ["parser"]
