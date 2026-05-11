# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income CLI entry points — PR-1 stubs.

Per ``MRE_FIXED_INCOME_INSTRUCTIONS.md §8``: the FI feature ships 7
``fi-*`` subcommands on the ``mre`` CLI. PR-1 lands the full argparse
surface (so flags don't shift across PRs and downstream automation can
hard-code the call sites today) plus stub handlers that emit the
canonical ``not_yet_implemented`` JSON payload and return exit code 0.

Subsequent PRs swap each stub for the real implementation:

- PR-3 ``fi-build-features``, ``fi-score-credit-regime``
- PR-4 ``fi-score-liquidity``
- PR-5 ``fi-score-execution-confidence``
- PR-6 ``fi-tca-segment``
- PR-7 ``fi-evidence-pack``, ``fi-report``

The dispatcher lives in :func:`run`; ``cli_dispatch.main`` routes any
argv whose first token starts with ``fi-`` here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

CLI_COMMANDS: tuple[str, ...] = (
    "fi-build-features",
    "fi-score-credit-regime",
    "fi-score-liquidity",
    "fi-score-execution-confidence",
    "fi-tca-segment",
    "fi-evidence-pack",
    "fi-report",
    "fi-evidence-resign",
)

# Commands fully implemented in this PR. Stub-emitting commands fall
# back to ``_emit_stub`` below.
_LIVE_COMMANDS: frozenset[str] = frozenset(
    {
        "fi-score-credit-regime",
        "fi-score-liquidity",
        "fi-score-execution-confidence",
        "fi-tca-segment",
        "fi-evidence-pack",
        "fi-report",
        "fi-evidence-resign",
    }
)

_VALID_EVIDENCE_COMPONENTS: frozenset[str] = frozenset(
    {
        "credit_regime",
        "liquidity_stress",
        "execution_confidence",
        "tca_segmentation",
        "other",
    }
)


def _emit_stub(command: str) -> int:
    """Print the canonical ``not_yet_implemented`` JSON envelope.

    Newline-terminated so shell pipelines can consume it without
    needing ``--no-pretty``. Exit code 0 mirrors the ``snapshot-build``
    stub pattern in v1.4 ``cli_dispatch._cmd_snapshot_build``.
    """
    payload = {"status": "not_yet_implemented", "command": command}
    print(json.dumps(payload, sort_keys=True))
    return 0


def _build_fi_build_features(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-build-features",
        help="Build PIT-safe FI features from TRACE/RFQ/quotes/curves (PR-3+).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument("--asof", help="ISO-8601 as-of timestamp; defaults to now (UTC).")
    parser.add_argument("--out", help="Optional output path for materialised features.")


def _build_fi_score_credit_regime(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-score-credit-regime",
        help="Score the latest credit-spread-regime index (PR-3).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument("--asof", help="ISO-8601 as-of timestamp; defaults to now (UTC).")
    parser.add_argument(
        "--profile",
        help="Operating profile (production | default).",
        default="production",
    )
    parser.add_argument(
        "--release-gate",
        help="Inbound governance flag (default: true).",
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument("--model-run-id", help="Explicit model_run_id (auto-generated if omitted).")
    parser.add_argument(
        "--lookback-days",
        help="Rolling window length (default 504 = ~2y).",
        type=int,
        default=504,
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the scoring envelope JSON.",
        dest="output_json",
    )
    # PR-1 alias kept for back-compat with the original stub flag name.
    parser.add_argument(
        "--out-json",
        help=argparse.SUPPRESS,
        dest="out_json_legacy",
    )


def _build_fi_score_liquidity(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-score-liquidity",
        help="Score liquidity-stress indices per scope (PR-4).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument("--asof", help="ISO-8601 as-of timestamp; defaults to now (UTC).")
    parser.add_argument(
        "--scope-type",
        help="Scope: market / sector / rating / cusip.",
        default="market",
        choices=["market", "sector", "rating", "cusip"],
    )
    parser.add_argument(
        "--scope-id",
        help="Scope id. Required for sector/rating/cusip; defaults to 'ALL' for market.",
        default="ALL",
    )
    parser.add_argument(
        "--profile",
        help="Operating profile (production | default).",
        default="production",
    )
    parser.add_argument(
        "--release-gate",
        help="Inbound governance flag (default: true).",
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--model-run-id",
        help="Explicit model_run_id (auto-generated if omitted).",
    )
    parser.add_argument(
        "--lookback-days",
        help="Rolling window length (default 30).",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--use-hierarchical",
        help="Enable the hierarchical Bayesian scorer (opt-in, default false).",
        action="store_true",
    )
    parser.add_argument(
        "--prev-label-from-warehouse",
        help="Read the previous label from the warehouse for hysteresis (default true).",
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the scoring envelope JSON.",
        dest="output_json",
    )
    # PR-1 alias kept for back-compat with the original stub flag name.
    parser.add_argument(
        "--out-json",
        help=argparse.SUPPRESS,
        dest="out_json_legacy",
    )


def _build_fi_score_execution_confidence(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-score-execution-confidence",
        help="Score execution-confidence for a candidate order (PR-5).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument(
        "--input",
        help="Path to the ExecutionConfidenceRequest JSON payload.",
        required=True,
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the scoring envelope JSON.",
        dest="output_json",
    )
    # PR-1 alias kept for back-compat.
    parser.add_argument(
        "--out-json",
        help=argparse.SUPPRESS,
        dest="out_json_legacy",
    )
    parser.add_argument(
        "--profile",
        help="Operating profile (production | default).",
        default="production",
    )
    parser.add_argument(
        "--release-gate",
        help="Inbound governance flag (default: true).",
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--request-id",
        help="Composite-PK request id (PR-15). Auto-generated UUID4 if omitted.",
        dest="request_id",
    )
    parser.add_argument(
        "--model-run-id",
        help="Explicit model_run_id (auto-generated if omitted).",
        dest="model_run_id",
    )


def _build_fi_tca_segment(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-tca-segment",
        help="Tag and aggregate TCA segments by regime/liquidity/confidence (PR-6).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument(
        "--date",
        help="Target trading day (YYYY-MM-DD). Defaults to the previous trading day.",
    )
    parser.add_argument(
        "--dimensions",
        help="Comma-separated segmentation dimensions (informational only — "
        "materialize writes every canonical dim-combo).",
        default="regime_label,liquidity_label",
    )
    parser.add_argument(
        "--soft-weighting",
        help="Enable soft regime weighting (default false).",
        action="store_true",
        dest="soft_weighting",
    )
    parser.add_argument(
        "--use-hysteresis",
        help="Apply asymmetric hysteresis to the regime label during tagging (default true).",
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--model-run-id",
        help="Explicit model_run_id (auto-generated when omitted).",
        dest="model_run_id",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the materialisation summary JSON.",
        dest="output_json",
    )
    # PR-1 alias kept for back-compat with the original stub flag name.
    parser.add_argument(
        "--out-json",
        help=argparse.SUPPRESS,
        dest="out_json_legacy",
    )
    # PR-1 placeholder flags retained as no-ops so downstream automation
    # that hard-coded the stub arg-list continues to parse.
    parser.add_argument("--start", help=argparse.SUPPRESS)
    parser.add_argument("--end", help=argparse.SUPPRESS)


def _build_fi_evidence_pack(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-evidence-pack",
        help="Generate, sign, and persist a Fixed-Income evidence pack (PR-7).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument(
        "--model-run-id",
        help="Model run id to render the pack for.",
        required=True,
    )
    parser.add_argument(
        "--component",
        help="Component: credit_regime|liquidity_stress|execution_confidence|tca_segmentation|other.",
        required=True,
        choices=sorted(_VALID_EVIDENCE_COMPONENTS),
    )
    parser.add_argument(
        "--request-id",
        help="Composite-PK request id (PR-15). Auto-generated UUID4 if omitted.",
        dest="request_id",
    )
    parser.add_argument(
        "--sign",
        help=(
            "Sign with HMAC. 'auto' (default) signs when keys are configured "
            "and follows production-mode requirements; 'true' forces a sign "
            "(raises if no keys); 'false' refuses to sign."
        ),
        default="auto",
        choices=["auto", "true", "false"],
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the evidence pack JSON.",
        dest="output_json",
    )
    # PR-1 alias kept for back-compat.
    parser.add_argument("--out", help=argparse.SUPPRESS, dest="out_json_legacy")


def _build_fi_report(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-report",
        help="Generate the Fixed-Income RCIE Markdown/HTML report (PR-7).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument(
        "--out",
        help="Output report path.",
        default="data/reports/fixed_income_rcie.md",
    )
    parser.add_argument(
        "--format",
        help="Report format: 'markdown' or 'html'.",
        default="markdown",
        choices=["markdown", "html"],
    )
    parser.add_argument(
        "--asof",
        help="Optional ISO-8601 as-of timestamp (defaults to now UTC).",
    )


def _build_fi_evidence_resign(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-evidence-resign",
        help="Bulk re-sign FI evidence packs from one HMAC version to another (PR-7 §F).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument(
        "--from-key",
        help="Existing key version that the packs are currently signed under.",
        required=True,
        dest="from_key",
    )
    parser.add_argument(
        "--to-key",
        help="Target key version to re-sign with.",
        required=True,
        dest="to_key",
    )
    parser.add_argument(
        "--dry-run",
        help="Print the count without writing.",
        action="store_true",
        dest="dry_run",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mre", description="Fixed-Income RCIE / X-Pro Auto-X CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    _build_fi_build_features(sub)
    _build_fi_score_credit_regime(sub)
    _build_fi_score_liquidity(sub)
    _build_fi_score_execution_confidence(sub)
    _build_fi_tca_segment(sub)
    _build_fi_evidence_pack(sub)
    _build_fi_report(sub)
    _build_fi_evidence_resign(sub)
    return parser


def _cmd_fi_score_credit_regime(ns: argparse.Namespace) -> int:
    """Run :func:`score_credit_regime` end-to-end and persist the row.

    Workflow:

    1. Open the DuckDB warehouse at ``--db``.
    2. Build PIT-safe credit features for the asof timestamp.
    3. Score via :func:`score_credit_regime` with the explicit profile,
       release_gate, and (optional) model_run_id.
    4. Persist via :func:`write_credit_regime_score` so the API and
       evidence-pack consumers see the new row.
    5. Print the canonical scoring envelope to stdout.
    6. (Optional) write the same envelope to ``--output-json``.

    Returns 0 on success, 2 on PIT or input validation failure. The
    exit code maps to the operator runbook: 0 = clean run (including
    release_gate=false), 2 = upstream contract broken (PIT, missing
    feature with audit policy, etc).
    """
    # Lazy imports keep ``fi-* stub commands`` cheap (they don't pay
    # the pandas / DuckDB import cost just to print ``not_yet_implemented``).
    from market_regime_engine.fixed_income.credit_spread_regime import (
        score_credit_regime,
        write_credit_regime_score,
    )
    from market_regime_engine.fixed_income.feature_builders import (
        build_credit_features,
    )
    from market_regime_engine.fixed_income.pit_guard import PitViolationError
    from market_regime_engine.fixed_income.timestamps import to_utc
    from market_regime_engine.frontier.data_cleaning import PitAuditFailure
    from market_regime_engine.storage import Warehouse

    asof_arg = getattr(ns, "asof", None)
    if asof_arg:
        try:
            asof_ts = to_utc(asof_arg)
        except ValueError as exc:
            print(json.dumps({"status": "error", "detail": str(exc)}, sort_keys=True))
            return 2
    else:
        asof_ts = pd.Timestamp.now(tz="UTC")
    if asof_ts is None:
        print(json.dumps({"status": "error", "detail": "asof must not be None"}, sort_keys=True))
        return 2

    release_gate = (getattr(ns, "release_gate", "true") or "true").lower() != "false"
    profile = getattr(ns, "profile", "production")

    wh = Warehouse(ns.db)
    try:
        try:
            features = build_credit_features(wh, asof_ts, lookback_days=int(getattr(ns, "lookback_days", 504)))
        except PitViolationError as exc:
            print(json.dumps({"status": "pit_violation", "detail": str(exc)}, sort_keys=True))
            return 2
        try:
            output = score_credit_regime(
                features,
                asof=asof_ts,
                model_run_id=getattr(ns, "model_run_id", None),
                release_gate=release_gate,
                profile=profile,
            )
        except PitViolationError as exc:
            print(json.dumps({"status": "pit_violation", "detail": str(exc)}, sort_keys=True))
            return 2
        except PitAuditFailure as exc:
            print(json.dumps({"status": "pit_audit_failed", "detail": str(exc)}, sort_keys=True))
            return 2
        write_credit_regime_score(wh, output)
    finally:
        wh.close()

    envelope = _envelope_from_output(output)
    print(json.dumps(envelope, sort_keys=True))

    output_path = getattr(ns, "output_json", None) or getattr(ns, "out_json_legacy", None)
    if output_path:
        _write_optional_json(output_path, envelope)
    return 0


def _envelope_from_output(output: Any) -> dict[str, Any]:
    """Stdout summary envelope per the scope spec.

    Includes the AGENT.md §6.1 fields plus the v1.5 governance triple.
    Drivers are emitted as a list (JSON-friendly).
    """
    return {
        "timestamp": output.timestamp,
        "regime_score": float(output.regime_score),
        "regime_label": output.regime_label,
        "confidence": float(output.confidence),
        "drivers": list(output.drivers),
        "component_scores": dict(output.component_scores),
        "model_run_id": output.model_run_id,
        "release_gate": bool(output.release_gate),
        "artifact_hash": output.artifact_hash,
    }


def _envelope_from_liquidity_output(output: Any) -> dict[str, Any]:
    """Stdout summary envelope for ``score_liquidity_stress`` results."""
    return {
        "timestamp": output.timestamp,
        "scope_type": output.scope_type,
        "scope_id": output.scope_id,
        "liquidity_index": float(output.liquidity_index),
        "liquidity_label": output.liquidity_label,
        "confidence": float(output.confidence),
        "drivers": list(output.drivers),
        "score_components": dict(output.metadata.get("score_components", {})),
        "model_run_id": output.model_run_id,
        "release_gate": bool(output.release_gate),
        "artifact_hash": output.artifact_hash,
    }


def _cmd_fi_score_liquidity(ns: argparse.Namespace) -> int:
    """Run :func:`score_liquidity_stress` end-to-end and persist the row.

    Workflow:

    1. Open the DuckDB warehouse at ``--db``.
    2. Build PIT-safe liquidity features for the asof timestamp.
    3. Optionally fetch the previous label for the scope from the
       warehouse so :func:`score_liquidity_stress` applies hysteresis.
    4. Score via :func:`score_liquidity_stress` (the
       ``--use-hierarchical`` flag is wired through; when set, the
       hierarchical model is constructed for the documented hybrid
       flow, but the deterministic composite remains the primary
       scorer per AGENT.md "explainable baselines first").
    5. Persist via :func:`write_liquidity_stress_score`.
    6. Print + optionally write the canonical envelope.

    Returns 0 on success, 2 on PIT or audit failure.
    """
    from market_regime_engine.fixed_income.feature_builders import (
        build_liquidity_features,
    )
    from market_regime_engine.fixed_income.liquidity_stress import (
        latest_liquidity_stress_score,
        score_liquidity_stress,
        write_liquidity_stress_score,
    )
    from market_regime_engine.fixed_income.pit_guard import PitViolationError
    from market_regime_engine.fixed_income.schemas import LiquidityLabel
    from market_regime_engine.fixed_income.timestamps import to_utc
    from market_regime_engine.frontier.data_cleaning import PitAuditFailure
    from market_regime_engine.storage import Warehouse

    asof_arg = getattr(ns, "asof", None)
    if asof_arg:
        try:
            asof_ts = to_utc(asof_arg)
        except ValueError as exc:
            print(json.dumps({"status": "error", "detail": str(exc)}, sort_keys=True))
            return 2
    else:
        asof_ts = pd.Timestamp.now(tz="UTC")
    if asof_ts is None:
        print(json.dumps({"status": "error", "detail": "asof must not be None"}, sort_keys=True))
        return 2

    release_gate = (getattr(ns, "release_gate", "true") or "true").lower() != "false"
    profile = getattr(ns, "profile", "production")
    scope_type_raw = getattr(ns, "scope_type", "market") or "market"
    if scope_type_raw not in {"market", "sector", "rating", "cusip"}:
        print(
            json.dumps(
                {"status": "error", "detail": f"unsupported scope_type: {scope_type_raw!r}"},
                sort_keys=True,
            )
        )
        return 2
    # Narrow the type so mypy is happy passing into the Literal-typed scorer.
    from typing import Literal, cast

    scope_type: Literal["market", "sector", "rating", "cusip"] = cast(
        Literal["market", "sector", "rating", "cusip"], scope_type_raw
    )
    scope_id = getattr(ns, "scope_id", "ALL") or "ALL"
    prev_from_wh = (getattr(ns, "prev_label_from_warehouse", "true") or "true").lower() != "false"
    use_hier = bool(getattr(ns, "use_hierarchical", False))

    wh = Warehouse(ns.db)
    prev_label: LiquidityLabel | None = None
    try:
        if prev_from_wh:
            prev = latest_liquidity_stress_score(wh, scope_type=scope_type, scope_id=scope_id)
            if prev is not None and prev.liquidity_label:
                try:
                    prev_label = next(lbl for lbl in LiquidityLabel if lbl.label == prev.liquidity_label)
                except StopIteration:
                    prev_label = None
        try:
            features = build_liquidity_features(
                wh,
                asof_ts,
                scope_type=scope_type,
                scope_id=scope_id,
                lookback_days=int(getattr(ns, "lookback_days", 30)),
            )
        except PitViolationError as exc:
            print(json.dumps({"status": "pit_violation", "detail": str(exc)}, sort_keys=True))
            return 2
        try:
            output = score_liquidity_stress(
                features,
                scope_type=scope_type,
                scope_id=scope_id,
                asof=asof_ts,
                model_run_id=getattr(ns, "model_run_id", None),
                release_gate=release_gate,
                profile=profile,
                prev_label=prev_label,
            )
        except PitViolationError as exc:
            print(json.dumps({"status": "pit_violation", "detail": str(exc)}, sort_keys=True))
            return 2
        except PitAuditFailure as exc:
            print(json.dumps({"status": "pit_audit_failed", "detail": str(exc)}, sort_keys=True))
            return 2
        # ``--use-hierarchical`` is wired through metadata so downstream
        # telemetry / evidence-pack code can see the flag; v1.5 keeps
        # the deterministic composite as the primary scorer per AGENT.md
        # non-negotiable "explainable baselines first". Activating the
        # hierarchical scorer in production lands behind validation in
        # PR-7.
        if use_hier:
            extra = dict(output.metadata)
            extra["use_hierarchical_requested"] = True
            output = type(output)(  # rebuild frozen dataclass with updated metadata
                timestamp=output.timestamp,
                scope_type=output.scope_type,
                scope_id=output.scope_id,
                liquidity_index=output.liquidity_index,
                liquidity_label=output.liquidity_label,
                confidence=output.confidence,
                drivers=output.drivers,
                model_run_id=output.model_run_id,
                release_gate=output.release_gate,
                artifact_hash=output.artifact_hash,
                metadata=extra,
            )
        write_liquidity_stress_score(wh, output)
    finally:
        wh.close()

    envelope = _envelope_from_liquidity_output(output)
    print(json.dumps(envelope, sort_keys=True))
    output_path = getattr(ns, "output_json", None) or getattr(ns, "out_json_legacy", None)
    if output_path:
        _write_optional_json(output_path, envelope)
    return 0


def _write_optional_json(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def _envelope_from_execution_confidence(output: Any) -> dict[str, Any]:
    """Stdout envelope for ``score_execution_confidence`` results."""
    return {
        "timestamp": output.timestamp,
        "cusip": output.cusip,
        "side": output.side,
        "notional": float(output.notional),
        "protocol": output.protocol,
        "confidence_score": float(output.confidence_score),
        "expected_slippage_bps": (
            float(output.expected_slippage_bps) if output.expected_slippage_bps is not None else None
        ),
        "confidence_interval": [
            output.confidence_interval_low,
            output.confidence_interval_high,
        ],
        "recommended_action": output.recommended_action,
        "human_review_required": bool(output.human_review_required),
        "model_run_id": output.model_run_id,
        "release_gate": bool(output.release_gate),
        "artifact_hash": output.artifact_hash,
        "metadata": dict(output.metadata),
    }


def _cmd_fi_score_execution_confidence(ns: argparse.Namespace) -> int:
    """Run :func:`score_execution_confidence` end-to-end for a JSON order.

    Workflow:

    1. Read ``--input`` JSON, route through the Pydantic v2 boundary
       model so naive timestamps / oversized notionals / non-alphanumeric
       cusips surface with exit code 2.
    2. Open the DuckDB warehouse at ``--db``.
    3. Score via :func:`score_execution_confidence`.
    4. Persist via :func:`write_execution_confidence_prediction` keyed by
       ``--request-id`` (auto-generated UUID4 if omitted).
    5. Print + optionally write the envelope to ``--output-json``.

    Exit codes: 0 on clean run (including release_gate=false stale-signal
    fail-closed), 2 on input / PIT / audit failure.
    """
    import uuid as _uuid

    from market_regime_engine.fixed_income.api import ExecutionConfidenceRequestModel
    from market_regime_engine.fixed_income.execution_confidence import (
        score_execution_confidence,
        write_execution_confidence_prediction,
    )
    from market_regime_engine.fixed_income.pit_guard import PitViolationError
    from market_regime_engine.frontier.data_cleaning import PitAuditFailure
    from market_regime_engine.storage import Warehouse

    input_path = getattr(ns, "input", None)
    if not input_path:
        print(json.dumps({"status": "error", "detail": "--input is required"}, sort_keys=True))
        return 2
    try:
        raw = Path(input_path).read_text(encoding="utf-8")
    except OSError as exc:
        print(json.dumps({"status": "error", "detail": str(exc)}, sort_keys=True))
        return 2
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"status": "error", "detail": str(exc)}, sort_keys=True))
        return 2

    request_id = getattr(ns, "request_id", None) or _uuid.uuid4().hex
    # ``request_id`` is required on the Pydantic model; inject ours if the
    # caller did not include it in the JSON payload.
    payload.setdefault("request_id", request_id)
    try:
        body = ExecutionConfidenceRequestModel(**payload)
    except Exception as exc:
        print(json.dumps({"status": "validation_error", "detail": str(exc)}, sort_keys=True))
        return 2

    release_gate = (getattr(ns, "release_gate", "true") or "true").lower() != "false"
    profile = getattr(ns, "profile", "production")

    wh = Warehouse(ns.db)
    try:
        try:
            output = score_execution_confidence(
                body.to_dataclass(),
                warehouse=wh,
                release_gate=release_gate,
                profile=profile,
                model_run_id=getattr(ns, "model_run_id", None),
            )
        except PitViolationError as exc:
            print(json.dumps({"status": "pit_violation", "detail": str(exc)}, sort_keys=True))
            return 2
        except PitAuditFailure as exc:
            print(json.dumps({"status": "pit_audit_failed", "detail": str(exc)}, sort_keys=True))
            return 2
        write_execution_confidence_prediction(wh, output, request_id=body.request_id)
    finally:
        wh.close()

    envelope = _envelope_from_execution_confidence(output)
    envelope["request_id"] = body.request_id
    print(json.dumps(envelope, sort_keys=True, default=str))
    output_path = getattr(ns, "output_json", None) or getattr(ns, "out_json_legacy", None)
    if output_path:
        _write_optional_json(output_path, envelope)
    return 0


def _cmd_fi_tca_segment(ns: argparse.Namespace) -> int:
    """Run :func:`materialize_tca_segments_for_day` end-to-end.

    Workflow:

    1. Resolve ``--date`` (defaults to the previous SIFMA bond trading day).
    2. Open the DuckDB warehouse at ``--db``.
    3. Call :func:`materialize_tca_segments_for_day` with the parsed
       flags; persist segment rows for every canonical dim-combo.
    4. Print a summary envelope to stdout.
    5. Optionally write the same envelope to ``--output-json``.

    Returns 0 on success, 2 on input validation / PIT failure.
    """
    from market_regime_engine.fixed_income.calendars import (
        TradingCalendar,
        previous_trading_day,
    )
    from market_regime_engine.fixed_income.pit_guard import PitViolationError
    from market_regime_engine.fixed_income.tca_segmentation import (
        DIMENSION_COLUMNS,
        materialize_tca_segments_for_day,
    )
    from market_regime_engine.storage import Warehouse

    date_arg = getattr(ns, "date", None)
    if date_arg:
        try:
            date_ts = pd.Timestamp(date_arg)
        except Exception as exc:
            print(json.dumps({"status": "error", "detail": str(exc)}, sort_keys=True))
            return 2
    else:
        date_ts = previous_trading_day(pd.Timestamp.now(tz="UTC"), TradingCalendar.SIFMA_BOND)
    if date_ts.tzinfo is None:
        date_ts = date_ts.tz_localize("UTC")

    dimensions_raw = (getattr(ns, "dimensions", "") or "").strip()
    if dimensions_raw:
        dim_list = [d.strip() for d in dimensions_raw.split(",") if d.strip()]
        invalid = [d for d in dim_list if d not in DIMENSION_COLUMNS]
        if invalid:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "detail": f"invalid --dimensions: {invalid!r}",
                        "valid_dimensions": sorted(DIMENSION_COLUMNS),
                    },
                    sort_keys=True,
                )
            )
            return 2
    else:
        dim_list = ["regime_label", "liquidity_label"]

    soft_weighting = bool(getattr(ns, "soft_weighting", False))
    use_hysteresis = (getattr(ns, "use_hysteresis", "true") or "true").lower() != "false"
    model_run_id = getattr(ns, "model_run_id", None)

    wh = Warehouse(ns.db)
    try:
        try:
            rows_written = materialize_tca_segments_for_day(
                wh,
                date=date_ts,
                soft_weighting=soft_weighting,
                use_hysteresis=use_hysteresis,
                model_run_id=model_run_id,
            )
        except PitViolationError as exc:
            print(json.dumps({"status": "pit_violation", "detail": str(exc)}, sort_keys=True))
            return 2
    finally:
        wh.close()

    envelope = {
        "status": "ok",
        "date": str(date_ts.date()),
        "rows_written": int(rows_written),
        "dimensions_requested": list(dim_list),
        "soft_weighting": bool(soft_weighting),
        "use_hysteresis": bool(use_hysteresis),
    }
    print(json.dumps(envelope, sort_keys=True))
    output_path = getattr(ns, "output_json", None) or getattr(ns, "out_json_legacy", None)
    if output_path:
        _write_optional_json(output_path, envelope)
    return 0


def _resolve_component_artifacts(warehouse: Any, component: str, model_run_id: str) -> dict[str, Any]:
    """Resolve component-specific output rows for evidence pack assembly.

    Returns a dict with keys:

    - ``timestamp``: ISO-8601 UTC string of the underlying signal.
    - ``output_hash``: ``"sha256:..."`` artifact hash of the row.
    - ``release_gate``: bool.
    - ``model_version``: best-effort version string.
    - ``input_features_hash``: ``"sha256:..."`` over the row's drivers /
      payload (best-effort; falls back to the component name when no
      payload column exists).

    Raises ``LookupError`` when ``model_run_id`` is unknown for the
    requested component.
    """
    from market_regime_engine.fixed_income.hashing import canonical_sha256

    component = component.lower()
    if component == "credit_regime":
        df = warehouse.read_credit_regime_scores()
        sub = df[df["model_run_id"] == model_run_id] if not df.empty else df
        if sub.empty:
            raise LookupError(f"no credit_regime_scores rows for model_run_id={model_run_id!r}")
        row = sub.iloc[-1]
        return {
            "timestamp": str(row["timestamp"]),
            "output_hash": str(row["artifact_hash"]),
            "release_gate": bool(int(row["release_gate"])),
            "model_version": "credit_regime_baseline_v0",
            "input_features_hash": canonical_sha256({"drivers": str(row.get("drivers_json", ""))}),
        }
    if component == "liquidity_stress":
        df = warehouse.read_liquidity_stress_scores()
        sub = df[df["model_run_id"] == model_run_id] if not df.empty else df
        if sub.empty:
            raise LookupError(f"no liquidity_stress_scores rows for model_run_id={model_run_id!r}")
        row = sub.iloc[-1]
        return {
            "timestamp": str(row["timestamp"]),
            "output_hash": str(row["artifact_hash"]),
            "release_gate": bool(int(row["release_gate"])),
            "model_version": "liquidity_stress_baseline_v0",
            "input_features_hash": canonical_sha256({"drivers": str(row.get("drivers_json", ""))}),
        }
    if component == "execution_confidence":
        df = warehouse.read_execution_confidence_predictions()
        sub = df[df["model_run_id"] == model_run_id] if not df.empty else df
        if sub.empty:
            raise LookupError(f"no execution_confidence_predictions rows for model_run_id={model_run_id!r}")
        row = sub.iloc[-1]
        return {
            "timestamp": str(row["timestamp"]),
            "output_hash": str(row["artifact_hash"]),
            "release_gate": bool(int(row["release_gate"])),
            "model_version": "execution_confidence_baseline_v0",
            "input_features_hash": canonical_sha256(
                {
                    "cusip": str(row.get("cusip", "")),
                    "side": str(row.get("side", "")),
                    "notional": float(row.get("notional", 0.0)),
                }
            ),
        }
    if component == "tca_segmentation":
        df = warehouse.read_tca_regime_segments()
        sub = df[df["model_run_id"] == model_run_id] if not df.empty else df
        if sub.empty:
            raise LookupError(f"no tca_regime_segments rows for model_run_id={model_run_id!r}")
        row = sub.iloc[-1]
        # TCA rows do not carry an artifact_hash column; derive from the
        # canonical row payload so the pack is still tamper-evident.
        row_hash = canonical_sha256({col: str(row.get(col, "")) for col in sub.columns if col != "metadata_json"})
        return {
            "timestamp": str(row["timestamp"]),
            "output_hash": row_hash,
            "release_gate": True,
            "model_version": "tca_segmentation_baseline_v0",
            "input_features_hash": canonical_sha256({"metric": str(row.get("metric_name", ""))}),
        }
    raise ValueError(f"unsupported component: {component!r}")


def _cmd_fi_evidence_pack(ns: argparse.Namespace) -> int:
    """Build, sign, and persist an FI evidence pack for ``model_run_id``.

    Workflow per AGENT.md PR-7 §"Evidence pack" + INSTRUCTIONS.md §6.5:

    1. Open the DuckDB warehouse.
    2. Resolve the component output row for ``--model-run-id``.
    3. Capture per-source vintages via ``capture_data_vintages``.
    4. Build the ``FixedIncomeEvidencePack`` dataclass with the
       canonical sha256 over the output row.
    5. Sign with HMAC if keys are configured (or required by env).
    6. Persist via ``write_evidence_pack`` (composite-PK
       ``(model_run_id, request_id)``).
    7. Print the pack JSON to stdout; optionally write to
       ``--output-json`` / ``--out``.

    Exit codes: 0 on success, 2 on lookup / signing failure, 3 on
    governance errors (production mode without keys).
    """
    import uuid as _uuid

    from market_regime_engine.fixed_income.evidence_pack import (
        build_evidence_pack,
        capture_data_vintages,
        evidence_pack_to_dict,
        require_production_hmac,
        write_evidence_pack,
    )
    from market_regime_engine.fixed_income.hashing import canonical_sha256
    from market_regime_engine.model_runs import _git_revision, _lockfile_hash
    from market_regime_engine.storage import Warehouse

    sign_arg = (getattr(ns, "sign", "auto") or "auto").lower()
    sign_value: bool | None
    if sign_arg == "auto":
        sign_value = None
    elif sign_arg == "true":
        sign_value = True
    elif sign_arg == "false":
        sign_value = False
    else:
        print(
            json.dumps(
                {"status": "error", "detail": f"invalid --sign value: {sign_arg!r}"},
                sort_keys=True,
            )
        )
        return 2

    component = (getattr(ns, "component", "") or "").lower()
    if component not in _VALID_EVIDENCE_COMPONENTS:
        print(
            json.dumps(
                {
                    "status": "error",
                    "detail": (
                        f"invalid --component {component!r}; expected one of {sorted(_VALID_EVIDENCE_COMPONENTS)!r}"
                    ),
                },
                sort_keys=True,
            )
        )
        return 2

    model_run_id = getattr(ns, "model_run_id", None)
    if not model_run_id:
        print(json.dumps({"status": "error", "detail": "--model-run-id required"}, sort_keys=True))
        return 2

    request_id = getattr(ns, "request_id", None) or _uuid.uuid4().hex

    wh = Warehouse(ns.db)
    try:
        try:
            artifacts = _resolve_component_artifacts(wh, component, model_run_id)
        except LookupError as exc:
            print(json.dumps({"status": "not_found", "detail": str(exc)}, sort_keys=True))
            return 2
        except ValueError as exc:
            print(json.dumps({"status": "error", "detail": str(exc)}, sort_keys=True))
            return 2

        try:
            asof_ts = pd.Timestamp(artifacts["timestamp"])
        except Exception:
            asof_ts = None

        vintages = capture_data_vintages(wh, asof=asof_ts)
        # The model_hash is canonical over the output_hash + component
        # so a signed pack can be verified standalone.
        model_hash = canonical_sha256({"component": component, "output_hash": artifacts["output_hash"]})
        pack = build_evidence_pack(
            model_run_id=model_run_id,
            component_name=component,
            model_version=str(artifacts["model_version"]),
            code_sha=_git_revision(short=True),
            model_hash=model_hash,
            input_features_hash=str(artifacts["input_features_hash"]),
            output_hash=str(artifacts["output_hash"]),
            release_gate=bool(artifacts["release_gate"]),
            data_vintages=vintages,
            timestamp=str(artifacts["timestamp"]),
            lockfile_hash=_lockfile_hash() or None,
        )
        try:
            persisted = write_evidence_pack(wh, pack, request_id=request_id, sign=sign_value)
        except RuntimeError as exc:
            payload = {"status": "error", "detail": str(exc)}
            if require_production_hmac():
                payload["governance"] = "production_requires_hmac"
                print(json.dumps(payload, sort_keys=True))
                return 3
            print(json.dumps(payload, sort_keys=True))
            return 2
    finally:
        wh.close()

    pack_dict = evidence_pack_to_dict(persisted)
    pack_dict["request_id"] = request_id
    print(json.dumps(pack_dict, sort_keys=True, default=str))

    output_path = getattr(ns, "output_json", None) or getattr(ns, "out_json_legacy", None)
    if output_path:
        _write_optional_json(output_path, pack_dict)
    return 0


def _cmd_fi_report(ns: argparse.Namespace) -> int:
    """Generate the FI RCIE report and write it to ``--out``."""
    from market_regime_engine.fixed_income.report import generate_fi_report
    from market_regime_engine.fixed_income.timestamps import to_utc
    from market_regime_engine.storage import Warehouse

    fmt_raw = (getattr(ns, "format", "markdown") or "markdown").lower()
    if fmt_raw not in {"markdown", "html"}:
        print(
            json.dumps(
                {"status": "error", "detail": f"invalid --format: {fmt_raw!r}"},
                sort_keys=True,
            )
        )
        return 2

    asof_arg = getattr(ns, "asof", None)
    asof_ts = None
    if asof_arg:
        try:
            asof_ts = to_utc(asof_arg)
        except ValueError as exc:
            print(json.dumps({"status": "error", "detail": str(exc)}, sort_keys=True))
            return 2

    out_path = Path(getattr(ns, "out", None) or "data/reports/fixed_income_rcie.md")
    wh = Warehouse(ns.db)
    try:
        # mypy: format is validated above so the cast is safe.
        from typing import Literal, cast

        fmt: Literal["markdown", "html"] = cast(Literal["markdown", "html"], fmt_raw)
        body = generate_fi_report(wh, asof=asof_ts, output_format=fmt)
    finally:
        wh.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "path": str(out_path),
                "format": fmt_raw,
                "bytes": len(body.encode("utf-8")),
            },
            sort_keys=True,
        )
    )
    return 0


def _cmd_fi_evidence_resign(ns: argparse.Namespace) -> int:
    """Bulk re-sign FI evidence packs from one HMAC version to another."""
    from market_regime_engine.fixed_income.evidence_pack import (
        evidence_pack_to_row,
        get_hmac_keys,
        sign_pack,
    )
    from market_regime_engine.storage import Warehouse

    keys = get_hmac_keys()
    from_key = getattr(ns, "from_key", None)
    to_key = getattr(ns, "to_key", None)
    if not from_key or not to_key:
        print(
            json.dumps(
                {"status": "error", "detail": "--from-key and --to-key are required"},
                sort_keys=True,
            )
        )
        return 2
    if to_key not in keys:
        print(
            json.dumps(
                {
                    "status": "error",
                    "detail": (f"--to-key {to_key!r} not in MRE_FI_HMAC_KEY_VERSIONS {sorted(keys)!r}"),
                },
                sort_keys=True,
            )
        )
        return 2

    dry_run = bool(getattr(ns, "dry_run", False))

    wh = Warehouse(ns.db)
    matched_count = 0
    resigned_count = 0
    try:
        df = wh.read_evidence_packs()
        if df is None or df.empty:
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "from_key": from_key,
                        "to_key": to_key,
                        "dry_run": dry_run,
                        "matched": 0,
                        "resigned": 0,
                    },
                    sort_keys=True,
                )
            )
            return 0
        # Filter to packs whose hmac_signature is "<from_key>:..." (signed
        # under from_key). Unsigned packs and packs signed under other
        # versions are skipped.
        sig_series = df["hmac_signature"].astype(str)
        prefix = f"{from_key}:"
        mask = sig_series.str.startswith(prefix)
        matching = df[mask]
        matched_count = len(matching)
        if not dry_run and not matching.empty:
            from dataclasses import replace as _dataclass_replace

            from market_regime_engine.fixed_income.evidence_pack import _row_to_pack

            rows: list[dict[str, Any]] = []
            for _, row in matching.iterrows():
                pack = _row_to_pack(row)
                # Strip the existing signature so canonical_pack_payload
                # doesn't re-include a stale value (defensive — the helper
                # already drops hmac_signature before signing).
                pack = _dataclass_replace(pack, hmac_signature=None)
                signed = sign_pack(pack, key_version=to_key)
                rows.append(evidence_pack_to_row(signed, request_id=str(row["request_id"])))
                resigned_count += 1
            wh.write_evidence_pack(pd.DataFrame(rows))
    finally:
        wh.close()

    print(
        json.dumps(
            {
                "status": "ok",
                "from_key": from_key,
                "to_key": to_key,
                "dry_run": dry_run,
                "matched": matched_count,
                "resigned": int(resigned_count),
            },
            sort_keys=True,
        )
    )
    return 0


def run(args: Sequence[str]) -> int:
    """Dispatch an ``fi-*`` subcommand.

    Returns the subprocess exit code. Stub commands return 0 + a
    ``not_yet_implemented`` JSON payload so downstream automation can
    detect the placeholder state via ``status`` rather than a non-zero
    exit. PR-7 closes out ``fi-evidence-pack``, ``fi-report``, and
    ``fi-evidence-resign``; the v1.5.0 release ships zero stubs.
    """
    parser = _build_parser()
    ns = parser.parse_args(list(args))
    command = ns.command
    if command == "fi-score-credit-regime":
        return _cmd_fi_score_credit_regime(ns)
    if command == "fi-score-liquidity":
        return _cmd_fi_score_liquidity(ns)
    if command == "fi-score-execution-confidence":
        return _cmd_fi_score_execution_confidence(ns)
    if command == "fi-tca-segment":
        return _cmd_fi_tca_segment(ns)
    if command == "fi-evidence-pack":
        return _cmd_fi_evidence_pack(ns)
    if command == "fi-report":
        return _cmd_fi_report(ns)
    if command == "fi-evidence-resign":
        return _cmd_fi_evidence_resign(ns)
    if command in CLI_COMMANDS:
        return _emit_stub(command)
    parser.error(f"unsupported fi-* command: {command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(sys.argv[1:]))


__all__ = ["CLI_COMMANDS", "run"]
