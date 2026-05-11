# SPDX-License-Identifier: Apache-2.0
"""PR-7 §C — FI RCIE report generator.

Per AGENT.md PR-7 §"Report output should include":

1. Latest credit regime index (score + label + drivers + signal age).
2. Latest liquidity stress index (per-scope summary).
3. Execution-confidence calibration summary (last N predictions vs.
   outcomes; success-rate roll-up).
4. TCA by regime (top buckets by sample count + slippage).
5. Release-gate status (latest release_gates row).
6. Evidence-pack references (latest packs per component).

The Markdown surface is the primary; HTML is rendered via the
``markdown`` Python library when available, with a no-deps Markdown→
HTML fallback that wraps the body in a ``<pre>`` block. The fallback
keeps the FI dependency surface optional-light; AGENT.md §"Final
acceptance command set" only requires the Markdown variant to round-
trip.

The function gracefully degrades on missing data: empty tables emit
"no data" sections rather than raising so the report can be generated
on a fresh install before any FI signals have landed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["FiReportSection", "generate_fi_report"]


@dataclass(frozen=True)
class FiReportSection:
    """One renderable section of the FI report."""

    heading: str
    body: str


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _signal_age_seconds(asof: pd.Timestamp, ts: pd.Timestamp | str | None) -> float | None:
    if ts is None:
        return None
    try:
        parsed = pd.Timestamp(ts)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    if asof.tzinfo is None:
        asof = asof.tz_localize("UTC")
    else:
        asof = asof.tz_convert("UTC")
    return float((asof - parsed).total_seconds())


def _credit_regime_section(
    df: pd.DataFrame, *, asof: pd.Timestamp
) -> FiReportSection:
    if df is None or df.empty:
        return FiReportSection(
            heading="Credit Regime Index",
            body="_No credit regime score has been recorded yet._\n",
        )
    row = df.iloc[-1]
    drivers_raw = row.get("drivers_json")
    drivers: list[str] = []
    if drivers_raw:
        try:
            drivers = list(json.loads(drivers_raw))
        except Exception:
            drivers = []
    age = _signal_age_seconds(asof, row.get("timestamp"))
    age_str = f"{age:.0f}s" if age is not None else "n/a"
    body_lines = [
        f"- **Score**: {float(row['regime_score']):.2f} / 100",
        f"- **Label**: {row['regime_label']}",
        f"- **Confidence**: {float(row['confidence']):.2f}",
        f"- **Release gate**: {bool(int(row['release_gate']))}",
        f"- **Timestamp**: {row['timestamp']}",
        f"- **Signal age (vs asof)**: {age_str}",
        f"- **Model run**: `{row['model_run_id']}`",
    ]
    if drivers:
        body_lines.append(f"- **Drivers**: {', '.join(drivers[:5])}")
    return FiReportSection(
        heading="Credit Regime Index",
        body="\n".join(body_lines) + "\n",
    )


def _liquidity_section(df: pd.DataFrame, *, asof: pd.Timestamp) -> FiReportSection:
    if df is None or df.empty:
        return FiReportSection(
            heading="Liquidity Stress Index",
            body="_No liquidity stress score has been recorded yet._\n",
        )
    # One summary row per scope_type (latest row per scope_type).
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df_sorted = df.sort_values("timestamp")
    grouped = df_sorted.groupby("scope_type", as_index=False).tail(1)
    lines = ["| Scope | Scope ID | Score | Label | Release gate | Age (s) |", "|---|---|---|---|---|---|"]
    for _, row in grouped.iterrows():
        age = _signal_age_seconds(asof, row.get("timestamp"))
        age_str = f"{age:.0f}" if age is not None else "n/a"
        lines.append(
            f"| {row['scope_type']} | {row['scope_id']} | "
            f"{float(row['liquidity_score']):.2f} | {row['liquidity_label']} | "
            f"{bool(int(row['release_gate']))} | {age_str} |"
        )
    return FiReportSection(
        heading="Liquidity Stress Index",
        body="\n".join(lines) + "\n",
    )


def _execution_confidence_section(
    predictions: pd.DataFrame, outcomes: pd.DataFrame, *, asof: pd.Timestamp
) -> FiReportSection:
    if predictions is None or predictions.empty:
        return FiReportSection(
            heading="Execution Confidence",
            body="_No execution-confidence predictions have been recorded yet._\n",
        )
    head = predictions.tail(50)
    actions = head["recommended_action"].astype(str).value_counts()
    lines = [
        f"- **Total predictions** (last {len(head)}): {len(head)}",
        f"- **Auto-X allowed**: {int(actions.get('Auto-X allowed', 0))}",
        f"- **Auto-X caution**: {int(actions.get('Auto-X caution / trader confirm', 0))}",
        f"- **Manual review required**: "
        f"{int(actions.get('Manual review required', 0))}",
    ]
    if outcomes is not None and not outcomes.empty:
        joined = predictions.merge(
            outcomes,
            left_on="request_id",
            right_on="request_id",
            how="inner",
            suffixes=("_pred", "_out"),
        )
        if not joined.empty:
            joined["filled"] = (
                joined["filled_quantity"].fillna(0).astype(float) > 0
            )
            success_rate = float(joined["filled"].mean())
            lines.append(f"- **Filled rate** (predictions ∩ outcomes): {success_rate:.1%}")
            lines.append(f"- **Sample size**: {len(joined)}")
    return FiReportSection(
        heading="Execution Confidence",
        body="\n".join(lines) + "\n",
    )


def _tca_section(df: pd.DataFrame) -> FiReportSection:
    if df is None or df.empty:
        return FiReportSection(
            heading="TCA By Regime",
            body="_No TCA segments have been materialised yet._\n",
        )
    df = df.copy()
    # Keep the latest snapshot only.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].notna().any():
        cutoff = df["timestamp"].max() - pd.Timedelta(days=1)
        df = df[df["timestamp"] >= cutoff]
    if df.empty:
        return FiReportSection(
            heading="TCA By Regime",
            body="_No TCA segments in the last 24h window._\n",
        )
    grouped = (
        df.groupby(["regime_label", "liquidity_label"], dropna=False)
        .agg(samples=("sample_count", "sum"), avg_value=("metric_value", "mean"))
        .sort_values("samples", ascending=False)
        .head(10)
        .reset_index()
    )
    lines = ["| Regime | Liquidity | Samples | Avg metric value |", "|---|---|---|---|"]
    for _, row in grouped.iterrows():
        lines.append(
            f"| {row['regime_label']} | {row['liquidity_label']} | "
            f"{int(row['samples'])} | {float(row['avg_value']):.4f} |"
        )
    return FiReportSection(
        heading="TCA By Regime",
        body="\n".join(lines) + "\n",
    )


def _release_gate_section(df: pd.DataFrame) -> FiReportSection:
    if df is None or df.empty:
        return FiReportSection(
            heading="Release Gate Status",
            body="_No release gate row has been recorded yet._\n",
        )
    row = df.iloc[-1]
    decision = str(row.get("decision", "unknown"))
    profile = str(row.get("resolved_profile", "n/a"))
    reasons = row.get("reasons")
    lines = [
        f"- **Latest decision**: {decision}",
        f"- **Resolved profile**: {profile}",
        f"- **Min confidence**: {row.get('min_confidence', 'n/a')}",
        f"- **Worst coverage**: {row.get('worst_coverage', 'n/a')}",
    ]
    if reasons:
        lines.append(f"- **Reasons**: {reasons}")
    return FiReportSection(
        heading="Release Gate Status",
        body="\n".join(lines) + "\n",
    )


def _evidence_pack_section(df: pd.DataFrame) -> FiReportSection:
    if df is None or df.empty:
        return FiReportSection(
            heading="Evidence Packs",
            body="_No evidence packs have been recorded yet._\n",
        )
    by_component = (
        df.sort_values("timestamp")
        .groupby("component_name", as_index=False)
        .tail(1)
    )
    lines = [
        "| Component | Model run | Request | Signed | Timestamp |",
        "|---|---|---|---|---|",
    ]
    for _, row in by_component.iterrows():
        signed = (
            str(row.get("hmac_signature", "")).split(":", 1)[0]
            if str(row.get("hmac_signature", ""))
            else "no"
        )
        lines.append(
            f"| {row['component_name']} | `{row['model_run_id']}` | "
            f"`{row['request_id']}` | {signed} | {row['timestamp']} |"
        )
    return FiReportSection(
        heading="Evidence Packs",
        body="\n".join(lines) + "\n",
    )


def _render_markdown(sections: list[FiReportSection], *, asof: pd.Timestamp) -> str:
    asof_str = pd.Timestamp(asof).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts: list[str] = [
        "# Fixed-Income RCIE Report",
        "",
        f"_Generated: {_now_iso()}; asof: {asof_str}_",
        "",
    ]
    for section in sections:
        parts.append(f"## {section.heading}")
        parts.append("")
        parts.append(section.body.rstrip("\n"))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _render_html(markdown_body: str) -> str:
    """Render Markdown to HTML.

    Prefers the optional ``markdown`` library; falls back to a minimal
    pass-through that wraps the Markdown body in a ``<pre>`` block so
    the report is still readable in a browser without the optional
    dep.
    """
    try:
        import markdown  # type: ignore[import-not-found]

        html_body = markdown.markdown(
            markdown_body,
            extensions=["tables", "fenced_code"],
        )
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>Fixed-Income RCIE Report</title></head><body>"
            f"{html_body}</body></html>\n"
        )
    except Exception:
        from html import escape

        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<title>Fixed-Income RCIE Report</title></head><body>"
            f"<pre>{escape(markdown_body)}</pre>"
            "</body></html>\n"
        )


def generate_fi_report(
    warehouse: Any,
    *,
    asof: pd.Timestamp | None = None,
    output_format: Literal["markdown", "html"] = "markdown",
) -> str:
    """Generate the FI RCIE Markdown / HTML report body.

    Parameters
    ----------
    warehouse
        DuckDB / SQLite-backed :class:`Warehouse`. Empty tables degrade
        gracefully ("no data" sections).
    asof
        Anchor timestamp for the signal-age columns; defaults to
        ``pd.Timestamp.now(tz="UTC")``.
    output_format
        ``"markdown"`` (default) or ``"html"``.
    """
    if asof is None:
        asof_ts = pd.Timestamp.now(tz="UTC")
    else:
        asof_ts = pd.Timestamp(asof)
        if asof_ts.tzinfo is None:
            asof_ts = asof_ts.tz_localize("UTC")
        else:
            asof_ts = asof_ts.tz_convert("UTC")

    def _safe_read(reader_name: str) -> pd.DataFrame:
        reader = getattr(warehouse, reader_name, None)
        if reader is None or not callable(reader):
            return pd.DataFrame()
        try:
            df = reader()
        except Exception as exc:  # pragma: no cover - storage-side failure
            log.warning("FI report read %s failed: %s", reader_name, exc)
            return pd.DataFrame()
        return df if df is not None else pd.DataFrame()

    sections = [
        _credit_regime_section(_safe_read("read_credit_regime_scores"), asof=asof_ts),
        _liquidity_section(_safe_read("read_liquidity_stress_scores"), asof=asof_ts),
        _execution_confidence_section(
            _safe_read("read_execution_confidence_predictions"),
            _safe_read("read_execution_outcomes"),
            asof=asof_ts,
        ),
        _tca_section(_safe_read("read_tca_regime_segments")),
        _release_gate_section(_safe_read("read_release_gates")),
        _evidence_pack_section(_safe_read("read_evidence_packs")),
    ]

    body = _render_markdown(sections, asof=asof_ts)
    if output_format == "html":
        return _render_html(body)
    return body
