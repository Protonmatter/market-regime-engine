# SPDX-License-Identifier: Apache-2.0
"""Institutional report writer.

The historical v0.5–v0.8 governance addendums each lived in their own
``report_writer_v{2..5}.py`` module. v1.3 consolidates the five files into
a single module with explicit section selection. The legacy entry points
(:func:`append_v05_sections`, ``v06``, ``v07``, ``v08``) remain as
deprecation shims so external automation that called them by name keeps
working for one release; they are scheduled for removal in v1.4.

The institutional-report markdown is contractually byte-stable: the
v1.3 implementation produces the exact same bytes as the v1.2.1
``write_institutional_report`` + ``append_v0X_sections`` chain. A
regression test (``tests/test_v1_3_report_writer_parity.py``) hashes
both outputs and asserts equality.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from market_regime_engine.analogs import analog_summary
from market_regime_engine.explain import latest_explanation


def _md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> str:
    if df is None or df.empty:
        return "_No data._\n"
    view = df[cols].head(max_rows).copy()
    return view.to_markdown(index=False) + "\n"


def _write_base_report(
    *,
    regimes: pd.DataFrame,
    model_outputs: pd.DataFrame,
    analogs: pd.DataFrame,
    domain_attribution: pd.DataFrame,
    feature_attribution: pd.DataFrame,
    validation_dir: str | Path | None,
    out: Path,
) -> Path:
    """Emit the base institutional report (no governance addendums)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    latest = latest_explanation(regimes)
    analog_stats = analog_summary(analogs)
    lines: list[str] = []
    lines.append("# Market Regime Engine Institutional Report\n")
    lines.append("## Executive snapshot\n")
    if latest.get("status"):
        lines.append("No regime scores available.\n")
    else:
        lines.append(f"- **Date:** {latest['date']}\n")
        lines.append(f"- **Decoded regime:** {latest['regime']}\n")
        lines.append(f"- **Raw regime:** {latest['raw_regime']}\n")
        lines.append(f"- **Stress score:** {latest['score']:.3f}\n")
        cp = latest.get("change_point_prob")
        lines.append(f"- **Change-point probability:** {'n/a' if cp is None else f'{cp:.1%}'}\n")

    lines.append("\n## Latest model outputs\n")
    if model_outputs is not None and not model_outputs.empty:
        latest_date = model_outputs["date"].max()
        mo = model_outputs[model_outputs["date"] == latest_date]
        lines.append(_md_table(mo, ["model_name", "horizon", "target", "value"], 50))
    else:
        lines.append("_No model outputs._\n")

    lines.append("\n## Historical analog summary\n")
    lines.append("```json\n")
    lines.append(json.dumps(analog_stats, indent=2, sort_keys=True))
    lines.append("\n```\n")

    lines.append("\n## Top historical analogs\n")
    lines.append(_md_table(analogs, ["rank", "analog_date", "distance", "similarity"], 15))

    lines.append("\n## Domain attribution\n")
    lines.append(_md_table(domain_attribution, ["rank", "domain", "score", "zscore", "change_3m"], 12))

    lines.append("\n## Feature attribution\n")
    lines.append(_md_table(feature_attribution, ["rank", "feature_name", "domain", "value", "zscore"], 20))

    if validation_dir:
        vdir = Path(validation_dir)
        lines.append("\n## Validation artifact inventory\n")
        if vdir.exists():
            for p in sorted(vdir.glob("*.csv")):
                lines.append(f"- `{p}`\n")
        else:
            lines.append(f"Validation directory not found: `{vdir}`\n")

    lines.append("\n## Model-risk notes\n")
    lines.append("- Forecasts are probability distributions, not price targets.\n")
    lines.append("- Promotion requires benchmark comparison and calibration evidence.\n")
    lines.append("- Point-in-time ingestion must be enforced before live historical claims are trusted.\n")
    lines.append("- Geopolitical event overlays are scenario signals, not deterministic forecasts.\n")

    out.write_text("".join(lines), encoding="utf-8")
    return out


def _v05_section(
    *,
    confidence: pd.DataFrame | None,
    invalidation: pd.DataFrame | None,
    model_runs: pd.DataFrame | None,
    calibrated_outputs: pd.DataFrame | None,
) -> str:
    lines = ["\n\n# v0.5 governance and confidence layer\n"]
    if confidence is not None and not confidence.empty:
        row = confidence.iloc[-1]
        lines.append(
            f"\n## Forecast confidence\n- **Grade:** {row['grade']}\n- **Score:** {float(row['confidence']):.3f}\n"
        )
        with contextlib.suppress(Exception):
            lines.append(
                "```json\n"
                + json.dumps(json.loads(row.get("metadata_json", "{}")), indent=2, sort_keys=True)
                + "\n```\n"
            )
    if invalidation is not None and not invalidation.empty:
        breached = invalidation[invalidation["status"] == "breached"]
        lines.append("\n## Forecast invalidation triggers\n")
        if breached.empty:
            lines.append("_No active breached invalidation triggers._\n")
        else:
            lines.append(breached[["trigger", "severity", "value", "threshold"]].to_markdown(index=False) + "\n")
    if calibrated_outputs is not None and not calibrated_outputs.empty:
        latest = calibrated_outputs[calibrated_outputs["date"] == calibrated_outputs["date"].max()]
        lines.append("\n## Calibrated probability outputs\n")
        lines.append(latest[["model_name", "horizon", "target", "value"]].to_markdown(index=False) + "\n")
    if model_runs is not None and not model_runs.empty:
        latest = model_runs.iloc[-1]
        lines.append("\n## Latest immutable model run\n")
        lines.append(
            f"- **Run ID:** `{latest['run_id']}`\n- **Artifact hash:** `{latest['artifact_hash']}`\n- **Created:** {latest['created_at_utc']}\n"
        )
    # The legacy v0.5 helper appended a trailing newline at write time
    # (see ``report_writer_v2.py``); preserve it exactly so the v1.3
    # consolidation is byte-identical to the v1.2.1 chain.
    return "".join(lines) + "\n"


def _v06_section(
    *,
    drift: pd.DataFrame | None,
    release_gates: pd.DataFrame | None,
    ensemble_weights: pd.DataFrame | None,
    stacking_diagnostics: pd.DataFrame | None,
) -> str:
    parts = ["\n\n# v0.6 Forecast Governance Layer\n"]
    if release_gates is not None and not release_gates.empty:
        latest = release_gates.iloc[-1]
        parts.append("\n## Release Gate\n")
        parts.append(f"- Decision: **{latest.get('decision')}**\n")
        parts.append(f"- Confidence: {latest.get('confidence')} / {latest.get('confidence_grade')}\n")
        parts.append(f"- Reasons: {latest.get('reasons')}\n")
    if drift is not None and not drift.empty:
        latest_date = drift["date"].max()
        d = drift[drift["date"] == latest_date]
        parts.append("\n## Drift Monitor\n")
        parts.append(f"- As of: {latest_date}\n")
        parts.append(f"- Major drift features: {int((d['status'] == 'major').sum())}\n")
        parts.append(f"- Max PSI: {float(d['psi'].max()):.3f}\n")
    if ensemble_weights is not None and not ensemble_weights.empty:
        parts.append("\n## Stacking Weights\n")
        for _, row in ensemble_weights.head(10).iterrows():
            parts.append(
                f"- {row.get('target')} {row.get('horizon')} {row.get('model_name')}: {float(row.get('weight', 0)):.3f}\n"
            )
    if stacking_diagnostics is not None and not stacking_diagnostics.empty:
        parts.append("\n## Stacking Diagnostics\n")
        for _, row in stacking_diagnostics.head(10).iterrows():
            parts.append(
                f"- {row.get('target')} {row.get('horizon')}: log_loss={float(row.get('log_loss', 0)):.4f}, brier={float(row.get('brier', 0)):.4f}\n"
            )
    return "".join(parts)


def _v07_section(
    *,
    alerts: pd.DataFrame,
    promotion_workflow: pd.DataFrame,
    hazard_diagnostics: pd.DataFrame,
    alfred_manifest: pd.DataFrame,
) -> str:
    parts = ["\n\n# v0.7 Governance, Hazard, and Vintage Ingestion\n"]
    parts.append("\n## Promotion workflow\n")
    parts.append(
        (
            promotion_workflow.tail(10).to_markdown(index=False)
            if promotion_workflow is not None and not promotion_workflow.empty
            else "No promotion workflow rows."
        )
        + "\n"
    )
    parts.append("\n## Alert routing\n")
    parts.append(
        (alerts.tail(20).to_markdown(index=False) if alerts is not None and not alerts.empty else "No alerts routed.")
        + "\n"
    )
    parts.append("\n## Fitted hazard diagnostics\n")
    parts.append(
        (
            hazard_diagnostics.tail(10).to_markdown(index=False)
            if hazard_diagnostics is not None and not hazard_diagnostics.empty
            else "No fitted hazard diagnostics."
        )
        + "\n"
    )
    parts.append("\n## ALFRED/FRED vintage ingestion manifest\n")
    parts.append(
        (
            alfred_manifest.tail(20).to_markdown(index=False)
            if alfred_manifest is not None and not alfred_manifest.empty
            else "No live vintage ingestion manifest."
        )
        + "\n"
    )
    return "".join(parts)


def _v08_section(
    *,
    vintage_audits: pd.DataFrame,
    feature_asof: pd.DataFrame,
    vintage_observations: pd.DataFrame,
) -> str:
    parts: list[str] = ["\n\n# v0.8 Real Point-in-Time Vintage Layer\n\n"]
    if vintage_audits is not None and not vintage_audits.empty:
        parts.append("## Vintage/as-of audit\n\n")
        parts.append(vintage_audits.tail(10).to_markdown(index=False))
        parts.append("\n\n")
    else:
        parts.append("No vintage audit rows found. Run `mre audit-vintage --enforce`.\n\n")
    if feature_asof is not None and not feature_asof.empty:
        latest = feature_asof[feature_asof["as_of_date"] == feature_asof["as_of_date"].max()]
        parts.append("## Feature as-of lineage\n\n")
        parts.append(f"Latest as-of date: `{latest['as_of_date'].iloc[0]}`  \n")
        parts.append(f"Feature rows on latest as-of date: `{len(latest)}`\n\n")
        parts.append(latest.head(20).to_markdown(index=False))
        parts.append("\n\n")
    else:
        parts.append("No feature-as-of rows found. Run `mre materialize-asof-features --write-features`.\n\n")
    if vintage_observations is not None and not vintage_observations.empty:
        parts.append("## Vintage observation coverage\n\n")
        coverage = pd.DataFrame(
            [
                {
                    "rows": len(vintage_observations),
                    "series": vintage_observations["series_id"].nunique(),
                    "min_observation_date": vintage_observations["observation_date"].min(),
                    "max_observation_date": vintage_observations["observation_date"].max(),
                    "min_vintage_date": vintage_observations["vintage_date"].min(),
                    "max_vintage_date": vintage_observations["vintage_date"].max(),
                }
            ]
        )
        parts.append(coverage.to_markdown(index=False))
        parts.append("\n\n")
    parts.append("## v0.8 hard invariant\n\n")
    parts.append(
        "Every feature entering the point-in-time pipeline must prove `observation_date <= as_of_date` and `vintage_date <= as_of_date`. Rows violating that condition fail `audit-vintage --enforce`.\n"
    )
    return "".join(parts)


_ALL_SECTIONS: tuple[str, ...] = ("v04", "v05", "v06", "v07", "v08")


def write_institutional_report(
    *,
    regimes: pd.DataFrame,
    model_outputs: pd.DataFrame,
    analogs: pd.DataFrame,
    domain_attribution: pd.DataFrame,
    feature_attribution: pd.DataFrame,
    validation_dir: str | Path | None = None,
    out: str | Path = "data/reports/institutional_report.md",
    sections: Iterable[str] = _ALL_SECTIONS,
    confidence: pd.DataFrame | None = None,
    invalidation: pd.DataFrame | None = None,
    model_runs: pd.DataFrame | None = None,
    calibrated_outputs: pd.DataFrame | None = None,
    drift: pd.DataFrame | None = None,
    release_gates: pd.DataFrame | None = None,
    ensemble_weights: pd.DataFrame | None = None,
    stacking_diagnostics: pd.DataFrame | None = None,
    alerts: pd.DataFrame | None = None,
    promotion_workflow: pd.DataFrame | None = None,
    hazard_diagnostics: pd.DataFrame | None = None,
    alfred_manifest: pd.DataFrame | None = None,
    vintage_audits: pd.DataFrame | None = None,
    feature_asof: pd.DataFrame | None = None,
    vintage_observations: pd.DataFrame | None = None,
) -> Path:
    """Write the institutional report.

    ``sections`` selects which historical addendums to append. The default
    matches the v1.2.1 ``cli.institutional_report_cmd`` chain (v04 base
    plus v05/v06/v07/v08 governance addendums). Pass a subset to suppress
    individual addendums; the remaining sections are still appended in
    canonical order.
    """
    out = Path(out)
    base = _write_base_report(
        regimes=regimes,
        model_outputs=model_outputs,
        analogs=analogs,
        domain_attribution=domain_attribution,
        feature_attribution=feature_attribution,
        validation_dir=validation_dir,
        out=out,
    )
    sections_set = set(sections)
    pieces: list[str] = []
    if "v05" in sections_set:
        pieces.append(
            _v05_section(
                confidence=confidence,
                invalidation=invalidation,
                model_runs=model_runs,
                calibrated_outputs=calibrated_outputs,
            )
        )
    if "v06" in sections_set:
        pieces.append(
            _v06_section(
                drift=drift,
                release_gates=release_gates,
                ensemble_weights=ensemble_weights,
                stacking_diagnostics=stacking_diagnostics,
            )
        )
    if "v07" in sections_set:
        pieces.append(
            _v07_section(
                alerts=alerts if alerts is not None else pd.DataFrame(),
                promotion_workflow=promotion_workflow if promotion_workflow is not None else pd.DataFrame(),
                hazard_diagnostics=hazard_diagnostics if hazard_diagnostics is not None else pd.DataFrame(),
                alfred_manifest=alfred_manifest if alfred_manifest is not None else pd.DataFrame(),
            )
        )
    if "v08" in sections_set:
        pieces.append(
            _v08_section(
                vintage_audits=vintage_audits if vintage_audits is not None else pd.DataFrame(),
                feature_asof=feature_asof if feature_asof is not None else pd.DataFrame(),
                vintage_observations=vintage_observations if vintage_observations is not None else pd.DataFrame(),
            )
        )
    if pieces:
        with base.open("a", encoding="utf-8") as f:
            f.write("".join(pieces))
    return base


__all__ = ["write_institutional_report"]
