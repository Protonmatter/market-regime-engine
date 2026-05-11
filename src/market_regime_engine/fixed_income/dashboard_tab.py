# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income Streamlit dashboard tab (PR-7 §H, AF-2).

Per plan §7 §H + REVIEW.md §3.1 AF-2: the macro Streamlit dashboard
gets an FI section with four sub-panels:

1. **Credit regime ribbon** — time × regime score line / scatter.
2. **Liquidity stress heatmap** — scope_id × time z-score / score
   intensity heatmap, falls back to a per-scope line if there's not
   enough breadth.
3. **Execution-confidence success-rate** — last-N predictions ∩
   outcomes filled-rate trend.
4. **Release-gate decision timeline** — annotated history of
   release_gates rows.

The render helper is decoupled from the top-level
:mod:`market_regime_engine.dashboard` script so unit tests can drive
it without spinning up the Streamlit server. Streamlit imports are
lazy so the module is importable on any environment.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "fi_dashboard_summary",
    "load_fi_tables",
    "render_fi_tab",
]


def load_fi_tables(db_path: str) -> dict[str, pd.DataFrame]:
    """Load the seven FI tables the dashboard renders.

    Returns a dict keyed on the warehouse table name; missing tables
    produce empty frames (the helper survives a fresh deployment).
    """
    from market_regime_engine.storage import Warehouse

    out: dict[str, pd.DataFrame] = {
        "credit_regime_scores": pd.DataFrame(),
        "liquidity_stress_scores": pd.DataFrame(),
        "execution_confidence_predictions": pd.DataFrame(),
        "execution_outcomes": pd.DataFrame(),
        "tca_regime_segments": pd.DataFrame(),
        "release_gates": pd.DataFrame(),
        "fixed_income_evidence_packs": pd.DataFrame(),
    }
    try:
        db = Warehouse(db_path)
    except Exception as exc:
        log.warning("FI dashboard could not open warehouse %s: %s", db_path, exc)
        return out
    try:
        for table, reader in (
            ("credit_regime_scores", "read_credit_regime_scores"),
            ("liquidity_stress_scores", "read_liquidity_stress_scores"),
            ("execution_confidence_predictions", "read_execution_confidence_predictions"),
            ("execution_outcomes", "read_execution_outcomes"),
            ("tca_regime_segments", "read_tca_regime_segments"),
            ("release_gates", "read_release_gates"),
            ("fixed_income_evidence_packs", "read_evidence_packs"),
        ):
            fn = getattr(db, reader, None)
            if fn is None:
                continue
            try:
                df = fn()
            except Exception as exc:
                log.warning("FI dashboard read %s failed: %s", reader, exc)
                continue
            if df is not None:
                out[table] = df
    finally:
        try:
            db.close()
        except Exception:
            pass
    return out


def fi_dashboard_summary(fi_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Return a JSON-friendly summary used by the Streamlit panels.

    Exposed as a separate function so unit tests can validate the
    underlying numbers without invoking Streamlit.
    """
    summary: dict[str, Any] = {
        "credit_regime_rows": int(len(fi_data.get("credit_regime_scores", pd.DataFrame()))),
        "liquidity_scopes": int(
            fi_data.get("liquidity_stress_scores", pd.DataFrame())
            .get("scope_id", pd.Series(dtype=str))
            .nunique()
            if not fi_data.get("liquidity_stress_scores", pd.DataFrame()).empty
            else 0
        ),
        "execution_predictions": int(
            len(fi_data.get("execution_confidence_predictions", pd.DataFrame()))
        ),
        "release_gate_rows": int(
            len(fi_data.get("release_gates", pd.DataFrame()))
        ),
        "evidence_pack_rows": int(
            len(fi_data.get("fixed_income_evidence_packs", pd.DataFrame()))
        ),
    }
    cr = fi_data.get("credit_regime_scores", pd.DataFrame())
    if not cr.empty:
        summary["latest_regime_score"] = float(cr.iloc[-1]["regime_score"])
        summary["latest_regime_label"] = str(cr.iloc[-1]["regime_label"])
    return summary


def render_fi_tab(fi_data: dict[str, pd.DataFrame]) -> None:
    """Render the FI tab into the current Streamlit container.

    Streamlit is imported lazily so this module is importable on any
    environment. The render helper is no-op when invoked outside a
    Streamlit context (tests use :func:`fi_dashboard_summary` for
    smoke checks instead).
    """
    try:
        import streamlit as st
    except Exception as exc:  # pragma: no cover - import path
        log.warning("Streamlit not installed; FI dashboard tab skipped: %s", exc)
        return

    st.header("Fixed-Income RCIE")

    summary = fi_dashboard_summary(fi_data)

    cols = st.columns(4)
    cols[0].metric(
        "Credit regime score",
        f"{summary.get('latest_regime_score', float('nan')):.1f}"
        if "latest_regime_score" in summary
        else "n/a",
    )
    cols[1].metric(
        "Liquidity scopes", summary.get("liquidity_scopes", 0)
    )
    cols[2].metric(
        "Execution predictions",
        summary.get("execution_predictions", 0),
    )
    cols[3].metric(
        "Evidence packs",
        summary.get("evidence_pack_rows", 0),
    )

    cr = fi_data.get("credit_regime_scores", pd.DataFrame())
    if not cr.empty:
        st.subheader("Credit regime ribbon")
        cr_plot = cr.copy()
        cr_plot["timestamp"] = pd.to_datetime(cr_plot["timestamp"], errors="coerce")
        cr_plot = cr_plot.dropna(subset=["timestamp"])
        if not cr_plot.empty:
            try:
                import plotly.express as px

                fig = px.scatter(
                    cr_plot,
                    x="timestamp",
                    y="regime_score",
                    color="regime_label",
                    height=320,
                    title="Credit regime score over time",
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.line_chart(
                    cr_plot.set_index("timestamp")[["regime_score"]],
                    use_container_width=True,
                )

    liq = fi_data.get("liquidity_stress_scores", pd.DataFrame())
    if not liq.empty:
        st.subheader("Liquidity stress heatmap")
        liq_plot = liq.copy()
        liq_plot["timestamp"] = pd.to_datetime(liq_plot["timestamp"], errors="coerce")
        try:
            pivot = liq_plot.pivot_table(
                index="scope_id",
                columns=liq_plot["timestamp"].dt.strftime("%Y-%m-%d"),
                values="liquidity_score",
                aggfunc="mean",
            )
            st.dataframe(pivot, use_container_width=True)
        except Exception:
            st.dataframe(liq_plot.tail(50), use_container_width=True)

    exec_pred = fi_data.get("execution_confidence_predictions", pd.DataFrame())
    exec_out = fi_data.get("execution_outcomes", pd.DataFrame())
    if not exec_pred.empty:
        st.subheader("Execution confidence")
        if not exec_out.empty:
            joined = exec_pred.merge(
                exec_out, on="request_id", how="inner", suffixes=("_p", "_o")
            )
            if not joined.empty:
                joined["filled"] = joined["filled_quantity"].fillna(0).astype(float) > 0
                joined["timestamp"] = pd.to_datetime(
                    joined["timestamp_p"], errors="coerce"
                )
                joined = joined.sort_values("timestamp")
                joined["rolling_fill_rate"] = (
                    joined["filled"].rolling(20, min_periods=1).mean()
                )
                st.line_chart(
                    joined.set_index("timestamp")[["rolling_fill_rate"]],
                    use_container_width=True,
                )
        st.caption(f"Total predictions: {summary.get('execution_predictions', 0)}")

    rel = fi_data.get("release_gates", pd.DataFrame())
    if not rel.empty:
        st.subheader("Release gate decisions")
        st.dataframe(rel.tail(20), use_container_width=True)
