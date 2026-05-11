# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from market_regime_engine import __version__
from market_regime_engine.analogs import analog_summary
from market_regime_engine.explain import latest_explanation
from market_regime_engine.storage import Warehouse

st.set_page_config(page_title="Market Regime Engine", layout="wide")
st.title(f"Market Regime Engine v{__version__}")
st.caption(
    "Macro regime, change-point, fitted hazard, analog, attribution, calibration, drift, "
    "release-gate, alerts, promotion workflow, confidence, and real point-in-time vintage audit."
)


@st.cache_data(ttl=120, show_spinner=False)
def _load_tables(db_path: str) -> dict[str, pd.DataFrame]:
    db = Warehouse(db_path)
    try:
        tables = {
            "regimes": db.read_regimes(),
            "outputs": db.read_model_outputs(),
            "calibrated": db.read_calibrated_outputs(),
            "features": db.read_features(),
            "analogs": db.read_historical_analogs(),
            "attribution": db.read_driver_attribution(),
            "confidence": db.read_confidence_scores(),
            "invalidation": db.read_invalidation_triggers(),
            "model_runs": db.read_model_runs(),
            "drift": db.read_model_drift(),
            "gates": db.read_release_gates(),
            "weights": db.read_ensemble_weights(),
            "alerts": db.read_routed_alerts(),
            "promotion_workflow": db.read_promotion_workflow(),
            "hazard": db.read_hazard_diagnostics(),
            "vintage_audits": db.read_vintage_audits(),
            "feature_asof": db.read_feature_asof_values(),
            "vintage_observations": db.read_vintage_observations(),
        }
    finally:
        db.close()
    return tables


db_path = st.sidebar.text_input("Database", os.getenv("MRE_DB_PATH", "data/mre.duckdb"))
if st.sidebar.button("Refresh data"):
    _load_tables.clear()

with st.spinner("Loading warehouse..."):
    data = _load_tables(db_path)


regimes = data["regimes"]
if regimes.empty:
    st.warning("No regime data. Run bootstrap/build/score first.")
    st.stop()


latest = latest_explanation(regimes)
gates = data["gates"]
confidence = data["confidence"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Regime", latest["regime"])
c2.metric("Stress score", f"{latest['score']:.2f}")
cp = latest.get("change_point_prob")
c3.metric("Change-point probability", "n/a" if cp is None else f"{cp:.1%}")
if not confidence.empty:
    crow = confidence.iloc[-1]
    c4.metric("Confidence", f"{crow['grade']} / {float(crow['confidence']):.2f}")
else:
    c4.metric("Confidence", "not scored")

if not gates.empty:
    decision = str(gates.iloc[-1].get("decision", "unknown"))
    color = "green" if decision == "release" else ("orange" if decision == "hold" else "red")
    st.markdown(
        f"<div style='padding:8px;border-radius:6px;background:{color};color:white;display:inline-block;'>"
        f"Release gate: <b>{decision}</b></div>",
        unsafe_allow_html=True,
    )

st.subheader("Top domain drivers from regime metadata")
st.dataframe(pd.DataFrame(latest["top_drivers"]), use_container_width=True)


def _plot_regime_ribbon(regimes: pd.DataFrame) -> None:
    try:
        import plotly.express as px
    except Exception:
        st.line_chart(regimes.set_index(pd.to_datetime(regimes["date"]))[["score", "change_point_prob"]])
        return
    df = regimes.copy()
    df["date"] = pd.to_datetime(df["date"])
    fig = px.scatter(
        df,
        x="date",
        y="score",
        color="decoded_regime",
        opacity=0.7,
        height=320,
        title="Regime path (color) vs. stress score",
    )
    fig.add_scatter(x=df["date"], y=df["change_point_prob"], mode="lines", name="CP probability", yaxis="y2")
    fig.update_layout(yaxis2={"title": "CP probability", "overlaying": "y", "side": "right", "range": [0, 1]})
    st.plotly_chart(fig, use_container_width=True)


st.subheader("Regime ribbon")
_plot_regime_ribbon(regimes)


calibrated = data["calibrated"]
outputs = data["outputs"]
if not calibrated.empty:
    st.subheader("Latest calibrated probability outputs")
    st.dataframe(calibrated[calibrated["date"] == calibrated["date"].max()], use_container_width=True)
elif not outputs.empty:
    st.subheader("Latest raw model outputs")
    st.dataframe(outputs[outputs["date"] == outputs["date"].max()], use_container_width=True)

invalidation = data["invalidation"]
if not invalidation.empty:
    st.subheader("Forecast invalidation triggers")
    latest_inv = invalidation[invalidation["date"] == invalidation["date"].max()]
    st.dataframe(latest_inv, use_container_width=True)

attribution = data["attribution"]
if not attribution.empty:
    st.subheader("Stored attribution")
    latest_attr = attribution[attribution["date"] == attribution["date"].max()]
    st.dataframe(latest_attr, use_container_width=True)

analogs = data["analogs"]
if not analogs.empty:
    st.subheader("Historical analogs")
    latest_analogs = analogs[analogs["as_of_date"] == analogs["as_of_date"].max()]
    st.json(analog_summary(latest_analogs))
    st.dataframe(latest_analogs, use_container_width=True)

if not gates.empty:
    st.subheader("Forecast release gate (history)")
    st.dataframe(gates.tail(10), use_container_width=True)

drift = data["drift"]
if not drift.empty:
    st.subheader("Model drift monitor")
    st.dataframe(drift.head(50), use_container_width=True)

weights = data["weights"]
if not weights.empty:
    st.subheader("Stacked ensemble weights")
    st.dataframe(weights, use_container_width=True)

alerts = data["alerts"]
if not alerts.empty:
    st.subheader("Routed alerts")
    st.dataframe(alerts.head(50), use_container_width=True)

promotion_workflow = data["promotion_workflow"]
if not promotion_workflow.empty:
    st.subheader("Promotion workflow")
    st.dataframe(promotion_workflow.tail(10), use_container_width=True)

hazard = data["hazard"]
if not hazard.empty:
    st.subheader("Fitted hazard diagnostics")
    st.dataframe(hazard.tail(10), use_container_width=True)

vintage_audits = data["vintage_audits"]
if not vintage_audits.empty:
    st.subheader("Vintage / point-in-time audit")
    st.dataframe(vintage_audits.tail(20), use_container_width=True)

feature_asof = data["feature_asof"]
if not feature_asof.empty:
    st.subheader("Feature as-of values")
    latest_asof = feature_asof[feature_asof["as_of_date"] == feature_asof["as_of_date"].max()]
    st.metric("Latest as-of feature count", len(latest_asof))
    st.dataframe(latest_asof.head(200), use_container_width=True)

vintage_observations = data["vintage_observations"]
if not vintage_observations.empty:
    st.subheader("Vintage observation coverage")
    st.json(
        {
            "rows": len(vintage_observations),
            "series": int(vintage_observations["series_id"].nunique()),
            "min_observation_date": str(vintage_observations["observation_date"].min()),
            "max_observation_date": str(vintage_observations["observation_date"].max()),
            "min_vintage_date": str(vintage_observations["vintage_date"].min()),
            "max_vintage_date": str(vintage_observations["vintage_date"].max()),
        }
    )

model_runs = data["model_runs"]
if not model_runs.empty:
    st.subheader("Immutable model runs")
    st.dataframe(model_runs.tail(20), use_container_width=True)

features = data["features"]
if not features.empty:
    st.subheader("Recent features")
    st.dataframe(features.tail(250), use_container_width=True)


# v1.5 PR-7 §H — Fixed-Income RCIE dashboard tab.
#
# Conditional on the presence of FI tables: skip when
# ``credit_regime_scores`` is empty so a fresh deployment renders only
# the macro surface. The render helper lives in
# ``fixed_income.dashboard_tab`` so the FI section stays decoupled
# from the macro page; tests can drive the helper without spinning up
# Streamlit.
try:
    from market_regime_engine.fixed_income.dashboard_tab import (
        load_fi_tables,
        render_fi_tab,
    )

    fi_data = load_fi_tables(db_path)
    if any(df is not None and not df.empty for df in fi_data.values()):
        st.divider()
        render_fi_tab(fi_data)
except Exception as _fi_exc:  # pragma: no cover - defensive
    st.caption(f"FI dashboard tab unavailable: {_fi_exc!r}")
