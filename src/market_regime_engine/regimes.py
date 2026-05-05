# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pandas as pd

from market_regime_engine.bocpd import DiagonalStudentTBOCPD, MultivariateNIWBOCPD
from market_regime_engine.changepoint import RollingMultivariateChangePoint
from market_regime_engine.features import feature_matrix
from market_regime_engine.hmm import REGIME_STATES, HMMRegimePosterior
from market_regime_engine.wfst import RegimeWFST, event_labels_from_scores


def domain_scores(features: pd.DataFrame) -> pd.DataFrame:
    mat = feature_matrix(features)
    rows = []
    for date, row in mat.iterrows():
        # Bind ``row`` explicitly via default arg so ruff B023 is happy and
        # the closure captures the per-iteration value rather than the loop
        # variable.
        def g(name: str, _row: pd.Series = row) -> float:
            val = _row.get(name, 0.0)
            return 0.0 if pd.isna(val) else float(val)

        scores = {
            "labor": g("UNRATE.diff_3m") + g("U6RATE.diff_3m") - 10 * g("PAYEMS.log_3m"),
            "rates": g("FEDFUNDS.level") / 5
            + g("DGS10.level") / 5
            + g("MORTGAGE30US.level") / 6
            - g("T10Y3M.level") / 2,
            "inflation": 25 * g("CPIAUCSL.log_yoy") + 30 * g("CPILFESL.log_yoy"),
            "credit": g("BAA10Y.level") / 3,
            "housing": -4 * g("PERMIT.log_yoy") - 4 * g("HOUST.log_yoy") + g("MORTGAGE30US.diff_12m") / 3,
            "energy": abs(g("DCOILWTICO.log_yoy")) * 2 + max(2 * g("DCOILWTICO.log_yoy"), 0),
            "fx": abs(g("DTWEXBGS.log_yoy")) * 3,
            "fiscal": g("GFDEGDQ188S.level") / 120,
        }
        for domain, score in scores.items():
            rows.append({"date": date, "domain": domain, "score": float(score)})
    return pd.DataFrame(rows)


def classify(scores: dict[str, float], cp_prob: float = 0.0) -> tuple[str, float]:
    labor = scores.get("labor", 0.0)
    rates = scores.get("rates", 0.0)
    inflation = scores.get("inflation", 0.0)
    credit = scores.get("credit", 0.0)
    housing = scores.get("housing", 0.0)
    energy = scores.get("energy", 0.0)
    fiscal = scores.get("fiscal", 0.0)
    total = labor + rates + inflation + credit + housing + energy + fiscal + cp_prob

    if credit > 1.2 and housing > 0.6:
        return "credit_stress", total
    if energy > 0.9 and inflation > 1.0:
        return "energy_shock", total
    if inflation > 1.2 and labor > 0.2:
        return "stagflation", total
    if inflation > 1.2 and rates > 1.2:
        return "sticky_inflation", total
    if labor > 0.8 and housing > 0.5:
        return "recessionary_bear", total
    if total < 1.0 and rates < 1.0:
        return "risk_on_expansion", total
    if inflation < 0.9 and labor < 0.4:
        return "soft_landing", total
    return "late_cycle", total


def _posterior_rows(hmm: pd.DataFrame) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if hmm.empty:
        return rows
    for _, r in hmm.iterrows():
        rows.append({state: float(r.get(f"regime_prob_{state}", 0.0)) for state in REGIME_STATES})
    return rows


def score_regimes(
    features: pd.DataFrame,
    *,
    use_bocpd: bool = True,
    bocpd_core: str = "niw",
    fit_hmm: bool = True,
    min_hmm_history: int = 60,
) -> pd.DataFrame:
    """Score regimes from a long-format feature frame.

    Parameters
    ----------
    use_bocpd:
        When True, run a BOCPD detector and merge its change-point probability
        with the rolling Mahalanobis detector by element-wise max.
    bocpd_core:
        ``"niw"`` (default) uses the v1.0 multivariate Normal-Inverse-Wishart
        BOCPD which captures cross-domain covariance. ``"diagonal"`` keeps the
        v0.8 Student-t diagonal fallback for back-compat or for very tiny
        windows where the NIW posterior is poorly conditioned.
    fit_hmm:
        When True (default in v1.0), fit the Gaussian HMM emissions and
        transition matrix on the domain-score panel via Baum-Welch before
        decoding. The post-fit pinning step keeps regime labels stable. When
        False, fall back to the v0.8 hand-prior centroids.
    min_hmm_history:
        Minimum number of monthly observations required before EM is run. Below
        this, the hand-prior HMM is used regardless of ``fit_hmm`` to avoid
        degenerate covariance estimates.
    """
    ds = domain_scores(features)
    if ds.empty:
        return pd.DataFrame(columns=["date", "regime", "decoded_regime", "score", "change_point_prob", "metadata_json"])

    pivot = ds.pivot(index="date", columns="domain", values="score").fillna(0.0).sort_index()

    rolling_cp = RollingMultivariateChangePoint().score(pivot).rename(columns={"change_point_prob": "rolling_cp_prob"})
    bocpd = pd.DataFrame()
    if use_bocpd:
        if str(bocpd_core).lower() == "diagonal":
            bocpd = DiagonalStudentTBOCPD().score(pivot)
        else:
            bocpd = MultivariateNIWBOCPD().score(pivot)
    if not bocpd.empty:
        cp = pd.merge(rolling_cp, bocpd, on="date", how="outer")
        cp["change_point_prob"] = cp[["rolling_cp_prob", "change_point_prob"]].max(axis=1)
    else:
        cp = rolling_cp.rename(columns={"rolling_cp_prob": "change_point_prob"})
        cp["rolling_cp_prob"] = cp["change_point_prob"]
    cp_map = dict(zip(pd.to_datetime(cp["date"]), cp["change_point_prob"], strict=False))
    cp_rows = {pd.to_datetime(r["date"]): r for _, r in cp.iterrows()}

    hmm_model = HMMRegimePosterior()
    if fit_hmm and len(pivot) >= max(min_hmm_history, 24):
        try:
            hmm_model = hmm_model.fit(pivot)
        except Exception:
            hmm_model = HMMRegimePosterior()
    hmm = hmm_model.score(pivot)
    hmm_map = {pd.to_datetime(r["date"]): r for _, r in hmm.iterrows()}

    rows = []
    labels = []
    posterior_payloads = []
    for date, row in pivot.iterrows():
        scores = {k: float(v) for k, v in row.to_dict().items()}
        cp_prob = float(cp_map.get(date, 0.0))
        regime, total = classify(scores, cp_prob)
        hrow = hmm_map.get(date)
        hmm_regime = str(hrow["hmm_regime"]) if hrow is not None else regime
        hmm_confidence = float(hrow["hmm_confidence"]) if hrow is not None else 0.0
        # If HMM is confident, allow it to override noisy threshold classifier. Otherwise keep classifier.
        observed_regime = hmm_regime if hmm_confidence >= 0.55 else regime
        event_labels = event_labels_from_scores(scores, cp_prob)
        labels.append(event_labels)
        posterior_payload = (
            {state: float(hrow.get(f"regime_prob_{state}", 0.0)) for state in REGIME_STATES} if hrow is not None else {}
        )
        posterior_payloads.append(posterior_payload)
        cprow = cp_rows.get(date, {})
        rows.append(
            {
                "date": date,
                "regime": observed_regime,
                "decoded_regime": observed_regime,
                "score": total,
                "change_point_prob": cp_prob,
                "metadata_json": json.dumps(
                    {
                        "domain_scores": scores,
                        "total_stress": total,
                        "threshold_regime": regime,
                        "hmm_regime": hmm_regime,
                        "hmm_confidence": hmm_confidence,
                        "hmm_posterior": posterior_payload,
                        "event_labels": sorted(event_labels),
                        "rolling_cp_prob": float(cprow.get("rolling_cp_prob", cp_prob))
                        if isinstance(cprow, dict)
                        else cp_prob,
                        "bocpd_run_length_mean": float(cprow.get("bocpd_run_length_mean", 0.0))
                        if isinstance(cprow, dict)
                        else 0.0,
                        "bocpd_map_run_length": int(cprow.get("bocpd_map_run_length", 0))
                        if isinstance(cprow, dict)
                        else 0,
                    },
                    sort_keys=True,
                ),
            }
        )

    out = pd.DataFrame(rows)
    decoder = RegimeWFST()
    out["decoded_regime"] = decoder.decode(
        out["regime"].tolist(), event_labels=labels, posterior_rows=posterior_payloads
    )
    return out
