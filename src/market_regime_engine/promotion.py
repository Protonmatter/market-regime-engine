# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PromotionGate:
    """Champion/challenger promotion rules for model-risk hygiene."""

    min_brier_improvement: float = 0.005
    min_logloss_improvement: float = 0.005
    max_ece: float = 0.12
    min_observations: int = 24

    def evaluate_binary(
        self,
        candidate: pd.DataFrame,
        benchmark: pd.DataFrame,
        *,
        mcs_membership: set[str] | None = None,
    ) -> pd.DataFrame:
        """Return a per-(target, horizon) promotion frame.

        ``mcs_membership`` is the optional set of model names that survived
        Hansen-MCS on the validation loss frame. When provided, each row is
        annotated with ``mcs_evidence`` ∈ ``{"in_set", "out_of_set"}``; when
        ``None``, the column is filled with ``"absent"`` so a downstream
        consumer can distinguish "no MCS run" from "MCS rejected".
        """
        rows = []
        if candidate.empty or benchmark.empty:
            return pd.DataFrame(columns=["target", "horizon", "promoted", "reason", "mcs_evidence"])
        keys = ["target", "horizon"]
        for _, c in candidate.iterrows():
            b = benchmark[(benchmark["target"] == c["target"]) & (benchmark["horizon"] == c["horizon"])]
            if b.empty:
                rows.append(
                    {
                        **{k: c[k] for k in keys},
                        "promoted": False,
                        "reason": "missing benchmark",
                        "mcs_evidence": _mcs_evidence(c.get("model"), mcs_membership),
                    }
                )
                continue
            b = b.iloc[0]
            reasons = []
            if int(c.get("observations", 0)) < self.min_observations:
                reasons.append("too few observations")
            if not math.isfinite(float(c.get("brier", float("nan")))):
                reasons.append("candidate brier not finite")
            if float(c.get("brier", float("inf"))) > float(b.get("brier", float("inf"))) - self.min_brier_improvement:
                reasons.append("brier improvement below gate")
            if (
                float(c.get("log_loss", float("inf")))
                > float(b.get("log_loss", float("inf"))) - self.min_logloss_improvement
            ):
                reasons.append("log-loss improvement below gate")
            if float(c.get("ece", float("inf"))) > self.max_ece:
                reasons.append("calibration error above gate")
            rows.append(
                {
                    "target": c["target"],
                    "horizon": c["horizon"],
                    "promoted": not reasons,
                    "reason": "; ".join(reasons) if reasons else "passed promotion gate",
                    "candidate_brier": float(c.get("brier", float("nan"))),
                    "benchmark_brier": float(b.get("brier", float("nan"))),
                    "candidate_log_loss": float(c.get("log_loss", float("nan"))),
                    "benchmark_log_loss": float(b.get("log_loss", float("nan"))),
                    "candidate_ece": float(c.get("ece", float("nan"))),
                    "mcs_evidence": _mcs_evidence(c.get("model"), mcs_membership),
                }
            )
        return pd.DataFrame(rows)


def _mcs_evidence(model_name: object, mcs_membership: set[str] | None) -> str:
    """Map (candidate model name, MCS set) to a short audit string."""
    if mcs_membership is None:
        return "absent"
    if model_name is None:
        # No model name on the candidate row; cannot pronounce on membership.
        return "absent"
    return "in_set" if str(model_name) in mcs_membership else "out_of_set"


def best_benchmark(binary_validation: pd.DataFrame) -> pd.DataFrame:
    """Select lowest-Brier benchmark per target/horizon."""
    if binary_validation.empty:
        return binary_validation
    frame = binary_validation.copy()
    frame = frame.sort_values(["target", "horizon", "brier", "log_loss"])
    return frame.groupby(["target", "horizon"], as_index=False).head(1).reset_index(drop=True)
