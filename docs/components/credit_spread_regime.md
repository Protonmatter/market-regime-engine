# `credit_spread_regime.py` — Credit Spread Regime Scorer

## Purpose

Deterministic explainable composite scorer for the credit-spread
regime index. Returns 0–100 score + label + drivers + governance
triple. PR-3 ships the deterministic baseline; v1.5.1+ may add a
calibrated model variant behind a feature flag.

## Inputs

`score_credit_regime(features, *, asof, model_run_id=None,
release_gate=True, profile="production", weights=None,
prev_label=None) -> CreditRegimeOutput`

- `features` — wide DataFrame indexed by date with the columns:
  `ust_slope`, `ust_curvature`, `cdx_ig_5y`, `cdx_hy_5y`, `vix`,
  `move`, `etf_prem_disc`, ...
- `asof` — UTC `pd.Timestamp` decision cutoff (PIT-asserted).
- `model_run_id` — optional; auto-generated if omitted.
- `release_gate` — fail-closed flag.
- `weights` — optional override for the five-component weighted
  composite.

## Outputs

`CreditRegimeOutput` carrying `regime_score` (0–100),
`regime_label` (`RegimeLabel`), `confidence` (0–1), `drivers`
tuple, `component_scores` dict, plus the governance triple.

## Validation rules

1. `assert_pit_safe(feature_timestamp, decision_timestamp)` rejects
   future-dated features.
2. Missing input columns trigger `PitAuditFailure` so
   `release_gate=False` propagates rather than emitting a fake
   "Normal" score.
3. `regime_score` is bounded `[0, 100]`; the label bucket function
   is `regime_label_from_score`.
4. Hysteresis: `classify_with_hysteresis` retains the previous
   label when the new score is within the asymmetric band around
   the boundary.

## References

- AGENT.md PR-3 + INSTRUCTIONS.md §6.1.
- `docs/V1_5_FIXED_INCOME_RCIE.md` PR-3 section.
