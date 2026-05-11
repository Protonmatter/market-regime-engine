# `liquidity_stress.py` — Liquidity Stress Scorer

## Purpose

Per-scope (`market` / `sector` / `rating` / `cusip`) liquidity stress
index. PR-4 ships the deterministic composite + label hysteresis;
the optional hierarchical Bayesian variant (PR-4) lands behind
`--use-hierarchical` and is opt-in until the validation harness
clears it.

## Inputs

`score_liquidity_stress(features, *, scope_type, scope_id, asof,
model_run_id=None, release_gate=True, profile="production",
prev_label=None, weights=None) -> LiquidityStressOutput`

Required feature columns:

- `bid_ask_bps`, `trade_count_velocity`,
  `time_since_last_trade_seconds`, `volume_to_adv`,
  `dealers_requested`, `dealers_responded`,
  `quote_dispersion_bps`, `amihud_illiquidity`,
  `axe_freshness_hours`.

PIT-asserted via `assert_pit_safe(...)`.

## Outputs

`LiquidityStressOutput` with `liquidity_index` (0–100),
`liquidity_label` (`LiquidityLabel`), `confidence`, `drivers`
tuple, plus the governance triple.

## Validation rules

1. `NAN_FAILS_PIT_AUDIT` is the FI default — a CUSIP with sparse
   data does not silently emit "Normal".
2. Hysteresis: asymmetric enter / exit thresholds (e.g. enter
   "Elevated Stress" at 60, exit at 50).
3. Scope-aware: `latest_liquidity_stress_score(warehouse, *,
   scope_type, scope_id)` returns the latest row per scope; the
   API endpoint surfaces `metadata.signal_age_seconds`.

## References

- AGENT.md PR-4 + INSTRUCTIONS.md §6.2.
- Deep research §2 (hierarchical Bayesian liquidity).
