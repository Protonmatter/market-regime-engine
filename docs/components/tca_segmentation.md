# `tca_segmentation.py` — Regime-Aware TCA Segmentation

## Purpose

Tag every trade with the prevailing credit-regime / liquidity /
execution-confidence context (PIT-safe), compute the 11 TCA metrics
in `Decimal` precision, and aggregate by the 9 spec-canonical
segmentation dimensions to the `tca_regime_segments` warehouse
table.

## Inputs

- `tag_trade_with_regime_context(trade: TradeRecord, *, warehouse,
  use_hysteresis=True, tolerance="5min") -> TaggedTrade`.
- `aggregate_tca_by_regime(trades, *, dimensions=DIMENSION_COLUMNS,
  metrics_names=TCA_METRICS, soft_weighting=False) -> pd.DataFrame`.
- `materialize_tca_segments_for_day(warehouse, *, date,
  soft_weighting, use_hysteresis, model_run_id) -> int`.

## Outputs

One `TcaRegimeSegment` row per `(dim-combo, metric_name)` per
`(model_run_id, timestamp)` carrying `metric_value` (Decimal),
`sample_count`, and `metadata_json` (drop counts + rejection
reasons).

## Validation rules

1. `outcome.observed_at > decision.timestamp` strict inequality at
   row construction (PR-6 Q-2).
2. NaN drops at the aggregation boundary; emit
   `tca_dropped_rows_total{metric=...}` counter.
3. Decimal arithmetic for stable bps accumulation at $1B daily
   notional × 0.5 bps target precision.

## References

- AGENT.md PR-6 + INSTRUCTIONS.md §6.4.
- Deep research §4 (Regime-Aware TCA Segmentation).
