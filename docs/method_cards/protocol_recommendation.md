# Protocol Recommendation

## Production status

Production-track Track B component. It is additive to the legacy execution-confidence scorer and does not persist scorer rows during counterfactual ranking.

## Module path

`src/market_regime_engine/fixed_income/protocol_recommendation.py`

## Mathematical equation

For candidate protocol `p`, score:

```text
s_p = score_execution_confidence(request with protocol=p).confidence_score
```

Eligible candidates satisfy:

```text
release_gate_p = true
```

The recommendation is:

```text
argmax_p (s_p, deterministic_tie_break_p)
```

Default tie-break order is `RFQ`, `Auto-X`, `Manual`; explicit candidate order overrides it.

## Inputs

- `ExecutionConfidenceRequest`
- candidate protocol labels: `Auto-X`, `RFQ`, `Manual`
- warehouse with point-in-time credit-regime and liquidity-stress rows
- optional scorer profile, weights, model run id, and calibration flag

## Outputs

- `ProtocolRecommendation`
- per-candidate `ProtocolScore`
- recommended protocol
- selected scorer response
- release-gate state
- fail-closed reason codes

## Assumptions

- Existing scorer release-gate and staleness checks remain authoritative.
- Candidate protocols are evaluated independently by copying only the protocol field.
- Recommendation itself is a decisioning layer, not a persistence path for scorer predictions.

## Failure modes

- No candidate passes release gates: return `Manual`, set `human_review_required=true`, and emit `no_candidate_release_gate_passed`.
- Missing upstream FI signals: scorer returns fail-closed stale/no-data responses.
- Invalid protocol set: empty candidates raise validation errors.

## Diagnostics

- Candidate score list shows protocol, score, release-gate state, action, and lineage hash.
- Reason codes distinguish ranked selection from fail-closed fallback.
- No rows are written to `execution_confidence_predictions` by recommendation.

## Release-gate requirements

- At least one candidate must have `release_gate=true`.
- Upstream credit and liquidity rows must be release-gated and within staleness thresholds.
- Manual fallback is not a release approval; it is a human-review fail-closed state.

## Tests that validate it

- `tests/test_protocol_recommendation.py`
- `tests/test_xpro_decision_artifact.py`
- `tests/test_xpro_decision_api_endpoint.py`
- `tests/test_xpro_decision_cli.py`

## Known limitations

- It ranks only the protocol labels supplied to the scorer.
- It does not optimize dealer choice, venue selection, child-order scheduling, or quantity slicing.
- Calibration quality depends on the existing execution-confidence calibration tables.
