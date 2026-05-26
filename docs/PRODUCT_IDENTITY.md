# Product identity: XPro 2.0 Execution Intelligence Layer

## Narrowed identity

Market Regime Engine Track B is an **XPro 2.0 execution intelligence layer** for fixed-income execution workflows.

It is not a broker, OMS, EMS, trading venue, portfolio optimizer, or autonomous order router.

```text
point-in-time FI signals and order context
        |
execution-confidence scorer
        |
counterfactual protocol ranking
        |
fixed-point XPro decision artifact
        |
release gate, HMAC evidence, validation report
        |
Auto-X / RFQ / manual execution operators and downstream APIs
```

The layer answers:

1. Which eligible protocol has the strongest governed execution-confidence evidence?
2. Did the selected protocol pass upstream release gates and staleness checks?
3. What fixed-point, HMAC-verifiable decision record should downstream systems consume?
4. Did realized outcomes validate the execution-confidence model for certification release?

## Execution decision contract

The canonical Track B output is `xpro_decision_artifact_v1`.

| Field | Meaning |
|---|---|
| `decision_id` | Immutable XPro decision identifier |
| `request_id` | Client/order request replay token |
| `asof_epoch_ns` | UTC decision timestamp as an epoch-nanosecond string |
| `numeric_policy` | Fixed-point scaling and canonical JSON policy |
| `input` | Quantized order context and metadata hash |
| `candidate_protocol_scores` | Counterfactual scores for candidate protocols |
| `decision.recommended_protocol` | Selected protocol: `Auto-X`, `RFQ`, or `Manual` |
| `decision.release_gate` | Final release-gate state |
| `auto_x_gate` | Whether Auto-X is selected and permitted |
| `lineage` | Selected and candidate scorer artifact hashes |
| `evidence.artifact_hash` | RFC8785/JCS SHA-256 hash |
| `evidence.hmac` | Optional FI HMAC signature |

## Adapter posture

Track B surfaces recommendations and evidence, not venue-side execution.

- Auto-X consumers must still enforce venue permissions, credit limits, and human-supervision policy.
- RFQ consumers receive a ranked recommendation plus reason codes, not a dealer allocation.
- Manual fallback is fail-closed when every candidate fails governance or staleness.
- Legacy `/v1/execution_confidence` remains available for compatibility; strict fixed-point output applies to the XPro surfaces.

## Production-mode contract

When `MRE_ENV=production`:

- API authentication remains mandatory unless explicitly overridden.
- FI HMAC signing follows the existing production HMAC requirements.
- Redis/API cache misconfiguration fails closed where configured.
- XPro artifacts must be persisted with lineage hashes and release-gate state.

## Evidence-pack contract

XPro decisions use the same audit posture as FI evidence packs:

- canonical JSON uses the project RFC8785/JCS v2 encoder;
- decision hashes are SHA-256 values with the `sha256:` prefix;
- optional HMAC signatures bind the full artifact hash payload;
- realized-outcome validation emits certification metadata consumed by `evaluate_release_gate(profile="certification")`.
