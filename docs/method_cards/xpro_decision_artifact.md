# XPro Decision Artifact

## Production status

Production-track Track B evidence component for signed execution-intelligence decisions.

## Module path

`src/market_regime_engine/fixed_income/xpro_decision.py`

## Mathematical equation

The selected protocol is inherited from `ProtocolRecommendation`. Numeric outputs are quantized:

```text
score_ppm = round_half_even(score * 1,000,000)
slippage_q4 = round_half_even(slippage_bps * 10,000)
price_q6 = round_half_even(price * 1,000,000)
notional_cents = round_half_even(notional * 100)
```

The artifact hash is:

```text
sha256(RFC8785_JCS(artifact_without_artifact_hash_or_hmac))
```

The HMAC is:

```text
HMAC_SHA256(versioned_key, RFC8785_JCS(artifact_without_hmac))
```

## Inputs

- `ExecutionConfidenceRequest`
- `request_id`
- optional `decision_id`
- warehouse-backed protocol recommendation
- FI HMAC key environment variables when signing is enabled

## Outputs

- `xpro_decision_artifact_v1`
- fixed-point input and model outputs
- candidate protocol scores
- decision and Auto-X gate sections
- lineage hashes
- canonical artifact hash
- optional HMAC signature

## Assumptions

- Internal scorer math may remain float-based.
- Published XPro artifact payloads must not contain raw float values.
- Metadata is represented through hashes instead of raw embedded values.

## Failure modes

- Non-finite numbers, invalid prices, negative money, or naive timestamps fail validation.
- Tampered payloads fail canonical hash verification.
- Missing HMAC key in non-production mode leaves artifacts unsigned and hash-verifiable; production HMAC enforcement remains controlled by the FI evidence-pack key policy.

## Diagnostics

- `evidence.artifact_hash` identifies the canonical payload.
- `evidence.hmac.key_version` identifies the signing key version.
- `lineage.candidate_execution_confidence_artifact_hashes` links every counterfactual candidate to scorer evidence.
- `decision.reason_codes` captures ranked or fail-closed selection.

## Release-gate requirements

- `decision.release_gate` is true only when the selected recommendation passed scorer release gates.
- `auto_x_gate.permitted` is true only when Auto-X is selected, release-gated, and does not require human review.
- Certification release consumes realized-outcome validation metadata persisted by `fi-validate-execution-confidence`.

## Tests that validate it

- `tests/test_xpro_decision_artifact.py`
- `tests/test_xpro_decision_api_endpoint.py`
- `tests/test_xpro_decision_cli.py`
- `tests/test_storage_xpro_decision_artifacts.py`
- `tests/test_execution_validation_certification_cli.py`

## Known limitations

- The artifact does not execute orders or prove venue acceptance.
- It does not embed raw free-form metadata to preserve the no-float artifact contract.
- HMAC verification requires the relevant FI HMAC key version to be available in the verifier environment when the artifact is signed or strict HMAC verification is required.
