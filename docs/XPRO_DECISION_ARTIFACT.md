# XPro Decision Artifact

## Version

`xpro_decision_artifact_v1`

## Scope

The artifact is the auditable output of Track B protocol decisioning. It records the input order context, counterfactual protocol scores, selected protocol, Auto-X permission gate, lineage hashes, numeric policy, canonical hash, and optional HMAC signature.

## Required structure

| Field | Description |
|---|---|
| `artifact_version` | Must be `xpro_decision_artifact_v1` |
| `decision_id` | Stable decision identifier |
| `request_id` | Request replay token |
| `asof_utc` / `asof_epoch_ns` | UTC decision time in text and epoch-ns string forms |
| `numeric_policy` | Fixed-point scale contract |
| `input` | Quantized order context |
| `candidate_protocol_scores` | Counterfactual protocol scores and release-gate states |
| `model_outputs.execution_confidence` | Selected scorer output in fixed-point form |
| `decision` | Recommended protocol, release gate, human-review flag, reason codes |
| `auto_x_gate` | Auto-X-specific permission summary |
| `lineage` | Selected and candidate scorer artifact hashes |
| `evidence.artifact_hash` | Canonical SHA-256 over the artifact without hash/HMAC fields |
| `evidence.hmac` | Optional versioned HMAC over the hashed artifact payload |

## Hashing and signing

`build_xpro_decision_artifact()` uses `canonical_sha256(..., version="v2")`, which routes through the project RFC8785/JCS encoder. `sign_xpro_decision_artifact()` uses the existing FI HMAC key environment variables:

- `MRE_FI_HMAC_KEY_VERSIONS`
- `MRE_FI_HMAC_KEY`
- `MRE_FI_HMAC_ACTIVE_VERSION`

`verify_xpro_decision_artifact()` recomputes the canonical artifact hash and verifies HMAC when present. In non-production/dev mode, an unsigned artifact can verify by hash alone; in production HMAC mode, or when the CLI is run with `--require-hmac`, missing HMAC fails verification. Any payload mutation produces `verified=false`.

## Persistence

Artifacts are stored in `xpro_decision_artifacts` with:

- `decision_id`
- `request_id`
- `timestamp`
- `model_run_id`
- `recommended_protocol`
- `release_gate`
- `artifact_hash`
- `hmac_signature`
- `payload_json`
- `metadata_json`

## Tests

Validated by `tests/test_xpro_decision_artifact.py`, `tests/test_xpro_decision_api_endpoint.py`, `tests/test_xpro_decision_cli.py`, and `tests/test_storage_xpro_decision_artifacts.py`.
