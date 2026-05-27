# Design

This document summarizes the system design implied by the README and existing
architecture docs. Use it as the first design review document before changing
storage, scoring, validation, API, or XPro decisioning behavior.

## Design goal

Market Regime Engine is a governed probabilistic signal engine. It produces
market-regime, risk, confidence, and execution-intelligence artifacts with
point-in-time lineage, validation evidence, release gates, and reproducibility
envelopes.

It is not an autonomous trading system. Downstream systems remain responsible
for venue permissions, limits, supervision policy, portfolio construction, and
actual order routing.

## System boundaries

| Boundary | Responsibility | Production posture |
|---|---|---|
| Stable core | Storage, PIT lineage, macro/regime scoring, validation, release gates, fixed-income XPro surfaces | Production target |
| Experimental frontier | Bayesian MS-VAR, deep-kernel GP-BOCPD, DFM-MQ variants, neural/distributional heads, frontier diagnostics | Explicit opt-in |
| API/CLI surface | Operator and integration commands, `/v1` API, legacy API gate | `/v1` preferred |
| Evidence layer | Model-run envelopes, evidence packs, XPro decision artifacts, HMAC verification | Fail-closed |
| Reporting layer | Institutional reports, warehouse exports, method cards, validation reports | Review artifact |

## Data-plane design

```text
observations / vintages / FI feeds
        |
        v
Warehouse repository API
        |
        +--> macro tables
        +--> fixed-income tables
        +--> validation and release-gate tables
        +--> evidence-pack and XPro artifact tables
        |
        v
read models and scoring modules
        |
        v
validated artifacts and API/CLI responses
```

The warehouse facade remains public for compatibility, but the implementation
is split across focused registry, backend, repository, and pool modules. New
internal code should prefer the focused modules when it needs internals and the
facade only when it needs the public repository contract.

## Control-plane design

Control decisions are made through explicit gates:

1. `audit-vintage --enforce` gates feature lineage.
2. `validate` and forecast-comparison outputs gate model evidence.
3. `score-confidence`, drift, invalidation, coverage, MCS, DSR/PBO, and TCA
   metadata feed `release-gate`.
4. `model-run` records immutable run identity and payload hashes.
5. `verify-run` re-derives the reproducibility envelope.
6. FI evidence packs and XPro artifacts bind the decision payload to canonical
   hashes and optional HMAC signatures.

The release gate must be treated as a control-plane decision. If it holds,
operators inspect evidence; they do not bypass it by deleting missing columns or
changing profiles without a documented rationale.

## Point-in-time design

The core invariant is:

```text
observation_date <= as_of_date
vintage_date     <= as_of_date
```

`feature_asof_values` is the governed training and validation source. Legacy
feature paths exist for compatibility and debugging, not production promotion.

Design implication: any new feature, model, or execution signal must carry a
source timestamp or explicit lineage hash that proves the data was available at
decision time.

## XPro decisioning design

```text
ExecutionConfidenceRequest
        |
        v
counterfactual protocol scorer
        |
        +--> Auto-X candidate
        +--> RFQ candidate
        +--> Manual candidate
        |
        v
rank release-gate-passing candidates
        |
        v
xpro_decision_artifact_v1
        |
        v
persist, verify, API/CLI response
```

Design rules:

- Preserve legacy execution-confidence behavior.
- Do not duplicate scorer persistence rows during counterfactual ranking.
- Fail closed to Manual with `human_review_required=true` when all candidates
  fail governance or staleness.
- Publish new XPro artifacts with deterministic scaled integers and timestamp
  strings, not raw floats.
- Verify hash first; require HMAC when production HMAC policy or
  `--require-hmac` says so.

## API design

Use `/v1` for production:

- optional API-key auth with `X-API-Key` when `MRE_API_KEY` is set;
- request-size cap and validation through Pydantic schemas;
- cache behavior isolated behind versioned cache helpers;
- metrics through the in-process/Prometheus/OTel observability adapters.

The legacy API is intentionally gated by `MRE_LEGACY_API_ALLOW_UNAUTH=1`.
Deployments should fail fast rather than expose unauthenticated governance
artifacts by accident.

## Observability design

Every production integration should preserve:

- structured logs with request or run correlation;
- release-gate reason codes;
- HMAC verification failures;
- evidence-pack and XPro artifact hashes;
- warehouse write/read failure counters where available;
- CI artifacts for ruff, mypy, pytest, coverage, SBOM, license audit, and
  security scan outputs.

## Failure-domain design

| Failure | Expected behavior |
|---|---|
| Missing PIT lineage | Fail closed before training or scoring |
| Missing validation artifact | Certification profile holds |
| Stale signal | XPro scorer emits fail-closed reason and human review |
| Missing HMAC in production | Verification fails |
| Invalid XPro payload | API returns fail-closed error; Auto-X not permitted |
| Frontier dependency missing | Soft-degrade or explicit `ImportError` with install hint |
| Warehouse read/write failure | Surface 5xx or CLI non-zero; do not emit release approval |

## Change design checklist

Before merging a design change, verify:

- The new behavior has a PIT story.
- The output has a stable schema or a documented version.
- The release gate either consumes the evidence or explicitly does not need it.
- The API/CLI response has fail-closed behavior.
- Tests cover the new production path and the failure path.
- Docs point operators to exact commands and expected outputs.
