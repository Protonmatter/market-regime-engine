# Handoff

This handoff reflects the current README-facing build shape: v1.6.1 stable
core, v1.5 fixed-income RCIE/XPro adapter, Track B XPro decision artifacts,
and certification-profile governance.

## Start here

Read these in order:

1. `README.md` - release narrative, feature inventory, quick start.
2. `docs/WORKFLOW.md` - executable operator workflows.
3. `docs/DESIGN.md` - system boundaries and failure-domain design.
4. `docs/MATHEMATICS_USED.md` - mathematical methods and validation evidence.
5. `docs/INSTRUCTIONS.md` - setup, validation, API, XPro, and publishing.
6. `docs/MATH_METHODS.md` and `docs/method_cards/` - method-level assumptions
   and production status.

## Current product shape

Market Regime Engine is a Python-first, Rust-ready market-regime intelligence
engine with a governed fixed-income XPro execution-intelligence layer.

Primary outputs:

- macro regime and change-point probabilities;
- drawdown, recession, hazard, quantile, and distributional forecasts;
- calibrated confidence, drift, invalidation, release-gate, and promotion rows;
- point-in-time vintage lineage and reproducibility envelopes;
- fixed-income credit-regime, liquidity-stress, execution-confidence, TCA, and
  XPro protocol-decision artifacts;
- HMAC-verifiable evidence packs and XPro decision artifacts.

## Stable-core versus frontier boundary

Stable core includes:

- warehouse repository APIs;
- point-in-time materialization and audits;
- HMM/MS-VAR/WFST/regime scoring paths;
- validation and release-gate controls;
- fixed-income XPro decision surfaces;
- API/CLI/reporting interfaces.

Experimental frontier includes:

- Bayesian MS-VAR;
- deep-kernel GP-BOCPD;
- frontier nowcasting and distributional heads;
- optional dependency paths requiring explicit install and review;
- diagnostics fenced by `MRE_ENABLE_EXPERIMENTAL_FRONTIER=1`.

Do not let stable-core release evidence silently depend on frontier behavior.

## Operational guardrails

- Always run `audit-vintage --enforce` before training, validation, or scoring.
- Treat `release-gate` holds as blockers unless a documented review explicitly
  changes the gate profile or threshold.
- Use `profile="certification"` for audit-grade stable-core evidence.
- Use `mre verify-run` before publishing reports or release artifacts.
- Use `/v1` API in production; legacy API import requires
  `MRE_LEGACY_API_ALLOW_UNAUTH=1`.
- In production FI/XPro mode, configure HMAC key versions and require strict
  artifact verification.
- Auto-X consumers must enforce venue, credit, and human-supervision policy
  outside this engine.

## Quick local validation

```powershell
python -m compileall -q src tests
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/market_regime_engine --show-error-codes --pretty
python -m pytest --collect-only -q
python -m pytest -q --durations=25
```

Track B focused validation:

```powershell
python -m pytest -q tests/test_numeric_contracts.py tests/test_protocol_recommendation.py tests/test_xpro_decision_artifact.py
python -m pytest -q tests/test_xpro_decision_api_endpoint.py tests/test_xpro_decision_cli.py
python -m pytest -q tests/test_execution_validation_certification_cli.py tests/test_storage_xpro_decision_artifacts.py
python -m pytest -q tests/test_execution_confidence.py tests/test_execution_confidence_api_endpoint.py tests/test_execution_confidence_cli.py
python -m pytest -q tests/test_canonical_json_rfc8785.py tests/test_fixed_income_evidence_pack_hmac.py tests/test_certification_release_and_execution_validation.py
python -m pytest -q tests/test_method_cards_docs_audit.py
```

## CI expectations

GitHub Actions should gate:

- version sanity across `pyproject.toml`, `__init__`, README, and tags;
- lockfile sanity and lockfile hash drift;
- ruff check and ruff format;
- pytest on Python 3.11/3.12 across Linux and Windows;
- security extras tests;
- mypy;
- smoke pipeline;
- golden trace;
- package sanity;
- SBOM, license audit, bandit, and best-effort Rust wheels.

If Actions cannot start because of GitHub billing/account limits, local
validation is still useful, but do not call the build CI-certified.

## Handoff risks

| Risk | Mitigation |
|---|---|
| README release narrative is dense | Use the new workflow/design/instruction docs as the operator entry point |
| Live vintage data can leak if lineage is skipped | Require `audit-vintage --enforce` and PIT training paths |
| Frontier methods can look production-ready by import path alone | Keep frontier boundary explicit and gated |
| XPro artifacts can be consumed without verification | Require artifact hash/HMAC verification in production |
| Release gates can be weakened by profile drift | Record resolved profile and reason codes with every release decision |
| Legacy API can expose unauthenticated outputs | Keep `MRE_LEGACY_API_ALLOW_UNAUTH` unset outside compatibility testing |

## Next high-ROI work

1. Keep README as a high-level product narrative and move operator detail into
   the focused docs created in this pass.
2. Add a CI check that verifies the README documentation map links exist.
3. Continue consolidating report-writer shims into one maintained reporting
   surface.
4. Add richer runbooks for HMAC key rotation, failed release-gate triage, and
   warehouse backup/restore.
5. Keep method cards synchronized with any model or validation behavior change.
