# v1.6.0 frontier math hardening

This build implements the first concrete pass from the deep-research roadmap for pushing Market Regime Engine toward frontier-grade validation and algorithmic rigor.

## Implemented

### Adapter correctness

- Strict boolean parsing replaces Python truthiness for release-gate fields.
- `"False"`, `"false"`, `"0"`, `0`, and `False` now remain false.
- LEAN custom-data stub uses `csv.reader` so quoted metadata with commas does not corrupt columns.
- vectorbt adapter exposes `entry_score` to preserve uncertainty rather than only using hard labels.
- PyPortfolioOpt adapter now returns `allocation_allowed` and `block_reason` instead of hiding governance blocks inside expected-return vectors.

### Evidence-pack hardening

Evidence packs now include:

- engine version
- git SHA and dirty state
- Python/platform metadata
- command line
- lockfile hashes
- redacted source map by default
- optional required HMAC signature
- unsafe `force=True` deletion guardrails

### Anti-overfit controls

New module: `market_regime_engine.frontier.overfit_control`.

Implemented:

- Deflated Sharpe Ratio approximation
- Probability of Backtest Overfitting via CSCV-style fold tournaments
- Minimum Track Record Length approximation
- Pre-registered model tournament manifest writer

### Online conformal primitives

New module: `market_regime_engine.frontier.online_conformal`.

Implemented:

- EnbPI-style residual interval wrapper
- Strongly adaptive ACI controller with expert gammas
- AgACI-style wrapper over adaptive ACI experts

## What this does not claim

This does not yet make the engine a fully validated frontier model. It adds the controls needed to make future frontier-model claims harder to fake.

## Next required work

1. Rebase the branch onto current `main`.
2. Restore the full operational README and update it to v1.6.0.
3. Add CI gates for the new v1.6 tests.
4. Add synthetic regime-recovery and change-point benchmark suites.
5. Add signed evidence-pack enforcement to release gates.
6. Add IMM/GPB regime filters and switching-covariance SVAR.
7. Add foundation-model challenger adapters behind PIT-safe release gates.
