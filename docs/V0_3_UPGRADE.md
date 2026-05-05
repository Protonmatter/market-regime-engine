# v0.3 Upgrade Notes

## Purpose

v0.3 upgrades the Market Regime Engine from a baseline macro dashboard into a more defensible probabilistic modeling scaffold.

## Added modules

| Module | Purpose |
|---|---|
| `bocpd.py` | Online structural-break detection using BOCPD mechanics |
| `hmm.py` | Regime posterior probabilities from domain stress vectors |
| `wfst.py` | Formal weighted regime-transition grammar |
| `baselines.py` | Naive benchmark models |
| `promotion.py` | Model promotion gates |
| `point_in_time.py` | Conservative release-lag and vintage/as-of controls |

## Regime pipeline

```text
features -> domain scores -> BOCPD CP_t -> HMM gamma_t -> event labels -> WFST decoded regime
```

## Validation pipeline

```text
features + targets
  -> candidate walk-forward models
  -> naive benchmarks
  -> validation metrics
  -> promotion gates
```

## Promotion rules

Candidate binary models are not promoted unless they beat the best naive benchmark on Brier score and log loss while staying below ECE limits.

## Future v0.4 targets

1. Full ALFRED vintage ingestion jobs.
2. Exact official release-calendar metadata.
3. Bayesian model averaging / dynamic stacking.
4. Real recession label integration.
5. Rust BOCPD/WFST kernels through PyO3.
6. Historical analog distance engine with Mahalanobis regime weighting.
7. Driver attribution report per forecast.
