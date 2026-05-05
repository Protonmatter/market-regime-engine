# Rust Extension (mre_rust_ext)

Validated PyO3 hot-path kernels for the Market Regime Engine. The Python
package (`market_regime_engine`) remains authoritative; the Rust kernels
are opt-in, parity-tested replacements for the inner loops that profile
hottest.

## What ships

| Kernel | Python reference | Status |
|---|---|---|
| `bocpd_diag_update` | `market_regime_engine.bocpd.DiagonalStudentTBOCPD` (per-step inner loop) | parity-tested at `atol=1e-9` |
| `wfst_viterbi_decode` | `market_regime_engine.wfst.RegimeWFST.decode` | parity-tested (path equality) |
| `population_stability_index_kernel` | `market_regime_engine.drift.population_stability_index` | parity-tested at `atol=1e-12` |
| `rolling_mahalanobis_distance_kernel` | `market_regime_engine.changepoint.RollingMultivariateChangePoint` (per-row distance) | parity-tested at `atol=1e-9` |
| `bocpd_change_probability` | (legacy v0.7 placeholder) | retained for back-compat only |
| `transition_cost` | (legacy v0.7 placeholder) | retained for back-compat only |

## Build and install

```bash
pip install maturin
cd rust_ext
maturin develop --release        # local dev install
# or:
maturin build --release           # wheel under target/wheels/
```

After a successful build, `python -c "import mre_rust_ext"` succeeds and
the Python wrapper in `market_regime_engine.rust_kernels` returns the
Rust implementation. Without a built extension, the wrapper falls back
to the Python reference and the parity tests skip cleanly.

## Run parity tests

```bash
pytest tests/test_rust_parity.py -q
# Or, with the `rust` marker explicitly:
pytest -q -m rust
```

Promotion criterion for a Rust kernel: parity tests pass **and**
`mre bench` shows a measurable speedup at every problem size in
`bench.py`.

## Bench

```bash
mre bench --out data/bench.csv
```

The harness reports `elapsed_seconds` and `peak_memory_mb` for each
kernel at three problem sizes (`small` / `medium` / `large`). The
`implementation` column distinguishes `python_reference` (default) from
`rust_extension` (when the bench is invoked with the extension built
and the wrapper is plumbed through).

## Engineering rule

Do not promote a Rust kernel without parity tests passing against the
Python reference implementation. Speed without parity is just faster
wrongness, which humanity already mastered.

## Future kernels (not yet implemented)

- Multivariate **NIW** BOCPD update (the diagonal Student-t kernel is in;
  the NIW posterior with full Cholesky update is the next step).
- Markov-Switching VAR Hamilton-Kim filter inner loop.
- Monte Carlo mixture-distribution simulation for ensemble forecast
  evaluation.
- Combinatorial Purged CV index generator (parallel-friendly Rust core).

When adding a new kernel, write the Python reference first, ship the
parity test, then implement the Rust kernel and gate promotion on the
parity test. Anything else is wishful thinking with a Cargo.toml.
