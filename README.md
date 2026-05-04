# Market Regime Engine v1.4.1

Python-first, Rust-ready probabilistic macro/market regime intelligence engine with a 2026-2027-frontier modeling layer.

> **v1.4.1** is a release-integrity patch on top of v1.4.0. It closes the audit-grade hygiene gaps found in v1.4.0: README/wheel metadata identity drift, `verify_run` drift checks for `extra` and `rng_seeds`, permissive release-gate defaults, and platform lockfile coverage for optional extras.

This repository was initialized from the reviewed v1.4.1 source release bundle.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
mre --help
```

## Release posture

Treat this build as a governed pre-production candidate until the remaining release-bundle integrity items are patched and a full CI run is reproduced from this repository.

## Next hardening target

v1.4.2 should add a release-bundle self-audit command, fix source-tree release-doc hash consistency, and include machine-verifiable CI artifacts.
