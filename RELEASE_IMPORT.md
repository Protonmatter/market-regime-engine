# Market Regime Engine v1.4.1 import manifest

This repository was seeded from the reviewed `market-regime-engine-v1.4.1-release.zip` bundle.

## Current import state

The repository currently contains a seed import through the GitHub connector. The connector can write text files and Git tree objects, but it does not expose a native release-asset upload path for binary ZIP/WHL/TAR artifacts in this session.

## Reviewed bundle contents

```text
market_regime_engine-1.4.1-py3-none-any.whl
market_regime_engine-1.4.1.tar.gz
market-regime-engine-1.4.1-source.zip
V1_4_1_FIXES.md
v141_demo_verify_run_extra_drift.json
v141_demo_release_gate_default_production.json
```

## Verified artifact hashes from review

```text
wheel:      0e37e3a028a5b777e8f2f48538796e36074282e5cfba1fc8e8dbb2e2d7de2c3d
sdist:      97bbc64fca739707f3e5546e4fc7f95c7ac260b7ae33e9448d8afd8aaeb4dbfa
source zip: 3133ae355d97fa287fa8b380d764e1bca9ec91701457ee01dbb183f92ad43561
```

## Known release-integrity note

The external `V1_4_1_FIXES.md` hash table matched the shipped artifacts during review, but the copy inside the source tree under `docs/V1_4_1_FIXES.md` had stale artifact hashes. The recommended next patch is v1.4.2 with a release-bundle self-audit command and source-doc hash consistency check.

## Recommended completion path

Complete the full source import from a local clone using the extracted `market-regime-engine-1.4.1-source.zip`, then run the repository CI from GitHub Actions and attach the resulting test, lint, type-check, SBOM, and security artifacts to the release.
