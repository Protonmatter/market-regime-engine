# v1.2.1 Patch Bundle — Production-Readiness Closure

> The v1.2 ZIP shipped successfully but a third-party reviewer caught a
> short list of production-readiness gaps: stale package metadata, an
> editable-install line in `requirements-lock.txt`, fail-open PIT
> training, a `verify_run` skip set that ignored the training audit, an
> unauthenticated legacy `api.py`, a quadratic-ish as-of materialisation
> loop, no top-level LICENSE, and a narrative "Build status" block. This
> document is the canonical changelog for what landed in v1.2.1 to close
> them.

The historical upgrade documents (`V1_0_UPGRADE.md`, `V1_1_FIXES.md`,
`V1_2_FRONTIER.md`, `V0_*_UPGRADE.md`) are intentionally untouched.
v1.2.1 is a focused patch on top of `904d058` (v1.2 frontier commit).

---

## Per-item table

| ID | What changed | Files (key lines) | Severity addressed | New regression test(s) |
|---|---|---|---|---|
| **A** | Version bump `1.0.0-dev → 1.2.1` everywhere it lives. `pyproject.toml [project] version`, `src/market_regime_engine/__init__.py __version__`, `api.py /health`, `api_v1.py /health` and `app(version=…)` all flow from a single source of truth. CI gains a `version-sanity` job that compares `pyproject` ↔ `__init__` ↔ `${{ github.ref_name }}` (when triggered by a tag) and fails on mismatch. | `pyproject.toml:7`, `src/market_regime_engine/__init__.py:8`, `src/market_regime_engine/api.py:64,73`, `.github/workflows/ci.yml:11-40` | Reviewer item 1 — package metadata lied about the shipped version | `tests/test_version_sanity.py` (5 tests) |
| **B** | `requirements-lock.txt` regenerated from `pyproject.toml` extras (`dev`, `dashboard`, `analytics`, `nowcast`, `observability`) via `pip-compile`. Zero `-e ` lines, zero local paths, internally consistent. New CI `lockfile-sanity` job greps the file (skipping comment lines) for `-e `, `c:\`, `/Users/`, `/home/`, `file://` and fails on any match. README documents that the lockfile is the canonical pinned manifest and how to regenerate it. The `frontier` extra (statsmodels, ngboost, torch) is intentionally excluded — torch wheels are platform-specific. | `requirements-lock.txt:1-292`, `.github/workflows/ci.yml:42-79`, `README.md:72-104` | Reviewer item 2 — editable Windows path `-e c:\users\mkang\…` baked into the lockfile broke `pip install` on every machine but the developer's | (covered by the CI `lockfile-sanity` job; nothing testable in pytest) |
| **C** | `training_data.load_training_panel` gains `allow_legacy_fallback: bool = False`. PIT mode with empty `feature_asof_values` now raises `RuntimeError` by default; the legacy fallback is preserved as an explicit opt-in that stamps `audit["mode_used"] = "legacy_fallback_explicit"` and `audit["fallback_authorized"] = True`. `cli._resolve_allow_legacy_fallback` threads the flag through `train-baseline` and `validate`; setting `--allow-legacy-fallback` without `--legacy-features` logs a WARNING. CLI surfaces the audit dict in error messages when training is empty. | `src/market_regime_engine/training_data.py:36-149`, `src/market_regime_engine/cli.py:188-330` | Reviewer item 3 — fail-open in the most sensitive part of the pipeline | `tests/test_training_data.py::test_pit_mode_fails_closed_by_default_when_asof_empty`, `::test_pit_mode_legacy_fallback_explicit_path_records_audit` (plus existing tests refactored to assert `fallback_authorized` everywhere) |
| **D** | `model_runs.verify_run` skip set narrowed from `{"rng_seeds", "extra"}` to `{"rng_seeds"}`. The `extra.training_audit` dict is now structurally compared: `training_mode_drift` is appended to `differences` whenever stored `mode_used != "point_in_time"`, and `legacy_fallback_authorized` is appended to `warnings` (non-fatal advisory) when the operator opted into the v1.2.1 explicit fallback. CLI `verify_run_cmd` prints warnings on stderr so the report's stdout JSON stays parseable. | `src/market_regime_engine/model_runs.py:262-322`, `src/market_regime_engine/cli.py:855-882` | Reviewer item 4 — verify_run could approve a run that was secretly trained on revised macro data | `tests/test_verify_run_training_audit.py` (8 tests covering PIT pass, legacy fail, fail-closed audit fail, authorized-fallback warning, missing audit, source-skip-set guard, round-trip) |
| **E** | Legacy `api.py` now refuses to import unless `MRE_LEGACY_API_ALLOW_UNAUTH=1`. Check fires at module load time so a misconfigured `uvicorn market_regime_engine.api:app` deploy fails fast rather than silently exposing governance artifacts. `app.version` and `/health` payload pull from `__version__`. README documents the env-var ack as a v1.2.1 breaking change. | `src/market_regime_engine/api.py:1-66`, `README.md:135-160`, `README.md:106-115` | Reviewer item 6 — unauthenticated legacy mount exposed `/regime/latest` etc. with no API-key check | `tests/test_api_legacy_gate.py` (6 tests covering missing env, env=1, env=truthy-but-not-1, /health version, end-to-end /regime/latest, instructive error message) |
| **F** | Apache-2.0 `LICENSE` added at repo root. `pyproject.toml [project]` declares `license = { text = "Apache-2.0" }` plus the matching `License :: OSI Approved :: Apache Software License` classifier and the standard `Development Status / Operating System / Topic` set. SPDX header `# SPDX-License-Identifier: Apache-2.0` inserted into all 82 `.py` files under `src/market_regime_engine/` via `scripts/add_spdx_headers.py` (idempotent — running again is a no-op). | `LICENSE:1-202`, `pyproject.toml:6-26`, `src/market_regime_engine/**/*.py` (82 files), `scripts/add_spdx_headers.py:1-92`, `MANIFEST.in:6,8` | Reviewer item 7 — no top-level LICENSE (institutional consumers can't legally use the package) | `tests/test_package_metadata.py::test_metadata_declares_apache_license` |
| **G** | `materialize_feature_asof_values` rewritten as a vectorised set-based pipeline. Detects whether the input has any revisions per `(series_id, observation_date)` and dispatches: in the no-revisions fast path the panel and feature transforms are computed exactly once on the full data, with lineage attached via a single `merge_asof`; in the revisions path the per-as-of rebuild is preserved but lineage is also vectorised and the panel build is cached when consecutive as-of dates share the same legal table. Output is byte-identical to the pre-v1.2.1 frame. New helper `latest_vintage_observations_per_asof_grid` exposed as the multi-date primitive. Architecture doc updated. | `src/market_regime_engine/asof.py:1-462`, `docs/ARCHITECTURE.md:104-135` | Reviewer item 5 — quadratic-ish runtime; smoke pipeline timed out | `tests/test_asof_perf.py::test_materialize_under_30s`, `::test_materialize_matches_legacy_per_loop_output`, `::test_latest_vintage_observations_per_asof_grid_returns_set_based_panel`, `::test_materialize_handles_empty_vintage_input`, `::test_materialize_with_revisions_path_handles_multi_vintage` (5 tests) |
| **H** | CI uploads `test-results.xml` (pytest junit), `ruff-results.json`, `mypy-results.json`, `coverage.xml`, `bench.csv` as workflow artifacts. New `scripts/refresh_build_status.py` rewrites a `<!-- ci-status-start --> ... <!-- ci-status-end -->` sentinel block in `README.md` from those artifacts. `--check` mode is non-destructive (exits 1 when the block diverges); the `refresh-build-status` CI job runs after `test`/`lint`/`mypy`/`bench` on `push` to `main` and commits the refreshed block when non-empty. | `.github/workflows/ci.yml:81-280`, `scripts/refresh_build_status.py:1-186`, `README.md:52-70` | Reviewer item 9 — narrative build-status block; auditors had to take the README on faith | (covered by `refresh_build_status.py --check` invocation in CI; the sentinel block guarantees the section is auditable) |
| **I** | New CI `package-sanity` job builds the wheel + sdist via `python -m build`, installs the wheel into a fresh venv on Ubuntu and Windows, then runs `tests/test_package_metadata.py` from a directory **outside** the source tree so the test exercises the installed package, not the editable source. `MANIFEST.in` ensures `LICENSE`, `README.md`, `requirements-lock.txt`, `config/`, `docs/`, `scripts/`, `MANIFEST.in`, `pyproject.toml` ship inside the sdist. `build` added to the `[dev]` extra. | `.github/workflows/ci.yml:222-275`, `MANIFEST.in:1-31`, `tests/test_package_metadata.py:1-119`, `pyproject.toml:38-43` | Reviewer-implied — wheel-install round-trip wasn't exercised, so any new metadata regression would only be caught in production | `tests/test_package_metadata.py` (5 tests covering metadata version, license declaration, clean import, public symbols, console script) |
| **J** | `build_zip.py` (the legacy v1.2 source-zip helper) preserved as a back-compat shim that delegates to `scripts/build_audit_zip.py`. Audit zip emits `dist/market-regime-engine-1.2.1-source.zip` with `.git/` preserved so `mre verify-run` works after extraction. New `scripts/build_release.py` is the dual-artifact driver: it runs `python -m build` (wheel + sdist) then `scripts/build_audit_zip.py` (audit zip), and prints SHA-256 for each artifact. New CI `release-artifacts` job runs `scripts/build_release.py --clean` on `workflow_dispatch` or any `v*` tag push and uploads all three artifacts. | `build_zip.py:1-19`, `scripts/build_audit_zip.py:1-160`, `scripts/build_release.py:1-128`, `.github/workflows/ci.yml:282-307` | Reviewer-adjacent — single-zip distribution made the wheel-vs-archive separation muddy | (covered by `package-sanity` for wheel install + audit zip is exercised by hand against the synthetic sample) |

---

## Test, lint, and mypy delta

| | Pre-v1.2.1 (v1.2 baseline) | v1.2.1 |
|---|---|---|
| `pytest tests/ -q -m "not slow"` | 162 passed, 1 skipped, 1 deselected | **192 passed**, 1 skipped, 1 deselected (+30 tests) |
| `ruff check src tests` | 0 | **0** |
| `ruff format --check src tests` | 0 | **0** (105 files clean) |
| `mypy src/market_regime_engine` | 35 | **35** (no regression; cap is ≤35) |
| End-to-end smoke `bootstrap-sample → … → verify-run` | `approved: true` | **`approved: true`** |
| Wheel install round-trip (`pip install dist/*.whl` → `pytest tests/test_package_metadata.py` from tempdir) | not exercised | **5 passed** |
| `materialize_feature_asof_values` wall-clock on synthetic sample (435 obs × 16 series × 8 transforms × 399 as-of dates) | **132.973 s** | **3.731 s** (≈35.6× faster) |

---

## Performance evidence — `materialize_feature_asof_values`

The reviewer's smoke pipeline timed out at this step; v1.2.1 brings the
synthetic-sample wall-clock from ~2 minutes to under 4 seconds while
producing **byte-identical** output. Both numbers are from
`scripts/compare_asof_implementations.py` on the same input
(`generate_sample_observations()` → `seed_vintage_observations_from_latest`
→ `materialize_feature_asof_values(..., min_history_months=36)`):

```
legacy elapsed: 132.973s   rows: 19551
fast   elapsed: 3.731s   rows: 19551
speedup: 35.6x
OK: vectorised output matches legacy per-loop output exactly.
```

The performance regression test
(`tests/test_asof_perf.py::test_materialize_under_30s`) sets the gate at
30 seconds — generous enough that a shared CI runner does not flake but
strict enough to catch any reintroduction of the per-asof Python loop.
The correctness regression test
(`tests/test_asof_perf.py::test_materialize_matches_legacy_per_loop_output`)
embeds a verbatim recreation of the pre-v1.2.1 implementation and
asserts equality (modulo float rounding within `1e-9`) on every output
row.

---

## verify-run fail-closed PIT demo (criterion 12)

The end-to-end demonstration that the fail-closed contract is wired
through the CLI:

```text
=== Step 1a: bootstrap-sample (NO materialize-asof-features) ===
Inserted 6960 sample observations into data\v121_demo.db

=== Step 1b: build legacy features (so legacy fallback has data to train on) ===
Built 21043 features

=== Step 2a: train-baseline WITHOUT --allow-legacy-fallback (must fail closed) ===
ERROR market_regime_engine.training_data :: PIT training failed closed because feature_asof_values is empty.
ERROR mre.cli :: train-baseline failed closed
POINT_IN_TIME mode requires non-empty feature_asof_values. Run `mre materialize-asof-features --write-features` first, or pass --allow-legacy-fallback to opt into the deprecated legacy path.
train-baseline (no flag) exit code: 1

=== Step 2b: train-baseline --allow-legacy-fallback (succeeds with audit) ===
WARNING mre.cli :: PIT path active but legacy fallback authorized as a safety net (--allow-legacy-fallback set without --legacy-features).
WARNING market_regime_engine.training_data :: POINT_IN_TIME mode active but legacy fallback explicitly authorized via allow_legacy_fallback=True. Training will proceed against the legacy features table.
INFO    mre.cli :: baseline training complete
Wrote 24 model outputs (mode=legacy_fallback_explicit, rows=21043, audit_path=data\training_audit.json)

=== Step 3: model-run ===
Wrote immutable model run rows: 1
... extra.training_audit captured: {"as_of_dates": 0, "fallback_authorized": true, "fallback_reason": "feature_asof_values empty", "mode": "point_in_time", "mode_used": "legacy_fallback_explicit", "rows": 21043} ...

=== Step 4: verify-run (MUST exit non-zero with training_mode_drift) ===
verify-run exit code: 2
```

The literal `mre verify-run --db data/v121_demo.db` JSON output (stdout):

```json
{
  "approved": false,
  "differences": {
    "training_mode_drift": {
      "expected": "point_in_time",
      "stored_mode": "legacy_fallback_explicit"
    }
  },
  "lockfile_present": true,
  "missing_envelope": false,
  "run_id": "13b81f812f960d2b",
  "warnings": [
    "legacy_fallback_authorized"
  ]
}
```

`differences.training_mode_drift` is present, `warnings` carries the
explicit `legacy_fallback_authorized` advisory, and the process exits
with code `2` — the verify-run gate refuses to approve a run that
trained on revised macro data, even when the fallback was authorized.

The verify-run JSON is also persisted at
[`docs/v121_demo_verify_run.json`](v121_demo_verify_run.json) for
auditors who want to diff it against future runs.

---

## Breaking-change advisory

Two changes in v1.2.1 are deliberately backwards-incompatible. Both
fail loud at the most useful moment (deploy time / training time)
rather than silent at the most expensive moment (production drift).

### 1. Legacy `api.py` requires `MRE_LEGACY_API_ALLOW_UNAUTH=1`

Pre-v1.2.1::

    uvicorn market_regime_engine.api:app
    # quietly serves /regime/latest, /model-outputs/latest, /release-gate/latest, ...
    # to anyone who can reach the port.

v1.2.1::

    uvicorn market_regime_engine.api:app
    # raises RuntimeError at module import time with a message
    # pointing at api_v1 and at the env-var override.

    MRE_LEGACY_API_ALLOW_UNAUTH=1 uvicorn market_regime_engine.api:app
    # works exactly as before, but the operator has acknowledged
    # in the deployment manifest that the surface is unauthenticated.

The recommended migration is to mount the v1 hardened app instead::

    MRE_API_KEY="rotate-me" uvicorn market_regime_engine.api_v1:app

`api_v1` honors `X-API-Key`, ships TTL caching, and exposes Prometheus
metrics. The legacy `api` mount is preserved only for back-compat with
read-only consumers that pre-date the v1 hardening.

### 2. PIT training fails closed without `--allow-legacy-fallback`

Pre-v1.2.1, a missing `materialize-asof-features --write-features` step
caused `train-baseline` and `validate` to silently swap the legacy
features table into PIT-mode training, log a warning, and continue. The
warning was easy to miss; the audit dict that recorded the swap was
embedded in `repro_envelope.extra` but `verify_run` skipped that field
entirely (item D). Net effect: a model could be trained on revised
macro data, recorded as PIT, and approved by verify-run — exactly the
look-ahead failure the router was supposed to eliminate.

v1.2.1 fails closed: PIT mode + empty `feature_asof_values` raises
`RuntimeError`. Operators who genuinely want the fallback (smoke tests,
bootstrap pipelines, demos) opt in with::

    mre train-baseline --db data/mre.db --allow-legacy-fallback
    mre validate --db data/mre.db --allow-legacy-fallback

The audit dict then records `mode_used = "legacy_fallback_explicit"`
and `fallback_authorized = True`. `mre verify-run` surfaces the
`training_mode_drift` difference (so the run is not approved) and a
non-fatal `legacy_fallback_authorized` warning so a change-management
gate can see the deliberate downgrade.

---

## Upgrade notes — v1.2 → v1.2.1

For an operator who already runs v1.2:

1. `git pull` (or extract the v1.2.1 source zip).
2. `pip install -e ".[dev,dashboard,analytics]"` to pick up the new
   `[project]` metadata (Apache-2.0 license, version `1.2.1`, `build`
   in the `[dev]` extra).
3. **If you previously deployed `uvicorn market_regime_engine.api:app`
   without an external auth proxy:** either set
   `MRE_LEGACY_API_ALLOW_UNAUTH=1` to keep the old behavior with an
   explicit ack, or migrate to
   `MRE_API_KEY="…" uvicorn market_regime_engine.api_v1:app`. Item E
   above documents the breaking change in detail.
4. **If your CI training step relies on the silent legacy fallback:**
   add `--allow-legacy-fallback` to your `mre train-baseline` and
   `mre validate` invocations, OR (preferred) add
   `mre materialize-asof-features --write-features` to your pipeline
   before training. The recommended end-to-end pipeline lives in the
   README's Quick start.
5. `pytest tests/` to confirm 192 passing, 1 skipped (Rust parity),
   1 deselected (slow). The new tests are
   `tests/test_version_sanity.py`,
   `tests/test_asof_perf.py`,
   `tests/test_api_legacy_gate.py`,
   `tests/test_verify_run_training_audit.py`,
   `tests/test_package_metadata.py`, plus the refactored
   `tests/test_training_data.py`.
6. Optional: `python scripts/build_release.py --clean` to produce the
   wheel + sdist + audit zip locally and verify SHA-256s.

---

## Out-of-scope deferrals (continued from v1.1)

These items appeared in the SOTA roadmap or v1.2 second-opinion review
but stay deferred per the user's "v1.2.1 patch bundle" scope. v1.1 and
v1.2 deferrals NOT closed by v1.2.1 are still tracked here for
continuity:

- DFM EM likelihood correctness (`dfm.py:116`, second-opinion #C/8) —
  math-heavy fix that touches the Watson-Engle approximation; needs a
  separate review pass.
- BOCPD-MUSE M2 ordering edge case (no test coverage gap currently
  observable).
- DuckDB-primary `Warehouse` (Phase B / scale-out work).
- Five `report_writer_v{1..5}.py` consolidation (code-rot smell). Pure
  cleanup, no functional impact.
- `_hash_frame` dtype-fragility — needs a switch from `astype(str)+csv`
  to `pd.util.hash_pandas_object` and a golden-trace migration.
- `apply_release_lag` only knows 8 series — needs a release-rule
  extension across the catalog.
- `MondrianBinaryConformal` docstring contradiction + non-binary `y`
  validation — cosmetic + small validation; not blocking.
- `_purge_and_embargo` O(|train|·|test|) Python loop — fine for
  monthly data; NumPy `searchsorted` migration deferred until daily
  workloads land.
- `# TODO(MRE-API-1)`: shared cache (Redis) for multi-worker uvicorn
  deployments. The lock-protection in v1.1 F is sufficient for
  `--workers 1`.

---

## How to consume this PR

1. `git checkout v1.1-fixes`.
2. `pip install -e ".[dev]"`.
3. `pytest tests/ -q -m "not slow"` → expect `192 passed, 1 skipped,
   1 deselected`.
4. `ruff check src tests` → 0 errors.
5. `ruff format --check src tests` → 0 changes.
6. `mypy src/market_regime_engine` → 35 errors (no regression).
7. End-to-end smoke per README Quick start → `verify-run.approved =
   true`.
8. Fail-closed PIT demo per criterion 12 → `verify-run` exits non-zero
   with `differences.training_mode_drift` present.
9. `python scripts/build_release.py --clean` → wheel, sdist, audit zip
   in `dist/`.
10. `pip install dist/market_regime_engine-1.2.1-py3-none-any.whl` in a
    fresh venv → `pytest tests/test_package_metadata.py` passes.

This patch bundle is forward-compatible with future v1.3 work; nothing
in v1.2.1 closes off a future migration to a DuckDB-primary warehouse,
a wider Rust kernel surface, or an additional ML head.
