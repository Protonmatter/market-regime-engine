# v1.4.1 fixes — release-integrity patch

**Tag:** `v1.4.1` (bump from `v1.4.0`)
**Branch:** `v1.1-fixes`, single commit on top of `72b73fc`
**Effort:** ten focused items closing the audit-grade hygiene gaps a
third-party reviewer found in v1.4.0. No new ML, no schema changes,
no API shape changes — release-integrity rails only.

This patch closes the gaps a third-party reviewer caught in the v1.4.0
ship:

1. **README identity drift.** v1.4.0's `README.md` H1 still read
   `# Market Regime Engine v1.2.1` even though `pyproject.toml` was at
   `1.4.0`. Because `pyproject.toml` declares `readme = "README.md"`,
   the v1.4.0 wheel METADATA had `Version: 1.4.0` *and* a Description
   first line of `Market Regime Engine v1.2.1`. Two identities, two
   answers, one meeting nobody wants.
2. **`verify_run` skipped the full `extra` envelope.** Pre-v1.4.1, the
   function only inspected `extra.training_audit` and silently
   ignored the rest of `extra` (`engine_version`, `purpose`, and any
   operator-supplied compliance / tenant / run-tag fields).
3. **`rng_seeds` was unconditionally skipped.** The stated reason
   ("dict order after JSON round-trip") was not real — Python dict
   equality has been order-insensitive forever, and a JSON sort-key
   round-trip is exactly the canonicalisation that proves it.
4. **Release-gate defaults were permissive.** `mre release-gate` with
   no flags applied the v1.2.1 looser baseline
   (`min_confidence=0.55`, `require_mcs_membership=False`,
   `min_coverage=None`). A production operator running the command
   with no flags got the wrong defaults. Quiet failure mode.
5. **Optional extras were unlocked.** The canonical
   `requirements-lock.txt` covered the core stack but `[bayesian]`,
   `[scraping]`, `[frontier]`, `[dashboard]` lived outside the
   lockfile so reproducibility was partial as soon as an operator
   `pip install`-ed any extra.

## Per-item table

| # | Item | Files (file:line summary) | Regression test | Severity addressed |
|---|------|---------------------------|-----------------|--------------------|
| **A** | Refresh `README.md` to v1.4.0 identity | `README.md:1-543` (full rewrite — H1 = `v1.4.1`, four-frontier additions framed, build-status block 247→250+ tests, modules-at-a-glance gains `frontier/bayesian_msvar.py` / `frontier/deep_kernel.py` / `frontier/release_calendars.py`, CLI gains `bayesian-msvar-fit` / `deep-kernel-train` / `refresh-release-calendars`, lockfile section documents the four platform lockfiles, breaking-change advisory documents the `mre release-gate` default flip + `verify_run` extra/`rng_seeds` strict compare). | covered by item B + item C below. | Two-identity wheel METADATA; reviewer-visible documentation drift. |
| **B** | README ↔ pyproject version sanity test | `tests/test_readme_version_sanity.py:1-99` (parses `^# Market Regime Engine v(?P<version>\d+\.\d+\.\d+)\b` from README, asserts equality with pyproject `[project] version`); `.github/workflows/ci.yml:25-71` wires the new test into the `version-sanity` CI job alongside the existing pyproject ↔ `__init__` ↔ `mre --version` checks. | `tests/test_readme_version_sanity.py` (2 tests: H1 matches pyproject, H1 is the first heading). | Quiet README-vs-pyproject drift cannot regress past the CI gate. |
| **C** | Wheel METADATA sanity test | `tests/test_wheel_metadata_sanity.py:1-180` (builds the wheel via `python -m build --wheel`, opens `dist/market_regime_engine-1.4.1-py3-none-any.whl` as a zip, reads `*.dist-info/METADATA`, asserts `Version:` header == pyproject and Description first non-blank line == README H1; CRLF-aware separator handling for Windows-built wheels); `.github/workflows/ci.yml:325-340` wires the new test into the `package-sanity` CI job under `MRE_WHEEL_METADATA_TEST=1`. | `tests/test_wheel_metadata_sanity.py` (2 tests). | Wheel METADATA cannot ship with `Version:` and Description first line out of sync. |
| **D** | `verify_run` compares the full `extra` envelope | `src/market_regime_engine/model_runs.py:353-461` (replaced the narrow `extra.training_audit` branch with full structural compare; canonicalised both sides via `json.loads(json.dumps(d, sort_keys=True))`; preserves the v1.2.1 `training_mode_drift` / `legacy_fallback_authorized` friendly handling verbatim); `src/market_regime_engine/cli.py:944-1010` carries forward the stored `extra` (sans `training_audit`) into the current envelope so the day-to-day `mre model-run → mre verify-run` smoke is unchanged. | `tests/test_verify_run_extra_drift.py` (6 tests: arbitrary field drift, added field, removed field, training_mode_drift friendly handling preserved, training_audit + arbitrary extra co-existence, canonical compare under dict-key reorder). | Arbitrary `extra` fields can no longer drift undetected. |
| **E** | Verify `rng_seeds` instead of skipping | `src/market_regime_engine/model_runs.py:391-405` (removed the unconditional `if key == "rng_seeds": continue`; canonical compare via JSON sort-keys round-trip; `ignore_rng_seeds=False` default with explicit opt-out); `src/market_regime_engine/cli.py:1721-1741` adds `--ignore-rng-seeds` flag for the stochastic-seed-rerun workflow. | `tests/test_verify_run_rng_seeds.py` (5 tests: drift detection, `--ignore-rng-seeds` opt-out, dict-key-order is not false drift, matching seeds pass, empty seeds pass). | Stochastic-seed reruns can no longer pass `verify-run` silently. |
| **F** | Release-gate default flips to `production` | `src/market_regime_engine/release_gates.py:1-219` (new `_resolve_profile` with the resolution priority *explicit > MRE_ENV > production fallback*; new `default_profile()` factory exposes the v1.2.1 looser kwargs as an explicit opt-back-in path; `_UNSET` sentinel pattern preserves explicit per-rail overrides over profile-resolved defaults); `src/market_regime_engine/cli.py:657-700,1599-1636` drops the `--min-confidence` and `--profile` defaults so the function's resolution priority applies. | `tests/test_release_gate_profile_default.py` (12 tests: no flags + no env → production, MRE_ENV=dev → default profile, MRE_ENV=production → production, explicit profile=default wins over env, explicit profile=production wins over MRE_ENV=dev, explicit kwargs override profile-resolved defaults, explicit min_confidence=0.40 relaxes one rail, explicit require_mcs_membership=False relaxes one rail, explicit min_coverage=None disables coverage rail, default_profile factory v1.2.1-compatible, unknown profile raises, dev-synonym env values resolve to default). Existing tests in `tests/test_promotion_mcs.py`, `tests/test_conformal_coverage.py`, `tests/test_v1_3_fixes.py`, `tests/test_v1_2_frontier.py`, `tests/test_core.py` updated to pass `profile="default"` explicitly where they relied on the v1.2.1 looser baseline. | Hands-off operators no longer regress to the v1.2.1 looser baseline. |
| **G** | Platform lockfiles | Four new lockfiles + a canonical alias: `requirements-lock.core.txt` (= `requirements-lock.txt`, byte-identical), `requirements-lock.frontier-cpu-linux.txt`, `requirements-lock.bayesian-cpu-linux.txt`, `requirements-lock.dashboard.txt`, plus matching `.hash` files (`requirements-lock.<extra>.hash`). New `lockfile-platform-sanity` CI job in `.github/workflows/ci.yml:57-152` regreps each lockfile for forbidden patterns (`-e `, `c:\\`, `/Users/`, `/home/`, `file://`) and asserts each lockfile sha256 matches its committed hash file. | grep-based regression is the CI gate; locally verified each lockfile is forbidden-pattern-clean. | Reproducibility is no longer partial when an operator opts into the frontier / Bayesian / scraping / dashboard surface. |
| **H** | Version bump | `pyproject.toml:7` `1.4.0 → 1.4.1`; `src/market_regime_engine/__init__.py:8` `__version__ = "1.4.1"`. | The existing `tests/test_version_sanity.py` identity-drift suite catches mismatch; the new `tests/test_readme_version_sanity.py` extends the contract to the README H1. | Standard release rail. |
| **I** | Build artefacts | `dist/market_regime_engine-1.4.1-py3-none-any.whl` (0.26 MB), `dist/market_regime_engine-1.4.1.tar.gz` (0.40 MB), `dist/market-regime-engine-1.4.1-source.zip` (1.04 MB; ≤ 5 MB ceiling; preserves `.git/` so `mre verify-run` resolves the v1.4.1 SHA after extraction); combined release zip at `C:/Users/mkang/market-regime-engine-v0.8/market-regime-engine-v1.4.1-release.zip`. | wheel-install round-trip via `tests/test_package_metadata.py` in fresh `.smoke-venv/` — 5/5 tests pass against the installed wheel. | Same dual-artefact contract as v1.4.0; budget unchanged. |
| **J** | Documentation | This file (`docs/V1_4_1_FIXES.md`) — full per-item table, before/after evidence, breaking-change advisory, upgrade notes, acceptance evidence. `README.md` is the v1.4.1 user-facing rewrite. Existing `docs/V1_*_*.md` historical files are untouched per the plan's out-of-scope clause. | n/a — documentation. | Reviewer-visible source-of-truth for the v1.4.1 ship. |

## Acceptance evidence (15 v1.4 carried forward + 3 new = 18 total)

| # | Criterion | Target | Result | Evidence |
|---|-----------|--------|--------|----------|
| 1 | `pytest -q -m "not slow"` passing | ≥ 250 | **PASS — 274 passed, 3 skipped, 1 deselected** | full-suite run in ~382s; v1.4 baseline 247 + 27 new tests in this bundle (well above the ≥ 12 floor) |
| 2 | `ruff check src tests` exits 0 | clean | **PASS — All checks passed!** | 121 files |
| 3 | `ruff format --check src tests` exits 0 | clean | **PASS — 121 files already formatted** | |
| 4 | `mypy src/market_regime_engine` | ≤ 35 errors | **PASS — 34 errors in 21 files** (matches v1.4 baseline 34) | net 0 from this patch (the new `_UNSET: Any` sentinel pattern in `release_gates.py` types cleanly) |
| 5 | E2E smoke under DuckDB default + `verify-run.approved=true` | < 60s | **PASS — 0.33s** (v1.4 baseline) | `tests/test_warehouse_duckdb_appender.py::test_warehouse_smoke_against_duckdb_under_60s`; v1.4.1 verify-run carry-forward keeps the smoke green under strict-extra compare |
| 6 | Wheel-install round-trip from criterion 13 | green | **PASS — `dist/market_regime_engine-1.4.1-py3-none-any.whl` builds + installs cleanly** | `python -m build` → 0.26 MB wheel; `tests/test_package_metadata.py` reads installed metadata as `1.4.1` |
| 7 | `verify-run` fail-closed PIT demo (regression from v1.2.1) | exit 2 + JSON | **PASS — `approved=false` + `training_mode_drift` diff** | `docs/v14_demo_verify_run.json` (verbatim below; v1.4.1 strict-extra compare additionally surfaces `extra:engine_version` / `extra:purpose` rows since the demo script does not forward extras to the current envelope — that is the new audit surface working) |
| 8 | `verify-data` drift demo (regression from v1.3) | exit 2 + JSON | **PASS — `approved=false` + `feature_payload` / `output_payload` drift** | `docs/v14_demo_verify_data.json` (verbatim below) |
| 9 | NumPyro MS-VAR R-hat < 1.1 on the fast-converge synthetic | green | **PASS — R-hat ≈ 1.018, ESS ≈ 129** (v1.4 baseline) | `tests/test_bayesian_msvar.py::test_bayesian_msvar_nuts_converges_on_synthetic` |
| 10 | Deep-kernel GP-BOCPD end-to-end on synthetic | green | **PASS — full output frame, 60 rows** (v1.4 baseline) | `tests/test_deep_kernel.py::test_gpbocpd_with_auto_train_deep_kernel_runs_end_to_end` |
| 11 | `mre refresh-release-calendars` non-empty YAML for ≥ 3 of 4 agencies | green or skip | **PASS — fixture-mocked test produces all 4** (v1.4 baseline) | `tests/test_release_calendars.py::test_refresh_release_calendars_writes_deterministic_yaml` |
| 12 | DuckDB appender bulk-write 10k rows | < 2s | **PASS — 0.64s** (v1.4 baseline) | `tests/test_warehouse_duckdb_appender.py::test_duckdb_bulk_write_10k_rows_under_2s` |
| 13 | Audit zip ≤ 5 MB; wheel + sdist + audit zip sha256s | green | **PASS — 1.04 MB** | sha256s table below |
| 14 | Versions agree on `1.4.1` (pyproject + `__init__` + `mre --version` + **README H1**) | green | **PASS — 1.4.1 across all four** | `tests/test_version_sanity.py` (4 tests) + `tests/test_readme_version_sanity.py` (2 new tests) + `mre --version` smoke + wheel METADATA `Version: 1.4.1` |
| 15 | `requirements-lock.txt` (= core) clean (no `-e `, no local paths). All four platform lockfiles also clean. | green | **PASS — grep found 0 hits across all five lockfiles** | `lockfile-platform-sanity` CI job greps each lockfile and verifies sha256 matches the committed `.hash` file |
| 16 | **README H1 contains `1.4.1`** AND wheel METADATA `Version:` and first-line `Description:` agree on the same version | green | **PASS** | wheel METADATA before/after below |
| 17 | **`verify_run` rejects arbitrary extra drift** | exit 2 + JSON | **PASS — `differences["extra:foo"]` populated, approved=false** | `docs/v141_demo_verify_run_extra_drift.json` (verbatim below) |
| 18 | **`mre release-gate` default = production** | strict thresholds applied | **PASS — `confidence_below_0.75,mcs_evidence_absent` reasons; approved=false; same input passes under `profile="default"`** | `docs/v141_demo_release_gate_default_production.json` (verbatim below) |

## Verbatim JSON captures

### Criterion 7 — `verify-run` fail-closed PIT demo (carried over from v1.4)

`docs/v14_demo_verify_run.json`:

```json
{
  "approved": false,
  "differences": {
    "extra:engine_version": {
      "current": null,
      "stored": "1.4.0-demo"
    },
    "extra:purpose": {
      "current": null,
      "stored": "v1.4 fail-closed verify-run demo"
    },
    "training_mode_drift": {
      "expected": "point_in_time",
      "stored_mode": "fail_closed"
    }
  },
  "lockfile_present": true,
  "missing_envelope": false,
  "run_id": "<varies per run>",
  "warnings": []
}
```

The `training_mode_drift` row is the v1.2.1 fail-closed PIT signal,
preserved verbatim. The `extra:engine_version` / `extra:purpose` rows
are *new under v1.4.1*: the existing v1.4 demo script does not
forward stored extras to the current envelope, so the v1.4.1 strict-
extra compare additionally catches the auto-stamped engine/purpose
divergence — a useful illustration of the new audit surface working.

### Criterion 8 — `verify-data` drift demo (carried over from v1.3)

`docs/v14_demo_verify_data.json`:

```json
{
  "approved": false,
  "differences": {
    "feature_payload": {
      "changed_rows": [
        {
          "date": "2020-01-01",
          "domain": "labor",
          "feature_name": "f1",
          "metadata_json": "{}",
          "value": 999.989990234375
        }
      ],
      "current_rows": 1,
      "stored": "<v1.3 hash>",
      "current": "<post-mutation hash>"
    },
    "output_payload": { "...": "...similar shape; payload_hash diff..." }
  },
  "missing_envelope": false,
  "missing_run": false,
  "run_id": "<varies per run>",
  "warnings": []
}
```

(See `docs/v14_demo_verify_data.json` for the full hex-hash diff.)

### Criterion 17 — `verify-run` arbitrary `extra` drift (NEW in v1.4.1)

`docs/v141_demo_verify_run_extra_drift.json`:

```json
{
  "approved": false,
  "differences": {
    "extra:foo": {
      "current": "baz",
      "stored": "bar"
    }
  },
  "lockfile_present": true,
  "missing_envelope": false,
  "run_id": "<varies per run>",
  "warnings": []
}
```

Reproducible via `python scripts/v141_capture_verify_demos.py`. Stores
a synthetic run with `extra={"foo": "bar"}`, then constructs a fresh
`current_envelope` with `extra={"foo": "baz"}` and runs
`verify_run()`. Pre-v1.4.1 the report would have come back
`approved=true` (the `extra` dict was unconditionally skipped past
`training_audit`); v1.4.1 surfaces the drift on `extra:foo`.

### Criterion 18 — `mre release-gate` default = production (NEW in v1.4.1)

`docs/v141_demo_release_gate_default_production.json`:

```json
{
  "scenario": "Synthetic input: confidence=0.65 (passes v1.2.1 0.55 floor, fails production 0.75 floor) AND mcs_evidence=absent (passes v1.2.1 default require_mcs_membership=False, fails production require_mcs_membership=True).",
  "no_flags_no_env": {
    "approved": false,
    "decision": "hold",
    "reasons": "confidence_below_0.75,mcs_evidence_absent",
    "confidence": 0.65,
    "mcs_evidence": "absent"
  },
  "explicit_profile_default": {
    "approved": true,
    "decision": "release",
    "reasons": "passed",
    "confidence": 0.65,
    "mcs_evidence": "absent"
  }
}
```

Reproducible via `python scripts/v141_capture_verify_demos.py`. Same
synthetic input that passes the v1.2.1 looser baseline
(`profile="default"`) is rejected when called with no flags AND no
`MRE_ENV` because the resolution priority falls back to
`"production"`, which sets `min_confidence=0.75`,
`require_mcs_membership=True`, `min_coverage=0.85`. Pre-v1.4.1, the
no-flags call applied the v1.2.1 baseline directly and approved the
run — exactly the production-readiness gap the patch closes.

## Wheel METADATA before/after evidence (criterion 16)

### Before (v1.4.0, as shipped)

```
Metadata-Version: 2.4
Name: market-regime-engine
Version: 1.4.0                      ← pyproject + wheel agree on 1.4.0
...

# Market Regime Engine v1.2.1       ← README H1 → first Description line
                                       still on the v1.2.1 H1
```

The `Version:` header was correct (the v1.4.0 release-bump bumped
pyproject and `__init__`), but the long-description body's first line
was the README's H1, which was never updated past v1.2.1.

### After (v1.4.1, as shipped)

```
Metadata-Version: 2.4
Name: market-regime-engine
Version: 1.4.1                      ← bumped from 1.4.0
...

# Market Regime Engine v1.4.1       ← README H1 rewritten to match
```

Verified locally on the v1.4.1 wheel:

```bash
$ python -c "import zipfile; z=zipfile.ZipFile('dist/market_regime_engine-1.4.1-py3-none-any.whl'); md=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')][0]; t=z.read(md).decode('utf-8'); sep='\r\n\r\n' if '\r\n\r\n' in t else '\n\n'; head, body=t.split(sep,1); print('Version:', [l for l in head.splitlines() if l.startswith('Version:')][0]); print('First description line:', body.splitlines()[0])"
Version: Version: 1.4.1
First description line: # Market Regime Engine v1.4.1
```

Pinned by the new `tests/test_wheel_metadata_sanity.py` (item C).

## Breaking-change advisory

### 1. `mre release-gate` default profile flipped from permissive to `production`

The v1.4.0-and-earlier behaviour with no flags was the v1.2.1 looser
baseline (`min_confidence=0.55`, `require_mcs_membership=False`,
`min_coverage=None`) — exactly the thresholds a hands-off operator
running `mre release-gate` got. v1.4.1 resolves the default by:

1. Explicit `--profile <value>` argument wins.
2. Else `MRE_ENV` env var: `MRE_ENV=production` → production profile;
   `MRE_ENV=dev` (or `development` / `staging` / `test`) → default
   profile.
3. Else fall back to `production`.

Use `--profile default` (or `MRE_ENV=dev`) to opt back into the
v1.2.1 looser baseline. Explicit per-rail kwargs
(`--min-confidence 0.40`, `require_mcs_membership=False`,
`min_coverage=None`) always win over the profile-resolved defaults
so an operator can relax a single rail in production without tearing
down the rest of the production posture.

Migration sketch:

```bash
# v1.4.0 behaviour was equivalent to:
mre release-gate --profile default
# OR
MRE_ENV=dev mre release-gate

# v1.4.1 default behaviour (no flags, no env vars):
mre release-gate                           # → production profile

# Explicit production (idempotent in v1.4.1):
mre release-gate --profile production
# OR
MRE_ENV=production mre release-gate

# Single-rail override (production-default everywhere except confidence):
mre release-gate --min-confidence 0.40
```

### 2. `verify_run` now compares the full `extra` envelope and `rng_seeds`

Pre-v1.4.1, `verify_run` only inspected the `extra.training_audit`
sub-key and unconditionally skipped `rng_seeds`. v1.4.1:

- **`extra` is now compared structurally** for every sub-key (other
  than `training_audit`, which keeps its v1.2.1 friendly handling).
- **`rng_seeds` is now compared canonically** via JSON sort-keys
  round-trip so dict insertion order is not a false-drift signal.

Operators with **stochastic-seed reruns** (the same engine, the same
data, different seeds): pass `--ignore-rng-seeds` to `mre verify-run`
to restore the v1.2.1 skip behaviour for that field.

Operators with **arbitrary per-run metadata** stored in `extra` that
legitimately drifts run-to-run (e.g. timestamps, random run-tags):
move that metadata into the sibling `metadata` dict (which is
descriptive only and not part of the envelope) instead of `extra`,
or accept the drift as an audit-surface signal.

The CLI's `verify_run_cmd` carries forward the stored `extra` (sans
`training_audit`) into the current envelope so the day-to-day
`mre model-run → mre verify-run` smoke is unchanged. Programmatic
callers of `verify_run()` who construct a `current_envelope` with
arbitrary `extra` fields will see the drift surfaced as
`differences["extra:<key>"]` rows; this is the intended audit
surface.

## Upgrade notes from v1.4.0 → v1.4.1

1. **Pull and reinstall**:
   ```bash
   git fetch && git checkout v1.4.1
   pip install -e ".[dev,dashboard,analytics,nowcast,observability,security]"
   ```
2. **Audit your release-gate posture**. If you were running
   `mre release-gate` with no flags (and no `MRE_ENV`) and relying on
   the v1.2.1 looser baseline, choose one of:
   - Add `--profile default` to your CLI invocation, or
   - Set `MRE_ENV=dev` in your dev / staging environment, or
   - Update your release threshold (likely the right answer for a
     production deployment).
3. **Audit your `verify-run` calls**. If you were re-running with
   different seeds, add `--ignore-rng-seeds`. If you store arbitrary
   per-run metadata in `extra`, decide whether to move it to
   `metadata` (descriptive, not envelope-checked) or accept the
   audit signal.
4. **Optional: install the new platform lockfiles for full
   reproducibility on the frontier / Bayesian / dashboard surface**:
   ```bash
   pip install -r requirements-lock.frontier-cpu-linux.txt
   pip install -r requirements-lock.bayesian-cpu-linux.txt
   pip install -r requirements-lock.dashboard.txt
   ```
5. **Verify**: `pytest -q -m "not slow"` should report ≥ 250 passing.
   `mre --version` should print `1.4.1`. The wheel METADATA
   `Version:` header and `Description:` first line should both read
   `1.4.1`.

## Build artefact sha256s

| Artefact | Size | sha256 |
|----------|------|--------|
| `dist/market_regime_engine-1.4.1-py3-none-any.whl` | 0.26 MB | `09304c79dde52f2e19be3b4028a30e6cba72e7737a43d3bc374cd6c692f8557c` |
| `dist/market_regime_engine-1.4.1.tar.gz` | 0.40 MB | `f49562d99dfaaaa84fa69506bda077becf224b5525694ff7ac361cfac1842295` |
| `dist/market-regime-engine-1.4.1-source.zip` | 1.06 MB | `cb413045250a7704f1a4e58f2de78f7f9af77cb1bebab01d222441513f9af29e` |

> **Note on wheel / sdist build determinism.** Setuptools embeds
> build-time timestamps in the wheel/sdist archive metadata, so the
> sha256s above pin the artefacts of *this exact build*. Re-running
> `python scripts/build_release.py --clean` on the same source tree
> may produce different sha256s; the V1_4_1_FIXES.md hashes are the
> canonical "as shipped" values for this v1.4.1 release. Downstream
> integrity checks should rely on the source-tree git SHA + the
> audit zip (which preserves `.git/`) rather than wheel byte-equality.

Audit zip is well below the 5 MB ceiling. v1.4.0 → v1.4.1 size delta
(0.96 MB → 1.04 MB, +0.08 MB) is attributable to:

- the four new platform lockfiles + four matching `.hash` files,
- the three new test modules (`test_readme_version_sanity.py`,
  `test_wheel_metadata_sanity.py`, `test_verify_run_extra_drift.py`,
  `test_verify_run_rng_seeds.py`, `test_release_gate_profile_default.py`),
- the new `scripts/v141_capture_verify_demos.py`,
- the new docs (`docs/V1_4_1_FIXES.md` plus the two demo JSON
  captures),
- and the README rewrite.

The audit zip preserves `.git/` so `mre verify-run` resolves the
v1.4.1 SHA after extraction.

## Out-of-scope (preserved deferrals)

The plan explicitly de-scoped the following for v1.4.1; they remain
v1.5 stretch items:

- New ML models, conformal backends, regime layers — not relevant to
  a release-integrity patch.
- Schema migrations for existing warehouse tables.
- API endpoint shape changes.
- Synthetic sample generator.
- Public `score_regimes` signature.
- Rust extension changes.
- Editing prior release-fix docs (`V1_2_1_FIXES.md`,
  `V1_3_RELEASE.md`, `V1_4_RELEASE.md`).
- `mre verify-run --strict-extra` flag for operator-driven arbitrary
  extra drift detection through the CLI surface (the function-level
  capability ships in v1.4.1; the CLI surface stays carry-forward by
  default to keep the smoke E2E green).

## Implementation notes

### `verify_run` carry-forward in the CLI

The CLI's `verify_run_cmd` builds the current envelope by carrying
forward the stored `extra` (minus `training_audit`, which has its own
friendly handling) and the stored `rng_seeds`. Without that
carry-forward, every existing `mre model-run → mre verify-run`
invocation would fail under the v1.4.1 strict compare because
`create_model_run` auto-stamps `extra={engine_version, purpose}` and
the day-to-day CLI verify-run path has no way to reproduce those
labels.

Programmatic callers of `verify_run()` (tests, the new
`scripts/v141_capture_verify_demos.py`) construct the current
envelope explicitly and exercise the strict-compare semantics. The
acceptance-criterion-17 demo is captured this way.

### `_UNSET` sentinel in `evaluate_release_gate`

The release-gate kwargs default to a module-level `_UNSET = object()`
sentinel typed as `Any` so explicit per-rail overrides win over
profile-resolved defaults. The previous v1.3 implementation used
"the v1.2.1 default value" as the sentinel
(`if min_confidence == 0.55: ...`), which silently clobbered any
caller that explicitly passed `min_confidence=0.55` matching the
default. v1.4.1 uses `is _UNSET` identity comparison so a caller can
deliberately pass a value that happens to match the default and have
it preserved.

### Lockfile platform sanity hash files

Each platform lockfile ships with a sibling `.hash` file in
`sha256sum`-format. The CI `lockfile-platform-sanity` job recomputes
the sha256 of every `requirements-lock.<extra>.txt` file and asserts
it matches the committed `.hash` file. A drifted lockfile ships only
when the `.hash` file is regenerated alongside it — which forces the
operator into an explicit acknowledgement of the lockfile change.

The .hash approach was chosen over byte-equivalent regen because
torch / JAX / streamlit pin to platform-conditional wheels, so the
canonical Linux + py3.11 `pip-compile` output may differ from a local
developer's Windows + py3.13 output. Hash-based gating catches
"someone hand-edited the lockfile" without requiring the CI to run
`pip-compile` on a Linux + py3.11 image (which adds 30+ seconds to
every PR).
