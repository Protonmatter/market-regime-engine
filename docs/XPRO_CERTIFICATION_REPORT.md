# XPro Certification Report

## Version

`xpro_certification_report_v1`

## Scope

The certification report is the release-level evidence envelope for XPro decisioning. It ties together realized execution-confidence validation, the certification release gate, method-card coverage, optional frontier diagnostics, optional XPro decision verification, build identity, lockfile hashes, and a canonical report hash.

It is not a live trading recommendation and it does not replace the underlying warehouse evidence. It is the machine-readable artifact CI and model-risk reviewers can archive for a specific build.

## Required structure

| Field | Description |
|---|---|
| `artifact_version` | Must be `xpro_certification_report_v1` |
| `asof_utc` | Report as-of timestamp in UTC |
| `approved` | Top-level release/hold boolean |
| `decision` | `release` when all required checks pass; otherwise `hold` |
| `profile` | Release-gate profile, normally `certification` |
| `reasons` | Namespaced fail-closed reasons |
| `build` | Engine version, git SHA, dirty flag, and lockfile hashes |
| `inputs` | Validation directory, model card path, XPro decision id, and frontier diagnostics path |
| `checks.execution_confidence` | Realized-outcome validation summary and persisted confidence-row fields |
| `checks.release_gate` | Certification release-gate row and reasons |
| `checks.method_cards` | Required method-card file, section, and test-reference audit |
| `checks.frontier` | Optional frontier diagnostic result |
| `checks.xpro_decision` | Optional stored XPro decision artifact verifier result |
| `artifact_hash` | Canonical SHA-256 over the report without `artifact_hash` |

## CLI

```powershell
mre certification-report `
  --db data/mre.duckdb `
  --validation-dir data/validation `
  --asof 2026-01-02T00:00:00Z `
  --out-json data/certification_report.json `
  --dsr 0.75 `
  --pbo 0.01 `
  --evidence-pack-hmac v1:<hmac> `
  --fail-on-hold
```

The command writes the report JSON, prints the same payload to stdout, and exits `2` when `--fail-on-hold` is set and the report is not approved.

## CI fixture

For CI-only certification plumbing checks, use the deterministic synthetic fixture:

```powershell
python scripts/build_xpro_certification_fixture.py `
  --db data/xpro-certification.duckdb `
  --validation-dir data/xpro-certification-validation `
  --force
```

Then run `mre certification-report` against that fixture. The GitHub Actions workflow publishes `.ci-artifacts/certification_report.json` as the `xpro-certification-report` artifact.

## Fail-Closed Behavior

The report is held when any required control fails:

- realized execution-confidence validation is missing, skipped, or outside thresholds;
- certification release-gate evidence is missing, including DSR, PBO, model card, validation hash, or evidence-pack HMAC;
- required method cards are missing required sections or concrete test references;
- optional frontier diagnostics are supplied and any diagnostic fails;
- an optional XPro decision id is supplied but the stored artifact cannot be read or verified.

## Tests

Validated by `tests/test_certification_report.py`, `tests/test_execution_validation_certification_cli.py`, `tests/test_certification_release_and_execution_validation.py`, and `tests/test_method_cards_docs_audit.py`.
