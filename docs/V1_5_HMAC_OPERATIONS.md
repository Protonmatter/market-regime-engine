# v1.5 Fixed-Income HMAC Operations Playbook

Operating procedure for the FI evidence-pack HMAC layer shipped in
v1.5.0 (per PR-7 §A + REVIEW.md §4.2 HMAC rotation cadence).

## 1. Key generation

Every key is 32 random bytes, base64-encoded:

```bash
python -c "import secrets, base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"
```

Treat the resulting string as a production secret. Use a vault
(HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager) for
custody; never commit a key to git.

## 2. Environment variable schema

Two env vars are recognised. **Both must agree** when both are set;
the multi-version form is the recommended production layout.

### Multi-version (recommended)

```
MRE_FI_HMAC_KEY_VERSIONS = {"v1": "<base64-key>", "v2": "<base64-key>"}
```

Versions follow lexicographic ordering — `latest_hmac_version()`
picks the lexicographic max (use `v01`, `v02`, ... when you expect
more than 9 active versions). The signer always uses the latest
version; the verifier accepts any version whose key is present in
the map, so keys can overlap during a rotation window.

### Singleton (legacy / dev)

```
MRE_FI_HMAC_KEY = <base64-key>
```

Registered as `v1` internally; equivalent to
`MRE_FI_HMAC_KEY_VERSIONS = {"v1": "..."}` and useful in dev /
single-key deployments.

### Production gate

```
MRE_ENV = production         # OR
MRE_FI_REQUIRE_HMAC = 1
```

When either is set and no keys are configured, `sign_pack` /
`write_evidence_pack` / the FI CLI all raise rather than silently
publishing unsigned packs.

## 3. Rotation cadence

Quarterly rotation per REVIEW.md §4.2:

| Step | Action | Window |
|------|--------|--------|
| 1 | Generate `vN+1` key, add to `MRE_FI_HMAC_KEY_VERSIONS` | T-7d |
| 2 | Roll workers (signer picks `vN+1` automatically) | T |
| 3 | Run `mre fi-evidence-resign --from-key vN --to-key vN+1` over historical packs | T+1h |
| 4 | Verify with `mre verify-run --model-run-id <id>` (sample) | T+1h |
| 5 | Remove `vN` from `MRE_FI_HMAC_KEY_VERSIONS` after audit window | T+30d |
| 6 | Rotate the env-var secret in the vault | T+30d |

`fi-evidence-resign` reads every pack signed under `--from-key`,
re-signs it with `--to-key`, and writes the new signature back to
`fixed_income_evidence_packs`. Use `--dry-run` first to preview the
matched count.

## 4. On-call playbook

### Symptom: `fi_hmac_signature_failures_total` rises above 1%

The `fi_hmac_signature_failures_total` counter (registered in
`fixed_income/observability_ext.py`) increments whenever
`verify_pack` rejects a signature. A sustained > 1% failure rate on
any FI consumer indicates either:

1. **Key drift between writer and verifier** — the writer is signing
   with a key version that the verifier's
   `MRE_FI_HMAC_KEY_VERSIONS` doesn't include. Check that the same
   JSON env-var is rolled out to every worker.

2. **Tampered packs** — someone wrote / replicated a pack outside
   the canonical `write_evidence_pack` path. Treat as a security
   incident: snapshot the warehouse, audit which writer process
   produced the rejected packs, and re-sign through
   `fi-evidence-resign` only after the offending writer is shut
   down.

3. **Time-skew artefacts** — extremely rare; the canonical bytestream
   is timestamp-aware and a clock skew > 1 second changes the
   payload. Stamp `timestamp` once via the scoring pipeline and
   never re-derive on read.

### Symptom: `mre verify-run` reports `fi_hmac_verified=False`

1. Run `mre fi-evidence-pack --db <path> --model-run-id <id>` to
   regenerate the pack from the live signal row. If the new pack
   verifies, the original was either tampered or signed under a
   key that's no longer configured.
2. Cross-reference `fixed_income_evidence_packs.hmac_signature` —
   the `v<ver>:` prefix tells you which key version was used.
3. Confirm that key version is still in
   `MRE_FI_HMAC_KEY_VERSIONS`.

### Symptom: production worker boot fails with `FI HMAC required`

`MRE_ENV=production` is set but `MRE_FI_HMAC_KEY_VERSIONS` /
`MRE_FI_HMAC_KEY` is not. Roll back the production env or inject
the key from the vault. The fail-closed design is intentional —
there is no soft-degrade path for unsigned packs in production.

## 5. Disaster recovery

If the active key is leaked or lost:

1. Rotate immediately: generate `vN+1`, add to
   `MRE_FI_HMAC_KEY_VERSIONS`.
2. Run `mre fi-evidence-resign --from-key vN --to-key vN+1` to
   re-sign every existing pack.
3. Remove the leaked `vN` from the env after every consumer has
   refreshed and stale verifier-only deployments have ack'd.
4. Audit the warehouse for any pack written outside the canonical
   path during the leak window.

If the leak window is unknown, treat all packs signed under `vN`
as suspect and rebuild the canonical bytestream from the underlying
signal rows (which carry their own `artifact_hash` from the
governance triple) before re-signing.

## 6. Key custody contract

- Production keys live only in the vault. Workers fetch via the
  platform's secret-injection (Kubernetes secrets, AWS task role,
  GCP secret manager).
- Dev keys are operator-generated; never rotate the same key into
  prod and dev. The `MRE_ENV` gate enforces production-only HMAC
  via the JSON env var.
- Every key rotation is logged via `fi_hmac_signature_failures_total`
  + the resign-tool stdout summary. Keep the resign output for the
  audit trail.

## 7. Verification

Smoke check after a rotation:

```bash
mre fi-evidence-pack --db data/mre.duckdb \
    --model-run-id <recent_run> \
    --component credit_regime \
    --request-id smoke-rotation-test \
    --sign true
mre verify-run --db data/mre.duckdb --run-id <recent_run>
# expect: "fi_hmac_verified": true
```
