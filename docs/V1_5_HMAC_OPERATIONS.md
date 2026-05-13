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

## 8. request_id binding (v1.5.1, PR-9 FIX 3)

v1.5.0 packs bound only `(model_run_id, component_name, output_hash,
timestamp)` into the HMAC payload. A replay of the same
`(model_run_id, output_hash)` under a different inbound `request_id`
verified silently. v1.5.1 closes that hole by threading `request_id`
into the canonical bytestream when:

- the pack was built via `build_evidence_pack(request_id=...)`, AND
- `metadata["_request_id_bound"]` is True (auto-stamped by the builder).

Legacy v1.5.0 packs lack the metadata flag; their canonical bytestream
remains byte-identical and the v1 signatures continue to verify. The
following matrix summarises the contract:

| Pack origin | metadata flag | request_id in canonical bytes | Key prefix |
|---|---|---|---|
| v1.5.0 sign + v1 key | absent | excluded | `v1:...` |
| v1.5.0 resign under v2 (`fi-evidence-resign --to-key v2`) | absent | excluded | `v2:...` |
| v1.5.1 sign with `request_id=X` | `True` | included as `"request_id":"X"` | `v2:...` (recommended) |
| v1.5.1 sign with `request_id=None` | absent | excluded | `v1:...` or `v2:...` |

### v1 → v2 rotation procedure (FIX 3)

1. Roll out the `v2` key via `MRE_FI_HMAC_KEY_VERSIONS`.
2. Update every signer to call `build_evidence_pack(request_id=...)`
   (the FastAPI execution-confidence path threads `X-Request-ID`; the
   `mre fi-evidence-pack` CLI exposes `--request-id`).
3. Re-sign historical packs:
   `mre fi-evidence-resign --from-key v1 --to-key v2 --db ...`. The
   command emits a `warning` field with a `null_request_id_count` and
   `null_request_id_sample` when re-signed packs preserve
   `request_id=null` semantics. Those packs are still replay-vulnerable
   for the legacy `(model_run_id, output_hash)` tuple; schedule them
   for a re-issue at source.
4. After `vN` retirement, regenerate any remaining null-`request_id`
   packs from the live signal rows via `mre fi-evidence-pack
   --request-id <id> --sign true`.

### Production guard

When `MRE_ENV=production` (or `MRE_FI_REQUIRE_HMAC=1`) AND the pack's
`component_name == "execution_confidence"`, `sign_pack` raises
`RuntimeError` if `pack.request_id` is `None`. This prevents a
production worker from publishing a replay-vulnerable
execution-confidence pack. Other components (`credit_regime`,
`liquidity_stress`, `tca_segmentation`) do not consume an inbound
request id and are exempt.

## Canonical-JSON encoder versions (v1.6.0)

Per `REVIEW_DEEP_V1_5_2.md` §2.5: the canonical-JSON encoder is now
versioned. The encoder version is independent of the HMAC key version
above — the HMAC prefix (`v1:` / `v2:`) routes the key, the metadata
key `_canonical_version` routes the encoder.

| Encoder | Implementation | Behaviour |
|---|---|---|
| `v1` (legacy) | `json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)` | Stable across same-CPython runs; v1.5.x wire format. NOT RFC 8785-compliant on numbers, non-ASCII strings, or NaN/Inf. |
| `v2` (RFC 8785) | Pure-Python implementation of the JCS spec. Numbers via ECMA-262 7.1.12.1 Number::toString (`1.0` → `"1"`); raw UTF-8 strings with minimal escapes; UTF-16 code-point key sort; rejects NaN/Inf/non-JSON-native types. | Cross-language verifiable. A Java JCS or Rust `serde_jcs` library reproduces the same bytes byte-for-byte. |

### v1 → v2 migration: when to migrate

Migrate when **any** of the following is true:

1. A non-Python consumer (Java JCS, Rust `serde_jcs`, Go JCS) needs
   to re-derive the canonical bytes / hash from the pack contents.
2. The pack contains floats whose representation matters at the ULP
   level (BLP DSR thresholds, percentile bps, etc.) and a downstream
   verifier may be running a different Python minor version. v1's
   `json.dumps` is stable across same-Python-minor runs but
   re-formatting `1.0` to `"1.0"` (vs ECMA's `"1"`) is a divergence
   waiting to happen on a Python upgrade.
3. The pack carries non-ASCII strings (`cusip`, dealer names with
   accents, etc.) and a downstream verifier compares manifests
   character-by-character. v1's `ensure_ascii=True` default escapes
   non-ASCII as `\uXXXX`; v2 emits raw UTF-8.

Do **NOT** migrate when:

1. You need byte-identical reproduction of a v1.5.x persisted hash
   (the v1 → v2 transition by design changes the canonical bytes,
   and therefore the pack hash, even when logical content is
   unchanged). Keep the v1 row alongside the v2 row.

### How to migrate

```bash
# 1. Generate a v2 HMAC key alongside the existing v1.
export MRE_FI_HMAC_KEY_VERSIONS='{"v1": "<v1 base64 key>", "v2": "<v2 base64 key>"}'

# 2. Bulk re-sign every v1-signed pack under v2 with the new encoder.
mre fi-evidence-resign \
    --db data/mre.duckdb \
    --from-key v1 \
    --to-key v2 \
    --to-version v2

# 3. Optional: retire v1 from the key environment after the audit
# window (operationally identical to the HMAC v1 → v2 rotation
# described above).
```

The `--to-version` flag is **independent** of `--to-key`:

| `--to-key` | `--to-version` | Effect |
|---|---|---|
| `v2` | (omitted) | Rotate HMAC key only; canonical encoder stays at whatever was stamped. Pack hash unchanged. |
| `v2` | `v2` | Rotate HMAC key AND upgrade canonical encoder to RFC 8785. Pack hash changes. |
| `v2` | `v1` | Rotate HMAC key AND explicitly downgrade canonical encoder (rare; usually for forensic reproduction). Pack hash changes. |

### Legacy-verify guarantee

A v1 pack (no `_canonical_version` metadata) verified with the v1
HMAC key continues to verify under v1.6.0+ code paths verbatim:

- `compute_pack_hash(pack)` reads the (absent) metadata key, defaults
  to `version="v1"`, computes the legacy bytes, hashes them.
- `verify_pack(pack)` reads the HMAC prefix (`v1:`), looks up the v1
  key, recomputes HMAC over the legacy bytes, compares with
  `hmac.compare_digest`.

This contract is pinned by `tests/test_fi_evidence_pack_canonical_v2.py`
and `tests/test_fi_evidence_resign.py`; any future refactor that
breaks legacy verification will fail those tests.
