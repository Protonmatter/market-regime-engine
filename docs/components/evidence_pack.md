# `evidence_pack.py` — FI Evidence Pack

## Purpose

Per AGENT.md non-negotiable 6 + INSTRUCTIONS.md §6.5: every signal
that goes external must be reproducible from a tamper-evident
`FixedIncomeEvidencePack`. PR-1 shipped the dataclass + canonical
SHA-256 hash; PR-7 hardens with HMAC-SHA-256 sign / verify and
warehouse persistence.

## Inputs

`build_evidence_pack(...)` accepts:

- `model_run_id`, `component_name`, `model_version` — governance
  identifiers.
- `code_sha`, `model_hash`, `input_features_hash`, `output_hash` —
  reproducibility hashes (canonical sha256 with `"sha256:"` prefix).
- `release_gate` (bool) — propagated from the underlying scorer.
- `data_vintages` — dict of `table → ISO-8601 latest-source-timestamp`.
- `validation_results`, `random_seeds`, `metadata` — optional dicts.
- `lockfile_hash`, `python_version` — optional reproducibility info.
- `hmac_signature` — initially `None`; populated by `sign_pack`.

`capture_data_vintages(warehouse, *, asof)` derives the
`data_vintages` dict from the seven FI source tables (`trace_trades`,
`rfq_events`, `curve_snapshots`, `cds_curve_snapshots`,
`bond_reference`, `dealer_quotes`, `dealer_response_stats`) capped at
`asof`. Missing tables emit `"1970-01-01T00:00:00Z"`.

## Outputs

`FixedIncomeEvidencePack` (frozen dataclass) — see
`schemas.py`.

`compute_pack_hash(pack)` returns `"sha256:<hex>"` over the canonical
JSON excluding `hmac_signature`. `canonical_pack_payload(pack)`
returns the canonical JSON bytestring used for hashing / signing.

`sign_pack(pack)` returns a new pack with `hmac_signature =
"v<ver>:<hex(hmac-sha256)>"`.

`verify_pack(pack)` returns `True` when the signature verifies via
`hmac.compare_digest`.

`write_evidence_pack(warehouse, pack, *, request_id, sign=None)`
persists the row. `read_evidence_pack(warehouse, *, model_run_id,
request_id=None)` round-trips the latest matching row.

## Validation rules

1. The canonical bytestream excludes `hmac_signature` so the same
   pack can be re-signed under a rotated key without invalidating
   the historical hash.
2. Production mode (`MRE_ENV=production` or `MRE_FI_REQUIRE_HMAC=1`)
   refuses to write or sign without a configured key.
3. Tampering with any byte of the canonical JSON fails
   `verify_pack`.
4. The warehouse roundtrip normalises the ISO-8601 timestamp on read
   so the canonical bytestream is bit-identical to the signed-at-
   write payload (DuckDB TIMESTAMP otherwise drops the `Z` and
   substitutes a space).

## References

- AGENT.md "Hashing rules" + "FixedIncomeEvidencePack".
- INSTRUCTIONS.md §6.5 + §10 governance rules 1, 5, 7.
- PR-7 §A in `docs/V1_5_FIXED_INCOME_RCIE.md`.
- `docs/V1_5_HMAC_OPERATIONS.md`.
