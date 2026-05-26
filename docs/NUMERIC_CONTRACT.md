# Fixed-Income Numeric Contract

## Purpose

Track B keeps internal model math as floats where the existing scorer already uses them, but new XPro decision surfaces publish deterministic fixed-point values only. This prevents JSON float rendering, language-runtime rounding, and NaN/Infinity edge cases from entering signed decision artifacts.

## Policy

The default policy is `NumericPolicy` in `src/market_regime_engine/fixed_income/numeric_contracts.py`.

| Field | Value | Use |
|---|---:|---|
| `prob_scale` | `1_000_000` | Probabilities as ppm |
| `bps_scale` | `10_000` | Basis points as q4 integers |
| `price_scale` | `1_000_000` | Prices as q6 integers |
| `money_scale` | `100` | Money/notional as cents |
| `rounding` | `half_even` | Decimal half-even quantization |
| `canonical_json` | `rfc8785-jcs-v2` | Artifact hash byte contract |

## Quantizers

- `prob_to_ppm(value)` converts `[0, 1]` probabilities to integer ppm.
- `bps_to_q4(value)` converts basis-point values to integer q4.
- `price_to_q6(value)` converts positive prices to integer q6.
- `money_to_cents(value)` converts non-negative monetary values to integer cents.
- `timestamp_to_epoch_ns_str(value)` converts tz-aware timestamps to UTC epoch-nanosecond strings.
- `assert_no_float_artifact(payload)` recursively rejects raw float values in new artifacts.

## Boundary rules

- Existing scorer dataclasses and legacy API responses keep float fields for backward compatibility.
- XPro decision artifacts, XPro CLI JSON, and new persisted quantized columns use integers/strings.
- NaN, Infinity, empty decimals, negative money, non-positive prices, and naive timestamps fail closed.
- Free-form metadata is represented by a canonical hash, not embedded as raw typed values.

## Tests

Validated by `tests/test_numeric_contracts.py`, `tests/test_xpro_decision_artifact.py`, and `tests/test_storage_xpro_decision_artifacts.py`.
