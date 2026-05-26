# Architecture Refactor and Boundary Notes

## Objective

The v1.6 refactor reduces review surface without changing the system identity. Large monolithic modules now expose compatibility facades while implementation details live behind smaller responsibility-specific modules.

## Storage split

| Module | Responsibility |
|---|---|
| `storage.py` | Backward-compatible facade. |
| `storage_registry.py` | `TableSpec`, table registration, legacy schema aggregate shims. |
| `storage_backends.py` | Backend protocol, SQLite adapter, DuckDB adapter, backend selection. |
| `storage_repositories.py` | `Warehouse` read/write repository API and migration helpers. |
| `storage_pool.py` | Per-process pooling, write locks, pool lifecycle. |

## CLI split

| Module | Responsibility |
|---|---|
| `cli.py` | Backward-compatible facade and `main()`. |
| `cli_parser.py` | Argparse tree construction. |
| `cli_handlers.py` | Command handlers. |
| `cli_helpers.py` | Shared helper functions and verification utilities. |

## Fixed-income API split

| Module | Responsibility |
|---|---|
| `fixed_income/api.py` | Backward-compatible facade. |
| `fixed_income/api_schemas.py` | Pydantic API boundary and response serialization. |
| `fixed_income/api_handlers.py` | FastAPI router and endpoint handlers. |
| `fixed_income/api_middleware.py` | Rate-limit and body-size guard logic. |
| `fixed_income/api_cache.py` | Per-process versioned read cache. |

## Stable core vs experimental frontier

The stable core is the default release-gate target. The frontier package is a separate research boundary. Frontier paths that can use retrospective information or optional experimental dependencies must require explicit opt-in through `MRE_ENABLE_EXPERIMENTAL_FRONTIER=1`.

Release gates now accept `gate_boundary="stable_core"` or `gate_boundary="experimental_frontier"`. The stable core keeps prior production/default profile resolution semantics. The frontier boundary requires explicit experimental enablement and records the non-production boundary in release-gate metadata.

## Review implications

- Production reviewers can focus first on stable-core modules and release-gate evidence.
- Research reviewers can evaluate frontier models without confusing retrospective diagnostics with production certifiability.
- Import compatibility is intentionally retained so downstream notebooks, CLIs, and API tests do not need immediate rewrites.

## v1.7 certification boundary enforcement

Stable-core certification now has two enforcement layers:

1. `evaluate_release_gate(..., profile="certification")` requires validation artifacts and evidence-pack material that the production profile previously treated as optional.
2. `tests/test_certification_import_boundary.py` audits stable-core imports so mature components do not silently depend on experimental frontier implementations. Utility imports such as frontier data-cleaning/release-calendar helpers remain allowed; model implementations stay behind command/orchestration/frontier boundaries.

The fixed-income execution-confidence validation path is intentionally separate from scoring. Scoring produces a decision-time output. Validation consumes later realized outcomes and creates a separate artifact hash, which is then attached to a certification confidence row. This prevents outcome-derived information from contaminating the decision-time scoring path.
