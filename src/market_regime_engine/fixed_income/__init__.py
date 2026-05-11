# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income RCIE / X-Pro Auto-X Adapter — v1.5 scaffolding.

Mission (per ``MRE_FIXED_INCOME_AGENT.md §"Mission"``): "Add fixed-income
RCIE / X-Pro Auto-X integration layer without destabilizing the existing
macro/regime engine. The repo already contains the core quantitative and
governance engine. Add fixed-income-specific modules, schemas, API
contracts, CLI commands, warehouse tables, tests, and documentation."

PR-1 (this commit) lands the package skeleton, data contracts, hashing,
PIT guard, posterior-mode enforcement, evidence-pack scaffolding, and
placeholder API/CLI surfaces. Subsequent PRs (per the v1.5 implementation
plan) fill in warehouse, scorers, execution-confidence, TCA segmentation,
HMAC signing, and reports.

Public surface (PR-1):

- Data contracts: :class:`CreditRegimeOutput`, :class:`LiquidityStressOutput`,
  :class:`ExecutionConfidenceRequest`, :class:`ExecutionConfidenceResponse`,
  :class:`FixedIncomeEvidencePack`.
- Label enums: :class:`RegimeLabel`, :class:`LiquidityLabel`,
  :class:`ExecutionRecommendation`.
- Posterior wrappers: :class:`PosteriorMode`, :class:`FilteredPosterior`,
  :class:`SmoothedPosterior`.
- Helpers: :func:`assert_pit_safe`, :func:`canonical_sha256`.
"""

from market_regime_engine.fixed_income.calendars import (
    TradingCalendar,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    trading_days_between,
)
from market_regime_engine.fixed_income.credit_spread_regime import (
    DEFAULT_WEIGHTS as CREDIT_REGIME_DEFAULT_WEIGHTS,
)
from market_regime_engine.fixed_income.credit_spread_regime import (
    HYSTERESIS_BANDS_CREDIT,
    latest_credit_regime_score,
    score_credit_regime,
    write_credit_regime_score,
)
from market_regime_engine.fixed_income.credit_spread_regime import (
    classify_with_hysteresis as classify_credit_with_hysteresis,
)
from market_regime_engine.fixed_income.feature_builders import (
    build_credit_features,
    build_liquidity_features,
)
from market_regime_engine.fixed_income.hashing import canonical_sha256
from market_regime_engine.fixed_income.liquidity_stress import (
    DEFAULT_WEIGHTS as LIQUIDITY_STRESS_DEFAULT_WEIGHTS,
)
from market_regime_engine.fixed_income.liquidity_stress import (
    HYSTERESIS_BANDS_LIQUIDITY,
    latest_liquidity_stress_score,
    list_recent_liquidity_stress_scores,
    score_liquidity_stress,
    write_liquidity_stress_score,
)
from market_regime_engine.fixed_income.liquidity_stress import (
    classify_with_hysteresis as classify_liquidity_with_hysteresis,
)
from market_regime_engine.fixed_income.pit_guard import assert_pit_safe, assert_trading_day
from market_regime_engine.fixed_income.posterior_mode import (
    FilteredPosterior,
    PosteriorMode,
    SmoothedPosterior,
)
from market_regime_engine.fixed_income.schema import FI_TABLE_NAMES
from market_regime_engine.fixed_income.schema import register as _register_fi_schema
from market_regime_engine.fixed_income.schemas import (
    CreditRegimeOutput,
    ExecutionConfidenceRequest,
    ExecutionConfidenceResponse,
    ExecutionRecommendation,
    FixedIncomeEvidencePack,
    LiquidityLabel,
    LiquidityStressOutput,
    RegimeLabel,
)
from market_regime_engine.fixed_income.timestamps import assert_utc, iso8601_z, to_utc

# v1.5 (PR-2 task B): register the 13 FI warehouse tables with the
# storage registry on package import. ``register_tables`` is idempotent
# on name so a re-import is a no-op; the resulting warehouse therefore
# carries all 34 core + 13 FI tables (47 total) whenever
# ``market_regime_engine.fixed_income`` is imported before
# ``Warehouse(...)`` is instantiated.
_register_fi_schema()

__all__ = [
    "CREDIT_REGIME_DEFAULT_WEIGHTS",
    "FI_TABLE_NAMES",
    "HYSTERESIS_BANDS_CREDIT",
    "HYSTERESIS_BANDS_LIQUIDITY",
    "LIQUIDITY_STRESS_DEFAULT_WEIGHTS",
    "CreditRegimeOutput",
    "ExecutionConfidenceRequest",
    "ExecutionConfidenceResponse",
    "ExecutionRecommendation",
    "FilteredPosterior",
    "FixedIncomeEvidencePack",
    "LiquidityLabel",
    "LiquidityStressOutput",
    "PosteriorMode",
    "RegimeLabel",
    "SmoothedPosterior",
    "TradingCalendar",
    "assert_pit_safe",
    "assert_trading_day",
    "assert_utc",
    "build_credit_features",
    "build_liquidity_features",
    "canonical_sha256",
    "classify_credit_with_hysteresis",
    "classify_liquidity_with_hysteresis",
    "is_trading_day",
    "iso8601_z",
    "latest_credit_regime_score",
    "latest_liquidity_stress_score",
    "list_recent_liquidity_stress_scores",
    "next_trading_day",
    "previous_trading_day",
    "score_credit_regime",
    "score_liquidity_stress",
    "to_utc",
    "trading_days_between",
    "write_credit_regime_score",
    "write_liquidity_stress_score",
]
