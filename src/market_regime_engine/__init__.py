# SPDX-License-Identifier: Apache-2.0
"""Market Regime Engine.

Governed macro regime signal layer with point-in-time lineage,
production guardrails, adapter exports, tamper-evident validation packs,
anti-overfit controls, and online conformal calibration primitives.
"""

__version__ = "1.6.0"

from market_regime_engine.logging_setup import configure_logging, get_logger

__all__ = ["__version__", "configure_logging", "get_logger"]
