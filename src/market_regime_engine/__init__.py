# SPDX-License-Identifier: Apache-2.0
"""Market Regime Engine.

Public API entry points are re-exported here for convenience. Submodules can
also be imported directly.
"""

__version__ = "1.6.1"

from market_regime_engine.logging_setup import configure_logging, get_logger

__all__ = ["__version__", "configure_logging", "get_logger"]
