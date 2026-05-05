# SPDX-License-Identifier: Apache-2.0
"""Deprecated v0.5 governance addendum shim.

The append_v05_sections function moved into the consolidated
:mod:`market_regime_engine.report_writer` in v1.3. This shim emits a
DeprecationWarning and forwards to the new module so external automation
keeps working for one release. Removal is scheduled for v1.4.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from market_regime_engine.report_writer import _v05_section


def append_v05_sections(
    path: str | Path,
    *,
    confidence: pd.DataFrame | None = None,
    invalidation: pd.DataFrame | None = None,
    model_runs: pd.DataFrame | None = None,
    calibrated_outputs: pd.DataFrame | None = None,
) -> Path:
    """Forward to the consolidated report writer.

    .. deprecated:: 1.3
       Use :func:`market_regime_engine.report_writer.write_institutional_report`
       with ``sections=("v05",)`` and pass the governance frames directly.
    """
    warnings.warn(
        "report_writer_v2.append_v05_sections is deprecated; pass sections=('v05',) "
        "to market_regime_engine.report_writer.write_institutional_report instead. "
        "This shim will be removed in v1.4.",
        DeprecationWarning,
        stacklevel=2,
    )
    p = Path(path)
    text = p.read_text(encoding="utf-8") if p.exists() else "# Market Regime Engine Institutional Report\n"
    body = _v05_section(
        confidence=confidence,
        invalidation=invalidation,
        model_runs=model_runs,
        calibrated_outputs=calibrated_outputs,
    )
    p.write_text(text.rstrip() + body, encoding="utf-8")
    return p


__all__ = ["append_v05_sections"]
