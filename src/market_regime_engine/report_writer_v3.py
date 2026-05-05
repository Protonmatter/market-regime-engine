# SPDX-License-Identifier: Apache-2.0
"""Deprecated v0.6 governance addendum shim.

The append_v06_sections function moved into the consolidated
:mod:`market_regime_engine.report_writer` in v1.3. This shim emits a
DeprecationWarning and forwards to the new module so external automation
keeps working for one release. Removal is scheduled for v1.4.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from market_regime_engine.report_writer import _v06_section


def append_v06_sections(
    path: str | Path,
    drift: pd.DataFrame | None = None,
    release_gates: pd.DataFrame | None = None,
    ensemble_weights: pd.DataFrame | None = None,
    stacking_diagnostics: pd.DataFrame | None = None,
) -> Path:
    """Forward to the consolidated report writer.

    .. deprecated:: 1.3
       Use :func:`market_regime_engine.report_writer.write_institutional_report`
       with ``sections=("v06",)``.
    """
    warnings.warn(
        "report_writer_v3.append_v06_sections is deprecated; use "
        "market_regime_engine.report_writer.write_institutional_report "
        "with sections=('v06',) instead. Removed in v1.4.",
        DeprecationWarning,
        stacklevel=2,
    )
    p = Path(path)
    body = _v06_section(
        drift=drift,
        release_gates=release_gates,
        ensemble_weights=ensemble_weights,
        stacking_diagnostics=stacking_diagnostics,
    )
    with p.open("a", encoding="utf-8") as f:
        f.write(body)
    return p


__all__ = ["append_v06_sections"]
