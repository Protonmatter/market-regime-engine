# SPDX-License-Identifier: Apache-2.0
"""Deprecated v0.7 governance addendum shim.

The append_v07_sections function moved into the consolidated
:mod:`market_regime_engine.report_writer` in v1.3. This shim emits a
DeprecationWarning and forwards to the new module so external automation
keeps working for one release. Removal is scheduled for v1.4.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from market_regime_engine.report_writer import _v07_section


def append_v07_sections(
    path: str | Path,
    *,
    alerts: pd.DataFrame,
    promotion_workflow: pd.DataFrame,
    hazard_diagnostics: pd.DataFrame,
    alfred_manifest: pd.DataFrame,
) -> None:
    """Forward to the consolidated report writer.

    .. deprecated:: 1.3
       Use :func:`market_regime_engine.report_writer.write_institutional_report`
       with ``sections=("v07",)``.
    """
    warnings.warn(
        "report_writer_v4.append_v07_sections is deprecated; use "
        "market_regime_engine.report_writer.write_institutional_report "
        "with sections=('v07',) instead. Removed in v1.4.",
        DeprecationWarning,
        stacklevel=2,
    )
    p = Path(path)
    body = _v07_section(
        alerts=alerts,
        promotion_workflow=promotion_workflow,
        hazard_diagnostics=hazard_diagnostics,
        alfred_manifest=alfred_manifest,
    )
    with p.open("a", encoding="utf-8") as f:
        f.write(body)


__all__ = ["append_v07_sections"]
