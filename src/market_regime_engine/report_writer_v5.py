# SPDX-License-Identifier: Apache-2.0
"""Deprecated v0.8 governance addendum shim.

The append_v08_sections function moved into the consolidated
:mod:`market_regime_engine.report_writer` in v1.3. This shim emits a
DeprecationWarning and forwards to the new module so external automation
keeps working for one release. Removal is scheduled for v1.4.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from market_regime_engine.report_writer import _v08_section


def append_v08_sections(
    path: str | Path,
    *,
    vintage_audits: pd.DataFrame,
    feature_asof: pd.DataFrame,
    vintage_observations: pd.DataFrame,
) -> None:
    """Forward to the consolidated report writer.

    .. deprecated:: 1.3
       Use :func:`market_regime_engine.report_writer.write_institutional_report`
       with ``sections=("v08",)``.
    """
    warnings.warn(
        "report_writer_v5.append_v08_sections is deprecated; use "
        "market_regime_engine.report_writer.write_institutional_report "
        "with sections=('v08',) instead. Removed in v1.4.",
        DeprecationWarning,
        stacklevel=2,
    )
    p = Path(path)
    body = _v08_section(
        vintage_audits=vintage_audits,
        feature_asof=feature_asof,
        vintage_observations=vintage_observations,
    )
    with p.open("a", encoding="utf-8") as f:
        f.write(body)


__all__ = ["append_v08_sections"]
