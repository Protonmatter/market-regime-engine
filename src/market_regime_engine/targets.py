# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime_engine.nber import add_forward_recession_targets, label_recession_months


def forward_log_return(price: pd.Series, horizon: int) -> pd.Series:
    return np.log(price.shift(-horizon) / price)


def forward_max_drawdown(price: pd.Series, horizon: int) -> pd.Series:
    out = []
    for i in range(len(price)):
        w = price.iloc[i : i + horizon + 1]
        if len(w) < horizon + 1 or w.isna().any():
            out.append(np.nan)
            continue
        dd = (w / w.cummax() - 1.0).min()
        out.append(float(dd))
    return pd.Series(out, index=price.index)


def make_targets(panel: pd.DataFrame, price_col: str = "SPX", horizons: tuple[int, ...] = (3, 6, 12)) -> pd.DataFrame:
    targets = pd.DataFrame(index=panel.index)
    if price_col in panel:
        price = panel[price_col].astype(float)
        for h in horizons:
            targets[f"ret_{h}m"] = forward_log_return(price, h)
            targets[f"dd_{h}m"] = forward_max_drawdown(price, h)
            targets[f"dd10_{h}m"] = (targets[f"dd_{h}m"] <= -0.10).astype(float)
    rec = add_forward_recession_targets(label_recession_months(panel.index), horizons=horizons)
    if not rec.empty:
        rec = rec.set_index(pd.to_datetime(rec["date"]))
        for h in horizons:
            col = f"recession_next_{h}m"
            if col in rec:
                targets[col] = rec[col].astype(float)
    return targets
