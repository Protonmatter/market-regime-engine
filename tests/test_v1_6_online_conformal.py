from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime_engine.frontier.online_conformal import AgACI, EnbPIInterval, StronglyAdaptiveACI


def test_enbpi_interval_fit_predict() -> None:
    idx = pd.date_range("2024-01-01", periods=20)
    preds = pd.DataFrame({"m1": np.arange(20), "m2": np.arange(20) + 0.5}, index=idx)
    y = np.arange(20) + 0.25
    model = EnbPIInterval(alpha=0.1).fit(preds, y)
    out = model.predict_interval(preds.tail(3))
    assert list(out.columns) == ["center", "lower", "upper", "alpha"]
    assert (out["upper"] >= out["lower"]).all()
    assert model.fitted_n == 20


def test_strongly_adaptive_aci_updates_alpha() -> None:
    controller = StronglyAdaptiveACI(alpha_target=0.1)
    out = controller.run([True, True, False, True, False, False])
    assert len(out) == 6
    assert out["alpha_t"].between(controller.alpha_min, controller.alpha_max).all()


def test_agaci_runs() -> None:
    out = AgACI(alpha_target=0.1).run([True, False, True])
    assert len(out) == 3
    assert "alpha_t" in out.columns
