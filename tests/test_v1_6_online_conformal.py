from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime_engine.frontier.online_conformal import (
    AgACI,
    EnbPIInterval,
    EnsembleMeanSplitConformal,
    MultiplicativeWeightsACI,
    StronglyAdaptiveACI,
)


def test_ensemble_mean_split_conformal_fit_predict() -> None:
    idx = pd.date_range("2024-01-01", periods=20)
    preds = pd.DataFrame({"m1": np.arange(20), "m2": np.arange(20) + 0.5}, index=idx)
    y = np.arange(20) + 0.25
    model = EnsembleMeanSplitConformal(alpha=0.1).fit(preds, y)
    out = model.predict_interval(preds.tail(3))
    assert list(out.columns) == ["center", "lower", "upper", "alpha"]
    assert (out["upper"] >= out["lower"]).all()
    assert model.fitted_n == 20


def test_multiplicative_weights_aci_updates_alpha() -> None:
    controller = MultiplicativeWeightsACI(alpha_target=0.1)
    out = controller.run([True, True, False, True, False, False])
    assert len(out) == 6
    assert out["alpha_t"].between(controller.alpha_min, controller.alpha_max).all()


def test_online_conformal_v1_5_x_aliases_resolve_to_renamed_classes() -> None:
    """REVIEW_DEEP_V1_5_2.md §1.9 / Findings #5, #23 — backwards-compat
    aliases preserve v1.5.x callers while the v1.6.0 names are honest
    about what each class actually does.

    AgACI is no longer a separate class (it was a no-op wrapper that
    delegated everything to StronglyAdaptiveACI without aggregating
    multiple controllers); aliased to the underlying multiplicative-
    weights controller. Faithful AgACI per Zaffran et al. 2022 is a
    v1.7.0 TODO.
    """
    assert EnbPIInterval is EnsembleMeanSplitConformal
    assert StronglyAdaptiveACI is MultiplicativeWeightsACI
    assert AgACI is MultiplicativeWeightsACI


def test_v1_5_x_alias_still_works_end_to_end() -> None:
    # v1.5.x callers using the old names continue to function.
    out = AgACI(alpha_target=0.1).run([True, False, True])
    assert len(out) == 3
    assert "alpha_t" in out.columns
