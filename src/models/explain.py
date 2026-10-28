"""SHAP explanations for the gradient-boosted-tree forecaster."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.engineering import supervised_frame
from src.models.forecasters import fit_gbm


def shap_for_series(
    category: str, y: pd.Series, cfg: dict[str, Any] | None = None, kind: str = "xgboost", sample: int = 400
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the tree model on the full history and return (importance, sample_of_shap_values).

    importance : mean |SHAP| per feature (average impact on the forecast, in units/day)
    sample     : long frame ``feature, shap, feature_value`` for a random subset of days
                 (used for a beeswarm-style plot in the dashboard)
    """
    import shap

    cfg = cfg or load_config()
    model, cols = fit_gbm(y, cfg, kind)
    X = supervised_frame(y, cfg).drop(columns="y")[cols]
    vals = shap.TreeExplainer(model).shap_values(X)
    imp = pd.DataFrame({"category": category, "feature": cols, "mean_abs_shap": np.abs(vals).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    rng = np.random.default_rng(cfg["project"]["seed"])
    ix = np.sort(rng.choice(len(X), size=min(sample, len(X)), replace=False))
    long = pd.DataFrame(
        {
            "category": category,
            "feature": np.tile(cols, len(ix)),
            "shap": vals[ix].ravel(),
            "feature_value": X.iloc[ix].to_numpy().ravel(),
            "row": np.repeat(ix, len(cols)),
        }
    )
    return imp, long
