"""Forecast accuracy metrics (zero-safe variants for intermittent daily demand)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y, yhat) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(_arr(y) - _arr(yhat))))


def rmse(y, yhat) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((_arr(y) - _arr(yhat)) ** 2)))


def mape(y, yhat) -> float:
    """MAPE in %, computed only over days with actual demand > 0 (undefined at 0)."""
    y, yhat = _arr(y), _arr(yhat)
    m = y > 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100)


def smape(y, yhat) -> float:
    """Symmetric MAPE in % (0-200 scale); a 0/0 term counts as a perfect forecast."""
    y, yhat = _arr(y), _arr(yhat)
    denom = np.abs(y) + np.abs(yhat)
    ratio = np.divide(2 * np.abs(y - yhat), denom, out=np.zeros_like(denom), where=denom > 0)
    return float(np.mean(ratio) * 100)


def mase(y, yhat, y_train, season: int = 7) -> float:
    """Mean absolute scaled error vs the in-sample seasonal-naive MAE (Hyndman & Koehler, 2006)."""
    y_train = _arr(y_train)
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if scale == 0:
        return float("nan")
    return float(mae(y, yhat) / scale)


def all_metrics(y, yhat, y_train=None, season: int = 7) -> dict[str, float]:
    """Dictionary with MAE, RMSE, MAPE, sMAPE (and MASE when ``y_train`` is given)."""
    out = {"mae": mae(y, yhat), "rmse": rmse(y, yhat), "mape": mape(y, yhat), "smape": smape(y, yhat)}
    if y_train is not None:
        out["mase"] = mase(y, yhat, y_train, season)
    return out


def summarize_cv(cv: pd.DataFrame, history: dict[str, pd.Series], season: int = 7) -> pd.DataFrame:
    """Aggregate CV predictions into one metric row per (category, model).

    ``history`` maps category -> full series; MASE scales by the data before the first origin.
    """
    rows = []
    for (cat, model), g in cv.groupby(["category", "model"]):
        first_origin = g["origin"].min()
        train = history[cat].loc[:first_origin].iloc[:-1]
        rows.append(
            {
                "category": cat,
                "model": model,
                "n_obs": len(g),
                **all_metrics(g["y"], g["yhat"], train, season),
            }
        )
    return pd.DataFrame(rows)
