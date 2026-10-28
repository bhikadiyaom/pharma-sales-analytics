"""Forecasting models behind one interface.

Every forecaster is a function ``f(y, horizon, cfg) -> np.ndarray`` that fits on the
daily series ``y`` (DatetimeIndex, freq D) and returns ``horizon`` non-negative forecasts
for the days following ``y.index[-1]``.  Refitting inside the function keeps rolling-origin
CV leak-free: nothing after the forecast origin is ever visible to the model.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.engineering import build_feature_frame, supervised_frame

log = logging.getLogger(__name__)

Forecaster = Callable[[pd.Series, int, dict[str, Any]], np.ndarray]


# ----------------------------------------------------------------- baselines
def naive_forecast(y: pd.Series, horizon: int, cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Repeat the last observed value."""
    return np.repeat(float(y.iloc[-1]), horizon)


def seasonal_naive_forecast(
    y: pd.Series, horizon: int, cfg: dict[str, Any] | None = None, season: int = 7
) -> np.ndarray:
    """Repeat the value observed one week earlier (weekly seasonality)."""
    last = y.to_numpy()[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last, reps)[:horizon]


# ------------------------------------------------------------------- SARIMAX
def fourier_terms(idx: pd.DatetimeIndex, k: int) -> pd.DataFrame:
    """Annual Fourier terms used as exogenous regressors for yearly seasonality."""
    t = idx.dayofyear.to_numpy() / 365.25
    cols = {}
    for i in range(1, k + 1):
        cols[f"sin{i}"] = np.sin(2 * np.pi * i * t)
        cols[f"cos{i}"] = np.cos(2 * np.pi * i * t)
    return pd.DataFrame(cols, index=idx)


def sarimax_forecast(y: pd.Series, horizon: int, cfg: dict[str, Any] | None = None) -> np.ndarray:
    """SARIMA(p,d,q)(P,D,Q,7) with annual Fourier regressors (weekly + yearly seasonality)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    cfg = cfg or load_config()
    c = cfg["forecasting"]["sarimax"]
    y = y.iloc[-c["max_train_days"] :]
    exog = fourier_terms(y.index, c["fourier_k"])
    fut_idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(
            y.to_numpy(),
            exog=exog.to_numpy(),
            order=tuple(c["order"]),
            seasonal_order=tuple(c["seasonal_order"]),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = mod.fit(disp=False, maxiter=60, method="lbfgs")
        fc = res.forecast(horizon, exog=fourier_terms(fut_idx, c["fourier_k"]).to_numpy())
    return np.clip(np.asarray(fc), 0, None)


# ------------------------------------------------------------------- Prophet
def prophet_forecast(y: pd.Series, horizon: int, cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Facebook/Meta Prophet with weekly + yearly seasonality and fixed-date holidays."""
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    cfg = cfg or load_config()
    cal = cfg["calendar"]
    years = range(y.index[0].year, y.index[-1].year + 2 + horizon // 365)
    hol = pd.DataFrame(
        [
            {"holiday": "festive", "ds": pd.Timestamp(f"{yr}-{md}"), "lower_window": 0, "upper_window": 0}
            for yr in years
            for md in cal["fixed_holidays"]
        ]
    )
    m = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        holidays=hol,
        changepoint_prior_scale=0.05,
        uncertainty_samples=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(pd.DataFrame({"ds": y.index, "y": y.to_numpy()}))
        fut = pd.DataFrame(
            {"ds": pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")}
        )
        pred = m.predict(fut)
    return np.clip(pred["yhat"].to_numpy(), 0, None)


# ------------------------------------------------- Gradient boosted trees
def make_gbm(kind: str, cfg: dict[str, Any]):
    """Create an XGBoost or LightGBM regressor from config."""
    g = cfg["forecasting"]["gbm"]
    seed = cfg["project"]["seed"]
    if kind == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=g["n_estimators"],
            learning_rate=g["learning_rate"],
            max_depth=g["max_depth"],
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )
    if kind == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=g["n_estimators"],
            learning_rate=g["learning_rate"],
            max_depth=g["max_depth"],
            num_leaves=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=seed,
            verbose=-1,
        )
    raise ValueError(f"unknown GBM kind {kind!r}")


def fit_gbm(y: pd.Series, cfg: dict[str, Any], kind: str = "xgboost"):
    """Fit a lag-feature GBM; returns ``(model, feature_names)``."""
    sf = supervised_frame(y, cfg)
    X, target = sf.drop(columns="y"), sf["y"]
    model = make_gbm(kind, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, target)
    return model, list(X.columns)


def gbm_recursive_predict(
    model, cols: list[str], y: pd.Series, horizon: int, cfg: dict[str, Any]
) -> np.ndarray:
    """Multi-step forecast: predict a day, append it to the history, build the next features."""
    keep = max(cfg["features"]["lags"] + cfg["features"]["rolling_windows"]) + 3
    buf = y.iloc[-keep:].copy()
    out = np.empty(horizon)
    for i in range(horizon):
        nxt = buf.index[-1] + pd.Timedelta(days=1)
        ext = pd.concat([buf, pd.Series([np.nan], index=[nxt])])
        X = build_feature_frame(ext, cfg).iloc[[-1]][cols]
        pred = max(float(model.predict(X)[0]), 0.0)
        out[i] = pred
        buf = ext.copy()
        buf.iloc[-1] = pred
        buf = buf.iloc[-keep:]
    return out


def _gbm_forecaster(kind: str) -> Forecaster:
    def _f(y: pd.Series, horizon: int, cfg: dict[str, Any] | None = None) -> np.ndarray:
        cfg = cfg or load_config()
        model, cols = fit_gbm(y, cfg, kind)
        return gbm_recursive_predict(model, cols, y, horizon, cfg)

    _f.__name__ = f"{kind}_forecast"
    return _f


FORECASTERS: dict[str, Forecaster] = {
    "naive": naive_forecast,
    "snaive": seasonal_naive_forecast,
    "sarimax": sarimax_forecast,
    "prophet": prophet_forecast,
    "xgboost": _gbm_forecaster("xgboost"),
    "lightgbm": _gbm_forecaster("lightgbm"),
}

MODEL_LABELS = {
    "naive": "Naive",
    "snaive": "Seasonal naive (7d)",
    "sarimax": "SARIMAX (weekly+Fourier)",
    "prophet": "Prophet",
    "xgboost": "XGBoost (lags)",
    "lightgbm": "LightGBM (lags)",
}
