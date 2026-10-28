"""Rolling-origin (expanding-window) cross-validation, model selection and prediction intervals."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.models.forecasters import FORECASTERS

log = logging.getLogger(__name__)


def rolling_origins(n: int, horizon: int, step: int, n_folds: int, min_train: int) -> list[int]:
    """Positional forecast origins (= size of the training set) for each fold.

    The last fold's test window ends exactly at the end of the series.  Every training
    set ``y[:origin]`` strictly precedes its test window, so there is no look-ahead.
    """
    origins = [n - horizon - (n_folds - 1 - i) * step for i in range(n_folds)]
    origins = [o for o in origins if o >= min_train]
    return origins


def run_cv(
    category: str, y: pd.Series, cfg: dict[str, Any] | None = None, models: list[str] | None = None
) -> pd.DataFrame:
    """Evaluate every model on all folds; returns long prediction frame.

    Columns: category, model, fold, origin (last train date), date, h, y, yhat.
    """
    cfg = cfg or load_config()
    f = cfg["forecasting"]
    cv = f["cv"]
    models = models or f["models"]
    origins = rolling_origins(len(y), cv["horizon"], cv["step"], cv["n_folds"], cv["min_train_days"])
    rows = []
    for name in models:
        fn = FORECASTERS[name]
        t0 = time.time()
        for k, o in enumerate(origins):
            train, test = y.iloc[:o], y.iloc[o : o + cv["horizon"]]
            try:
                pred = fn(train, len(test), cfg)
            except Exception as exc:  # a single failing fit must not kill the run
                log.warning("%s/%s fold %d failed (%s); using seasonal naive", category, name, k, exc)
                pred = FORECASTERS["snaive"](train, len(test), cfg)
            rows.append(
                pd.DataFrame(
                    {
                        "category": category,
                        "model": name,
                        "fold": k,
                        "origin": train.index[-1],
                        "date": test.index,
                        "h": np.arange(1, len(test) + 1),
                        "y": test.to_numpy(),
                        "yhat": pred,
                    }
                )
            )
        log.info("CV %s | %-9s %d folds in %.1fs", category, name, len(origins), time.time() - t0)
    return pd.concat(rows, ignore_index=True)


def select_best(metrics: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """Lowest-``metric`` model per category."""
    idx = metrics.groupby("category")[metric].idxmin()
    return metrics.loc[idx].reset_index(drop=True)


def improvement_vs_baseline(metrics: pd.DataFrame, baseline: str = "snaive") -> pd.DataFrame:
    """Percentage reduction of each error metric relative to ``baseline`` (positive = better)."""
    base = metrics[metrics["model"] == baseline].set_index("category")
    out = metrics.copy()
    for m in ["mae", "rmse", "mape", "smape"]:
        out[f"{m}_improv_pct"] = (1 - out[m] / out["category"].map(base[m])) * 100
    return out


# --------------------------------------------------------------------------- #
# Prediction intervals (model agnostic, empirical)
# --------------------------------------------------------------------------- #
H_BUCKETS = [(1, 14), (15, 30), (31, 60), (61, 90)]


def _std_resid(y, yhat) -> np.ndarray:
    """Residuals scaled by sqrt of the forecast level (count data: variance grows with mean)."""
    return (np.asarray(y) - np.asarray(yhat)) / np.sqrt(np.maximum(np.asarray(yhat), 1.0))


def calibration_residuals(y: pd.Series, model: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Out-of-sample forecasts from several *older* origins with the full horizon."""
    ic = cfg["forecasting"]["interval"]
    H = ic["calibration_horizon"]
    rows = []
    for k in range(ic["calibration_origins"]):
        o = len(y) - H - k * ic["calibration_step"]
        if o < cfg["forecasting"]["cv"]["min_train_days"]:
            break
        train, test = y.iloc[:o], y.iloc[o : o + H]
        pred = FORECASTERS[model](train, len(test), cfg)
        rows.append(
            pd.DataFrame(
                {"origin_k": k, "h": np.arange(1, len(test) + 1), "y": test.to_numpy(), "yhat": pred}
            )
        )
    return pd.concat(rows, ignore_index=True)


def interval_quantiles(cal: pd.DataFrame, level: float) -> dict[tuple[int, int], tuple[float, float]]:
    """Lower/upper quantiles of standardised residuals per horizon bucket."""
    a = (1 - level) / 2
    out = {}
    for lo, hi in H_BUCKETS:
        g = cal[(cal["h"] >= lo) & (cal["h"] <= hi)]
        if g.empty:
            continue
        r = _std_resid(g["y"], g["yhat"])
        out[(lo, hi)] = (float(np.quantile(r, a)), float(np.quantile(r, 1 - a)))
    return out


def apply_intervals(
    yhat: np.ndarray, q: dict[tuple[int, int], tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert point forecasts + bucket quantiles into (lower, upper) bounds, floored at 0."""
    h = np.arange(1, len(yhat) + 1)
    lo, hi = np.zeros(len(yhat)), np.zeros(len(yhat))
    last = next(iter(q.values()))
    for i, hh in enumerate(h):
        for (a, b), v in q.items():
            if a <= hh <= b:
                last = v
                break
        scale = np.sqrt(max(yhat[i], 1.0))
        lo[i] = max(yhat[i] + last[0] * scale, 0.0)
        hi[i] = yhat[i] + last[1] * scale
    return lo, hi


def interval_coverage_check(cal: pd.DataFrame, level: float) -> float:
    """Honest coverage: calibrate on older origins, score on the most recent half."""
    ks = sorted(cal["origin_k"].unique())
    if len(ks) < 4:
        return float("nan")
    recent = ks[: len(ks) // 2]  # k=0 is the most recent origin
    train_part, test_part = cal[~cal["origin_k"].isin(recent)], cal[cal["origin_k"].isin(recent)]
    q = interval_quantiles(train_part, level)
    hits = []
    for _, g in test_part.groupby("origin_k"):
        lo, hi = apply_intervals(g["yhat"].to_numpy(), q)
        hits.append(((g["y"].to_numpy() >= lo) & (g["y"].to_numpy() <= hi)))
    return float(np.mean(np.concatenate(hits)))
