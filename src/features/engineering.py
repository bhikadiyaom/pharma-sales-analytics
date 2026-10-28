"""Feature engineering: calendar, lag, rolling and (optional) price/promotion features.

Leakage rule: every lag / rolling feature at day *t* only uses information up to
day *t-1* (rolling statistics are computed on ``y.shift(1)``).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.config import load_config


def holiday_flags(idx: pd.DatetimeIndex, fixed: Sequence[str], window: int = 1) -> pd.DataFrame:
    """Flag fixed-date holidays (``MM-DD``) and the +/- ``window`` days around them."""
    md = idx.strftime("%m-%d")
    is_hol = pd.Series(md.isin(list(fixed)), index=idx)
    near = is_hol.copy()
    for k in range(1, window + 1):
        near |= is_hol.shift(k, fill_value=False) | is_hol.shift(-k, fill_value=False)
    return pd.DataFrame({"is_holiday": is_hol.astype(int), "near_holiday": near.astype(int)}, index=idx)


def calendar_features(idx: pd.DatetimeIndex, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Weekday, month, quarter, cyclic day-of-year and holiday flags."""
    cfg = cfg or load_config()
    cal = cfg["calendar"]
    doy = idx.dayofyear.to_numpy()
    out = pd.DataFrame(
        {
            "weekday": idx.weekday,
            "is_weekend": (idx.weekday >= 5).astype(int),
            "month": idx.month,
            "quarter": idx.quarter,
            "day_of_month": idx.day,
            "week_of_year": idx.isocalendar().week.to_numpy().astype(int),
            "is_month_end": idx.is_month_end.astype(int),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=idx,
    )
    return out.join(holiday_flags(idx, cal["fixed_holidays"], cal["holiday_window_days"]))


def lag_features(y: pd.Series, lags: Sequence[int]) -> pd.DataFrame:
    """``lag_k`` = demand k days earlier."""
    return pd.DataFrame({f"lag_{k}": y.shift(k) for k in lags})


def rolling_features(y: pd.Series, windows: Sequence[int]) -> pd.DataFrame:
    """Rolling mean / std over the *previous* ``w`` days (shifted by one to avoid leakage)."""
    prev = y.shift(1)
    cols = {}
    for w in windows:
        cols[f"roll_mean_{w}"] = prev.rolling(w, min_periods=max(2, w // 2)).mean()
        cols[f"roll_std_{w}"] = prev.rolling(w, min_periods=max(2, w // 2)).std()
    return pd.DataFrame(cols)


def price_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price and promotion features, created only when those columns exist.

    The real Kaggle data has no price/promotion columns, so this is a no-op there;
    it is exercised by tests and used automatically if a richer dataset is plugged in.
    """
    out = pd.DataFrame(index=df.index)
    if "price" in df.columns:
        out["price"] = df["price"]
        out["price_change_pct"] = df["price"].pct_change().fillna(0.0)
        out["price_rel_28"] = df["price"] / df["price"].rolling(28, min_periods=1).mean()
    if "promo_flag" in df.columns:
        out["promo_flag"] = df["promo_flag"].astype(int)
    return out


def build_feature_frame(
    y: pd.Series, cfg: dict[str, Any] | None = None, extra: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return the model matrix (features only, no target) indexed like ``y``."""
    cfg = cfg or load_config()
    f = cfg["features"]
    parts = [
        calendar_features(y.index, cfg),
        lag_features(y, f["lags"]).set_axis(y.index),
        rolling_features(y, f["rolling_windows"]).set_axis(y.index),
    ]
    if extra is not None:
        parts.append(price_promo_features(extra.reindex(y.index)))
    return pd.concat(parts, axis=1)


def feature_columns(cfg: dict[str, Any] | None = None) -> list[str]:
    """Names of the model features in a fixed order."""
    dummy = pd.Series(np.arange(60, dtype=float), index=pd.date_range("2020-01-01", periods=60))
    return list(build_feature_frame(dummy, cfg).columns)


def supervised_frame(y: pd.Series, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Features + target ``y`` with rows lacking full lag history dropped."""
    X = build_feature_frame(y, cfg)
    X["y"] = y
    return X.dropna()
