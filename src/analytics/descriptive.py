"""Descriptive analytics: KPIs, STL decomposition, ABC-XYZ, demand-class features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import category_assumptions, load_config


def add_revenue(daily: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Revenue = units x *assumed* unit price (the dataset has no price column)."""
    cfg = cfg or load_config()
    price = {c: category_assumptions(cfg, c)["unit_price"] for c in daily["category"].unique()}
    out = daily.copy()
    out["unit_price"] = out["category"].map(price)
    out["revenue"] = out["units"] * out["unit_price"]
    return out


def kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Headline KPIs for a (filtered) daily frame with ``units`` and ``revenue``.

    ``avg_daily_units`` replaces "average basket": the data has no transaction ids,
    so basket size cannot be derived.
    """
    if df.empty:
        return {}
    by_cat = df.groupby("category")[["units", "revenue"]].sum().sort_values("revenue", ascending=False)
    monthly = df.set_index("date").resample("MS")["revenue"].sum()
    # drop incomplete first/last month so growth is not distorted
    first, last = df["date"].min(), df["date"].max()
    if first.day > 1 and len(monthly):
        monthly = monthly.iloc[1:]
    if (last + pd.Timedelta(days=1)).day != 1 and len(monthly):
        monthly = monthly.iloc[:-1]
    mom = float((monthly.iloc[-1] / monthly.iloc[-2] - 1) * 100) if len(monthly) >= 2 else np.nan
    yoy = float((monthly.iloc[-1] / monthly.iloc[-13] - 1) * 100) if len(monthly) >= 13 else np.nan
    days = df["date"].nunique()
    return {
        "total_revenue": float(df["revenue"].sum()),
        "total_units": float(df["units"].sum()),
        "avg_daily_units": float(df["units"].sum() / max(days, 1)),
        "top_categories": by_cat.head(3),
        "mom_growth_pct": mom,
        "yoy_growth_pct": yoy,
        "last_complete_month": monthly.index[-1] if len(monthly) else None,
        "n_days": int(days),
    }


def drop_partial_months(monthly: pd.DataFrame | pd.Series, df: pd.DataFrame, tol: float = 0.9):
    """Remove months for which ``df`` has fewer than ``tol`` of the calendar days (partial months)."""
    days = df.groupby(df["date"].dt.to_period("M").dt.to_timestamp())["date"].nunique()
    dim = days.index.days_in_month
    ok = days.index[(days / dim) >= tol]
    idx = monthly.index.get_level_values(-1) if isinstance(monthly.index, pd.MultiIndex) else monthly.index
    return monthly[idx.isin(ok)]


def stl_decompose(y: pd.Series, period: int = 7, robust: bool = True):
    """STL decomposition (statsmodels). Returns the result object."""
    from statsmodels.tsa.seasonal import STL

    return STL(y, period=period, robust=robust).fit()


def seasonal_strength(y: pd.Series, period: int) -> float:
    """Wang-Smith-Hyndman seasonal strength F_s = max(0, 1 - Var(R)/Var(S+R)) in [0, 1]."""
    res = stl_decompose(y, period)
    return float(max(0.0, 1 - res.resid.var() / (res.seasonal + res.resid).var()))


def trend_strength(y: pd.Series, period: int = 7) -> float:
    res = stl_decompose(y, period)
    return float(max(0.0, 1 - res.resid.var() / (res.trend + res.resid).var()))


# ------------------------------------------------------------------ ABC / XYZ
def abc_classification(revenue: pd.Series, thresholds: tuple[float, float] = (0.80, 0.95)) -> pd.DataFrame:
    """Pareto/ABC analysis on revenue contribution.

    Items are sorted by revenue; the class is decided by the cumulative share *before*
    adding the item, so the first item that crosses the 80 % line is still class A.
    """
    s = revenue.sort_values(ascending=False)
    share = s / s.sum()
    cum = share.cumsum()
    prev = cum - share
    cls = np.where(prev < thresholds[0], "A", np.where(prev < thresholds[1], "B", "C"))
    return pd.DataFrame({"revenue": s, "share": share, "cum_share": cum, "abc": cls})


def xyz_classification(
    demand: pd.DataFrame, freq: str = "W", thresholds: tuple[float, float] = (0.30, 0.45)
) -> pd.DataFrame:
    """XYZ by coefficient of variation of aggregated demand (daily CV is dominated by noise)."""
    wide = demand.pivot(index="date", columns="category", values="units")
    agg = wide.resample(freq).sum()
    agg = agg.iloc[1:-1] if len(agg) > 2 else agg  # drop partial first/last period
    cv = agg.std() / agg.mean()
    cls = np.where(cv <= thresholds[0], "X", np.where(cv <= thresholds[1], "Y", "Z"))
    return pd.DataFrame({"cv": cv, "xyz": cls})


STOCKING_POLICY = {
    "AX": "High value, stable: continuous review, lean safety stock, automate replenishment (JIT).",
    "AY": "High value, variable: forecast-driven replenishment, moderate safety stock, weekly review.",
    "AZ": "High value, erratic: tight monitoring, higher safety stock, short order cycles, review forecasts often.",
    "BX": "Medium value, stable: periodic review with fixed order quantities (EOQ).",
    "BY": "Medium value, variable: periodic review, seasonal safety-stock adjustment.",
    "BZ": "Medium value, erratic: buffer stock with caps, order on demand signals.",
    "CX": "Low value, stable: bulk/infrequent ordering, minimal attention (two-bin).",
    "CY": "Low value, variable: simple min-max rules, review monthly.",
    "CZ": "Low value, erratic: stock on request or small buffer; consider delisting if slow.",
}


def abc_xyz_matrix(daily_rev: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Combine ABC (revenue) and XYZ (weekly demand CV) with a recommended stocking policy."""
    cfg = cfg or load_config()
    a = cfg["abc_xyz"]
    abc = abc_classification(daily_rev.groupby("category")["revenue"].sum(), tuple(a["abc_thresholds"]))
    xyz = xyz_classification(
        daily_rev[["date", "category", "units"]], a["xyz_aggregation"], tuple(a["xyz_cv_thresholds"])
    )
    out = abc.join(xyz)
    out["class"] = out["abc"] + out["xyz"]
    out["policy"] = out["class"].map(STOCKING_POLICY)
    return out.reset_index(names="category")


# ------------------------------------------------------- demand-class features
def demand_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-category behavioural features used for clustering / demand typing."""
    rows = []
    for cat, g in daily.groupby("category"):
        y = g.set_index("date")["units"].asfreq("D")
        weekly = y.resample("W").sum().iloc[1:-1]
        monthly = y.resample("MS").sum().iloc[1:-1]
        nz = y[y > 0]
        adi = len(y) / max(len(nz), 1)  # average demand interval
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan  # squared CV of non-zero sizes
        t = np.arange(len(weekly))
        slope = np.polyfit(t, weekly.to_numpy(), 1)[0] / weekly.mean()  # relative weekly trend
        wk = y.groupby(y.index.weekday).mean()
        rows.append(
            {
                "category": cat,
                "mean_daily": y.mean(),
                "cv_weekly": weekly.std() / weekly.mean(),
                "cv_daily": y.std() / y.mean(),
                "adi": adi,
                "cv2_nonzero": cv2,
                "zero_share": float((y == 0).mean()),
                "weekly_seasonality": seasonal_strength(y, 7),
                "annual_seasonality": float(
                    monthly.groupby(monthly.index.month).mean().std() / monthly.mean()
                ),
                "trend_slope_rel": slope,
                "weekend_ratio": float(wk.loc[[5, 6]].mean() / wk.mean()),
                "skew": float(y.skew()),
            }
        )
    return pd.DataFrame(rows).set_index("category")


def rule_based_label(row: pd.Series, volume_median: float) -> str:
    """Interpretable demand label (thresholds are heuristics; volume is relative to the portfolio).

    Intermittent        > 25 % zero-sales days
    Seasonal & erratic  weekly CV > 0.45 and strong annual pattern
    Erratic             weekly CV > 0.45
    Seasonal            strong annual pattern (annual-seasonality index > 0.22)
    Fast / Slow-moving  mean daily demand above / below the portfolio median
    """
    seasonal = row["annual_seasonality"] > 0.22
    if row["zero_share"] > 0.25:
        return "Intermittent"
    if row["cv_weekly"] > 0.45:
        return "Seasonal & erratic" if seasonal else "Erratic"
    if seasonal:
        return "Seasonal"
    return "Fast-moving" if row["mean_daily"] >= volume_median else "Slow-moving"
