"""Back-test of inventory policies on the out-of-sample (CV) period.

Two periodic-review order-up-to policies are replayed day by day against the *actual*
demand of the last CV year, with identical lead time and target service level:

* ``baseline``  - classic practice: trailing 28-day mean and std at each review date
* ``forecast``  - mean from the selected forecasting model, sigma from that model's
                  *previous* out-of-sample errors (no look-ahead)

Order-up-to level  S = demand over (L + R) days + z * sigma * sqrt(L + R)
Unmet demand is lost (no backorders).  Everything is computed from data; the only
assumptions are the lead time / review period taken from ``config.yaml``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import category_assumptions, load_config
from src.inventory.policy import z_value


def replay_policy(
    demand: np.ndarray, review_days: list[int], targets: list[float], lead_time: int
) -> dict[str, float]:
    """Simulate one policy. ``review_days[i]`` is the index at which order-up-to ``targets[i]`` is set."""
    n = len(demand)
    on_hand = targets[0]  # start the test year fully stocked
    pipeline: dict[int, float] = {}
    served = lost = 0.0
    stockout_days = 0
    inv_sum = 0.0
    review_set = dict(zip(review_days, targets))
    for t in range(n):
        on_hand += pipeline.pop(t, 0.0)  # arrivals at start of day
        if t in review_set:
            position = on_hand + sum(pipeline.values())
            order = max(0.0, review_set[t] - position)
            if order > 0:
                pipeline[t + lead_time] = pipeline.get(t + lead_time, 0.0) + order
        d = demand[t]
        sold = min(on_hand, d)
        served += sold
        lost += d - sold
        stockout_days += int(d > on_hand + 1e-9)
        on_hand -= sold
        inv_sum += on_hand
    total = served + lost
    return {
        "fill_rate": served / total if total else 1.0,
        "stockout_day_share": stockout_days / n,
        "avg_inventory": inv_sum / n,
        "lost_units": lost,
        "demand_units": total,
    }


def backtest_category(
    cat: str, cv_best: pd.DataFrame, history: pd.Series, cfg: dict[str, Any], service_level: float
) -> dict[str, dict[str, float]]:
    """Replay baseline and forecast-driven policies for one category."""
    inv = cfg["inventory"]
    a = category_assumptions(cfg, cat)
    L = int(round(a["lead_time_days"]))
    R = int(inv["review_period_days"])
    z = z_value(service_level)
    win = inv["simulation"]["baseline_window_days"]

    folds = sorted(cv_best["fold"].unique())
    demand, review_days = [], []
    t_f, t_b, prev_err = [], [], []
    pos = 0
    for k in folds:
        g = cv_best[cv_best["fold"] == k].sort_values("h")
        origin = g["origin"].iloc[0]
        trail = history.loc[:origin].iloc[-win:]
        yhat = g["yhat"].to_numpy()

        # demand over lead time + review: forecast window + flat extension for the L overshoot
        def cover(level_path: np.ndarray, mean_level: float) -> float:
            return float(level_path[:R].sum() + mean_level * L)

        sig_b = float(trail.std(ddof=1))
        t_b.append(cover(np.repeat(trail.mean(), R), trail.mean()) + z * sig_b * np.sqrt(L + R))
        sig_f = float(np.sqrt(np.mean(np.square(prev_err)))) if len(prev_err) >= 3 else sig_b
        t_f.append(cover(yhat, yhat.mean()) + z * sig_f * np.sqrt(L + R))
        prev_err.extend((g["y"] - g["yhat"]).tolist())
        demand.extend(g["y"].tolist())
        review_days.append(pos)
        pos += len(g)
    d = np.asarray(demand)
    return {
        "baseline": replay_policy(d, review_days, t_b, L),
        "forecast": replay_policy(d, review_days, t_f, L),
    }


def backtest_all(
    cv_best: pd.DataFrame,
    history: dict[str, pd.Series],
    cfg: dict[str, Any] | None = None,
    service_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Tidy table: category x service level x policy -> fill rate, stockout share, avg inventory."""
    cfg = cfg or load_config()
    service_levels = service_levels or cfg["inventory"]["service_levels"]
    rows = []
    for sl in service_levels:
        for cat, g in cv_best.groupby("category"):
            res = backtest_category(cat, g, history[cat], cfg, sl)
            cost = category_assumptions(cfg, cat)["unit_cost"]
            for pol, m in res.items():
                rows.append(
                    {
                        "category": cat,
                        "service_level": sl,
                        "policy": pol,
                        **m,
                        "avg_inventory_value": m["avg_inventory"] * cost,
                    }
                )
    return pd.DataFrame(rows)
