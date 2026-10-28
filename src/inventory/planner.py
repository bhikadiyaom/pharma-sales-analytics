"""Turn forecasts into an inventory plan table (reorder list + risk flags)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import category_assumptions
from src.inventory.policy import expiry_risk_units, plan_item, stock_status


def default_inputs(
    cfg: dict[str, Any], fb: pd.DataFrame, categories: list[str], window: int, cover_days: float | None = None
) -> pd.DataFrame:
    """Editable per-category inputs: lead time, stock on hand, remaining shelf life.

    Stock on hand is NOT in the dataset: default = ``cover_days`` (config
    ``default_days_of_cover`` if omitted) x mean forecast.
    """
    cover_days = cfg["inventory"]["default_days_of_cover"] if cover_days is None else cover_days
    rows = []
    for c in categories:
        a = category_assumptions(cfg, c)
        d = float(fb[fb["category"] == c].sort_values("h")["yhat"].head(window).mean())
        rows.append(
            {
                "category": c,
                "lead_time_days": float(a["lead_time_days"]),
                "on_hand": round(d * cover_days, 0),
                "shelf_life_days": float(a["shelf_life_remaining_days"]),
            }
        )
    return pd.DataFrame(rows)


def build_plan_table(
    cfg: dict[str, Any],
    fb: pd.DataFrame,
    best: pd.DataFrame,
    inputs: pd.DataFrame,
    service_level: float,
    window: int,
) -> pd.DataFrame:
    """One row per category: mean demand, SS, ROP, EOQ, status, expiry risk, suggested order.

    ``fb``  best-model forecasts (columns category, h, yhat)
    ``best`` best-model metrics indexed by category (needs ``rmse`` and ``model``)
    """
    inv = cfg["inventory"]
    rows = []
    for r in inputs.itertuples():
        c = r.category
        a = category_assumptions(cfg, c)
        f = fb[fb["category"] == c].sort_values("h")["yhat"].to_numpy()
        d = float(f[:window].mean())
        sigma = float(best.loc[c, "rmse"])
        p = plan_item(
            d,
            sigma,
            r.lead_time_days,
            service_level,
            a["order_cost"],
            a["unit_cost"],
            inv["holding_rate_annual"],
            a["lead_time_sd_days"],
        )
        status = stock_status(r.on_hand, p, r.lead_time_days, inv["overstock_cover_days"])
        exp_units = expiry_risk_units(r.on_hand, f, int(r.shelf_life_days))
        order_qty = max(0.0, p.max_stock - r.on_hand) if status in ("REORDER NOW", "STOCKOUT RISK") else 0.0
        rows.append(
            {
                "category": c,
                "model": best.loc[c, "model"],
                "mean_daily_forecast": d,
                "daily_error_rmse": sigma,
                "lead_time_days": r.lead_time_days,
                "safety_stock": p.safety_stock,
                "reorder_point": p.reorder_point,
                "eoq": p.eoq,
                "max_stock": p.max_stock,
                "on_hand": r.on_hand,
                "days_of_cover": r.on_hand / d if d > 0 else np.inf,
                "status": status,
                "suggested_order_qty": order_qty,
                "shelf_life_days": r.shelf_life_days,
                "expiry_risk_units": exp_units,
                "expiry_risk_value": exp_units * a["unit_cost"],
                "service_level": service_level,
            }
        )
    return pd.DataFrame(rows)
