"""Inventory formulas: safety stock, reorder point, EOQ, and risk flags.

Notation (all per day unless stated):
    d   mean forecast demand        sigma  std of the *daily* forecast error
    L   supplier lead time (days)   sigma_L std of lead time (days)
    z   standard-normal quantile of the target cycle service level
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def z_value(service_level: float) -> float:
    """z such that P(Z <= z) = service level."""
    if not 0.5 <= service_level < 1:
        raise ValueError("service level must be in [0.5, 1)")
    return float(norm.ppf(service_level))


def safety_stock(
    sigma_daily: float,
    lead_time: float,
    service_level: float,
    mean_daily: float = 0.0,
    sigma_lead_time: float = 0.0,
) -> float:
    """SS = z * sqrt(L*sigma^2 + d^2 * sigma_L^2).

    The first term is demand/forecast-error variability over the lead time (errors of
    consecutive days assumed independent); the second is supplier variability.
    """
    var = lead_time * sigma_daily**2 + (mean_daily**2) * sigma_lead_time**2
    return z_value(service_level) * math.sqrt(max(var, 0.0))


def reorder_point(mean_daily: float, lead_time: float, ss: float) -> float:
    """ROP = d * L + SS."""
    return mean_daily * lead_time + ss


def eoq(annual_demand: float, order_cost: float, unit_cost: float, holding_rate: float) -> float:
    """Economic order quantity sqrt(2 D S / H) with H = holding_rate * unit_cost per unit-year."""
    h = holding_rate * unit_cost
    if annual_demand <= 0 or h <= 0:
        return 0.0
    return math.sqrt(2 * annual_demand * order_cost / h)


@dataclass
class InventoryPlan:
    """Result of :func:`plan_item`."""

    mean_daily: float
    sigma_daily: float
    safety_stock: float
    reorder_point: float
    eoq: float
    max_stock: float  # ROP + EOQ, the (s, Q) upper level
    z: float
    service_level: float


def plan_item(
    mean_daily: float,
    sigma_daily: float,
    lead_time: float,
    service_level: float,
    order_cost: float,
    unit_cost: float,
    holding_rate: float,
    sigma_lead_time: float = 0.0,
) -> InventoryPlan:
    """Compute SS, ROP, EOQ for one item."""
    ss = safety_stock(sigma_daily, lead_time, service_level, mean_daily, sigma_lead_time)
    rop = reorder_point(mean_daily, lead_time, ss)
    q = eoq(mean_daily * 365, order_cost, unit_cost, holding_rate)
    return InventoryPlan(mean_daily, sigma_daily, ss, rop, q, rop + q, z_value(service_level), service_level)


# -------------------------------------------------------------- risk flags
def stock_status(on_hand: float, plan: InventoryPlan, lead_time: float, overstock_cover_days: float) -> str:
    """Classify stock position.

    * STOCKOUT RISK : stock will not even cover expected lead-time demand
    * REORDER NOW   : below the reorder point
    * OVERSTOCK     : more than ``overstock_cover_days`` of demand (or above ROP+EOQ)
    * OK            : otherwise
    """
    if on_hand < plan.mean_daily * lead_time:
        return "STOCKOUT RISK"
    if on_hand < plan.reorder_point:
        return "REORDER NOW"
    cover = on_hand / plan.mean_daily if plan.mean_daily > 0 else np.inf
    if cover > overstock_cover_days or on_hand > plan.max_stock * 1.5:
        return "OVERSTOCK"
    return "OK"


def expiry_risk_units(on_hand: float, forecast_daily: np.ndarray, remaining_shelf_days: int) -> float:
    """Units expected to expire unsold: ``max(0, on_hand - forecast demand until expiry)``.

    If the shelf life extends beyond the forecast, the mean forecast level is extrapolated.
    """
    if remaining_shelf_days <= 0:
        return float(on_hand)
    f = np.asarray(forecast_daily, dtype=float)
    if remaining_shelf_days <= len(f):
        demand = f[:remaining_shelf_days].sum()
    else:
        demand = f.sum() + f.mean() * (remaining_shelf_days - len(f))
    return float(max(0.0, on_hand - demand))
