"""What-if simulator: price, promotion, lead time and demand-surge scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from src.inventory.policy import InventoryPlan, plan_item


@dataclass
class Scenario:
    """User-controlled changes relative to the baseline."""

    price_change_pct: float = 0.0  # +10 => price 10 % higher
    promo_uplift_pct: float = 0.0  # demand uplift caused by a promotion
    demand_surge_pct: float = 0.0  # exogenous surge, e.g. epidemic +30 %
    lead_time_change_days: float = 0.0
    price_elasticity: float = -0.6  # assumed own-price elasticity of demand


@dataclass
class WhatIfResult:
    base_demand: float
    new_demand: float
    base_price: float
    new_price: float
    base_revenue_30d: float
    new_revenue_30d: float
    base_plan: InventoryPlan
    new_plan: InventoryPlan
    base_lead_time: float
    new_lead_time: float


def adjusted_demand(base_daily: float, sc: Scenario) -> float:
    """Demand multiplier: (1+elasticity*dP) * (1+promo) * (1+surge), floored at 0."""
    mult = 1 + sc.price_elasticity * sc.price_change_pct / 100.0
    mult *= (1 + sc.promo_uplift_pct / 100.0) * (1 + sc.demand_surge_pct / 100.0)
    return max(base_daily * mult, 0.0)


def simulate(
    mean_daily: float,
    sigma_daily: float,
    unit_price: float,
    unit_cost: float,
    lead_time: float,
    sigma_lead_time: float,
    service_level: float,
    order_cost: float,
    holding_rate: float,
    sc: Scenario,
    days: int = 30,
) -> WhatIfResult:
    """Compare baseline vs scenario for stock requirements and revenue over ``days``.

    Demand variability is assumed to scale proportionally with the demand level.
    """
    new_d = adjusted_demand(mean_daily, sc)
    scale = new_d / mean_daily if mean_daily > 0 else 1.0
    new_lt = max(lead_time + sc.lead_time_change_days, 0.0)
    new_price = unit_price * (1 + sc.price_change_pct / 100.0)
    base = plan_item(
        mean_daily,
        sigma_daily,
        lead_time,
        service_level,
        order_cost,
        unit_cost,
        holding_rate,
        sigma_lead_time,
    )
    new = plan_item(
        new_d,
        sigma_daily * scale,
        new_lt,
        service_level,
        order_cost,
        unit_cost,
        holding_rate,
        sigma_lead_time,
    )
    return WhatIfResult(
        mean_daily,
        new_d,
        unit_price,
        new_price,
        mean_daily * unit_price * days,
        new_d * new_price * days,
        base,
        new,
        lead_time,
        new_lt,
    )
