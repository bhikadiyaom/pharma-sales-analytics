import math

import numpy as np
import pytest

from src.inventory import policy as P
from src.inventory import simulate as S
from src.inventory.whatif import Scenario, adjusted_demand, simulate


def test_z_values():
    assert P.z_value(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert P.z_value(0.99) == pytest.approx(2.3263, abs=1e-3)
    with pytest.raises(ValueError):
        P.z_value(1.0)


def test_safety_stock_formula():
    # z * sqrt(L * sigma^2) with no lead-time variability
    assert P.safety_stock(10, 4, 0.95) == pytest.approx(1.6449 * math.sqrt(4) * 10, rel=1e-3)
    # lead-time variability adds d^2 * sigma_L^2 under the root
    ss = P.safety_stock(10, 4, 0.95, mean_daily=20, sigma_lead_time=1)
    assert ss == pytest.approx(1.6449 * math.sqrt(4 * 100 + 400), rel=1e-3)
    # higher service level => more safety stock
    assert P.safety_stock(10, 4, 0.99) > P.safety_stock(10, 4, 0.90) > 0


def test_reorder_point_and_eoq():
    assert P.reorder_point(20, 5, 30) == 130
    # D=36500/yr, S=500, H=0.2*30=6 => sqrt(2*36500*500/6)
    assert P.eoq(36500, 500, 30, 0.2) == pytest.approx(math.sqrt(2 * 36500 * 500 / 6))
    assert P.eoq(0, 500, 30, 0.2) == 0.0


def test_plan_item_consistency():
    p = P.plan_item(20, 8, 5, 0.95, 500, 30, 0.2, 1.0)
    assert p.reorder_point == pytest.approx(20 * 5 + p.safety_stock)
    assert p.max_stock == pytest.approx(p.reorder_point + p.eoq)


def test_stock_status_classification():
    p = P.plan_item(20, 8, 5, 0.95, 500, 30, 0.2)
    assert P.stock_status(50, p, 5, 60) == "STOCKOUT RISK"  # < 100 lead-time demand
    assert P.stock_status(p.reorder_point - 1, p, 5, 60) == "REORDER NOW"
    assert P.stock_status(p.reorder_point + 10, p, 5, 60) == "OK"
    assert P.stock_status(20 * 61, p, 5, 60) == "OVERSTOCK"


def test_expiry_risk_units():
    f = np.full(30, 10.0)
    assert P.expiry_risk_units(500, f, 30) == 200  # 300 sold, 200 expire
    assert P.expiry_risk_units(100, f, 30) == 0
    assert P.expiry_risk_units(500, f, 60) == 0  # extrapolated demand 600
    assert P.expiry_risk_units(50, f, 0) == 50


def test_whatif_directionality():
    base = dict(
        mean_daily=100,
        sigma_daily=20,
        unit_price=40,
        unit_cost=28,
        lead_time=5,
        sigma_lead_time=1,
        service_level=0.95,
        order_cost=500,
        holding_rate=0.2,
    )
    surge = simulate(**base, sc=Scenario(demand_surge_pct=30))
    assert surge.new_demand == pytest.approx(130)
    assert surge.new_plan.reorder_point > surge.base_plan.reorder_point
    assert surge.new_revenue_30d == pytest.approx(surge.base_revenue_30d * 1.3)
    longer = simulate(**base, sc=Scenario(lead_time_change_days=3))
    assert longer.new_plan.safety_stock > longer.base_plan.safety_stock
    assert simulate(**base, sc=Scenario()).new_plan.reorder_point == pytest.approx(
        simulate(**base, sc=Scenario()).base_plan.reorder_point
    )
    # price rise with negative elasticity lowers demand
    assert adjusted_demand(100, Scenario(price_change_pct=10, price_elasticity=-0.6)) == pytest.approx(94)


def test_replay_policy_conservation_and_service():
    d = np.full(28, 10.0)
    full = S.replay_policy(d, [0, 14], [200.0, 200.0], lead_time=3)
    assert full["fill_rate"] == 1.0 and full["lost_units"] == 0
    starved = S.replay_policy(d, [0], [20.0], lead_time=3)
    assert starved["fill_rate"] < 1.0
    assert starved["lost_units"] + (starved["demand_units"] - starved["lost_units"]) == pytest.approx(280)
