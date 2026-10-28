import numpy as np
import pandas as pd

from src.analytics.anomaly import detect_anomalies
from src.config import load_config
from src.inventory.planner import build_plan_table, default_inputs

CFG = load_config()


def _fb(cats, level=50.0):
    return pd.DataFrame([{"category": c, "h": h, "yhat": level} for c in cats for h in range(1, 91)])


def _best(cats):
    return pd.DataFrame({"category": cats, "model": "sarimax", "rmse": 5.0}).set_index("category")


def test_plan_flags_follow_stock_level():
    cats = ["R03", "N02BE"]
    fb, best = _fb(cats), _best(cats)
    low = build_plan_table(CFG, fb, best, default_inputs(CFG, fb, cats, 30, cover_days=2), 0.95, 30)
    assert set(low["status"]) == {"STOCKOUT RISK"} and (low["suggested_order_qty"] > 0).all()
    ok = build_plan_table(CFG, fb, best, default_inputs(CFG, fb, cats, 30, cover_days=25), 0.95, 30)
    assert set(ok["status"]) == {"OK"} and (ok["suggested_order_qty"] == 0).all()
    high = build_plan_table(CFG, fb, best, default_inputs(CFG, fb, cats, 30, cover_days=150), 0.95, 30)
    assert set(high["status"]) == {"OVERSTOCK"}
    assert (high["expiry_risk_units"] > 0).all()  # 150 days of stock, <=150 d shelf life


def test_higher_service_level_raises_reorder_point():
    fb, best = _fb(["R03"]), _best(["R03"])
    inp = default_inputs(CFG, fb, ["R03"], 30)
    r90 = build_plan_table(CFG, fb, best, inp, 0.90, 30)["reorder_point"].iloc[0]
    r99 = build_plan_table(CFG, fb, best, inp, 0.99, 30)["reorder_point"].iloc[0]
    assert r99 > r90


def test_anomaly_detector_finds_injected_spike_and_explains_it():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2018-01-01", periods=600, freq="D")
    base = 30 + 8 * (idx.weekday < 5)
    y = pd.Series(rng.poisson(base).astype(float), index=idx)
    y.iloc[300] = 200.0
    out = detect_anomalies(y, CFG)
    assert out["anomaly"].iloc[300] and out["direction"].iloc[300] == "spike"
    assert "expected" in out["explanation"].iloc[300]
    assert out["anomaly"].mean() < 0.05  # few false alarms on plain Poisson noise
