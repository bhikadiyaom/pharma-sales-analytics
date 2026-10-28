import numpy as np
import pandas as pd
import pytest

from src.analytics import descriptive as desc
from src.models import cv as cvm
from src.models import metrics as M
from src.models.forecasters import naive_forecast, seasonal_naive_forecast


def test_basic_metrics_known_values():
    y, p = [10, 20, 30], [12, 18, 33]
    assert M.mae(y, p) == pytest.approx((2 + 2 + 3) / 3)
    assert M.rmse(y, p) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert M.mape(y, p) == pytest.approx(np.mean([0.2, 0.1, 0.1]) * 100)
    assert M.smape(y, p) == pytest.approx(np.mean([2 * 2 / 22, 2 * 2 / 38, 2 * 3 / 63]) * 100)


def test_mape_ignores_zero_actuals_and_smape_handles_zero_zero():
    assert M.mape([0, 10], [5, 5]) == pytest.approx(50.0)
    assert np.isnan(M.mape([0, 0], [1, 1]))
    assert M.smape([0, 0], [0, 0]) == 0.0


def test_mase_scaling():
    train = np.arange(1, 15, dtype=float)  # weekly diffs are all 7
    assert M.mase([10], [3], train, season=7) == pytest.approx(7 / 7)


def test_baseline_forecasters():
    y = pd.Series(np.arange(14.0), index=pd.date_range("2020-01-01", periods=14))
    assert (naive_forecast(y, 3) == 13).all()
    assert seasonal_naive_forecast(y, 9).tolist() == [7, 8, 9, 10, 11, 12, 13, 7, 8]


def test_rolling_origins_no_overlap_no_leakage():
    n, h, step, k = 1000, 14, 14, 10
    origins = cvm.rolling_origins(n, h, step, k, min_train=500)
    assert len(origins) == k
    assert origins[-1] + h == n  # last fold ends at the series end
    for a, b in zip(origins, origins[1:]):
        assert b - a == step and a + h <= b  # test windows do not overlap
    assert all(o >= 500 for o in origins)


def test_run_cv_trains_only_on_past_data():
    idx = pd.date_range("2018-01-01", periods=900, freq="D")
    y = pd.Series(np.arange(900.0), index=idx)
    cfg = {
        "forecasting": {
            "models": ["naive"],
            "cv": {"horizon": 7, "step": 7, "n_folds": 5, "min_train_days": 100},
        }
    }
    out = cvm.run_cv("X", y, cfg)
    # naive forecast of a ramp = value at the origin; must equal y just before the first test day
    g = out[out.fold == 0]
    assert (g["yhat"] == y.loc[g["origin"].iloc[0]]).all()
    assert (g["date"] > g["origin"]).all()


def test_select_best_and_improvement():
    m = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B"],
            "model": ["snaive", "xgb", "snaive", "xgb"],
            "mae": [10.0, 8.0, 5.0, 6.0],
            "rmse": [12.0, 9.0, 6.0, 7.0],
            "mape": [50.0, 40.0, 30.0, 33.0],
            "smape": [40.0, 30.0, 20.0, 22.0],
        }
    )
    best = cvm.select_best(m, "mae")
    assert dict(zip(best.category, best.model)) == {"A": "xgb", "B": "snaive"}
    imp = cvm.improvement_vs_baseline(m, "snaive")
    assert imp.loc[(imp.category == "A") & (imp.model == "xgb"), "mae_improv_pct"].iloc[0] == pytest.approx(
        20.0
    )


def test_interval_bounds_are_ordered_and_nonnegative():
    rng = np.random.default_rng(1)
    cal = pd.DataFrame({"origin_k": np.repeat(np.arange(8), 90), "h": np.tile(np.arange(1, 91), 8)})
    cal["yhat"] = 20.0
    cal["y"] = rng.poisson(20, len(cal))
    q = cvm.interval_quantiles(cal, 0.9)
    lo, hi = cvm.apply_intervals(np.full(90, 20.0), q)
    assert (lo <= 20.0).all() and (hi >= 20.0).all() and (lo >= 0).all()
    assert 0.8 <= cvm.interval_coverage_check(cal, 0.9) <= 1.0


def test_abc_classification_pareto():
    rev = pd.Series({"a": 700.0, "b": 150.0, "c": 100.0, "d": 50.0})
    out = desc.abc_classification(rev, (0.8, 0.95))
    # class is decided by the cumulative share *before* the item: b crosses 80 % so it is still A
    assert out["abc"].tolist() == ["A", "A", "B", "C"]
    assert out["cum_share"].iloc[-1] == pytest.approx(1.0)


def test_xyz_classification_by_variability():
    idx = pd.date_range("2020-01-01", periods=140, freq="D")
    rng = np.random.default_rng(0)
    df = pd.concat(
        [
            pd.DataFrame({"date": idx, "category": "stable", "units": 50 + rng.normal(0, 1, 140)}),
            pd.DataFrame({"date": idx, "category": "wild", "units": rng.choice([0, 200], 140)}),
        ]
    )
    out = desc.xyz_classification(df, "W", (0.3, 0.45))
    assert out.loc["stable", "xyz"] == "X"
