import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import engineering as fe

CFG = load_config()


def _series(n=120):
    idx = pd.date_range("2019-12-01", periods=n, freq="D")
    return pd.Series(np.arange(n, dtype=float), index=idx)


def test_calendar_features_and_holidays():
    idx = pd.date_range("2019-12-23", "2020-01-02")
    cal = fe.calendar_features(idx, CFG)
    assert cal.loc["2019-12-25", "is_holiday"] == 1
    assert cal.loc["2019-12-23", "is_holiday"] == 0
    assert cal.loc["2019-12-23", "near_holiday"] == 1  # day before 12-24
    assert cal.loc["2020-01-01", "is_holiday"] == 1
    assert cal.loc["2019-12-28", "weekday"] == 5 and cal.loc["2019-12-28", "is_weekend"] == 1
    assert {"month", "quarter", "doy_sin", "doy_cos"} <= set(cal.columns)


def test_lag_features_values():
    y = _series()
    lag = fe.lag_features(y, [1, 7])
    assert lag["lag_1"].iloc[10] == y.iloc[9]
    assert lag["lag_7"].iloc[10] == y.iloc[3]
    assert lag["lag_7"].iloc[:7].isna().all()


def test_rolling_features_do_not_leak_current_day():
    y = _series()
    roll = fe.rolling_features(y, [7])
    t = 50
    # mean of the 7 previous days, excluding day t itself
    assert np.isclose(roll["roll_mean_7"].iloc[t], y.iloc[t - 7 : t].mean())
    y2 = y.copy()
    y2.iloc[t] = 10_000  # changing today's value must not change today's features
    assert np.isclose(fe.rolling_features(y2, [7])["roll_mean_7"].iloc[t], roll["roll_mean_7"].iloc[t])


def test_build_feature_frame_is_causal_and_stable():
    y = _series()
    X = fe.build_feature_frame(y, CFG)
    assert list(X.columns) == fe.feature_columns(CFG)
    y_future_changed = y.copy()
    y_future_changed.iloc[-10:] += 999
    X2 = fe.build_feature_frame(y_future_changed, CFG)
    pd.testing.assert_frame_equal(X.iloc[:-10], X2.iloc[:-10])  # past rows unaffected by the future


def test_supervised_frame_drops_incomplete_history():
    y = _series()
    sf = fe.supervised_frame(y, CFG)
    assert sf.notna().all().all()
    assert len(sf) == len(y) - max(CFG["features"]["lags"])


def test_price_promo_features_only_when_columns_exist():
    df = pd.DataFrame({"price": [10.0, 10.0, 12.0]}, index=pd.date_range("2020-01-01", periods=3))
    out = fe.price_promo_features(df)
    assert "price_change_pct" in out and out["price_change_pct"].iloc[2] == pytest.approx(0.2)
    assert fe.price_promo_features(pd.DataFrame(index=df.index)).shape[1] == 0
