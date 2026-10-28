import numpy as np
import pandas as pd
import pytest

from src.data import ingest


def _long(dates, cat="A", units=None):
    units = units if units is not None else np.arange(len(dates), dtype=float)
    return pd.DataFrame({"date": pd.to_datetime(dates), "category": cat, "units": units})


def test_validate_schema_raises_on_missing_column():
    df = pd.DataFrame({"datum": [1], "M01AB": [1]})
    with pytest.raises(ingest.SchemaError):
        ingest.validate_schema(df, ["datum", "M01AB", "R03"])
    ingest.validate_schema(df, ["datum", "M01AB"])  # no error


def test_remove_duplicates_exact_and_keys():
    df = _long(["2020-01-01", "2020-01-01", "2020-01-02"], units=[5.0, 5.0, 3.0])
    rep = ingest.QualityReport()
    out = ingest.remove_duplicates(df, rep)
    assert len(out) == 2
    assert rep.items["exact_duplicate_rows_removed"] == 1
    # same key, different value -> aggregated
    df2 = _long(["2020-01-01", "2020-01-01"], units=[5.0, 2.0])
    assert ingest.remove_duplicates(df2)["units"].iloc[0] == 7.0


def test_complete_calendar_and_missing_imputation():
    df = _long(["2020-01-01", "2020-01-04"], units=[1.0, 4.0])
    rep = ingest.QualityReport()
    full = ingest.complete_calendar(df, "D", rep)
    assert len(full) == 4 and rep.items["calendar_days_missing_before_fill"] == 2
    filled = ingest.handle_missing(full, rep)
    assert filled["units"].tolist() == [1.0, 2.0, 3.0, 4.0]  # linear interpolation
    assert rep.items["missing_values_imputed"] == 2


def test_negative_values_are_treated_as_missing():
    df = _long(["2020-01-01", "2020-01-02", "2020-01-03"], units=[2.0, -5.0, 4.0])
    out = ingest.handle_missing(df)
    assert out["units"].tolist() == [2.0, 3.0, 4.0]


def test_cap_outliers_caps_only_extreme_points():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.poisson(20, 300).astype(float))
    s.iloc[150] = 500.0  # injected data-entry error
    capped, mask = ingest.cap_outliers(s, window=29, z=6.0)
    assert mask.iloc[150] and mask.sum() <= 2
    assert capped.iloc[150] < 100
    # untouched points keep their value
    assert (capped[~mask] == s[~mask]).all()


def test_cap_outliers_preserves_intermittent_peaks():
    # zero-inflated, over-dispersed counts like the R03 series: peaks are genuine demand
    rng = np.random.default_rng(3)
    base = rng.negative_binomial(2, 0.3, 1500).astype(float)
    base[rng.random(1500) < 0.25] = 0.0
    s = pd.Series(base)
    capped, mask = ingest.cap_outliers(s, window=29, z=6.0)
    assert mask.mean() < 0.01  # (almost) nothing is capped
    assert capped.max() >= 0.9 * s.max() or mask.sum() == 0
