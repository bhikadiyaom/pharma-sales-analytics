"""Anomaly detection on sales: robust z-score of STL residuals + Isolation Forest."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.analytics.descriptive import stl_decompose
from src.config import load_config
from src.features.engineering import holiday_flags


def detect_anomalies(y: pd.Series, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return per-day expected value, residual z-score, flags and a text explanation.

    * expected = STL (period 7, robust) trend + seasonal
    * z        = Pearson-style residual (y - expected) / sqrt(expected), robustly standardised
                 with MAD (scale floored at half the residual std, because MAD collapses to ~0 on
                 zero-inflated series).  Dividing by sqrt(expected) accounts for count variance
                 growing with the level.
    * anomaly  = |z| > threshold; ``confirmed`` additionally requires an Isolation-Forest outlier.
    """
    cfg = cfg or load_config()
    a = cfg["anomaly"]
    res = stl_decompose(y, a["stl_period"], robust=True)
    expected = (res.trend + res.seasonal).clip(lower=0)
    resid = y - expected
    pearson = resid / np.sqrt(np.maximum(expected, 1.0))
    mad = (pearson - pearson.median()).abs().median() * 1.4826
    scale = max(float(mad), 0.5 * float(pearson.std()), 1e-9)
    z = (pearson - pearson.median()) / scale

    # Isolation Forest on [pearson residual, level vs previous 28-day mean, weekday]
    ratio = y / y.rolling(28, min_periods=7).mean().shift(1).replace(0, np.nan)
    feat = pd.DataFrame(
        {"pearson": pearson, "ratio": ratio.fillna(1.0).clip(0, 10), "weekday": y.index.weekday},
        index=y.index,
    )
    iso = IsolationForest(contamination=a["contamination"], random_state=cfg["project"]["seed"])
    iso_flag = iso.fit_predict(feat) == -1

    hol = holiday_flags(y.index, cfg["calendar"]["fixed_holidays"], 0)["is_holiday"].astype(bool)
    out = pd.DataFrame(
        {"actual": y, "expected": expected, "residual": resid, "z": z, "iso_flag": iso_flag, "holiday": hol},
        index=y.index,
    )
    out["anomaly"] = out["z"].abs() > a["z_threshold"]
    out["confirmed"] = out["anomaly"] & out["iso_flag"]
    out["direction"] = np.where(out["residual"] > 0, "spike", "drop")
    out["explanation"] = [_explain(r) for r in out.itertuples()]
    return out


def _explain(r) -> str:
    """Plain-language reason for a flagged day."""
    if not r.anomaly:
        return ""
    kind = r.direction.capitalize()
    if r.expected >= 1:
        pct = (r.actual / r.expected - 1) * 100
        bits = [
            f"{kind}: {r.actual:.1f} units vs {r.expected:.1f} expected ({pct:+.0f}%, residual z = {r.z:+.1f})"
        ]
    else:  # near-zero baseline: percentages are meaningless
        bits = [
            f"{kind}: {r.actual:.1f} units on a day when only ~{r.expected:.1f} was expected "
            f"(normally near zero; residual z = {r.z:+.1f})"
        ]
    if r.holiday:
        bits.append("falls on a festive date")
    if r.actual == 0:
        bits.append("zero sales (possible stock-out, closure or data gap)")
    return "; ".join(bits)
