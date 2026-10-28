"""Ingestion, schema validation and cleaning of the raw sales files.

The real Kaggle "Pharma Sales Data" ships one wide table per frequency
(``datum`` + one column per ATC class).  Everything dataset specific is driven
by ``config.yaml`` so another source can be plugged in by editing the config.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve, results_dir

log = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Collects data-quality findings so they can be logged and saved."""

    items: dict[str, Any] = field(default_factory=dict)

    def add(self, key: str, value: Any) -> None:
        self.items[key] = value
        log.info("DQ | %s: %s", key, value)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.items, indent=2, default=str), encoding="utf-8")


class SchemaError(ValueError):
    """Raised when the raw file does not match the configured schema."""


# --------------------------------------------------------------------------- #
# Loading + validation
# --------------------------------------------------------------------------- #
def validate_schema(df: pd.DataFrame, required: list[str]) -> None:
    """Raise :class:`SchemaError` if a required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")


def load_raw(cfg: dict[str, Any] | None = None, file_key: str = "daily_file") -> pd.DataFrame:
    """Read a raw file and return a tidy *long* frame ``date, category, units``."""
    cfg = cfg or load_config()
    d = cfg["data"]
    path = resolve(cfg["paths"]["raw_dir"]) / d[file_key]
    raw = pd.read_csv(path)
    fmt = d["hourly_date_format"] if file_key == "hourly_file" else d["date_format"]

    if d["layout"] == "wide":
        validate_schema(raw, [d["date_col"], *d["category_cols"]])
        if file_key == "monthly_file":  # monthly file is ISO formatted
            raw["date"] = pd.to_datetime(raw[d["date_col"]], errors="coerce")
        else:
            raw["date"] = pd.to_datetime(raw[d["date_col"]], format=fmt, errors="coerce")
        long = raw.melt(
            id_vars="date", value_vars=d["category_cols"], var_name="category", value_name="units"
        )
    else:  # long layout
        validate_schema(raw, [d["date_col"], d["category_col"], d["units_col"]])
        long = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[d["date_col"]], format=fmt, errors="coerce"),
                "category": raw[d["category_col"]],
                "units": pd.to_numeric(raw[d["units_col"]], errors="coerce"),
            }
        )
    long["units"] = pd.to_numeric(long["units"], errors="coerce")
    return long


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def remove_duplicates(long: pd.DataFrame, rep: QualityReport | None = None) -> pd.DataFrame:
    """Drop exact duplicates, then collapse repeated (date, category) keys by sum."""
    n0 = len(long)
    long = long.drop_duplicates()
    n_exact = n0 - len(long)
    dup_keys = int(long.duplicated(["date", "category"], keep=False).sum())
    if dup_keys:
        long = long.groupby(["date", "category"], as_index=False)["units"].sum(min_count=1)
    if rep:
        rep.add("exact_duplicate_rows_removed", int(n_exact))
        rep.add("duplicate_key_rows_aggregated", dup_keys)
    return long


def complete_calendar(long: pd.DataFrame, freq: str = "D", rep: QualityReport | None = None) -> pd.DataFrame:
    """Reindex every category to a gap-free calendar; absent days become NaN."""
    idx = pd.date_range(long["date"].min(), long["date"].max(), freq=freq)
    full = long.pivot(index="date", columns="category", values="units").reindex(idx).rename_axis("date")
    if rep:
        rep.add("calendar_days_missing_before_fill", int(len(idx) - long["date"].nunique()))
    out = full.stack(future_stack=True).rename("units").reset_index()
    return out


def handle_missing(long: pd.DataFrame, rep: QualityReport | None = None) -> pd.DataFrame:
    """Fill NaN by linear interpolation per category (edges: nearest); negatives -> imputed."""
    long = long.sort_values(["category", "date"]).copy()
    n_nan = int(long["units"].isna().sum())
    n_neg = int((long["units"] < 0).sum())
    long.loc[long["units"] < 0, "units"] = np.nan
    long["units"] = long.groupby("category")["units"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    if rep:
        rep.add("missing_values_imputed", n_nan)
        rep.add("negative_values_set_missing_then_imputed", n_neg)
    return long


def cap_outliers(series: pd.Series, window: int = 29, z: float = 6.0) -> tuple[pd.Series, pd.Series]:
    """Winsorise *extreme* points using a robust z-score against a rolling median.

    The local scale (1.4826 x rolling MAD) collapses for intermittent series (many zeros),
    which would flag ordinary peaks, so it is floored at the series' global standard
    deviation (of the 99%-trimmed series).  Only |z| > ``z`` are touched, hence genuine seasonal peaks are preserved.
    Returns (capped_series, boolean_mask_of_capped_points).
    """
    minp = max(window // 2, 1)
    med = series.rolling(window, center=True, min_periods=minp).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=minp).median()
    # global spread measured after trimming the top 1 % so a single huge error cannot inflate it
    floor = float(series.clip(upper=series.quantile(0.99)).std())
    floor = floor if floor > 0 else 1.0
    scale = (1.4826 * mad).clip(lower=floor)
    zs = (series - med) / scale
    mask = zs.abs() > z
    capped = series.copy()
    capped[mask] = (med + np.sign(zs) * z * scale)[mask]
    return capped.clip(lower=0), mask


def handle_outliers(
    long: pd.DataFrame, cfg: dict[str, Any], rep: QualityReport | None = None
) -> pd.DataFrame:
    """Cap extreme outliers per category; keep the raw value in ``units_raw``."""
    o = cfg["data"]["outlier"]
    parts, counts = [], {}
    for cat, g in long.groupby("category"):
        g = g.sort_values("date").copy()
        g["units_raw"] = g["units"]
        g["units"], mask = cap_outliers(g["units"], o["window"], o["z_threshold"])
        counts[cat] = int(mask.sum())
        parts.append(g)
    if rep:
        rep.add("outliers_capped_by_category", counts)
    return pd.concat(parts, ignore_index=True)


def reconcile_with_aggregates(daily: pd.DataFrame, cfg: dict[str, Any], rep: QualityReport) -> None:
    """Cross-check daily totals against the monthly file shipped with the data."""
    try:
        m = load_raw(cfg, "monthly_file")
        m["date"] = m["date"] + pd.offsets.MonthEnd(0)
        d = daily.assign(date=daily["date"] + pd.offsets.MonthEnd(0))
        d = d.groupby(["date", "category"], as_index=False)["units_raw"].sum()
        j = m.merge(d, on=["date", "category"], how="inner", suffixes=("", "_daily"))
        j = j[j["units"] > 0]
        rel = (j["units_raw"] - j["units"]).abs() / j["units"]
        rep.add(
            "monthly_reconciliation",
            {
                "months_compared": int(j["date"].nunique()),
                "median_abs_rel_diff": round(float(rel.median()), 4),
                "share_within_1pct": round(float((rel < 0.01).mean()), 4),
            },
        )
    except Exception as exc:  # optional check; never block the pipeline
        rep.add("monthly_reconciliation", f"skipped ({exc})")


def clean_daily(cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, QualityReport]:
    """Full daily pipeline: load -> validate -> dedupe -> calendar -> impute -> outliers."""
    cfg = cfg or load_config()
    rep = QualityReport()
    long = load_raw(cfg)
    rep.add("rows_loaded", int(len(long)))
    rep.add("unparseable_dates", int(long["date"].isna().sum()))
    long = long.dropna(subset=["date"])
    rep.add("date_range", [str(long["date"].min().date()), str(long["date"].max().date())])
    rep.add("categories", sorted(long["category"].unique().tolist()))
    long = remove_duplicates(long, rep)
    long = complete_calendar(long, "D", rep)
    long = handle_missing(long, rep)
    long = handle_outliers(long, cfg, rep)
    zero_share = long.groupby("category")["units"].apply(lambda s: float((s == 0).mean()))
    rep.add("zero_sales_day_share", {k: round(v, 3) for k, v in zero_share.items()})
    reconcile_with_aggregates(long, cfg, rep)
    return long[["date", "category", "units", "units_raw"]].reset_index(drop=True), rep


def clean_hourly(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Hourly sales (only pharmacy-open hours are present) used for heatmaps."""
    cfg = cfg or load_config()
    h = load_raw(cfg, "hourly_file").dropna(subset=["date"])
    h = h.drop_duplicates(["date", "category"])
    h["units"] = h["units"].fillna(0).clip(lower=0)
    return h


def save_quality_report(rep: QualityReport, cfg: dict[str, Any] | None = None) -> Path:
    path = results_dir(cfg) / "data_quality_report.json"
    rep.save(path)
    return path
