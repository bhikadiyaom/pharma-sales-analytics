"""Reproducible pipeline.

    python -m src.pipeline data      # ingest -> clean -> parquet + SQLite + DQ report
    python -m src.pipeline train     # CV, model selection, forecasts, intervals, SHAP
    python -m src.pipeline analyze   # ABC-XYZ, clustering, anomalies, inventory back-test
    python -m src.pipeline all

Use ``--fast`` for a quick smoke run (4 folds, fewer models).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.analytics import anomaly as anom
from src.analytics import clustering, descriptive
from src.config import load_config, processed_dir, results_dir
from src.data import ingest, store
from src.features.engineering import build_feature_frame
from src.inventory import simulate
from src.inventory.planner import build_plan_table, default_inputs
from src.models import cv as cvm
from src.models import explain, metrics
from src.models.forecasters import FORECASTERS

log = logging.getLogger("pipeline")


def _series(daily: pd.DataFrame) -> dict[str, pd.Series]:
    return {c: g.set_index("date")["units"].asfreq("D") for c, g in daily.groupby("category")}


def _save(df: pd.DataFrame, name: str, cfg) -> None:
    p = results_dir(cfg) / name
    df.to_parquet(p, index=False) if name.endswith(".parquet") else df.to_csv(p, index=False)


def _load(name: str, cfg) -> pd.DataFrame:
    p = results_dir(cfg) / name
    return pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p)


# --------------------------------------------------------------------------- #
# Stage 1: data
# --------------------------------------------------------------------------- #
def stage_data(cfg: dict[str, Any]) -> None:
    daily, rep = ingest.clean_daily(cfg)
    hourly = ingest.clean_hourly(cfg)
    daily = descriptive.add_revenue(daily, cfg)
    pdir = processed_dir(cfg)
    daily.to_parquet(pdir / "daily_clean.parquet", index=False)
    hourly.to_parquet(pdir / "hourly_clean.parquet", index=False)
    store.write_sales_sqlite(daily, hourly, cfg)
    feats = []
    for cat, y in _series(daily).items():
        f = build_feature_frame(y, cfg)
        f.insert(0, "units", y.to_numpy())
        f.insert(0, "category", cat)
        feats.append(f.reset_index(names="date"))
    pd.concat(feats).to_parquet(pdir / "features.parquet", index=False)
    rep.add("hourly_rows", int(len(hourly)))
    rep.add("is_synthetic", bool(cfg["data"]["is_synthetic"]))
    ingest.save_quality_report(rep, cfg)
    log.info("data stage done: %d daily rows, %d hourly rows", len(daily), len(hourly))


# --------------------------------------------------------------------------- #
# Stage 2: train / evaluate / forecast
# --------------------------------------------------------------------------- #
def _final_forecast(cat: str, model: str, y: pd.Series, cfg) -> pd.DataFrame:
    H = cfg["forecasting"]["max_horizon"]
    pred = FORECASTERS[model](y, H, cfg)
    idx = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    return pd.DataFrame(
        {"category": cat, "model": model, "date": idx, "h": np.arange(1, H + 1), "yhat": pred}
    )


def stage_train(cfg: dict[str, Any], fast: bool = False) -> None:
    daily = store.read_daily(cfg)
    hist = _series(daily)
    cats = list(hist)
    f = cfg["forecasting"]
    n_jobs = f["n_jobs"]
    t0 = time.time()

    # ---- rolling-origin CV --------------------------------------------------
    parts = Parallel(n_jobs=n_jobs, verbose=0)(delayed(cvm.run_cv)(c, hist[c], cfg) for c in cats)
    cv_pred = pd.concat(parts, ignore_index=True)
    _save(cv_pred, "cv_predictions.parquet", cfg)
    log.info("CV done in %.0fs", time.time() - t0)

    # ---- metrics, comparison, selection ------------------------------------
    m = metrics.summarize_cv(cv_pred, hist)
    m = cvm.improvement_vs_baseline(m, f["baseline_model"])
    _save(m, "cv_metrics.csv", cfg)
    best = cvm.select_best(m, f["selection_metric"])
    _save(best, "best_models.csv", cfg)
    hm = (
        cv_pred.assign(ae=(cv_pred["y"] - cv_pred["yhat"]).abs())
        .groupby(["category", "model", "h"], as_index=False)["ae"]
        .mean()
    )
    _save(hm, "cv_error_by_horizon.csv", cfg)
    best_map = dict(zip(best["category"], best["model"]))

    # ---- prediction intervals (calibration CV, best model only) -------------
    level = f["interval"]["level"]

    def _cal(c):
        cal = cvm.calibration_residuals(hist[c], best_map[c], cfg)
        return c, cvm.interval_quantiles(cal, level), cvm.interval_coverage_check(cal, level)

    cal_out = Parallel(n_jobs=n_jobs)(delayed(_cal)(c) for c in cats)
    qmap = {c: q for c, q, _ in cal_out}
    cov = pd.DataFrame(
        {
            "category": [c for c, _, _ in cal_out],
            "model": [best_map[c] for c, _, _ in cal_out],
            "nominal": level,
            "empirical_coverage": [v for _, _, v in cal_out],
        }
    )
    _save(cov, "interval_coverage.csv", cfg)

    # ---- final fits + 90-day forecasts --------------------------------------
    jobs = [(c, mname) for c in cats for mname in f["models"]]
    fcs = Parallel(n_jobs=n_jobs)(delayed(_final_forecast)(c, mn, hist[c], cfg) for c, mn in jobs)
    fc_all = pd.concat(fcs, ignore_index=True)
    _save(fc_all, "forecasts_all.parquet", cfg)
    rows = []
    for c in cats:
        g = fc_all[(fc_all["category"] == c) & (fc_all["model"] == best_map[c])].copy()
        lo, hi = cvm.apply_intervals(g["yhat"].to_numpy(), qmap[c])
        g["lower"], g["upper"] = lo, hi
        rows.append(g)
    _save(pd.concat(rows, ignore_index=True), "forecasts_best.parquet", cfg)

    # ---- SHAP for the XGBoost model -----------------------------------------
    sh = Parallel(n_jobs=n_jobs)(delayed(explain.shap_for_series)(c, hist[c], cfg) for c in cats)
    _save(pd.concat([a for a, _ in sh], ignore_index=True), "shap_importance.csv", cfg)
    _save(pd.concat([b for _, b in sh], ignore_index=True), "shap_sample.parquet", cfg)
    log.info("train stage done in %.0fs", time.time() - t0)


# --------------------------------------------------------------------------- #
# Stage 3: analytics + inventory
# --------------------------------------------------------------------------- #
def stage_analyze(cfg: dict[str, Any]) -> None:
    daily = store.read_daily(cfg)
    hist = _series(daily)

    matrix = descriptive.abc_xyz_matrix(daily, cfg)
    _save(matrix, "abc_xyz.csv", cfg)

    feats = descriptive.demand_features(daily)
    cl = clustering.cluster_demand(feats, cfg)
    _save(cl.reset_index(), "demand_clusters.csv", cfg)
    (results_dir(cfg) / "clustering_meta.json").write_text(
        json.dumps({"k": int(cl.attrs["k"]), "silhouette": float(cl.attrs["silhouette"])})
    )

    an = []
    for c, y in hist.items():
        a = anom.detect_anomalies(y, cfg)
        a.insert(0, "category", c)
        an.append(a.reset_index(names="date"))
    _save(pd.concat(an, ignore_index=True), "anomalies.parquet", cfg)

    cv_pred = _load("cv_predictions.parquet", cfg)
    best = _load("best_models.csv", cfg)
    bm = dict(zip(best["category"], best["model"]))
    cv_best = pd.concat([cv_pred[(cv_pred.category == c) & (cv_pred.model == m)] for c, m in bm.items()])
    bt = simulate.backtest_all(cv_best, hist, cfg)
    _save(bt, "inventory_backtest.csv", cfg)

    plan = default_reorder_list(cfg)
    _save(plan, "reorder_list_default.csv", cfg)
    write_headline(cfg)


def default_reorder_list(cfg: dict[str, Any], service_level: float | None = None) -> pd.DataFrame:
    """Plan table for every category at the default service level (default stock inputs)."""
    sl = service_level or cfg["inventory"]["default_service_level"]
    fb = _load("forecasts_best.parquet", cfg)
    best = _load("best_models.csv", cfg).set_index("category")
    win = cfg["inventory"]["planning_horizon_days"]
    inputs = default_inputs(cfg, fb, sorted(best.index), win)
    return build_plan_table(cfg, fb, best, inputs, sl, win)


def write_headline(cfg: dict[str, Any]) -> dict[str, Any]:
    """Collect headline numbers (all computed from the saved result files)."""
    m = _load("cv_metrics.csv", cfg)
    best = _load("best_models.csv", cfg)
    base = cfg["forecasting"]["baseline_model"]
    bt = _load("inventory_backtest.csv", cfg)
    sl = cfg["inventory"]["default_service_level"]
    b = bt[np.isclose(bt["service_level"], sl)].pivot(index="category", columns="policy")
    mb = m[m.model == base].set_index("category")
    bb = best.set_index("category")
    h = {
        "best_model_per_category": dict(zip(best["category"], best["model"])),
        "baseline_model": base,
        "mean_mae_improvement_pct": float(((1 - bb["mae"] / mb["mae"]) * 100).mean()),
        "mean_rmse_improvement_pct": float(((1 - bb["rmse"] / mb["rmse"]) * 100).mean()),
        "mean_smape_improvement_pct": float(((1 - bb["smape"] / mb["smape"]) * 100).mean()),
        "mean_mape_improvement_pct": float(((1 - bb["mape"] / mb["mape"]) * 100).mean()),
        "mean_best_mape": float(bb["mape"].mean()),
        "mean_baseline_mape": float(mb["mape"].mean()),
        "mean_best_smape": float(bb["smape"].mean()),
        "mean_baseline_smape": float(mb["smape"].mean()),
        "backtest_service_level": sl,
        "fill_rate_baseline": float(
            1 - b["lost_units"]["baseline"].sum() / b["demand_units"]["baseline"].sum()
        ),
        "fill_rate_forecast": float(
            1 - b["lost_units"]["forecast"].sum() / b["demand_units"]["forecast"].sum()
        ),
        "stockout_days_baseline": float(b["stockout_day_share"]["baseline"].mean()),
        "stockout_days_forecast": float(b["stockout_day_share"]["forecast"].mean()),
        "lost_units_baseline": float(b["lost_units"]["baseline"].sum()),
        "lost_units_forecast": float(b["lost_units"]["forecast"].sum()),
        "avg_inventory_value_baseline": float(b["avg_inventory_value"]["baseline"].sum()),
        "avg_inventory_value_forecast": float(b["avg_inventory_value"]["forecast"].sum()),
    }
    h["stockout_day_reduction_pct"] = (1 - h["stockout_days_forecast"] / h["stockout_days_baseline"]) * 100
    h["lost_units_reduction_pct"] = (1 - h["lost_units_forecast"] / h["lost_units_baseline"]) * 100
    h["inventory_value_change_pct"] = (
        h["avg_inventory_value_forecast"] / h["avg_inventory_value_baseline"] - 1
    ) * 100
    (results_dir(cfg) / "headline.json").write_text(json.dumps(h, indent=2))
    return h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["data", "train", "analyze", "all"])
    ap.add_argument("--fast", action="store_true", help="smoke run: 4 folds, no SARIMAX/Prophet")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = copy.deepcopy(load_config())
    if args.fast:
        cfg["forecasting"]["cv"]["n_folds"] = 4
        cfg["forecasting"]["models"] = ["naive", "snaive", "xgboost", "lightgbm"]
        cfg["forecasting"]["interval"]["calibration_origins"] = 4
    if args.stage in ("data", "all"):
        stage_data(cfg)
    if args.stage in ("train", "all"):
        stage_train(cfg, args.fast)
    if args.stage in ("analyze", "all"):
        stage_analyze(cfg)


if __name__ == "__main__":
    main()
