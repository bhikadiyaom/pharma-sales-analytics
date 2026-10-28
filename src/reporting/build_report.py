"""Fill ``docs/*.template.md`` with numbers/tables read from ``reports/results``.

    python -m src.reporting.build_report

Every ``{{name}}`` placeholder is replaced by a value computed here from files written by the
pipeline, so no number in the final documents is typed by hand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, category_assumptions, load_config, results_dir
from src.models.forecasters import MODEL_LABELS


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}", index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    d = df.reset_index() if index else df
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in d.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("n/a" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def collect(cfg: dict) -> dict[str, str]:
    R = results_dir(cfg)
    h = json.loads((R / "headline.json").read_text())
    dq = json.loads((R / "data_quality_report.json").read_text())
    met = pd.read_csv(R / "cv_metrics.csv")
    best = pd.read_csv(R / "best_models.csv")
    cov = pd.read_csv(R / "interval_coverage.csv")
    abc = pd.read_csv(R / "abc_xyz.csv")
    cl = pd.read_csv(R / "demand_clusters.csv")
    cl_meta = json.loads((R / "clustering_meta.json").read_text())
    bt = pd.read_csv(R / "inventory_backtest.csv")
    an = pd.read_parquet(R / "anomalies.parquet")
    shap = pd.read_csv(R / "shap_importance.csv")
    plan = pd.read_csv(R / "reorder_list_default.csv")
    daily = pd.read_parquet(ROOT / cfg["paths"]["processed_dir"] / "daily_clean.parquet")
    base = cfg["forecasting"]["baseline_model"]
    v: dict[str, str] = {}

    # ---- data facts
    v["date_start"], v["date_end"] = dq["date_range"]
    v["n_days"] = f"{daily['date'].nunique():,}"
    v["n_cats"] = str(daily["category"].nunique())
    v["n_hourly"] = f"{dq['hourly_rows']:,}"
    v["total_units"] = f"{daily['units'].sum():,.0f}"
    v["cv_folds"] = str(cfg["forecasting"]["cv"]["n_folds"])
    v["cv_horizon"] = str(cfg["forecasting"]["cv"]["horizon"])
    v["oos_days"] = str(cfg["forecasting"]["cv"]["n_folds"] * cfg["forecasting"]["cv"]["step"])
    v["selection_metric"] = cfg["forecasting"]["selection_metric"].upper()
    v["outliers_total"] = str(sum(dq["outliers_capped_by_category"].values()))
    v["recon_months"] = str(dq["monthly_reconciliation"]["months_compared"])
    v["recon_within1"] = f"{dq['monthly_reconciliation']['share_within_1pct'] * 100:.1f}"
    v["dq_table"] = md_table(
        pd.DataFrame(
            {
                "category": list(dq["zero_sales_day_share"]),
                "zero-sales days %": [x * 100 for x in dq["zero_sales_day_share"].values()],
                "outliers capped": [dq["outliers_capped_by_category"][c] for c in dq["zero_sales_day_share"]],
            }
        ),
        "{:.1f}",
    )

    # ---- model comparison
    lab = met.assign(model=met["model"].map(MODEL_LABELS))
    order = [MODEL_LABELS[m] for m in cfg["forecasting"]["models"]]
    for m_ in ["mae", "rmse", "smape", "mape", "mase"]:
        piv = lab.pivot(index="category", columns="model", values=m_)[order]
        v[f"{m_}_table"] = md_table(piv, index=True)
    b = best.copy()
    b["model"] = b["model"].map(MODEL_LABELS)
    v["best_table"] = md_table(
        b[
            ["category", "model", "mae", "rmse", "smape", "mape", "mase", "mae_improv_pct", "rmse_improv_pct"]
        ].rename(columns={"mae_improv_pct": "MAE gain %", "rmse_improv_pct": "RMSE gain %"})
    )
    wins = best["model"].map(MODEL_LABELS).value_counts()
    v["win_counts"] = ", ".join(f"{k}: {n}" for k, n in wins.items())
    bm = best.set_index("category")["model"].map(MODEL_LABELS)
    v["best_by_cat"] = "; ".join(f"{c} → {m}" for c, m in bm.items())
    avg = lab.groupby("model")[["mae", "rmse", "smape", "mase"]].mean().loc[order]
    v["avg_by_model_table"] = md_table(avg, index=True)
    # models that beat the baseline on RMSE in how many series
    rm = met.pivot(index="category", columns="model", values="rmse")
    beat = {MODEL_LABELS[m]: int((rm[m] < rm[base]).sum()) for m in rm.columns if m not in (base,)}
    v["beat_baseline_table"] = md_table(
        pd.DataFrame({"model": list(beat), "series where RMSE < seasonal-naive": list(beat.values())})
    )

    # ---- headline numbers
    for k in [
        "mean_mae_improvement_pct",
        "mean_rmse_improvement_pct",
        "mean_smape_improvement_pct",
        "mean_mape_improvement_pct",
        "mean_best_mape",
        "mean_baseline_mape",
        "mean_best_smape",
        "mean_baseline_smape",
        "stockout_day_reduction_pct",
        "lost_units_reduction_pct",
        "inventory_value_change_pct",
    ]:
        v[k] = f"{h[k]:.1f}"
    v["fill_baseline"], v["fill_forecast"] = (
        f"{h['fill_rate_baseline'] * 100:.2f}",
        f"{h['fill_rate_forecast'] * 100:.2f}",
    )
    v["stockout_base"], v["stockout_fc"] = (
        f"{h['stockout_days_baseline'] * 100:.2f}",
        f"{h['stockout_days_forecast'] * 100:.2f}",
    )
    v["lost_base"], v["lost_fc"] = f"{h['lost_units_baseline']:,.0f}", f"{h['lost_units_forecast']:,.0f}"
    v["inv_base"], v["inv_fc"] = (
        f"{h['avg_inventory_value_baseline']:,.0f}",
        f"{h['avg_inventory_value_forecast']:,.0f}",
    )
    v["bt_sl"] = f"{h['backtest_service_level']:.0%}"

    agg = (
        bt.groupby(["service_level", "policy"])
        .agg(
            lost=("lost_units", "sum"),
            demand=("demand_units", "sum"),
            so=("stockout_day_share", "mean"),
            inv=("avg_inventory_value", "sum"),
        )
        .reset_index()
    )
    agg["fill-rate %"] = (1 - agg["lost"] / agg["demand"]) * 100
    agg["stock-out days %"] = agg["so"] * 100
    agg["policy"] = agg["policy"].map({"baseline": "28-day moving average", "forecast": "Model forecast"})
    agg["service_level"] = agg["service_level"].map(lambda x: f"{x:.0%}")
    v["backtest_table"] = md_table(
        agg[["service_level", "policy", "fill-rate %", "stock-out days %", "lost", "inv"]].rename(
            columns={"service_level": "target SL", "lost": "lost units", "inv": "avg inventory value (cost)"}
        )
    )
    pc = bt[np.isclose(bt["service_level"], h["backtest_service_level"])].pivot(
        index="category", columns="policy", values=["fill_rate", "avg_inventory"]
    )
    pc.columns = [f"{a}_{b}" for a, b in pc.columns]
    v["backtest_cat_table"] = md_table(
        pc.rename(
            columns={
                "fill_rate_baseline": "fill rate (MA28)",
                "fill_rate_forecast": "fill rate (model)",
                "avg_inventory_baseline": "avg inv. (MA28)",
                "avg_inventory_forecast": "avg inv. (model)",
            }
        ),
        "{:,.3f}",
        index=True,
    )

    zs = daily.groupby("category")["units"].apply(lambda x: (x == 0).mean())
    regular = [c for c in zs.index if zs[c] < 0.5]
    bb = best.set_index("category")
    base_m = met[met["model"] == base].set_index("category")
    v["n_intermittent"] = str(len(zs) - len(regular))
    v["smape_gain_regular"] = (
        f"{((1 - bb.loc[regular, 'smape'] / base_m.loc[regular, 'smape']) * 100).mean():.1f}"
    )
    v["mape_gain_regular"] = (
        f"{((1 - bb.loc[regular, 'mape'] / base_m.loc[regular, 'mape']) * 100).mean():.1f}"
    )
    v["mae_gain_min"] = f"{((1 - bb['mae'] / base_m['mae']) * 100).min():.1f}"
    v["mae_gain_max"] = f"{((1 - bb['mae'] / base_m['mae']) * 100).max():.1f}"
    fr = bt[np.isclose(bt["service_level"], h["backtest_service_level"])].pivot(
        index="category", columns="policy", values="fill_rate"
    )
    better = fr.index[fr["forecast"] > fr["baseline"] + 1e-9].tolist()
    worse = fr.index[fr["forecast"] < fr["baseline"] - 1e-9].tolist()
    v["bt_better_n"], v["bt_worse_list"] = str(len(better)), (", ".join(worse) if worse else "none")
    v["bt_n_cats"] = str(len(fr))

    # ---- intervals
    v["coverage_table"] = md_table(cov.assign(model=cov["model"].map(MODEL_LABELS)), "{:.3f}")
    v["coverage_mean"] = f"{cov['empirical_coverage'].mean() * 100:.1f}"
    v["coverage_nominal"] = f"{cfg['forecasting']['interval']['level'] * 100:.0f}"

    # ---- ABC-XYZ, clusters, anomalies, SHAP, plan
    v["abc_table"] = md_table(
        abc[["category", "revenue", "share", "abc", "cv", "xyz", "class"]]
        .assign(share=lambda d: d["share"] * 100)
        .rename(columns={"share": "revenue share %", "cv": "weekly CV"})
    )
    v["abc_policy_table"] = md_table(abc[["category", "class", "policy"]])
    v["cluster_table"] = md_table(
        cl[
            [
                "category",
                "demand_label",
                "kmeans_cluster",
                "hier_cluster",
                "mean_daily",
                "cv_weekly",
                "zero_share",
                "weekly_seasonality",
                "annual_seasonality",
            ]
        ],
        "{:.3f}",
    )
    v["cluster_k"], v["cluster_sil"] = str(cl_meta["k"]), f"{cl_meta['silhouette']:.2f}"
    fl = an[an["anomaly"]]
    cnt = fl.groupby(["category", "direction"]).size().unstack(fill_value=0)
    cnt["share of days %"] = fl.groupby("category").size() / an.groupby("category").size() * 100
    v["anomaly_table"] = md_table(cnt.fillna(0), "{:.2f}", index=True)
    v["anomaly_total"] = str(len(fl))
    top = fl.reindex(fl["z"].abs().sort_values(ascending=False).index).head(8)
    v["anomaly_top_table"] = md_table(
        top.assign(date=top["date"].dt.date)[["date", "category", "actual", "expected", "z", "explanation"]]
    )
    ti = shap.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).head(10)
    v["shap_table"] = md_table(
        ti.reset_index().rename(columns={"mean_abs_shap": "mean |SHAP| (units/day, avg over classes)"}),
        "{:.3f}",
    )
    v["shap_top3"] = ", ".join(ti.index[:3])
    v["plan_table"] = md_table(
        plan[
            [
                "category",
                "model",
                "mean_daily_forecast",
                "daily_error_rmse",
                "lead_time_days",
                "safety_stock",
                "reorder_point",
                "eoq",
            ]
        ],
        "{:,.1f}",
    )
    ass = pd.DataFrame(
        [
            {
                "category": c,
                **{
                    k: category_assumptions(cfg, c)[k]
                    for k in [
                        "unit_price",
                        "unit_cost",
                        "lead_time_days",
                        "order_cost",
                        "shelf_life_remaining_days",
                    ]
                },
            }
            for c in cfg["categories"]
        ]
    )
    v["assumption_table"] = md_table(ass, "{:,.1f}")
    return v


def fill(template: Path, values: dict[str, str], out: Path) -> None:
    text = template.read_text(encoding="utf-8")
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)) - set(values))
    if missing:
        raise KeyError(f"placeholders without values in {template.name}: {missing}")
    out.write_text(re.sub(r"\{\{(\w+)\}\}", lambda m: values[m.group(1)], text), encoding="utf-8")
    print("wrote", out.relative_to(ROOT))


def main() -> None:
    cfg = load_config()
    vals = collect(cfg)
    docs = ROOT / "docs"
    for t in docs.glob("*.template.md"):
        fill(t, vals, docs / t.name.replace(".template", ""))


if __name__ == "__main__":
    main()
