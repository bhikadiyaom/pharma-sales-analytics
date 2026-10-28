"""Build and execute the three explanatory notebooks.

    python notebooks/build_notebooks.py

Notebooks call the same ``src`` functions as the pipeline/app, so they can never drift from
the production code.  Commentary contains no hard-coded results: every number is printed by
code that was actually executed (run after ``make train``).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SETUP = """import sys, warnings
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from src.config import load_config, results_dir
cfg = load_config()
RES, FIG = results_dir(cfg), ROOT / cfg["paths"]["figures_dir"]
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
pd.options.display.float_format = "{:,.2f}".format
def savefig(name):
    plt.tight_layout(); plt.savefig(FIG / f"{name}.png", dpi=150); plt.show()"""


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s: str):
    return nbf.v4.new_code_cell(s.strip())


def nb_eda():
    return [
        md("""# 01 - Exploratory & Descriptive Analytics
**Goal:** understand the real pharmacy sales data before modelling: quality, trend, seasonality, product mix,
and how stable each therapeutic class is (ABC-XYZ).

> Data: Kaggle *Pharma Sales Data* (real, not synthetic): quantities per ATC class. Prices used for revenue are
> **assumptions** from `config.yaml` because the data has no price column."""),
        code(SETUP),
        md(
            "## 1. Ingestion, validation and cleaning\n`src/data/ingest.py` validates the schema, removes duplicates, "
            "completes the calendar, imputes gaps and winsorises only *extreme* outliers. The quality report is saved as JSON."
        ),
        code("""import json
dq = json.loads((RES / "data_quality_report.json").read_text())
pd.Series({k: v for k, v in dq.items() if not isinstance(v, (dict, list))}).to_frame("value")"""),
        code("""pd.DataFrame({k: dq[k] for k in ["outliers_capped_by_category", "zero_sales_day_share"]})"""),
        md("`units_raw` vs `units` shows exactly what the cleaning changed:"),
        code("""from src.data.store import read_daily
daily = read_daily(cfg)
chg = daily[daily.units != daily.units_raw]
print(f"{len(chg)} of {len(daily):,} rows changed by outlier capping")
chg[["date", "category", "units_raw", "units"]].head(10)"""),
        md("## 2. KPIs"),
        code("""from src.analytics.descriptive import kpis
k = kpis(daily)
print({a: (round(b, 2) if isinstance(b, float) else b) for a, b in k.items() if a != "top_categories"})
k["top_categories"]"""),
        md("## 3. Trend and category mix"),
        code("""piv = daily.pivot(index="date", columns="category", values="units")
m = piv.resample("MS").sum().iloc[1:-1]
m.plot(figsize=(11, 4.5), lw=1.4, title="Monthly units sold by ATC class (complete months)")
plt.ylabel("units"); savefig("eda_monthly_units")"""),
        code("""share = daily.groupby("category")["revenue"].sum().sort_values()
share.plot.barh(figsize=(7, 3.5), color="#0F5C8C", title="Revenue by class (ASSUMED prices)")
savefig("eda_revenue_share")"""),
        md(
            "## 4. Seasonality: STL decomposition\nSTL (Cleveland et al., 1990) separates trend, seasonal and remainder. "
            "Weekly cycle on daily data; annual cycle on weekly totals."
        ),
        code("""from src.analytics.descriptive import stl_decompose, seasonal_strength
cat = "R03"
y = piv[cat]
res = stl_decompose(y.resample("W").sum().iloc[1:-1], 52)
fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
for a, (n, s) in zip(ax, [("observed", res.observed), ("trend", res.trend), ("seasonal", res.seasonal), ("resid", res.resid)]):
    a.plot(s, lw=1); a.set_ylabel(n)
ax[0].set_title(f"{cat}: STL decomposition of weekly totals (period 52)")
savefig("eda_stl_R03")"""),
        code("""strength = pd.DataFrame({
    "weekly_cycle": {c: seasonal_strength(piv[c], 7) for c in piv},
    "annual_cycle": {c: seasonal_strength(piv[c].resample("W").sum().iloc[1:-1], 52) for c in piv}})
strength.sort_values("annual_cycle", ascending=False)"""),
        md("### Weekday x month and weekday x hour heatmaps"),
        code("""d = daily.assign(weekday=daily.date.dt.dayofweek, month=daily.date.dt.month)
order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, axes = plt.subplots(2, 4, figsize=(15, 6))
for ax, c in zip(axes.ravel(), sorted(d.category.unique())):
    p = d[d.category == c].pivot_table(index="weekday", columns="month", values="units", aggfunc="mean")
    p.index = order
    sns.heatmap(p, ax=ax, cmap="Blues", cbar=False); ax.set_title(c); ax.set_xlabel(""); ax.set_ylabel("")
savefig("eda_weekday_month_heatmaps")"""),
        code(
            """from src.data.store import read_hourly
h = read_hourly(cfg)
h = h.assign(weekday=h.date.dt.dayofweek, hour=h.date.dt.hour)
p = h.groupby(["weekday", "hour"])["units"].mean().unstack()
p.index = order
plt.figure(figsize=(11, 3.8)); sns.heatmap(p, cmap="Oranges")
plt.title("Average units per open hour (all classes): weekday x hour"); savefig("eda_weekday_hour_heatmap")"""
        ),
        md("## 5. Correlation between classes"),
        code("""wk = piv.resample("W").sum().iloc[1:-1]
plt.figure(figsize=(6, 5)); sns.heatmap(wk.corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1)
plt.title("Correlation of weekly totals"); savefig("eda_correlation")"""),
        md(
            "## 6. ABC (Pareto) and XYZ (variability) analysis\nABC uses revenue contribution (assumed prices); "
            "XYZ uses the coefficient of variation of *weekly* demand (daily CV is dominated by noise)."
        ),
        code("""from src.analytics.descriptive import abc_xyz_matrix
mx = abc_xyz_matrix(daily, cfg)
mx[["category", "revenue", "share", "cum_share", "abc", "cv", "xyz", "class"]]"""),
        code("""mx[["category", "class", "policy"]]"""),
        code("""fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(mx.category, mx.share * 100, color="#0F5C8C"); ax2 = ax.twinx()
ax2.plot(mx.category, mx.cum_share * 100, "o-", color="#E4572E"); ax2.set_ylim(0, 105)
ax.set_ylabel("share %"); ax2.set_ylabel("cumulative %"); ax.set_title("Pareto chart (ABC)")
savefig("eda_pareto")"""),
        md(
            "## 7. Demand behaviour clusters\nBehavioural features per class -> KMeans (k by silhouette) and Ward clustering, "
            "next to an interpretable rule-based label. With only 8 series this is exploratory."
        ),
        code("""cl = pd.read_csv(RES / "demand_clusters.csv")
cl[["category", "demand_label", "kmeans_cluster", "hier_cluster", "mean_daily", "cv_weekly", "zero_share",
    "weekly_seasonality", "annual_seasonality"]]"""),
        code("""plt.figure(figsize=(6.5, 5))
sns.scatterplot(data=cl, x="pc1", y="pc2", hue="kmeans_cluster", style="demand_label", s=120, palette="Set2")
for r in cl.itertuples(): plt.text(r.pc1 + .05, r.pc2 + .05, r.category)
plt.title("Demand-behaviour map (PCA)"); savefig("eda_clusters")"""),
    ]


def nb_model():
    return [
        md(
            """# 02 - Demand Forecasting
Six forecasters per series, compared with **rolling-origin cross-validation** (no leakage), automatic
best-model selection, empirical prediction intervals and SHAP.

| Model | Idea |
|---|---|
| Naive / Seasonal naive (7 d) | Baselines: last value / same weekday last week |
| SARIMAX | SARIMA(1,0,1)(1,1,1)7 + annual Fourier terms |
| Prophet | Trend + weekly + yearly seasonality + festive dates |
| XGBoost / LightGBM | Gradient-boosted trees on calendar, lag and rolling features (recursive multi-step) |"""
        ),
        code(SETUP),
        md(
            "## 1. Features and leakage control\nEvery lag/rolling feature at day *t* uses only data up to *t-1*."
        ),
        code("""from src.features.engineering import build_feature_frame
from src.data.store import read_daily
daily = read_daily(cfg)
y = daily[daily.category == "N02BE"].set_index("date")["units"].asfreq("D")
X = build_feature_frame(y, cfg)
print(X.shape); X.tail(3).T"""),
        code("""# leakage test: changing today's value must not change today's features
y2 = y.copy(); y2.iloc[-1] += 1000
print("features identical:", build_feature_frame(y2, cfg).iloc[-1].equals(X.iloc[-1]))"""),
        md(
            "## 2. Rolling-origin cross-validation\nExpanding window; each fold refits on the past only and predicts the next 14 days. "
            "26 contiguous folds = the final 364 days are out-of-sample."
        ),
        code(
            """from src.models.cv import rolling_origins
cv = cfg["forecasting"]["cv"]
o = rolling_origins(len(y), cv["horizon"], cv["step"], cv["n_folds"], cv["min_train_days"])
print("folds:", len(o), "| first origin:", y.index[o[0]-1].date(), "| last test day:", y.index[-1].date())"""
        ),
        md(
            "### Live demonstration on one series (fast models only)\nThe full six-model CV is produced by `make train`."
        ),
        code("""from src.models.cv import run_cv
from src.models import metrics as M
demo = run_cv("N02BE", y, cfg, models=["naive", "snaive", "xgboost", "lightgbm"])
pd.DataFrame({m: M.all_metrics(g.y, g.yhat) for m, g in demo.groupby("model")}).T"""),
        md("## 3. Results of the full cross-validation (`reports/results/cv_metrics.csv`)"),
        code("""mt = pd.read_csv(RES / "cv_metrics.csv")
mt.pivot(index="category", columns="model", values="mae").assign(best=lambda d: d.idxmin(axis=1))"""),
        code(
            """mt.pivot(index="category", columns="model", values="rmse").assign(best=lambda d: d.idxmin(axis=1))"""
        ),
        code("""mt.pivot(index="category", columns="model", values="smape")"""),
        code(
            """best = pd.read_csv(RES / "best_models.csv")
best[["category", "model", "mae", "rmse", "smape", "mape", "mase", "mae_improv_pct", "rmse_improv_pct", "smape_improv_pct"]]"""
        ),
        code("""fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=mt, x="category", y="rmse", hue="model", ax=ax, palette="viridis")
ax.set_title("Out-of-sample RMSE by model (lower is better)"); ax.legend(ncol=3, fontsize=8)
savefig("model_rmse_comparison")"""),
        md("### Error by forecast horizon"),
        code(
            """eh = pd.read_csv(RES / "cv_error_by_horizon.csv")
c = eh[eh.category == "N02BE"]
sns.lineplot(data=c, x="h", y="ae", hue="model"); plt.title("N02BE: MAE vs days ahead"); savefig("model_error_by_horizon")"""
        ),
        md("### Out-of-sample forecasts of the selected model"),
        code("""cvp = pd.read_parquet(RES / "cv_predictions.parquet")
bm = dict(zip(best.category, best.model))
fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True)
for ax, c in zip(axes.ravel(), sorted(bm)):
    g = cvp[(cvp.category == c) & (cvp.model == bm[c])].sort_values("date")
    ax.plot(g.date, g.y, lw=.8, color="#444"); ax.plot(g.date, g.yhat, lw=.8, color="#E4572E")
    ax.set_title(f"{c} - {bm[c]}", fontsize=10)
savefig("model_oos_forecasts")"""),
        md(
            "## 4. Prediction intervals\nModel-agnostic empirical intervals: residuals from several older forecast origins (up to 90 days ahead), "
            "scaled by sqrt(level) because count variance grows with the mean. Coverage is scored on the most recent origins "
            "after calibrating on older ones."
        ),
        code("""pd.read_csv(RES / "interval_coverage.csv")"""),
        code("""fb = pd.read_parquet(RES / "forecasts_best.parquet")
c = "R03"; g = fb[fb.category == c]; hist = daily[daily.category == c].set_index("date")["units"].iloc[-90:]
plt.figure(figsize=(11, 4)); plt.plot(hist.index, hist, color="#444", lw=1, label="actual")
plt.plot(g.date, g.yhat, color="#E4572E", label="forecast"); plt.fill_between(g.date, g.lower, g.upper, alpha=.2, color="#E4572E", label="90% interval")
plt.legend(); plt.title(f"{c}: 90-day forecast with interval"); savefig("model_forecast_interval")"""),
        md("## 5. Explainability: SHAP for the tree model"),
        code(
            """imp = pd.read_csv(RES / "shap_importance.csv")
top = imp.groupby("feature").mean_abs_shap.mean().sort_values().tail(12)
top.plot.barh(figsize=(7, 4.5), color="#0F5C8C", title="Mean |SHAP| averaged over classes (XGBoost)"); savefig("model_shap_global")"""
        ),
        code(
            """imp.pivot(index="feature", columns="category", values="mean_abs_shap").loc[top.index[::-1]]"""
        ),
    ]


def nb_inv():
    return [
        md(
            """# 03 - Inventory Decisions
Turning forecasts into replenishment decisions.

**Formulas** (per item; d = mean forecast demand per day, sigma = daily forecast RMSE, L = lead time, z = service-level quantile):

* Safety stock  `SS = z * sqrt(L*sigma^2 + d^2*sigma_L^2)`
* Reorder point `ROP = d*L + SS`
* Economic order quantity `EOQ = sqrt(2*D*S/H)`, `H = holding_rate * unit_cost`

Lead time, costs, shelf life and stock on hand are **assumptions** (`config.yaml`); the data has none of them."""
        ),
        code(SETUP),
        code("""from src.inventory.planner import build_plan_table, default_inputs
fb = pd.read_parquet(RES / "forecasts_best.parquet")
best = pd.read_csv(RES / "best_models.csv").set_index("category")
cats = sorted(best.index)
window = cfg["inventory"]["planning_horizon_days"]
inputs = default_inputs(cfg, fb, cats, window)
inputs"""),
        md("## 1. Plan at 90 / 95 / 99 % service level"),
        code(
            """plans = pd.concat([build_plan_table(cfg, fb, best, inputs, sl, window) for sl in cfg["inventory"]["service_levels"]])
plans.pivot(index="category", columns="service_level", values="safety_stock")"""
        ),
        code("""plans[plans.service_level == cfg["inventory"]["default_service_level"]][
    ["category", "mean_daily_forecast", "daily_error_rmse", "lead_time_days", "safety_stock", "reorder_point", "eoq",
     "on_hand", "status", "suggested_order_qty", "expiry_risk_units"]]"""),
        md("## 2. Stockout / overstock / expiry flags"),
        code("""p = plans[plans.service_level == cfg["inventory"]["default_service_level"]]
colors = {"OK": "#2E8B57", "REORDER NOW": "#E69F00", "STOCKOUT RISK": "#C0392B", "OVERSTOCK": "#6C5CE7"}
ax = p.set_index("category")[["on_hand", "reorder_point", "safety_stock"]].plot.bar(figsize=(10, 4), color=["#0F5C8C", "#E4572E", "#999"])
ax.set_title("Stock on hand (assumed default cover) vs reorder point and safety stock"); savefig("inv_stock_position")
p[["category", "status", "days_of_cover", "expiry_risk_units", "expiry_risk_value"]]"""),
        md("## 3. What-if: epidemic surge and supplier delay"),
        code("""from src.inventory.whatif import Scenario, simulate
from src.config import category_assumptions
rows = []
for name, sc in {"baseline": Scenario(), "epidemic +30%": Scenario(demand_surge_pct=30),
                 "lead time +4d": Scenario(lead_time_change_days=4), "price -10%": Scenario(price_change_pct=-10),
                 "promo +25%": Scenario(promo_uplift_pct=25)}.items():
    c = "R03"; a = category_assumptions(cfg, c)
    d = float(fb[fb.category == c].sort_values("h").yhat.head(window).mean())
    r = simulate(d, float(best.loc[c, "rmse"]), a["unit_price"], a["unit_cost"], a["lead_time_days"], a["lead_time_sd_days"],
                 0.95, a["order_cost"], cfg["inventory"]["holding_rate_annual"], sc, days=window)
    rows.append({"scenario": name, "demand/day": r.new_demand, "revenue_30d": r.new_revenue_30d,
                 "safety_stock": r.new_plan.safety_stock, "ROP": r.new_plan.reorder_point, "EOQ": r.new_plan.eoq})
pd.DataFrame(rows).set_index("scenario")"""),
        md("## 4. Anomalies"),
        code(
            """an = pd.read_parquet(RES / "anomalies.parquet")
flag = an[an.anomaly]
print(flag.groupby(["category", "direction"]).size().unstack(fill_value=0))
flag.sort_values("z", key=abs, ascending=False)[["date", "category", "actual", "expected", "z", "explanation"]].head(10)"""
        ),
        md(
            "## 5. Does forecast-driven replenishment help? Back-test on the out-of-sample year\n"
            "Both policies are periodic-review order-up-to rules with identical lead times and service-level target; "
            "only the demand mean/error estimates differ (28-day moving average vs. model forecast with its previous out-of-sample error). "
            "Unmet demand is lost."
        ),
        code("""bt = pd.read_csv(RES / "inventory_backtest.csv")
agg = bt.groupby(["service_level", "policy"]).agg(lost=("lost_units", "sum"), demand=("demand_units", "sum"),
        stockout_days=("stockout_day_share", "mean"), inv_value=("avg_inventory_value", "sum"))
agg["fill_rate_%"] = (1 - agg.lost / agg.demand) * 100
agg["stockout_days_%"] = agg.stockout_days * 100
agg[["fill_rate_%", "stockout_days_%", "lost", "inv_value"]]"""),
        code("""# per-category view at the default service level
bt[bt.service_level == cfg["inventory"]["default_service_level"]].pivot(
    index="category", columns="policy", values=["fill_rate", "avg_inventory"])"""),
        code("""import json
print(json.dumps(json.loads((RES / "headline.json").read_text()), indent=2))"""),
    ]


def build() -> None:
    specs = {"01_eda.ipynb": nb_eda, "02_modeling.ipynb": nb_model, "03_inventory.ipynb": nb_inv}
    for name, fn in specs.items():
        nb = nbf.v4.new_notebook()
        nb.cells = fn()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": str(HERE)}}
        )
        nbf.write(nb, HERE / name)
        print("built", name)


if __name__ == "__main__":
    build()
