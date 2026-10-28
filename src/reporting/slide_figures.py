"""Slide-ready (16:9) figures and a one-page summary, all drawn from reports/results.

    python -m src.reporting.slide_figures
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT, load_config, results_dir
from src.models.forecasters import MODEL_LABELS
from src.viz.theme import ACCENT, CATEGORY_COLORS, PRIMARY

W, H = 13.33, 7.5
plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 22, "axes.titleweight": "bold", "figure.dpi": 100})
CFG = load_config()
R = results_dir(CFG)
OUT = ROOT / CFG["paths"]["figures_dir"] / "slides"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)


def fig_gain():
    b = pd.read_csv(R / "best_models.csv")
    fig, ax = plt.subplots(figsize=(W, H))
    cols = [PRIMARY if m == "sarimax" else ACCENT for m in b["model"]]
    ax.bar(b["category"], b["rmse_improv_pct"], color=cols)
    for i, (v, m) in enumerate(zip(b["rmse_improv_pct"], b["model"])):
        ax.text(i, v + 0.5, f"{v:.0f}%\n{MODEL_LABELS[m].split(' ')[0]}", ha="center", fontsize=13)
    ax.set_ylabel("RMSE reduction vs seasonal naive (%)")
    ax.set_ylim(0, b["rmse_improv_pct"].max() + 8)
    ax.set_title("Selected model beats the baseline in every drug class")
    save(fig, "1_model_gain")


def fig_forecast():
    d = pd.read_parquet(ROOT / CFG["paths"]["processed_dir"] / "daily_clean.parquet")
    fb = pd.read_parquet(R / "forecasts_best.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    for ax, c in zip(axes, ["N02BE", "R03"]):
        h = d[d.category == c].set_index("date")["units"].iloc[-120:]
        f = fb[fb.category == c]
        ax.plot(h.index, h, color="#555", lw=1, label="actual")
        ax.fill_between(f["date"], f["lower"], f["upper"], color=ACCENT, alpha=0.2, label="90% interval")
        ax.plot(f["date"], f["yhat"], color=ACCENT, lw=2.5, label="forecast")
        ax.set_title(c, fontsize=18)
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=13)
    fig.suptitle("90-day demand forecast with prediction interval", fontsize=22, fontweight="bold")
    save(fig, "2_forecast_interval")


def fig_backtest():
    bt = pd.read_csv(R / "inventory_backtest.csv")
    g = bt.groupby(["service_level", "policy"]).agg(lost=("lost_units", "sum"), dem=("demand_units", "sum"),
                                                    inv=("avg_inventory_value", "sum")).reset_index()
    g["fill"] = (1 - g["lost"] / g["dem"]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    x = np.arange(3)
    for ax, col, ttl in [(axes[0], "fill", "Fill rate (%)"), (axes[1], "inv", "Avg inventory value at cost (assumed)")]:
        for k, (pol, color, lab) in enumerate([("baseline", "#999", "28-day moving avg"), ("forecast", PRIMARY, "Model forecast")]):
            v = g[g.policy == pol].sort_values("service_level")[col].to_numpy()
            ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=color, label=lab)
        ax.set_xticks(x, ["90%", "95%", "99%"])
        ax.set_xlabel("Target service level")
        ax.set_title(ttl, fontsize=18)
        if col == "fill":
            ax.set_ylim(94, 100)
    axes[0].legend(fontsize=13)
    fig.suptitle("Back-test, last 364 days: forecast-driven replenishment", fontsize=22, fontweight="bold")
    save(fig, "3_inventory_backtest")


def fig_abcxyz():
    m = pd.read_csv(R / "abc_xyz.csv")
    fig, ax = plt.subplots(figsize=(W * 0.7, H))
    for i, a in enumerate("ABC"):
        for j, x in enumerate("XYZ"):
            cats = m[(m.abc == a) & (m.xyz == x)]["category"].tolist()
            ax.add_patch(plt.Rectangle((j, 2 - i), 1, 1, fc=plt.cm.Blues(0.15 + 0.2 * len(cats)), ec="white", lw=3))
            ax.text(j + 0.5, 2 - i + 0.5, "\n".join(cats) or "-", ha="center", va="center", fontsize=18)
    ax.set_xlim(0, 3), ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5], ["X stable", "Y variable", "Z erratic"])
    ax.set_yticks([2.5, 1.5, 0.5], ["A high value", "B medium", "C low"])
    ax.set_title("ABC-XYZ matrix (revenue uses assumed prices)")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "4_abc_xyz")


def fig_seasonality():
    d = pd.read_parquet(ROOT / CFG["paths"]["processed_dir"] / "daily_clean.parquet")
    d = d[(d.date >= "2014-02-01") & (d.date < "2019-10-01")]
    fig, ax = plt.subplots(figsize=(W, H))
    for c in ["R06", "R03", "N02BE", "M01AB"]:
        s = d[d.category == c].groupby(d.date.dt.month)["units"].mean()
        ax.plot(s.index, s / s.mean(), marker="o", lw=3, color=CATEGORY_COLORS[c], label=c)
    ax.axhline(1, color="#aaa", ls=":")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg daily units / yearly average")
    ax.legend(fontsize=14)
    ax.set_title("Annual seasonality differs by drug class")
    save(fig, "5_seasonality")


def summary():
    h = json.loads((R / "headline.json").read_text())
    txt = f"""# Results at a glance (generated from reports/results)

Data: real Kaggle Pharma Sales, 8 ATC classes, daily 2014-01-02 to 2019-10-08.
Prices, costs, lead times and stock are assumptions (config.yaml).

* Best model per class: {", ".join(f"{k}: {MODEL_LABELS[v].split(' ')[0]}" for k, v in h["best_model_per_category"].items())}
* Gain over seasonal naive (mean over classes): MAE {h["mean_mae_improvement_pct"]:.1f}%, RMSE {h["mean_rmse_improvement_pct"]:.1f}%
* sMAPE gain {h["mean_smape_improvement_pct"]:.1f}% (misleading for intermittent N05C; see report)
* Back-test at {h["backtest_service_level"]:.0%} service level, last 364 days, forecast vs 28-day moving average:
  fill rate {h["fill_rate_baseline"] * 100:.2f}% -> {h["fill_rate_forecast"] * 100:.2f}%,
  stock-out days {h["stockout_day_reduction_pct"]:.1f}% lower, lost units {h["lost_units_reduction_pct"]:.1f}% lower,
  avg inventory value {h["inventory_value_change_pct"]:+.1f}%
* Caveats: one pharmacy, one year, assumed lead times/costs; top models are within a few percent of each other.
"""
    (OUT / "summary.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    for f in (fig_gain, fig_forecast, fig_backtest, fig_abcxyz, fig_seasonality, summary):
        f()
    print("wrote", OUT)
