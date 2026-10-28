"""Model Performance: cross-validated comparison of all forecasters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.models.forecasters import MODEL_LABELS  # noqa: E402
from src.viz.app_data import cfg, data_badge, page_setup, result, result_json, sidebar_filters  # noqa: E402
from src.viz.theme import ACCENT, style_fig  # noqa: E402

page_setup("Model Performance", "🏁")
_, cats = sidebar_filters()
st.title("🏁 Model Performance")
data_badge()
cv = cfg()["forecasting"]["cv"]
st.caption(
    f"Rolling-origin cross-validation: expanding training window, {cv['n_folds']} folds × {cv['horizon']}-day horizon "
    f"(the last {cv['n_folds'] * cv['step']} days are out-of-sample). Every fold refits on the past only - no leakage."
)

m = result("cv_metrics.csv")
m = m[m["category"].isin(cats)]
m["model_label"] = m["model"].map(MODEL_LABELS)
best = result("best_models.csv")
best = best[best["category"].isin(cats)]
sel = cfg()["forecasting"]["selection_metric"]

h = result_json("headline.json")
k = st.columns(4)
k[0].metric(
    "MAE gain vs seasonal-naive",
    f"{h['mean_mae_improvement_pct']:.1f}%",
    help="Mean over categories, best model.",
)
k[1].metric("RMSE gain", f"{h['mean_rmse_improvement_pct']:.1f}%")
k[2].metric("sMAPE gain", f"{h['mean_smape_improvement_pct']:.1f}%")
k[3].metric(
    "MAPE gain",
    f"{h['mean_mape_improvement_pct']:.1f}%",
    help="MAPE only counts days with sales > 0 and is unstable for slow movers; prefer MAE/RMSE/sMAPE.",
)

t1, t2, t3, t4 = st.tabs(["Comparison", "Best per series", "Error by horizon", "Inventory back-test"])
with t1:
    metric = st.selectbox(
        "Metric",
        ["mae", "rmse", "smape", "mape", "mase"],
        index=0,
        help="MAE/RMSE in units per day; sMAPE/MAPE in %; MASE < 1 beats the in-sample seasonal naive.",
    )
    piv = m.pivot(index="category", columns="model_label", values=metric)
    fig = px.imshow(piv, text_auto=".2f", color_continuous_scale="RdYlGn_r", aspect="auto")
    st.plotly_chart(
        style_fig(fig, f"{metric.upper()} by category and model (lower is better)", 420), width="stretch"
    )
    st.dataframe(
        m[["category", "model_label", "mae", "rmse", "mape", "smape", "mase"]]
        .round(3)
        .sort_values(["category", "mae"]),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "⬇ Download metrics (CSV)",
        m.drop(columns="model_label").to_csv(index=False).encode(),
        file_name="cv_metrics.csv",
        mime="text/csv",
    )
with t2:
    show = best[
        ["category", "model", "mae", "rmse", "smape", "mape", "mase", "mae_improv_pct", "rmse_improv_pct"]
    ].round(2)
    show["model"] = show["model"].map(MODEL_LABELS)
    st.dataframe(
        show.rename(
            columns={"mae_improv_pct": "MAE gain vs baseline %", "rmse_improv_pct": "RMSE gain vs baseline %"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        f"Selection rule: lowest cross-validated {sel.upper()} per series (RMSE rewards unbiased mean forecasts, "
        "which is what replenishment needs). The same folds are used to select and to report, so reported gains "
        "are slightly optimistic - see Limitations in About."
    )
    fig = px.bar(
        best,
        x="category",
        y="rmse_improv_pct",
        color="model",
        labels={"rmse_improv_pct": "RMSE gain vs seasonal naive %"},
    )
    st.plotly_chart(
        style_fig(fig, "Gain of the selected model over the seasonal-naive baseline", 380), width="stretch"
    )
    cov = result("interval_coverage.csv")
    st.markdown(
        "**Prediction-interval calibration** (calibrated on older origins, scored on the most recent):"
    )
    st.dataframe(cov[cov["category"].isin(cats)].round(3), width="stretch", hide_index=True)
with t3:
    eh = result("cv_error_by_horizon.csv")
    cat = st.selectbox("Category", cats, key="hcat")
    e = eh[eh["category"] == cat].assign(model=lambda d: d["model"].map(MODEL_LABELS))
    fig = px.line(e, x="h", y="ae", color="model", labels={"h": "Days ahead", "ae": "Mean absolute error"})
    st.plotly_chart(style_fig(fig, f"{cat}: error vs forecast horizon", 400), width="stretch")
    cvp = result("cv_predictions.parquet")
    mdl = best.set_index("category").loc[cat, "model"]
    g = cvp[(cvp["category"] == cat) & (cvp["model"] == mdl)].sort_values("date")
    fig = go.Figure()
    fig.add_scatter(x=g["date"], y=g["y"], name="Actual", line=dict(color="#444", width=1))
    fig.add_scatter(
        x=g["date"],
        y=g["yhat"],
        name=f"Out-of-sample forecast ({MODEL_LABELS[mdl]})",
        line=dict(color=ACCENT, width=1.2),
    )
    st.plotly_chart(
        style_fig(fig, f"{cat}: stitched out-of-sample forecasts (14-day windows)", 380), width="stretch"
    )
with t4:
    bt = result("inventory_backtest.csv")
    bt = bt[bt["category"].isin(cats)]
    agg = (
        bt.groupby(["service_level", "policy"])
        .agg(
            lost=("lost_units", "sum"),
            demand=("demand_units", "sum"),
            stockout_days=("stockout_day_share", "mean"),
            inv_value=("avg_inventory_value", "sum"),
        )
        .reset_index()
    )
    agg["fill_rate_%"] = (1 - agg["lost"] / agg["demand"]) * 100
    agg["stockout_days_%"] = agg["stockout_days"] * 100
    agg["policy"] = agg["policy"].map({"baseline": "28-day moving average", "forecast": "Model forecast"})
    st.dataframe(
        agg[["service_level", "policy", "fill_rate_%", "stockout_days_%", "lost", "inv_value"]].round(2),
        width="stretch",
        hide_index=True,
    )
    fig = px.bar(agg, x="service_level", y="fill_rate_%", color="policy", barmode="group")
    fig.update_yaxes(range=[max(agg["fill_rate_%"].min() - 3, 0), 100.5])
    st.plotly_chart(style_fig(fig, "Fill rate by target service level", 360), width="stretch")
    st.caption(
        "Same order-up-to logic, lead times and review period for both policies; only the demand mean and "
        "error estimate differ. Lead time, review period and costs are assumptions from config.yaml."
    )
