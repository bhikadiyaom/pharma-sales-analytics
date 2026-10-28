"""Demand forecasting with prediction intervals and SHAP explanation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.models.forecasters import MODEL_LABELS  # noqa: E402
from src.viz.app_data import cfg, daily, data_badge, page_setup, result, sidebar_filters  # noqa: E402
from src.viz.theme import ACCENT, PRIMARY, style_fig  # noqa: E402

page_setup("Forecasting", "📈")
_, cats = sidebar_filters()
st.title("📈 Demand Forecasting")
data_badge()

best = result("best_models.csv").set_index("category")
fb = result("forecasts_best.parquet")
fa = result("forecasts_all.parquet")
hist_all = daily()

c1, c2, c3 = st.columns(3)
cat = c1.selectbox("Category", cats)
horizon = c2.select_slider("Forecast horizon (days)", options=cfg()["forecasting"]["horizons"], value=30)
best_model = best.loc[cat, "model"]
choice = c3.selectbox(
    "Model",
    ["Best (auto-selected)"] + [MODEL_LABELS[m] for m in cfg()["forecasting"]["models"]],
    help=f"Auto-selection picks the lowest cross-validated {cfg()['forecasting']['selection_metric'].upper()} "
    f"per series → {MODEL_LABELS[best_model]} for {cat}.",
)
label_to_key = {v: k for k, v in MODEL_LABELS.items()}
model = best_model if choice.startswith("Best") else label_to_key[choice]

hist = hist_all[hist_all["category"] == cat].set_index("date")["units"]
look = st.slider("History shown (days)", 30, 365, 120)
if model == best_model:
    f = fb[fb["category"] == cat].sort_values("h").head(horizon)
else:
    f = fa[(fa["category"] == cat) & (fa["model"] == model)].sort_values("h").head(horizon)

fig = go.Figure()
fig.add_scatter(
    x=hist.index[-look:], y=hist.iloc[-look:], name="Actual", mode="lines", line=dict(color="#444", width=1.4)
)
if "lower" in f:
    fig.add_scatter(
        x=f["date"], y=f["upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
    )
    fig.add_scatter(
        x=f["date"],
        y=f["lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(228,87,46,0.18)",
        name=f"{int(cfg()['forecasting']['interval']['level'] * 100)}% interval",
    )
fig.add_scatter(
    x=f["date"],
    y=f["yhat"],
    name=f"Forecast ({MODEL_LABELS[model]})",
    mode="lines",
    line=dict(color=ACCENT, width=2),
)
st.plotly_chart(style_fig(fig, f"{cat}: {horizon}-day demand forecast (units/day)", 440), width="stretch")
if "lower" not in f:
    st.caption(
        "Prediction intervals are calibrated for the auto-selected model only; choose 'Best' to see them."
    )

m = st.columns(4)
m[0].metric("Forecast total", f"{f['yhat'].sum():,.0f} units", help=f"Sum over the next {horizon} days.")
m[1].metric("Mean per day", f"{f['yhat'].mean():,.1f}")
last = hist.iloc[-horizon:].sum()
m[2].metric(
    "vs. previous period",
    f"{(f['yhat'].sum() / last - 1) * 100:+.1f}%" if last > 0 else "n/a",
    help=f"Compared with the last {horizon} observed days.",
)
m[3].metric(
    "CV error (RMSE)",
    f"{best.loc[cat, 'rmse']:.2f}",
    help="Daily RMSE of the best model over the out-of-sample year.",
)

out = f[[c for c in ["date", "yhat", "lower", "upper"] if c in f]].copy()
out[[c for c in out.columns if c != "date"]] = out[[c for c in out.columns if c != "date"]].round(2)
out.insert(1, "category", cat)
out.insert(2, "model", model)
st.download_button(
    "⬇ Download forecast (CSV)",
    out.to_csv(index=False).encode(),
    file_name=f"forecast_{cat}_{horizon}d.csv",
    mime="text/csv",
)

st.subheader("What drives the tree-model forecast? (SHAP)")
imp = result("shap_importance.csv")
imp = imp[imp["category"] == cat].head(12).iloc[::-1]
a, b = st.columns(2)
with a:
    fig = px.bar(
        imp,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color_discrete_sequence=[PRIMARY],
        labels={"mean_abs_shap": "mean |SHAP| (units/day)", "feature": ""},
    )
    st.plotly_chart(
        style_fig(fig, "Feature importance (XGBoost)", 400).update_layout(hovermode="closest"),
        width="stretch",
    )
with b:
    sh = result("shap_sample.parquet")
    sh = sh[(sh["category"] == cat) & sh["feature"].isin(imp["feature"])].copy()
    sh["rank"] = sh.groupby("feature")["feature_value"].rank(pct=True)  # colour by relative value
    pos = {f_: i for i, f_ in enumerate(imp["feature"])}  # bottom-to-top order
    rng = np.random.default_rng(0)
    sh["ypos"] = sh["feature"].map(pos) + rng.uniform(-0.3, 0.3, len(sh))
    fig = go.Figure(
        go.Scattergl(
            x=sh["shap"],
            y=sh["ypos"],
            mode="markers",
            marker=dict(
                size=4,
                opacity=0.65,
                color=sh["rank"],
                colorscale="RdBu_r",
                colorbar=dict(title="feature value<br>(percentile)"),
            ),
            customdata=sh["feature"],
            hovertemplate="%{customdata}: SHAP %{x:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(tickmode="array", tickvals=list(pos.values()), ticktext=list(pos.keys()))
    fig.update_xaxes(title="SHAP value (units/day)")
    st.plotly_chart(
        style_fig(fig, "SHAP summary (red = high feature value)", 400).update_layout(hovermode="closest"),
        width="stretch",
    )
st.caption(
    "SHAP values show how much each feature moved a prediction up or down. Lags and rolling means dominate for "
    "short-term demand; calendar features capture weekday/seasonal effects."
)
