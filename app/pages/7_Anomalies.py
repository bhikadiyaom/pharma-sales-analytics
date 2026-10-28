"""Anomaly detection on daily sales."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.viz.app_data import cfg, data_badge, page_setup, result, sidebar_filters  # noqa: E402
from src.viz.theme import ACCENT, PRIMARY, style_fig  # noqa: E402

page_setup("Anomalies", "🚨")
df, cats = sidebar_filters()
st.title("🚨 Anomaly Detection")
data_badge()
st.caption(
    "Expected demand = STL trend + weekly seasonality. A day is flagged when its Pearson-style residual "
    "(÷ √expected) has a large robust z-score. An Isolation Forest gives a second opinion: days flagged by both "
    "are 'confirmed'."
)

an = result("anomalies.parquet")
an = an[an["category"].isin(cats) & (an["date"] >= df["date"].min()) & (an["date"] <= df["date"].max())]

c1, c2, c3 = st.columns(3)
cat = c1.selectbox("Category", cats)
thr = c2.slider(
    "|z| threshold",
    3.0,
    10.0,
    float(cfg()["anomaly"]["z_threshold"]),
    0.1,
    help="Lower = more sensitive. Default from config.yaml.",
)
use_iso = c3.checkbox(
    "Only Isolation-Forest-confirmed days",
    value=False,
    help="Keep only days that the Isolation Forest also marks as outliers (fewer, higher confidence).",
)

a = an[an["category"] == cat].copy()
a["flag"] = (a["z"].abs() > thr) & (a["iso_flag"] if use_iso else True)
fl = a[a["flag"]]

m = st.columns(3)
m[0].metric("Flagged days", len(fl), f"{len(fl) / max(len(a), 1) * 100:.1f}% of days")
m[1].metric("Spikes", int((fl["residual"] > 0).sum()))
m[2].metric("Drops", int((fl["residual"] < 0).sum()))

fig = go.Figure()
fig.add_scatter(x=a["date"], y=a["actual"], mode="lines", name="Actual", line=dict(color=PRIMARY, width=1))
fig.add_scatter(
    x=a["date"], y=a["expected"], mode="lines", name="Expected", line=dict(color="#999", dash="dot")
)
fig.add_scatter(
    x=fl["date"],
    y=fl["actual"],
    mode="markers",
    name="Anomaly",
    marker=dict(color=ACCENT, size=9, symbol="x"),
    text=fl["explanation"],
    hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>",
)
st.plotly_chart(
    style_fig(fig, f"{cat}: actual vs expected demand", 440).update_layout(hovermode="closest"),
    width="stretch",
)

b1, b2 = st.columns([3, 2])
with b1:
    st.subheader("Flagged days and explanations")
    tab = fl.assign(date=fl["date"].dt.date)[["date", "actual", "expected", "z", "direction", "explanation"]]
    tab = tab.sort_values("z", key=lambda s: -s.abs()).round(2)
    st.dataframe(tab, width="stretch", hide_index=True, height=360)
    st.download_button(
        "⬇ Download anomalies (CSV)",
        tab.to_csv(index=False).encode(),
        file_name=f"anomalies_{cat}.csv",
        mime="text/csv",
    )
with b2:
    per = fl.assign(year=fl["date"].dt.year).groupby(["year", "direction"]).size().reset_index(name="n")
    if not per.empty:
        fig = px.bar(
            per, x="year", y="n", color="direction", color_discrete_map={"spike": ACCENT, "drop": PRIMARY}
        )
        fig.update_xaxes(type="category")
        st.plotly_chart(style_fig(fig, "Anomalies per year", 360), width="stretch")
st.caption(
    "Possible causes to check: stock-outs (zero-sales drops), closures/holidays, outbreaks or promotions (spikes), "
    "data-entry errors. The data has no event log, so causes are hypotheses, not facts."
)
