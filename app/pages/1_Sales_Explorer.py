"""Sales Explorer: aggregate, compare and download."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.viz.app_data import data_badge, page_setup, sidebar_filters  # noqa: E402
from src.viz.theme import CATEGORY_COLORS, style_fig  # noqa: E402

page_setup("Sales Explorer", "🔎")
df, cats = sidebar_filters()
st.title("🔎 Sales Explorer")
data_badge()

c1, c2, c3 = st.columns(3)
metric = c1.radio(
    "Metric", ["units", "revenue"], horizontal=True, help="Revenue uses assumed unit prices from config.yaml."
)
freq = c2.radio("Aggregation", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True)
smooth = c3.slider("Moving average (periods)", 1, 30, 1, help="1 = no smoothing")
rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq]

wide = df.pivot_table(index="date", columns="category", values=metric, aggfunc="sum")
agg = wide.resample(rule).sum()
if freq != "Daily" and len(agg) > 2:  # drop partial first/last period
    agg = agg.iloc[1:-1]
if smooth > 1:
    agg = agg.rolling(smooth, min_periods=1).mean()
long = agg.reset_index().melt("date", var_name="category", value_name=metric)
fig = px.line(long, x="date", y=metric, color="category", color_discrete_map=CATEGORY_COLORS)
st.plotly_chart(
    style_fig(fig, f"{freq} {metric}" + (f" ({smooth}-period moving average)" if smooth > 1 else "")),
    width="stretch",
)

t1, t2, t3 = st.tabs(["Correlation", "Weekday profile", "Data table"])
with t1:
    if len(cats) < 2:
        st.info("Select at least two categories to see correlations.")
    else:
        basis = st.radio(
            "Correlation on",
            ["Weekly totals", "Daily values"],
            horizontal=True,
            help="Weekly totals remove the day-of-week effect shared by all categories.",
        )
        base = wide.resample("W").sum().iloc[1:-1] if basis == "Weekly totals" else wide
        corr = base.corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"
        )
        st.plotly_chart(style_fig(fig, "Correlation between categories", 480), width="stretch")
        st.caption(
            "High positive correlation = categories rise and fall together (shared seasonality or "
            "pharmacy traffic); they can be forecast/stocked with similar logic."
        )
with t2:
    d = df.assign(weekday=df["date"].dt.day_name())
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig = px.box(
        d,
        x="weekday",
        y="units",
        color="category",
        category_orders={"weekday": order},
        color_discrete_map=CATEGORY_COLORS,
    )
    st.plotly_chart(style_fig(fig, "Daily units by weekday", 460), width="stretch")
with t3:
    out = agg.reset_index()
    st.dataframe(out.round({c: 2 for c in out.columns if c != "date"}), width="stretch", height=360)
    st.download_button(
        "⬇ Download aggregated table (CSV)",
        out.to_csv(index=False).encode(),
        file_name=f"sales_{freq.lower()}_{metric}.csv",
        mime="text/csv",
    )
