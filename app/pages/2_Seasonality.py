"""Seasonality: STL decomposition and calendar heatmaps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from src.analytics.descriptive import seasonal_strength, stl_decompose  # noqa: E402
from src.viz.app_data import data_badge, hourly, page_setup, sidebar_filters  # noqa: E402
from src.viz.theme import PRIMARY, style_fig  # noqa: E402

page_setup("Seasonality", "🌦️")
df, cats = sidebar_filters()
st.title("🌦️ Trend & Seasonality")
data_badge()

cat = st.selectbox("Category", cats)
g = df[df["category"] == cat].set_index("date")["units"].asfreq("D")
mode = st.radio(
    "STL view",
    ["Weekly cycle (daily data, period 7)", "Annual cycle (weekly totals, period 52)"],
    horizontal=True,
    help="STL = Seasonal-Trend decomposition using LOESS (Cleveland et al., 1990).",
)
y = g if mode.startswith("Weekly") else g.resample("W").sum().iloc[1:-1]
period = 7 if mode.startswith("Weekly") else 52

if len(y) < 2 * period + 1:
    st.warning("Selected date range is too short for this decomposition.")
else:
    res = stl_decompose(y, period)
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
    )
    for i, s in enumerate([y, res.trend, res.seasonal, res.resid], 1):
        fig.add_trace(
            go.Scatter(x=s.index, y=s, mode="lines", line=dict(color=PRIMARY, width=1.2), showlegend=False),
            row=i,
            col=1,
        )
    st.plotly_chart(style_fig(fig, None, 640).update_layout(hovermode="x"), width="stretch")
    fs = seasonal_strength(y, period)
    st.metric(
        "Seasonal strength",
        f"{fs:.2f}",
        help="0 = no seasonality, 1 = fully seasonal (Wang, Smith & Hyndman, 2006).",
    )

a, b = st.columns(2)
with a:
    wk = df[df["category"] == cat].assign(
        weekday=lambda d: d["date"].dt.day_name(), month=lambda d: d["date"].dt.month
    )
    pv = wk.pivot_table(index="weekday", columns="month", values="units", aggfunc="mean")
    pv = pv.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig = px.imshow(
        pv, color_continuous_scale="Blues", aspect="auto", labels=dict(x="Month", y="", color="Avg units")
    )
    st.plotly_chart(style_fig(fig, f"{cat}: average units, weekday × month", 380), width="stretch")
with b:
    h = hourly()
    h = h[
        (h["category"] == cat)
        & (h["date"] >= df["date"].min())
        & (h["date"] <= df["date"].max() + np.timedelta64(1, "D"))
    ]
    h = h.assign(weekday=h["date"].dt.day_name(), hour=h["date"].dt.hour)
    pv = h.pivot_table(index="weekday", columns="hour", values="units", aggfunc="mean")
    pv = pv.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig = px.imshow(
        pv,
        color_continuous_scale="Oranges",
        aspect="auto",
        labels=dict(x="Hour of day", y="", color="Avg units/h"),
    )
    st.plotly_chart(style_fig(fig, f"{cat}: average units, weekday × hour", 380), width="stretch")

m = df[df["category"] == cat].assign(year=lambda d: d["date"].dt.year, month=lambda d: d["date"].dt.month)
pv = m.pivot_table(index="year", columns="month", values="units", aggfunc="sum")
pv = pv.astype(float)
if len(pv) > 2:  # mask the partial first/last year
    pv.loc[[pv.index.min(), pv.index.max()]] = np.nan
fig = px.imshow(
    pv,
    color_continuous_scale="YlGnBu",
    aspect="auto",
    text_auto=".0f",
    labels=dict(x="Month", y="Year", color="Units"),
)
st.plotly_chart(
    style_fig(fig, f"{cat}: monthly units by year (partial first/last years masked)", 360), width="stretch"
)
st.caption(
    "Reading guide: weekday patterns drive short-term safety stock; any annual pattern is what the yearly "
    "terms in SARIMAX / Prophet capture. Compare the seasonal strength across categories in notebook 01."
)
