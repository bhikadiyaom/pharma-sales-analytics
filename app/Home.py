"""Overview: KPI cards, revenue trend, category share."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.analytics.descriptive import drop_partial_months, kpis  # noqa: E402
from src.viz.app_data import cfg, data_badge, fmt_money, page_setup, sidebar_filters  # noqa: E402
from src.viz.theme import CATEGORY_COLORS, style_fig  # noqa: E402

page_setup("Overview")
df, cats = sidebar_filters()

st.title("💊 Pharmaceutical Sales & Medicine Demand Analytics")
data_badge()
st.caption(
    "From historical sales to demand forecasts and inventory decisions. "
    "Use the sidebar filters; every page respects them."
)

k = kpis(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Total revenue",
    fmt_money(k["total_revenue"]),
    help="Units × ASSUMED unit price (config.yaml). The dataset records quantities only.",
)
c2.metric("Units sold", f"{k['total_units']:,.0f}", help="Sum of reported quantities in the selected period.")
c3.metric(
    "Avg units / day",
    f"{k['avg_daily_units']:,.1f}",
    help="Replaces 'average basket': the data has no transaction ids, so basket size cannot be computed.",
)
c4.metric(
    "MoM growth",
    "n/a" if pd.isna(k["mom_growth_pct"]) else f"{k['mom_growth_pct']:+.1f}%",
    help="Revenue of the last complete month vs the month before.",
)
c5.metric(
    "YoY growth",
    "n/a" if pd.isna(k["yoy_growth_pct"]) else f"{k['yoy_growth_pct']:+.1f}%",
    help="Revenue of the last complete month vs the same month a year earlier.",
)
if k.get("last_complete_month") is not None:
    st.markdown(
        f'<span class="small-note">Growth measured on {k["last_complete_month"]:%b %Y}, '
        f"the last complete month in the selected range.</span>",
        unsafe_allow_html=True,
    )

left, right = st.columns([2, 1])
with left:
    monthly = df.set_index("date").groupby("category")["revenue"].resample("MS").sum()
    monthly = drop_partial_months(monthly, df).reset_index()
    fig = px.area(
        monthly,
        x="date",
        y="revenue",
        color="category",
        color_discrete_map=CATEGORY_COLORS,
        labels={"revenue": f"Revenue ({cfg()['assumptions']['currency']})", "date": ""},
    )
    st.plotly_chart(
        style_fig(fig, "Monthly revenue by category (assumed prices, complete months)"), width="stretch"
    )
with right:
    share = df.groupby("category")["revenue"].sum().reset_index()
    fig = px.pie(
        share,
        names="category",
        values="revenue",
        hole=0.55,
        color="category",
        color_discrete_map=CATEGORY_COLORS,
    )
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(style_fig(fig, "Revenue share", 420).update_layout(showlegend=False), width="stretch")

a, b = st.columns(2)
with a:
    st.subheader("Top-selling categories")
    top = df.groupby("category")[["units", "revenue"]].sum().sort_values("units", ascending=False)
    top["share_%"] = top["units"] / top["units"].sum() * 100
    top.insert(0, "description", [cfg()["categories"].get(c, "") for c in top.index])
    st.dataframe(
        top.style.format({"units": "{:,.0f}", "revenue": "{:,.0f}", "share_%": "{:.1f}"}), width="stretch"
    )
with b:
    yearly = df.assign(year=df["date"].dt.year).groupby(["year", "category"])["units"].sum().reset_index()
    fig = px.bar(yearly, x="year", y="units", color="category", color_discrete_map=CATEGORY_COLORS)
    fig.update_xaxes(type="category")
    st.plotly_chart(
        style_fig(fig, "Units sold per year (2014 and 2019 are partial years)", 360), width="stretch"
    )

st.info(
    "**Where to go next →** Sales Explorer for drill-down · Forecasting for demand predictions · "
    "Inventory Planner for reorder points · Model Performance for the evidence behind the forecasts."
)
