"""Inventory Planner: safety stock, reorder point, EOQ, risk flags."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.inventory.planner import build_plan_table, default_inputs  # noqa: E402
from src.viz.app_data import cfg, data_badge, fmt_money, page_setup, result, sidebar_filters  # noqa: E402
from src.viz.theme import STATUS_COLORS, style_fig  # noqa: E402

page_setup("Inventory Planner", "📦")
_, cats = sidebar_filters()
st.title("📦 Inventory Planner")
data_badge()
st.caption(
    "Forecasts + forecast error + supplier lead time → safety stock, reorder point and EOQ. "
    "**Stock on hand, lead times, shelf life and costs are assumptions** (not in the dataset) - edit them below."
)

inv = cfg()["inventory"]
best = result("best_models.csv").set_index("category")
fb = result("forecasts_best.parquet")

c1, c2, c3 = st.columns(3)
sl = c1.radio(
    "Target service level",
    inv["service_levels"],
    index=inv["service_levels"].index(inv["default_service_level"]),
    format_func=lambda x: f"{x:.0%}",
    horizontal=True,
    help="Probability of not stocking out during a replenishment cycle. Higher → more safety stock.",
)
window = c2.slider(
    "Demand window for the mean forecast (days)",
    7,
    90,
    inv["planning_horizon_days"],
    help="Mean daily demand is the average of the best-model forecast over this window.",
)

cover = c3.slider(
    "Default stock on hand (days of cover)",
    1,
    120,
    inv["default_days_of_cover"],
    help="Stock on hand is not in the dataset. This sets the starting value in the table below "
    "(= days × forecast demand). Try 4-6 days to see stock-out alerts, 100+ to see overstock.",
)
defaults = default_inputs(cfg(), fb, cats, window, cover)
st.markdown("**Editable inputs** (double-click a cell)")
edited = st.data_editor(
    defaults,
    hide_index=True,
    width="stretch",
    disabled=["category"],
    column_config={
        "lead_time_days": st.column_config.NumberColumn(
            "Lead time (days)", min_value=0.0, max_value=60.0, help="Supplier lead time (assumption)."
        ),
        "on_hand": st.column_config.NumberColumn(
            "Stock on hand (units)",
            min_value=0.0,
            help="Current stock. Default = days-of-cover × forecast demand.",
        ),
        "shelf_life_days": st.column_config.NumberColumn(
            "Remaining shelf life (days)", min_value=0.0, help="Days until the stock on hand expires."
        ),
    },
    key=f"inv_editor_{window}_{cover}_{'_'.join(cats)}",
)

plan = build_plan_table(cfg(), fb, best, edited, sl, window)

k = st.columns(4)
k[0].metric("Reorder / stockout risk", int(plan["status"].isin(["REORDER NOW", "STOCKOUT RISK"]).sum()))
k[1].metric("Overstocked", int((plan["status"] == "OVERSTOCK").sum()))
k[2].metric("Units at expiry risk", f"{plan['expiry_risk_units'].sum():,.0f}")
k[3].metric("Expiry risk value (cost)", fmt_money(plan["expiry_risk_value"].sum()))

fig = go.Figure()
fig.add_bar(
    x=plan["category"],
    y=plan["on_hand"],
    name="On hand",
    marker_color=[STATUS_COLORS[s] for s in plan["status"]],
)
fig.add_scatter(
    x=plan["category"],
    y=plan["reorder_point"],
    mode="markers",
    name="Reorder point",
    marker=dict(symbol="line-ew", size=28, line=dict(width=3, color="#111")),
)
fig.add_scatter(
    x=plan["category"],
    y=plan["safety_stock"],
    mode="markers",
    name="Safety stock",
    marker=dict(symbol="diamond", size=10, color="#555"),
)
fig.add_scatter(
    x=plan["category"],
    y=plan["max_stock"],
    mode="markers",
    name="Max (ROP+EOQ)",
    marker=dict(symbol="triangle-up", size=10, color="#0F5C8C"),
)
st.plotly_chart(
    style_fig(fig, "Stock position vs reorder point (bar colour = status)", 420).update_layout(
        hovermode="closest"
    ),
    width="stretch",
)
st.caption(
    "🟢 OK · 🟠 REORDER NOW (below ROP) · 🔴 STOCKOUT RISK (cannot cover expected lead-time demand) · "
    "🟣 OVERSTOCK (above the cover limit in config)."
)

show = plan[
    [
        "category",
        "model",
        "mean_daily_forecast",
        "daily_error_rmse",
        "lead_time_days",
        "safety_stock",
        "reorder_point",
        "eoq",
        "on_hand",
        "days_of_cover",
        "status",
        "suggested_order_qty",
        "expiry_risk_units",
    ]
].round(1)
st.dataframe(show, width="stretch", hide_index=True)
reorder = plan[plan["suggested_order_qty"] > 0]
d1, d2 = st.columns(2)
d1.download_button(
    "⬇ Reorder list (CSV)",
    reorder.round(1).to_csv(index=False).encode(),
    file_name="reorder_list.csv",
    mime="text/csv",
    disabled=reorder.empty,
    help="Categories at/below the reorder point with the suggested order-up-to quantity.",
)
d2.download_button(
    "⬇ Full inventory plan (CSV)",
    plan.round(2).to_csv(index=False).encode(),
    file_name="inventory_plan.csv",
    mime="text/csv",
)

with st.expander("Formulas used"):
    st.latex(
        r"SS = z_{SL}\sqrt{L\,\sigma_d^2 + \bar d^{\,2}\sigma_L^2},\quad ROP = \bar d\,L + SS,\quad "
        r"EOQ=\sqrt{\tfrac{2DS}{H}}"
    )
    st.markdown(
        "σ_d = out-of-sample daily RMSE of the selected model (forecast error, not raw demand variance); "
        "L = lead time; σ_L = lead-time sd; D = annual demand; S = order cost; H = holding rate × unit cost. "
        "Expiry risk = max(0, stock − forecast demand until expiry)."
    )

st.subheader("Does forecast-driven replenishment beat the usual rule? (back-test)")
bt = result("inventory_backtest.csv")
bt = bt[(bt["service_level"] == sl) & bt["category"].isin(cats)]
agg = bt.groupby("policy").agg(
    lost_units=("lost_units", "sum"),
    demand=("demand_units", "sum"),
    stockout_days=("stockout_day_share", "mean"),
    avg_inventory_value=("avg_inventory_value", "sum"),
)
agg["fill_rate_%"] = (1 - agg["lost_units"] / agg["demand"]) * 100
agg["stockout_days_%"] = agg["stockout_days"] * 100
st.dataframe(
    agg[["fill_rate_%", "stockout_days_%", "lost_units", "avg_inventory_value"]]
    .round(2)
    .rename(index={"baseline": "28-day moving average", "forecast": "Model forecast"}),
    width="stretch",
)
st.caption(
    "Replay of the last 364 days (out-of-sample) with identical lead times and service-level targets; "
    "unmet demand is lost. See Model Performance for details."
)
