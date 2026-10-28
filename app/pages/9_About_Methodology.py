"""About / methodology / assumptions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import category_assumptions  # noqa: E402
from src.viz.app_data import cfg, data_badge, page_setup, result_json, sidebar_filters  # noqa: E402

page_setup("About", "ℹ️")
sidebar_filters()
st.title("ℹ️ About & Methodology")
data_badge()

st.markdown("""
**Pharmaceutical Sales and Medicine Demand Analytics** - B.Tech Data Science (Sem 7) individual project.

**Problem.** Pharmacies lose revenue and endanger patients through stock-outs, and lose money through expired overstock.
This system forecasts demand per therapeutic class and converts the forecasts into stocking decisions.

**Pipeline.** ingest & validate → clean (duplicates, gaps, outliers) → features (calendar, lags 1/7/14/28, rolling stats)
→ 6 forecasters evaluated by rolling-origin CV → automatic best-model selection → empirical prediction intervals
→ inventory formulas (safety stock, ROP, EOQ) → risk flags, what-if and back-test.

**Models.** Naive · Seasonal naive (7 d) · SARIMAX (weekly seasonal + annual Fourier terms) · Prophet ·
XGBoost and LightGBM on lag/rolling/calendar features (recursive multi-step). SHAP explains the tree model.
""")

st.subheader("Data")
dq = result_json("data_quality_report.json")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
* **Source:** Kaggle *Pharma Sales Data* - quantities sold per ATC class (8 classes) at daily/hourly/weekly/monthly level.
* **Range:** {dq['date_range'][0]} → {dq['date_range'][1]} (daily rows: {dq['rows_loaded']:,}; hourly rows: {dq['hourly_rows']:,}).
* **Not in the data:** prices, costs, stock levels, expiry dates, branches, promotions, transactions.
* **Synthetic?** `{cfg()['data']['is_synthetic']}` - all sales figures are the real file; business parameters are assumptions.
""")
with c2:
    st.json(
        {
            k: dq[k]
            for k in [
                "exact_duplicate_rows_removed",
                "calendar_days_missing_before_fill",
                "missing_values_imputed",
                "outliers_capped_by_category",
                "zero_sales_day_share",
                "monthly_reconciliation",
            ]
            if k in dq
        },
        expanded=False,
    )

st.subheader("Assumptions (editable in config.yaml)")
rows = []
for c, desc in cfg()["categories"].items():
    a = category_assumptions(cfg(), c)
    rows.append({"category": c, "description": desc, **a})
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
st.warning(
    "Revenue, EOQ, holding cost and expiry value depend on these **assumed** parameters; unit forecasts, "
    "forecast errors, ABC-XYZ variability (XYZ) and anomalies depend only on the real sales data."
)

st.subheader("Limitations")
st.markdown("""
* Only 8 aggregated classes, not individual medicines/brands; clustering of 8 series is exploratory.
* One sales point; branch-level effects cannot be studied.
* No price/promotion/weather/epidemic covariates - the what-if elasticity is assumed, not estimated.
* Model selection and reporting use the same CV folds (mild optimism); long-horizon intervals rely on 8 calibration origins.
* Stock on hand and shelf life are user inputs; the back-test assumes lost sales (no backorders) and a fixed review period.
""")
