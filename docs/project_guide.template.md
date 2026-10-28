# Project Study Guide: everything you need for the viva

**Project:** Pharmaceutical Sales and Medicine Demand Analytics (B.Tech Data Science, Sem 7)
Read Parts 0 to 3 first (10 minutes), then use the rest as reference. Numbers in this guide are generated from the executed pipeline, so they match your dashboard and report.

---

## Part 0. The 60-second pitch (memorise this)

"Pharmacies lose money and endanger patients when medicines run out, and waste money when stock expires. I used the real Kaggle *Pharma Sales Data* ({{n_days}} days of daily sales for {{n_cats}} drug classes) to build a system that **describes** demand, **forecasts** it with six models compared by time-series cross-validation, and **turns the forecast and its error into inventory decisions**: safety stock, reorder point, order quantity and risk alerts. It is delivered as a tested pipeline and a 10-page Streamlit dashboard. The selected models cut forecast error by about {{mean_rmse_improvement_pct}} % (RMSE) versus the seasonal-naive baseline, and in a replay of the last {{oos_days}} days forecast-driven ordering raised the fill rate from {{fill_baseline}} % to {{fill_forecast}} % at lower inventory value. Prices, costs and lead times are assumptions because the dataset only has quantities."

**Three things to always say honestly:** (1) data is real, not synthetic; (2) business parameters (price, cost, lead time, stock, shelf life) are assumptions; (3) results come from one pharmacy and one out-of-sample year.

---

## Part 1. The problem in plain words

* **Stock-out** = demand arrives, no stock. Lost sale + patient not treated.
* **Overstock** = too much stock. Capital locked, and medicines **expire** (waste).
* **Goal:** keep just enough stock. Needs (a) a good forecast of demand and (b) knowledge of how wrong the forecast tends to be, because the "extra" buffer (safety stock) covers that error.
* **Why data science:** demand has patterns (weekday, season) that a simple "average of last 4 weeks" reacts to slowly.

Three analytics levels (say these words):
1. **Descriptive** - what happened (KPIs, trends, seasonality, ABC-XYZ).
2. **Predictive** - what will happen (forecasting, intervals, SHAP).
3. **Prescriptive** - what should we do (reorder point, safety stock, what-if, alerts).

---

## Part 2. The dataset

### 2.1 What it is
Kaggle **"Pharma Sales Data"** (real). Four files in `data/raw/` giving the *same* sales at different resolutions:

| File | Grain | Rows | Used for |
|---|---|---|---|
| `salesdaily.csv` | day | {{n_days}} | **all modelling** |
| `saleshourly.csv` | hour (open hours only) | {{n_hourly}} class-hour rows after reshaping | weekday × hour heatmaps |
| `salesweekly.csv` | week | 302 | not used (aggregate view) |
| `salesmonthly.csv` | month | 70 | reconciliation check of daily sums |

Period: **{{date_start}} to {{date_end}}** ({{n_days}} days, no gaps). Total units: {{total_units}}.

### 2.2 Columns
`datum` (date) + 8 numeric columns, one per **ATC class**. ATC = *Anatomical Therapeutic Chemical* classification (WHO), a code system for drug groups. Value = quantity sold (can be fractional, e.g. 3.67, so call it "units"). The files also contain `Year`, `Month`, `Hour`, `Weekday Name` helper columns that I re-derived and did not use.

| Code | Group | Typical examples (general knowledge, not from the file) |
|---|---|---|
| M01AB | Anti-inflammatory/antirheumatic, acetic acid derivatives | diclofenac-type |
| M01AE | Anti-inflammatory/antirheumatic, propionic acid derivatives | ibuprofen/naproxen-type |
| N02BA | Analgesics/antipyretics, salicylic acid derivatives | aspirin-type |
| N02BE | Analgesics/antipyretics, anilides | paracetamol-type |
| N05B | Psycholeptics, anxiolytics | anti-anxiety drugs |
| N05C | Psycholeptics, hypnotics and sedatives | sleep aids |
| R03 | Drugs for obstructive airway diseases | asthma/COPD drugs |
| R06 | Antihistamines for systemic use | allergy drugs |

### 2.3 What the data does NOT have (very likely viva question)
No prices, costs, stock levels, expiry dates, branches, promotions, supplier lead times, or transaction ids. The files also do not say which pharmacy or country. Therefore:
* **Revenue** = units × *assumed* price. ABC ranking, revenue KPIs, EOQ and expiry value depend on assumptions in `config.yaml`.
* **Stock on hand** is a user input in the Inventory Planner (default = days of cover × forecast).
* **"Average basket"** cannot be computed (no transactions) so I show average units per day.
* **Branch filter** is hidden because there is a single sales point.
* **Holidays:** no national calendar assumed; only fixed festive dates (1 Jan, 24-26 Dec, 31 Dec).
* **Parts that stay valid regardless of assumptions:** forecasts, forecast errors, XYZ variability, clusters, anomalies, seasonality.

### 2.4 Behaviour of each class (from the data)
{{cluster_table}}

Read: `mean_daily` = average units/day; `cv_weekly` = variability of weekly totals; `zero_share` = fraction of days with no sale; the seasonality columns are strength indices (0 = none). N05C is **intermittent** (about 68 % zero days). R03 and R06 are **seasonal and erratic**. N02BE is the highest-volume class.

### 2.5 Data quality
No duplicates, no missing calendar days, no missing or negative values. {{outliers_total}} extreme values were capped. Daily sums match the monthly file for {{recon_within1}} % of {{recon_months}} month-class pairs (within 1 %); the files do not explain the rest.

{{dq_table}}

---

## Part 3. Pipeline at a glance

```
raw CSV -> ingest/validate/clean -> features -> 6 models x rolling CV -> select best per series
        -> intervals + SHAP -> ABC-XYZ, clusters, anomalies -> inventory formulas
        -> what-if + back-test -> Streamlit dashboard + report
```

| Command | What it does |
|---|---|
| `python -m src.pipeline data` | clean data, write Parquet, SQLite (`pharma.db`), data-quality JSON |
| `python -m src.pipeline train` | cross-validation, metrics, selection, 90-day forecasts, intervals, SHAP (about 7 min, CPU) |
| `python -m src.pipeline analyze` | ABC-XYZ, clusters, anomalies, inventory back-test, headline numbers |
| `python -m streamlit run app/Home.py` | dashboard |
| `python -m pytest -q` | tests |
| `python -m src.reporting.build_report` | regenerate report, viva notes and this guide from results |

### File map
| Folder/file | Purpose |
|---|---|
| `config.yaml` | every setting: data mapping, assumptions, models, CV, inventory, thresholds, seed |
| `src/data/ingest.py` | load, validate, dedupe, calendar, impute, outliers, quality report |
| `src/features/engineering.py` | calendar, holiday, lag, rolling features |
| `src/models/forecasters.py` | the 6 models |
| `src/models/cv.py` | rolling-origin CV, selection, prediction intervals |
| `src/models/metrics.py` | MAE, RMSE, MAPE, sMAPE, MASE |
| `src/models/explain.py` | SHAP |
| `src/analytics/` | `descriptive.py` (KPIs, STL, ABC-XYZ), `clustering.py`, `anomaly.py` |
| `src/inventory/` | `policy.py` (formulas, flags), `planner.py`, `whatif.py`, `simulate.py` (back-test) |
| `app/` | `Home.py` + 9 pages |
| `tests/` | pytest suite |
| `reports/results/` | every result file; `reports/figures/` plots |

---

## Part 4. Data cleaning (what, why, and the mistakes I fixed)

Steps: schema validation, duplicate removal, calendar completion, imputation (linear interpolation), outlier capping, reconciliation with monthly file, everything logged to JSON.

**Outlier rule:** robust z-score against a centred 29-day rolling median; scale = max(1.4826 × rolling MAD, 99 %-trimmed global standard deviation); cap only if |z| > 6.

*Why this form (good viva story):*
1. Version 1 used only local MAD. On intermittent series (many zeros) MAD is about 0, so ordinary peaks looked extreme: 15 genuine R03 peaks were capped (31 units cut to 14). Wrong.
2. Version 2 floored the scale with the raw global SD, but a huge outlier inflates the SD itself (a 500 spike was cut only to about 190). A unit test caught it.
3. Final: trimmed SD floor. Only {{outliers_total}} points changed.

*Why winsorise (cap) instead of delete?* Deleting would break the daily calendar that lags and seasonality need.

---

## Part 5. Feature engineering

| Group | Features | Why |
|---|---|---|
| Calendar | weekday, is_weekend, month, quarter, day_of_month, week_of_year, is_month_end | weekly/annual patterns |
| Cyclic | doy_sin, doy_cos | day 365 is next to day 1 |
| Holiday | is_holiday, near_holiday (fixed dates +/- 1 day) | festive effects |
| Lags | lag_1, 7, 14, 28 | yesterday, last week, 2 and 4 weeks ago |
| Rolling | mean and std over previous 7, 14, 28 days | recent level and volatility |
| Price/promo | built but inactive | dataset lacks them |

**Leakage rule:** rolling features are computed on `y.shift(1)`, so day t never sees day t. A unit test changes future values and checks past features stay identical.

*Leakage = information from the future entering training; it makes scores look great and then fails in real use.*

---

## Part 6. The six models (know each one)

**General note:** all models are refit on the training part of every fold; each predicts the next 14 days (and 90 days for final forecasts).

### 6.1 Naive
Forecast = last observed value. Baseline; useless for seasonal data but a sanity check.

### 6.2 Seasonal naive (period 7) - *the baseline for all "gain" numbers*
Forecast = value from the same weekday last week. Strong, simple benchmark. If a complex model cannot beat it, it is not worth using.

### 6.3 SARIMAX
ARIMA family. Notation (p,d,q)(P,D,Q)s:
* AR(p): depends on past values; MA(q): depends on past errors; d: differencing for trend.
* Seasonal part with period s = 7 (weekly): seasonal differencing D=1.
* Mine: **(1,0,1)(1,1,1)7 + exogenous annual Fourier terms (3 sine/cosine pairs)**. Plain seasonal ARIMA with period 365 is impractical, so yearly seasonality enters through Fourier regressors (the "X"). Fitted on the trailing 3 years.
* Strengths: explicit weekly + annual seasonality, little data needed, robust. Weakness: linear, one series at a time, slow-ish (about 8 s per fit).
* **It won {{win_counts}}.**

### 6.4 Prophet (Meta)
Additive model: trend + weekly seasonality + yearly seasonality + holidays + noise. Fits by optimisation. Strengths: easy, handles calendar effects. Weakness: less precise for short-term autocorrelation. Here: weekly + yearly on, festive dates as holidays.

### 6.5 XGBoost (and 6.6 LightGBM)
Gradient-boosted decision trees: many small trees added one by one, each correcting the previous errors. Input = the features of Part 5. 300 trees, depth 4, learning rate 0.05, row/column subsampling 0.8.
* **Recursive multi-step:** predict tomorrow, append it to history, build tomorrow+1 features, repeat. (Alternative: direct models per horizon.) Errors can accumulate.
* Strengths: nonlinear, many features, SHAP-explainable. Weakness: cannot extrapolate trend, needs enough data, recursive error build-up.
* LightGBM: similar, leaf-wise growth, faster. XGBoost won N02BE (highest volume class).

### 6.7 Why not LSTM / deep learning?
8 series x about 2,100 noisy small-count points is little data; deep models need much more, are harder to explain and tune, often do not beat tuned statistical/tree models at this scale (see M5 literature), and I use a CPU-only laptop. Listed as future work.

### 6.8 Why these families?
One baseline family, one classical statistical, one decomposition-based, two tree-based. Fair comparison across paradigms, same folds.

---

## Part 7. Evaluation: cross-validation and metrics

### 7.1 Rolling-origin cross-validation
Expanding training window; test = next {{cv_horizon}} days; repeated **{{cv_folds}}** times, contiguous, so the last **{{oos_days}} days** are out-of-sample. Min training size 730 days. Each fold refits from scratch on data before the fold.
*Why not k-fold?* It shuffles time, so the model trains on the future (leakage).

### 7.2 Metrics (y = actual, f = forecast, n = number of points)
* **MAE** = mean(|y - f|). Units/day. Robust; optimal forecast is the *median*.
* **RMSE** = sqrt(mean((y - f)^2)). Penalises big errors; optimal forecast is the *mean*.
* **MAPE** = mean(|y - f| / y) x 100, computed only where y > 0. Breaks for small/zero counts.
* **sMAPE** = mean(2|y - f| / (|y| + |f|)) x 100 (0 to 200 %). Treats 0/0 as perfect.
* **MASE** = MAE / (in-sample MAE of seasonal naive). Below 1 = better than seasonal naive. Scale-free and zero-safe (Hyndman and Koehler 2006).

### 7.3 Why select by RMSE
MAE is minimised by the median; for intermittent series the median is 0, so MAE favours forecasting zero (a smoke run chose *naive* for N05C and R03). Inventory needs the **mean** demand, which RMSE rewards. So selection metric = {{selection_metric}}.

### 7.4 Selection optimism (say this before they ask)
Selection and reporting use the same folds, so the winner's score is slightly optimistic. A nested scheme or a final untouched test period would fix it. No significance test (e.g. Diebold-Mariano) was run.

---

## Part 8. Results

### 8.1 Average over classes
{{avg_by_model_table}}

All four non-baseline families beat seasonal naive on RMSE in how many series:

{{beat_baseline_table}}

### 8.2 Selected model per class
{{best_table}}

Mean gain of selected model vs seasonal naive: MAE **{{mean_mae_improvement_pct}} %**, RMSE **{{mean_rmse_improvement_pct}} %**. MAE gain range {{mae_gain_min}} % to {{mae_gain_max}} %.

### 8.3 How to read it honestly
* Top four models are within a few percent of each other; the *big* gain is over baselines. Do not claim SARIMAX is dramatically better than LightGBM.
* MAPE is high ({{mean_best_mape}} % selected vs {{mean_baseline_mape}} % baseline) because daily counts are small and noisy (single digits for most classes).
* sMAPE gain is {{mean_smape_improvement_pct}} % overall but {{smape_gain_regular}} % excluding the intermittent N05C, where a mean forecast is penalised on zero-sales days.

### 8.4 Full per-series tables
**RMSE**

{{rmse_table}}

**MAE**

{{mae_table}}

**MASE**

{{mase_table}}

---

## Part 9. Prediction intervals and SHAP

### 9.1 Intervals
Model-agnostic and empirical. From 8 older forecast origins (up to 90 days ahead) take residuals divided by sqrt(forecast) (count data variance grows with the mean). Take the 5 % and 95 % quantiles per horizon bucket (1-14, 15-30, 31-60, 61-90 days) and apply to new forecasts, floored at 0. Nominal {{coverage_nominal}} %; checked by calibrating on older origins and scoring recent ones: mean coverage **{{coverage_mean}} %**, but it varies by series:

{{coverage_table}}

Caveat: about 360 autocorrelated days per series, so these are approximate.

### 9.2 SHAP
SHAP (Shapley values) splits each prediction into contributions per feature: positive pushes demand up, negative down. Mean |SHAP| averaged over classes:

{{shap_table}}

Meaning: recent level (rolling means), weekday and annual cycle drive predictions. SHAP explains the *model*, not real-world causes.

---

## Part 10. Descriptive analytics

### 10.1 STL decomposition
STL = Seasonal-Trend decomposition using LOESS. Splits a series into trend + seasonal + remainder. Used with period 7 (daily data) or 52 (weekly totals). Seasonal strength = max(0, 1 - Var(remainder)/Var(seasonal + remainder)).

### 10.2 ABC analysis (Pareto)
Rank classes by revenue; cumulative share; A = first 80 %, B = next 15 %, C = rest. Rule used: class decided by the cumulative share *before* adding the item. Uses **assumed prices**.

### 10.3 XYZ analysis
Coefficient of variation (SD/mean) of **weekly** demand: X <= 0.30 stable, Y <= 0.45 variable, Z above = erratic. (Daily CV is mostly noise, so weekly is used.) Real data only.

### 10.4 ABC-XYZ result and stocking policy
{{abc_table}}

{{abc_policy_table}}

### 10.5 Clustering
Features per class (mean demand, CV, zero share, weekly and annual seasonality, weekend ratio, trend slope), log-scaled and standardised; KMeans with k chosen by **silhouette** (k = {{cluster_k}}, silhouette {{cluster_sil}}); Ward hierarchical clustering for comparison; PCA for the 2-D map. With only 8 series it is exploratory, so an interpretable rule-based label is also given (Intermittent, Seasonal and erratic, Erratic, Seasonal, Fast/Slow-moving; fast/slow is relative to the portfolio median). The clusters broadly separate stable classes, seasonal-erratic R03/R06, and intermittent N05C.

---

## Part 11. Anomaly detection

1. Expected value = STL (period 7, robust) trend + seasonal.
2. Residual = actual - expected; Pearson-style residual = residual / sqrt(expected).
3. Robust z = (residual - median) / scale; scale = max(1.4826 x MAD, 0.5 x SD) (MAD collapses on zero-heavy series).
4. Flag |z| > 5. **Isolation Forest** (2 % contamination) is a second opinion; flagged by both = "confirmed".
5. Each flagged day gets a text explanation.

*History:* first version (MAD z > 3.5) flagged 5-12 % of days, too many because count residuals are heavy-tailed; fixed by the Pearson scaling and threshold 5. Result: {{anomaly_total}} candidate days:

{{anomaly_table}}

Almost all are spikes (counts cannot go below zero). No event log exists, so causes (outbreak, promotion, stock-out, data error) are hypotheses.

---

## Part 12. Inventory maths (with a worked example)

Symbols: d = mean forecast demand per day; sigma = daily forecast RMSE; L = lead time (days); sigma_L = lead-time SD; z = normal quantile of service level (90 % = 1.28, 95 % = 1.645, 99 % = 2.33); D = annual demand; S = order cost; H = holding cost per unit-year = holding rate x unit cost.

| Quantity | Formula | Meaning |
|---|---|---|
| Safety stock | z x sqrt(L x sigma^2 + d^2 x sigma_L^2) | buffer against forecast error + supplier variability |
| Reorder point | d x L + SS | order when stock falls to this |
| EOQ | sqrt(2 D S / H) | order size balancing ordering vs holding cost |
| Max stock | ROP + EOQ | upper level |

**Service level** = probability of not stocking out in a replenishment cycle.

**Worked example (R03, 95 %, assumed inputs):** d about 7.9/day, sigma about 8.1, L = 8, sigma_L = 1, z = 1.645, S = 500, unit cost 98, holding 20 %/yr.
* SS = 1.645 x sqrt(8 x 8.1^2 + 7.9^2 x 1^2) = 1.645 x sqrt(525 + 62) = about 40
* ROP = 7.9 x 8 + 40 = about 103
* D = 7.9 x 365 = about 2,884; H = 0.2 x 98 = 19.6; EOQ = sqrt(2 x 2884 x 500 / 19.6) = about 384 (table below: 382, small rounding)

Default plan (all classes, 95 %):

{{plan_table}}

**Why forecast error, not raw demand SD?** The predictable part (weekday pattern) needs no buffer; only the unpredictable part does. Better forecast = smaller sigma = less safety stock.

**Status flags:** STOCKOUT RISK if stock < d x L; REORDER NOW if below ROP; OVERSTOCK if more than 60 days of demand (or far above max); OK otherwise. **Expiry risk** = max(0, stock - forecast demand until expiry).

**What-if simulator:** demand = base x (1 + elasticity x price change) x (1 + promo uplift) x (1 + surge); lead time shifts; SS/ROP/EOQ and 30-day revenue recomputed. Elasticity default -0.6 is an **assumption**, so it shows sensitivity, not an estimate.

---

## Part 13. The inventory back-test (the "business impact" evidence)

Question: does forecast-driven ordering beat the usual rule?
* Replay the last {{oos_days}} days day by day against **actual** demand.
* Both policies: periodic review every 14 days, order-up-to level S = demand over (L + 14 days) + z x sigma x sqrt(L + 14); orders arrive after lead time; unmet demand is lost.
* **Baseline:** trailing 28-day mean and SD. **Forecast policy:** model forecast, sigma from the model's *previous* out-of-sample errors (no look-ahead).

Results at {{bt_sl}} service level:

| Metric | Moving average | Model forecast |
|---|---|---|
| Fill rate | {{fill_baseline}} % | {{fill_forecast}} % |
| Stock-out days | {{stockout_base}} % | {{stockout_fc}} % |
| Lost units | {{lost_base}} | {{lost_fc}} |
| Avg inventory value (assumed cost) | {{inv_base}} | {{inv_fc}} |

Stock-out days {{stockout_day_reduction_pct}} % lower, lost units {{lost_units_reduction_pct}} % lower, inventory value {{inventory_value_change_pct}} %. Better fill rate in {{bt_better_n}} of {{bt_n_cats}} classes; worse in: {{bt_worse_list}}.

{{backtest_table}}

**Limits:** one year, one pharmacy, assumed lead times/costs/review period, lost-sales assumption, no confidence intervals. "Overstock reduction" is proxied by lower inventory value (no expiry data).

---

## Part 14. The dashboard (10 pages)

| Page | What it shows | Key interaction |
|---|---|---|
| Overview (Home) | KPI cards (revenue, units, avg/day, MoM, YoY), monthly revenue, share, top sellers | global filters |
| Sales Explorer | trends, correlation heatmap, weekday box plots, table | metric, aggregation, smoothing, CSV download |
| Seasonality | STL, weekday x month and weekday x hour heatmaps | weekly vs annual view |
| ABC-XYZ | Pareto, 3x3 matrix, policies, clusters | recalculates on filters |
| Forecasting | forecast with interval, SHAP | horizon 7/14/30/90, model choice, CSV download |
| Inventory Planner | SS, ROP, EOQ, flags, reorder list, back-test | service level 90/95/99, editable stock/lead time/shelf life |
| What-If Simulator | price, promo, surge, lead-time effects | sliders, presets (epidemic +30 %) |
| Anomalies | flagged days + explanations | z threshold slider |
| Model Performance | model comparison, best model, error by horizon, back-test | metric selector |
| About | method, assumptions, limitations | - |

Engineering: `st.cache_data` for data loading, shared sidebar filters kept across pages, consistent colour theme, tooltips, download buttons.

---

## Part 15. Testing and reproducibility
* pytest suite: cleaning (duplicates, gaps, outlier behaviour), features (calendar, lags, **no-leakage**), metrics (known values), CV splits (no overlap, train before test), inventory formulas, planner flags, anomaly detector, and a **headless render test of every dashboard page**.
* ruff and black for style. Seeds fixed (42). `config.yaml` controls everything.
* Report numbers are generated from result files by a script, not typed.

---

## Part 16. Decisions and corrections (shows you understand the work)
| Issue found | Fix | Lesson |
|---|---|---|
| Outlier rule capped real R03 peaks | trimmed-SD floor, z > 6, tests | robust scale collapses on zero-heavy data |
| MAE chose naive for intermittent series | select by RMSE | metric must match the decision (mean for stocking) |
| Anomaly rate 5-12 % | Pearson scaling, z > 5, IF as second opinion | count residuals are heavy-tailed |
| sMAPE bad for N05C | report MAE/RMSE/MASE alongside; explain | percentage errors fail near zero |
| Chart/KPI glitches (partial last month) | drop partial months | partial periods distort trends |
| Synthetic generator requested | not used; real data supplied | state clearly |

---

## Part 17. Limitations and future work
**Limitations:** 8 aggregated classes, one sales point; assumed business parameters; no price/promo/weather data; selection optimism and no significance tests; approximate intervals; one-year back-test; heuristic anomaly thresholds; data origin unspecified.
**Future:** medicine/brand/branch hierarchy; Croston/TSB or negative-binomial models for intermittent demand; exogenous drivers; Diebold-Mariano tests and nested selection; batch/expiry-aware simulation; deep global models when many series exist; live data feed.

---

## Part 18. Glossary
* **ATC** - WHO drug classification code. **SKU** - single stock item (not used here; classes are the unit).
* **Lead time** - delay between ordering and receiving stock. **Safety stock** - buffer for uncertainty.
* **Reorder point (ROP)** - stock level that triggers an order. **EOQ** - cost-minimising order size.
* **Service level / fill rate** - service level is the chance of no stock-out per cycle; fill rate is the share of demand actually served.
* **Intermittent demand** - many zero days. **Stationary** - statistical properties constant over time.
* **Differencing** - subtracting previous values to remove trend/seasonality. **Fourier terms** - sine/cosine pairs representing a smooth seasonal pattern.
* **Rolling-origin CV** - time-ordered cross-validation. **Leakage** - future information used in training.
* **Winsorise** - cap extreme values instead of deleting. **MAD** - median absolute deviation (robust spread).
* **Silhouette** - clustering quality score (-1 to 1). **Isolation Forest** - anomaly detector isolating odd points with random splits.
* **Elasticity** - % change in demand for a 1 % price change.
* **Recursive forecasting** - feeding predictions back as inputs for later steps.

---

## Part 19. Tough questions and safe answers

**"Is your data synthetic?"** No, real Kaggle data; only prices/costs/lead times are assumed.
**"Why is MAPE so high?"** Small noisy daily counts; relative gain and MASE matter; error feeds safety stock.
**"Which model is best?"** SARIMAX in most classes, XGBoost in N02BE; but the top four are close; the clear result is that seasonality-aware models beat baselines everywhere.
**"Is the improvement statistically significant?"** I did not run a formal test; gain over baseline is consistent across all 8 series; differences among top models are small.
**"How do you know there is no leakage?"** Rolling-origin refits, shift(1) rolling features, a leakage unit test, intervals calibrated on older origins, back-test sigma from earlier errors only.
**"Why 14-day horizon in CV but 90-day forecasts?"** 14 days matches the replenishment horizon for model selection; 90 days is for planning, with intervals calibrated up to 90 days separately.
**"Can this be used in a real pharmacy?"** As decision support, after replacing assumptions with real prices, costs, lead times, and stock and validating on their data.
**"What if demand suddenly jumps (epidemic)?"** Models do not know; the anomaly page flags it and the what-if simulator shows needed buffer; future work: external signals.
**"Why is R03 worse in the back-test?"** Erratic and seasonal with a large forecast error; one year of evidence; I report it rather than hide it.
**"What did you personally decide vs library?"** Data rules, feature design, CV protocol, selection metric, interval method, anomaly scoring, inventory formulas, back-test design, dashboard.
**"Where did the prices come from?"** Illustrative assumptions in `config.yaml`, labelled everywhere; change them and everything recomputes.
**"Why SARIMAX over plain SARIMA?"** The X adds annual Fourier regressors since period 365 is impractical for seasonal ARIMA.
**"Explain overfitting here."** Model fits noise instead of pattern; guarded by out-of-sample CV, shallow trees (depth 4), regularised subsampling.

---

## Part 20. Cheat-sheet numbers
| Fact | Value |
|---|---|
| Data | {{n_days}} days, {{n_cats}} classes, {{date_start}} to {{date_end}}, {{total_units}} units |
| CV | {{cv_folds}} folds x {{cv_horizon}} days; last {{oos_days}} days out-of-sample; selection {{selection_metric}} |
| Models | naive, seasonal naive, SARIMAX, Prophet, XGBoost, LightGBM |
| Winners | {{win_counts}} |
| Gain vs seasonal naive | MAE {{mean_mae_improvement_pct}} %, RMSE {{mean_rmse_improvement_pct}} % |
| Interval coverage | {{coverage_mean}} % (nominal {{coverage_nominal}} %) |
| Clusters | k = {{cluster_k}}, silhouette {{cluster_sil}} |
| Anomalies | {{anomaly_total}} candidate days |
| Back-test ({{bt_sl}}) | fill {{fill_baseline}} % to {{fill_forecast}} %; lost units {{lost_units_reduction_pct}} % lower |
| Seed | 42 |

**Where to look during the viva:** report `docs/project_report.md`; 25 Q&A `docs/viva_notes.md`; numbers `reports/results/`; code entry point `src/pipeline.py`; dashboard `app/Home.py`; settings `config.yaml`.
