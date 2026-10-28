# Project Study Guide: everything you need for the viva

**Project:** Pharmaceutical Sales and Medicine Demand Analytics (B.Tech Data Science, Sem 7)
Read Parts 0 to 3 first (10 minutes), then use the rest as reference. Numbers in this guide are generated from the executed pipeline, so they match your dashboard and report.

---

## Part 0. The 60-second pitch (memorise this)

"Pharmacies lose money and endanger patients when medicines run out, and waste money when stock expires. I used the real Kaggle *Pharma Sales Data* (2,106 days of daily sales for 8 drug classes) to build a system that **describes** demand, **forecasts** it with six models compared by time-series cross-validation, and **turns the forecast and its error into inventory decisions**: safety stock, reorder point, order quantity and risk alerts. It is delivered as a tested pipeline and a 10-page Streamlit dashboard. The selected models cut forecast error by about 27.3 % (RMSE) versus the seasonal-naive baseline, and in a replay of the last 364 days forecast-driven ordering raised the fill rate from 96.77 % to 98.01 % at lower inventory value. Prices, costs and lead times are assumptions because the dataset only has quantities."

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
| `salesdaily.csv` | day | 2,106 | **all modelling** |
| `saleshourly.csv` | hour (open hours only) | 404,256 class-hour rows after reshaping | weekday × hour heatmaps |
| `salesweekly.csv` | week | 302 | not used (aggregate view) |
| `salesmonthly.csv` | month | 70 | reconciliation check of daily sums |

Period: **2014-01-02 to 2019-10-08** (2,106 days, no gaps). Total units: 127,567.

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
| category | demand_label | kmeans_cluster | hier_cluster | mean_daily | cv_weekly | zero_share | weekly_seasonality | annual_seasonality |
|---|---|---|---|---|---|---|---|---|
| M01AB | Fast-moving | 1 | 0 | 5.034 | 0.238 | 0.019 | 0.222 | 0.051 |
| M01AE | Slow-moving | 1 | 0 | 3.896 | 0.253 | 0.017 | 0.214 | 0.083 |
| N02BA | Slow-moving | 1 | 0 | 3.880 | 0.294 | 0.037 | 0.217 | 0.097 |
| N02BE | Seasonal | 1 | 0 | 29.917 | 0.364 | 0.012 | 0.232 | 0.292 |
| N05B | Fast-moving | 1 | 0 | 8.845 | 0.359 | 0.020 | 0.229 | 0.112 |
| N05C | Intermittent | 2 | 2 | 0.588 | 0.731 | 0.679 | 0.051 | 0.155 |
| R03 | Seasonal & erratic | 0 | 1 | 5.512 | 0.595 | 0.230 | 0.107 | 0.304 |
| R06 | Seasonal & erratic | 0 | 1 | 2.900 | 0.558 | 0.122 | 0.137 | 0.447 |

Read: `mean_daily` = average units/day; `cv_weekly` = variability of weekly totals; `zero_share` = fraction of days with no sale; the seasonality columns are strength indices (0 = none). N05C is **intermittent** (about 68 % zero days). R03 and R06 are **seasonal and erratic**. N02BE is the highest-volume class.

### 2.5 Data quality
No duplicates, no missing calendar days, no missing or negative values. 7 extreme values were capped. Daily sums match the monthly file for 90.2 % of 70 month-class pairs (within 1 %); the files do not explain the rest.

| category | zero-sales days % | outliers capped |
|---|---|---|
| M01AB | 1.9 | 0 |
| M01AE | 1.7 | 0 |
| N02BA | 3.7 | 0 |
| N02BE | 1.2 | 0 |
| N05B | 2.0 | 1 |
| N05C | 67.9 | 6 |
| R03 | 23.0 | 0 |
| R06 | 12.2 | 0 |

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
3. Final: trimmed SD floor. Only 7 points changed.

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
* **It won SARIMAX (weekly+Fourier): 7, XGBoost (lags): 1.**

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
Expanding training window; test = next 14 days; repeated **26** times, contiguous, so the last **364 days** are out-of-sample. Min training size 730 days. Each fold refits from scratch on data before the fold.
*Why not k-fold?* It shuffles time, so the model trains on the future (leakage).

### 7.2 Metrics (y = actual, f = forecast, n = number of points)
* **MAE** = mean(|y - f|). Units/day. Robust; optimal forecast is the *median*.
* **RMSE** = sqrt(mean((y - f)^2)). Penalises big errors; optimal forecast is the *mean*.
* **MAPE** = mean(|y - f| / y) x 100, computed only where y > 0. Breaks for small/zero counts.
* **sMAPE** = mean(2|y - f| / (|y| + |f|)) x 100 (0 to 200 %). Treats 0/0 as perfect.
* **MASE** = MAE / (in-sample MAE of seasonal naive). Below 1 = better than seasonal naive. Scale-free and zero-safe (Hyndman and Koehler 2006).

### 7.3 Why select by RMSE
MAE is minimised by the median; for intermittent series the median is 0, so MAE favours forecasting zero (a smoke run chose *naive* for N05C and R03). Inventory needs the **mean** demand, which RMSE rewards. So selection metric = RMSE.

### 7.4 Selection optimism (say this before they ask)
Selection and reporting use the same folds, so the winner's score is slightly optimistic. A nested scheme or a final untouched test period would fix it. No significance test (e.g. Diebold-Mariano) was run.

---

## Part 8. Results

### 8.1 Average over classes
| model | mae | rmse | smape | mase |
|---|---|---|---|---|
| Naive | 4.53 | 6.23 | 76.01 | 1.10 |
| Seasonal naive (7d) | 4.45 | 5.93 | 76.46 | 1.11 |
| SARIMAX (weekly+Fourier) | 3.32 | 4.40 | 66.41 | 0.82 |
| Prophet | 3.45 | 4.53 | 66.98 | 0.84 |
| XGBoost (lags) | 3.47 | 4.51 | 67.22 | 0.87 |
| LightGBM (lags) | 3.38 | 4.45 | 67.01 | 0.84 |

All four non-baseline families beat seasonal naive on RMSE in how many series:

| model | series where RMSE < seasonal-naive |
|---|---|
| LightGBM (lags) | 8 |
| Naive | 2 |
| Prophet | 8 |
| SARIMAX (weekly+Fourier) | 8 |
| XGBoost (lags) | 8 |

### 8.2 Selected model per class
| category | model | mae | rmse | smape | mape | mase | MAE gain % | RMSE gain % |
|---|---|---|---|---|---|---|---|---|
| M01AB | SARIMAX (weekly+Fourier) | 2.21 | 2.84 | 45.71 | 67.02 | 0.77 | 25.14 | 25.93 |
| M01AE | SARIMAX (weekly+Fourier) | 1.67 | 2.14 | 48.41 | 105.79 | 0.75 | 24.42 | 24.54 |
| N02BA | SARIMAX (weekly+Fourier) | 1.52 | 2.00 | 55.41 | 96.89 | 0.60 | 27.69 | 26.14 |
| N02BE | XGBoost (lags) | 9.38 | 12.23 | 33.69 | 35.84 | 0.80 | 20.33 | 22.19 |
| N05B | SARIMAX (weekly+Fourier) | 3.25 | 4.14 | 41.91 | 57.65 | 0.62 | 28.38 | 28.96 |
| N05C | SARIMAX (weekly+Fourier) | 0.85 | 1.14 | 158.00 | 59.92 | 1.04 | 23.06 | 31.47 |
| R03 | SARIMAX (weekly+Fourier) | 5.80 | 8.14 | 86.22 | 123.40 | 1.08 | 31.66 | 29.72 |
| R06 | SARIMAX (weekly+Fourier) | 1.72 | 2.29 | 61.95 | 59.27 | 0.85 | 28.90 | 29.41 |

Mean gain of selected model vs seasonal naive: MAE **26.2 %**, RMSE **27.3 %**. MAE gain range 20.3 % to 31.7 %.

### 8.3 How to read it honestly
* Top four models are within a few percent of each other; the *big* gain is over baselines. Do not claim SARIMAX is dramatically better than LightGBM.
* MAPE is high (75.7 % selected vs 98.4 % baseline) because daily counts are small and noisy (single digits for most classes).
* sMAPE gain is 16.6 % overall but 25.4 % excluding the intermittent N05C, where a mean forecast is penalised on zero-sales days.

### 8.4 Full per-series tables
**RMSE**

| category | Naive | Seasonal naive (7d) | SARIMAX (weekly+Fourier) | Prophet | XGBoost (lags) | LightGBM (lags) |
|---|---|---|---|---|---|---|
| M01AB | 3.85 | 3.83 | 2.84 | 2.84 | 2.97 | 2.95 |
| M01AE | 2.90 | 2.83 | 2.14 | 2.18 | 2.16 | 2.18 |
| N02BA | 2.97 | 2.71 | 2.00 | 2.03 | 2.20 | 2.16 |
| N02BE | 18.16 | 15.71 | 12.50 | 13.12 | 12.23 | 12.26 |
| N05B | 6.19 | 5.83 | 4.14 | 4.38 | 4.51 | 4.32 |
| N05C | 1.47 | 1.67 | 1.14 | 1.14 | 1.18 | 1.15 |
| R03 | 11.01 | 11.58 | 8.14 | 8.22 | 8.54 | 8.28 |
| R06 | 3.26 | 3.24 | 2.29 | 2.29 | 2.31 | 2.31 |

**MAE**

| category | Naive | Seasonal naive (7d) | SARIMAX (weekly+Fourier) | Prophet | XGBoost (lags) | LightGBM (lags) |
|---|---|---|---|---|---|---|
| M01AB | 3.02 | 2.96 | 2.21 | 2.21 | 2.31 | 2.29 |
| M01AE | 2.24 | 2.21 | 1.67 | 1.68 | 1.70 | 1.70 |
| N02BA | 2.28 | 2.11 | 1.52 | 1.53 | 1.74 | 1.72 |
| N02BE | 12.84 | 11.78 | 9.52 | 9.87 | 9.38 | 9.27 |
| N05B | 4.83 | 4.54 | 3.25 | 3.50 | 3.65 | 3.46 |
| N05C | 0.93 | 1.11 | 0.85 | 0.87 | 0.93 | 0.88 |
| R03 | 7.58 | 8.49 | 5.80 | 6.16 | 6.28 | 5.95 |
| R06 | 2.50 | 2.41 | 1.72 | 1.75 | 1.79 | 1.77 |

**MASE**

| category | Naive | Seasonal naive (7d) | SARIMAX (weekly+Fourier) | Prophet | XGBoost (lags) | LightGBM (lags) |
|---|---|---|---|---|---|---|
| M01AB | 1.06 | 1.03 | 0.77 | 0.77 | 0.81 | 0.80 |
| M01AE | 1.01 | 1.00 | 0.75 | 0.76 | 0.77 | 0.77 |
| N02BA | 0.90 | 0.83 | 0.60 | 0.61 | 0.69 | 0.68 |
| N02BE | 1.10 | 1.01 | 0.82 | 0.85 | 0.80 | 0.79 |
| N05B | 0.92 | 0.86 | 0.62 | 0.67 | 0.69 | 0.66 |
| N05C | 1.13 | 1.35 | 1.04 | 1.05 | 1.13 | 1.07 |
| R03 | 1.41 | 1.58 | 1.08 | 1.14 | 1.17 | 1.11 |
| R06 | 1.24 | 1.20 | 0.85 | 0.87 | 0.89 | 0.88 |

---

## Part 9. Prediction intervals and SHAP

### 9.1 Intervals
Model-agnostic and empirical. From 8 older forecast origins (up to 90 days ahead) take residuals divided by sqrt(forecast) (count data variance grows with the mean). Take the 5 % and 95 % quantiles per horizon bucket (1-14, 15-30, 31-60, 61-90 days) and apply to new forecasts, floored at 0. Nominal 90 %; checked by calibrating on older origins and scoring recent ones: mean coverage **91.0 %**, but it varies by series:

| category | model | nominal | empirical_coverage |
|---|---|---|---|
| M01AB | SARIMAX (weekly+Fourier) | 0.900 | 0.839 |
| M01AE | SARIMAX (weekly+Fourier) | 0.900 | 0.928 |
| N02BA | SARIMAX (weekly+Fourier) | 0.900 | 0.928 |
| N02BE | XGBoost (lags) | 0.900 | 0.958 |
| N05B | SARIMAX (weekly+Fourier) | 0.900 | 0.928 |
| N05C | SARIMAX (weekly+Fourier) | 0.900 | 0.881 |
| R03 | SARIMAX (weekly+Fourier) | 0.900 | 0.961 |
| R06 | SARIMAX (weekly+Fourier) | 0.900 | 0.861 |

Caveat: about 360 autocorrelated days per series, so these are approximate.

### 9.2 SHAP
SHAP (Shapley values) splits each prediction into contributions per feature: positive pushes demand up, negative down. Mean |SHAP| averaged over classes:

| feature | mean |SHAP| (units/day, avg over classes) |
|---|---|
| roll_mean_7 | 0.560 |
| roll_mean_14 | 0.552 |
| roll_mean_28 | 0.367 |
| weekday | 0.367 |
| doy_sin | 0.285 |
| doy_cos | 0.274 |
| roll_std_28 | 0.250 |
| roll_std_14 | 0.242 |
| roll_std_7 | 0.241 |
| lag_7 | 0.225 |

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
| category | revenue | revenue share % | abc | weekly CV | xyz | class |
|---|---|---|---|---|---|---|
| R03 | 1,625,235.21 | 31.53 | A | 0.59 | Z | AZ |
| N02BE | 1,134,097.25 | 22.00 | A | 0.36 | Y | AY |
| N05B | 1,117,678.68 | 21.68 | A | 0.36 | Y | AY |
| M01AE | 451,254.03 | 8.75 | A | 0.25 | X | AX |
| M01AB | 371,032.80 | 7.20 | B | 0.24 | X | BX |
| R06 | 183,234.52 | 3.55 | B | 0.56 | Z | BZ |
| N02BA | 179,788.60 | 3.49 | B | 0.29 | X | BX |
| N05C | 92,938.63 | 1.80 | C | 0.73 | Z | CZ |

| category | class | policy |
|---|---|---|
| R03 | AZ | High value, erratic: tight monitoring, higher safety stock, short order cycles, review forecasts often. |
| N02BE | AY | High value, variable: forecast-driven replenishment, moderate safety stock, weekly review. |
| N05B | AY | High value, variable: forecast-driven replenishment, moderate safety stock, weekly review. |
| M01AE | AX | High value, stable: continuous review, lean safety stock, automate replenishment (JIT). |
| M01AB | BX | Medium value, stable: periodic review with fixed order quantities (EOQ). |
| R06 | BZ | Medium value, erratic: buffer stock with caps, order on demand signals. |
| N02BA | BX | Medium value, stable: periodic review with fixed order quantities (EOQ). |
| N05C | CZ | Low value, erratic: stock on request or small buffer; consider delisting if slow. |

### 10.5 Clustering
Features per class (mean demand, CV, zero share, weekly and annual seasonality, weekend ratio, trend slope), log-scaled and standardised; KMeans with k chosen by **silhouette** (k = 3, silhouette 0.43); Ward hierarchical clustering for comparison; PCA for the 2-D map. With only 8 series it is exploratory, so an interpretable rule-based label is also given (Intermittent, Seasonal and erratic, Erratic, Seasonal, Fast/Slow-moving; fast/slow is relative to the portfolio median). The clusters broadly separate stable classes, seasonal-erratic R03/R06, and intermittent N05C.

---

## Part 11. Anomaly detection

1. Expected value = STL (period 7, robust) trend + seasonal.
2. Residual = actual - expected; Pearson-style residual = residual / sqrt(expected).
3. Robust z = (residual - median) / scale; scale = max(1.4826 x MAD, 0.5 x SD) (MAD collapses on zero-heavy series).
4. Flag |z| > 5. **Isolation Forest** (2 % contamination) is a second opinion; flagged by both = "confirmed".
5. Each flagged day gets a text explanation.

*History:* first version (MAD z > 3.5) flagged 5-12 % of days, too many because count residuals are heavy-tailed; fixed by the Pearson scaling and threshold 5. Result: 584 candidate days:

| category | drop | spike | share of days % |
|---|---|---|---|
| M01AB | 0 | 45 | 2.14 |
| M01AE | 1 | 53 | 2.56 |
| N02BA | 0 | 51 | 2.42 |
| N02BE | 28 | 63 | 4.32 |
| N05B | 2 | 67 | 3.28 |
| N05C | 0 | 120 | 5.70 |
| R03 | 0 | 96 | 4.56 |
| R06 | 0 | 58 | 2.75 |

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

| category | model | mean_daily_forecast | daily_error_rmse | lead_time_days | safety_stock | reorder_point | eoq |
|---|---|---|---|---|---|---|---|
| M01AB | sarimax | 5.0 | 2.8 | 5.0 | 13.3 | 38.2 | 615.4 |
| M01AE | sarimax | 3.8 | 2.1 | 6.0 | 10.6 | 33.4 | 426.7 |
| N02BA | sarimax | 3.0 | 2.0 | 4.0 | 8.3 | 20.5 | 608.7 |
| N02BE | xgboost | 35.8 | 12.2 | 3.0 | 68.4 | 175.6 | 2,332.1 |
| N05B | sarimax | 8.7 | 4.1 | 7.0 | 23.0 | 83.7 | 614.0 |
| N05C | sarimax | 0.6 | 1.1 | 7.0 | 5.1 | 9.5 | 148.4 |
| R03 | sarimax | 7.9 | 8.1 | 8.0 | 40.0 | 102.8 | 382.4 |
| R06 | sarimax | 1.8 | 2.3 | 4.0 | 8.1 | 15.1 | 401.5 |

**Why forecast error, not raw demand SD?** The predictable part (weekday pattern) needs no buffer; only the unpredictable part does. Better forecast = smaller sigma = less safety stock.

**Status flags:** STOCKOUT RISK if stock < d x L; REORDER NOW if below ROP; OVERSTOCK if more than 60 days of demand (or far above max); OK otherwise. **Expiry risk** = max(0, stock - forecast demand until expiry).

**What-if simulator:** demand = base x (1 + elasticity x price change) x (1 + promo uplift) x (1 + surge); lead time shifts; SS/ROP/EOQ and 30-day revenue recomputed. Elasticity default -0.6 is an **assumption**, so it shows sensitivity, not an estimate.

---

## Part 13. The inventory back-test (the "business impact" evidence)

Question: does forecast-driven ordering beat the usual rule?
* Replay the last 364 days day by day against **actual** demand.
* Both policies: periodic review every 14 days, order-up-to level S = demand over (L + 14 days) + z x sigma x sqrt(L + 14); orders arrive after lead time; unmet demand is lost.
* **Baseline:** trailing 28-day mean and SD. **Forecast policy:** model forecast, sigma from the model's *previous* out-of-sample errors (no look-ahead).

Results at 95% service level:

| Metric | Moving average | Model forecast |
|---|---|---|
| Fill rate | 96.77 % | 98.01 % |
| Stock-out days | 2.85 % | 2.34 % |
| Lost units | 741 | 455 |
| Avg inventory value (assumed cost) | 25,612 | 23,110 |

Stock-out days 18.1 % lower, lost units 38.5 % lower, inventory value -9.8 %. Better fill rate in 7 of 8 classes; worse in: R03.

| target SL | policy | fill-rate % | stock-out days % | lost units | avg inventory value (cost) |
|---|---|---|---|---|---|
| 90% | 28-day moving average | 95.99 | 3.50 | 917.76 | 23,571.39 |
| 90% | Model forecast | 97.14 | 3.37 | 654.82 | 21,145.96 |
| 95% | 28-day moving average | 96.77 | 2.85 | 740.63 | 25,611.72 |
| 95% | Model forecast | 98.01 | 2.34 | 455.48 | 23,109.93 |
| 99% | 28-day moving average | 97.84 | 1.85 | 495.15 | 29,588.49 |
| 99% | Model forecast | 98.95 | 1.03 | 239.46 | 27,310.56 |

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
| Data | 2,106 days, 8 classes, 2014-01-02 to 2019-10-08, 127,567 units |
| CV | 26 folds x 14 days; last 364 days out-of-sample; selection RMSE |
| Models | naive, seasonal naive, SARIMAX, Prophet, XGBoost, LightGBM |
| Winners | SARIMAX (weekly+Fourier): 7, XGBoost (lags): 1 |
| Gain vs seasonal naive | MAE 26.2 %, RMSE 27.3 % |
| Interval coverage | 91.0 % (nominal 90 %) |
| Clusters | k = 3, silhouette 0.43 |
| Anomalies | 584 candidate days |
| Back-test (95%) | fill 96.77 % to 98.01 %; lost units 38.5 % lower |
| Seed | 42 |

**Where to look during the viva:** report `docs/project_report.md`; 25 Q&A `docs/viva_notes.md`; numbers `reports/results/`; code entry point `src/pipeline.py`; dashboard `app/Home.py`; settings `config.yaml`.
