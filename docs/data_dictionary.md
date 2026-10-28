# Data Dictionary

**Dataset:** Kaggle "Pharma Sales Data" (real data, *not* synthetic). Files live in `data/raw/`.
Nothing in this project was simulated: the synthetic generator described in the brief was not needed.

The raw files carry only **quantities sold per therapeutic (ATC) class**. They contain no prices, costs,
stock levels, expiry dates, branches, promotions or transaction ids. Any value derived from such fields
(revenue, EOQ, holding cost, expiry risk) uses **assumptions declared in `config.yaml → assumptions`**
and is labelled as such in the dashboard and report.

## 1. Raw files (`data/raw/`)

| File | Grain | Rows | Notes |
|---|---|---|---|
| `salesdaily.csv` | day | 2,106 | **Primary modelling table**, 2014-01-02 → 2019-10-08, no calendar gaps |
| `saleshourly.csv` | hour (open hours only) | 50,532 | Used for weekday × hour heatmaps |
| `salesweekly.csv` | week | 302 | Not used for modelling (aggregated view of the same sales) |
| `salesmonthly.csv` | month | 70 | Used only to reconcile daily totals (data-quality check) |

### Columns

| Column | Type | Description |
|---|---|---|
| `datum` | date / datetime | Period timestamp. Daily/hourly files use `M/D/YYYY[ H:MM]`; the monthly file is ISO `YYYY-MM-DD` (month end). |
| `M01AB` | float ≥ 0 | Quantity sold - anti-inflammatory/antirheumatic, acetic-acid derivatives & related |
| `M01AE` | float ≥ 0 | Anti-inflammatory/antirheumatic, propionic-acid derivatives (e.g. ibuprofen class) |
| `N02BA` | float ≥ 0 | Other analgesics/antipyretics, salicylic-acid derivatives |
| `N02BE` | float ≥ 0 | Other analgesics/antipyretics, anilides (paracetamol class) |
| `N05B` | float ≥ 0 | Psycholeptics - anxiolytics |
| `N05C` | float ≥ 0 | Psycholeptics - hypnotics & sedatives (highly intermittent: ~68 % zero days) |
| `R03` | float ≥ 0 | Drugs for obstructive airway diseases |
| `R06` | float ≥ 0 | Antihistamines for systemic use |
| `Year`, `Month`, `Hour`, `Weekday Name` | int / str | Calendar helper columns shipped with the daily/hourly files. Re-derived from `datum` in this project and not used as inputs. (`Hour` in the daily file is not a sales measure and is ignored.) |

Quantities are fractional in places (e.g. `3.67`), so they are best read as normalised dispensed quantity
rather than integer packs; the project calls them **units**.

## 2. Processed tables (`data/processed/`, also `pharma.db`)

### `daily_clean.parquet` / SQLite `sales_daily`

| Column | Description |
|---|---|
| `date` | Calendar day (complete, gap-free) |
| `category` | ATC class (long format) |
| `units` | Cleaned quantity (extreme outliers winsorised, see below) |
| `units_raw` | Quantity exactly as in the raw file |
| `unit_price` | **Assumed** price per unit (`config.yaml`) |
| `revenue` | `units × unit_price` (**assumed prices**) |

### `hourly_clean.parquet` / SQLite `sales_hourly`
`date` (hour timestamp), `category`, `units`.

### `features.parquet`
One row per day and category: `units` plus the model features below.

| Feature | Definition |
|---|---|
| `weekday`, `is_weekend`, `month`, `quarter`, `day_of_month`, `week_of_year`, `is_month_end` | Calendar |
| `doy_sin`, `doy_cos` | Cyclic day-of-year encoding |
| `is_holiday`, `near_holiday` | Fixed-date festive flags from `config.yaml → calendar` (the files do not state a country, so no national holiday list is assumed) |
| `lag_1`, `lag_7`, `lag_14`, `lag_28` | Demand *k* days earlier |
| `roll_mean_w`, `roll_std_w` (w = 7, 14, 28) | Mean / std of the **previous** *w* days (shifted by one → no leakage) |
| `price`, `price_change_pct`, `price_rel_28`, `promo_flag` | Created automatically **only if** such columns exist in a plugged-in dataset; absent in this data |

## 3. Cleaning rules (`src/data/ingest.py`)

1. **Schema validation** - required columns from `config.yaml` must exist (`SchemaError` otherwise).
2. **Duplicates** - exact duplicates dropped; repeated (date, category) keys aggregated.
3. **Calendar completion** - every category re-indexed to a gap-free daily calendar.
4. **Missing / negative values** - set to missing, then linearly interpolated.
5. **Outliers** - only *extreme* points are winsorised: robust z-score vs a centred 29-day rolling median with the
   scale floored at the 99 %-trimmed global standard deviation (protects intermittent series); |z| > 6.
6. **Reconciliation** - daily sums are compared with `salesmonthly.csv`.
7. Every step writes to `reports/results/data_quality_report.json`.

## 4. Result files (`reports/results/`)

| File | Content |
|---|---|
| `cv_predictions.parquet` | Every out-of-sample prediction: category, model, fold, origin, date, h, y, yhat |
| `cv_metrics.csv` | MAE, RMSE, MAPE, sMAPE, MASE per category × model + gain vs baseline |
| `best_models.csv` | Auto-selected model per category |
| `cv_error_by_horizon.csv` | MAE by days-ahead |
| `forecasts_all.parquet` | 90-day forecasts, all models |
| `forecasts_best.parquet` | 90-day forecast of the best model with 90 % interval |
| `interval_coverage.csv` | Out-of-sample coverage of the intervals |
| `shap_importance.csv`, `shap_sample.parquet` | SHAP for the XGBoost model |
| `abc_xyz.csv`, `demand_clusters.csv`, `clustering_meta.json` | Classification and clusters |
| `anomalies.parquet` | Expected value, residual z-score, flags, explanations |
| `inventory_backtest.csv`, `reorder_list_default.csv` | Policy replay and default reorder table |
| `headline.json` | Headline numbers quoted in the report |

## 5. Assumed business parameters (`config.yaml → assumptions`)

`unit_price`, `unit_cost`, `lead_time_days`, `lead_time_sd_days`, `order_cost`, `shelf_life_remaining_days`
(per category, with defaults), plus `holding_rate_annual`, `default_days_of_cover`, `overstock_cover_days` and
`price_elasticity` under `inventory:`. These are illustrative planning inputs, **not measurements**.
