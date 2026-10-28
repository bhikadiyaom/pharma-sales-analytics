# Pharmaceutical Sales and Medicine Demand Analytics

B.Tech Data Science (Semester 7) individual project: an end-to-end system that **analyses** historical pharmacy sales,
**forecasts** demand per therapeutic class, and turns the forecasts into **inventory decisions** (safety stock, reorder
point, EOQ, stock-out / overstock / expiry risk) in an interactive Streamlit dashboard.

> **Problem.** Pharmacies lose revenue and risk patient health through stock-outs, and lose money through expired
> overstock. Demand forecasting plus inventory optimisation reduces both.

## Data (real, not synthetic)

The project uses the **real Kaggle "Pharma Sales Data"** already placed in `data/raw/`
(`salesdaily/hourly/weekly/monthly.csv`; 8 ATC classes: M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06;
2014-01-02 → 2019-10-08). **No synthetic data is used.**

The files contain *quantities only*. Prices, costs, lead times, stock on hand and shelf life are **assumptions**
in `config.yaml → assumptions` and are labelled as assumptions everywhere (dashboard badges, report).
Details and every column: [docs/data_dictionary.md](docs/data_dictionary.md).
To plug in another dataset, edit the `data:` block of `config.yaml` (wide or long layout, column names, date format).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

make data      # ingest -> validate -> clean -> parquet + SQLite + data-quality report
make train     # rolling-origin CV (6 models x 8 series), selection, forecasts, SHAP, ABC-XYZ, anomalies, back-test
make app       # Streamlit dashboard at http://localhost:8501
```

No GNU `make` (typical on Windows)? Run the equivalent commands:

```bash
python -m src.pipeline data
python -m src.pipeline train && python -m src.pipeline analyze      # ~15-20 min on a 4-core laptop, CPU only
python -m streamlit run app/Home.py
python -m pytest -q                                                 # unit tests
python -m src.reporting.build_report                                # regenerate docs/project_report.md + viva_notes.md
python notebooks/build_notebooks.py                                 # rebuild + execute the notebooks
```

`python -m src.pipeline train --fast` is a ~1-minute smoke run (4 folds, no SARIMAX/Prophet); it overwrites
the result files, so re-run the full `train` before using the dashboard or report.

## What is implemented

| Area | Feature | Where |
|---|---|---|
| A. Pipeline | Ingestion, schema validation, duplicates/gaps/outliers, logged data-quality report | `src/data/ingest.py` |
| | Calendar / holiday, lag (1,7,14,28), rolling mean/std, price & promo features | `src/features/engineering.py` |
| B. Descriptive | KPI cards (revenue, units, top sellers, MoM/YoY), STL, weekday×month/hour heatmaps, share, correlation | `src/analytics/descriptive.py`, dashboard |
| | Pareto/ABC + XYZ → ABC-XYZ matrix with stocking policy | `descriptive.abc_xyz_matrix` |
| C. Predictive | Naive, Seasonal-naive, SARIMAX, Prophet, XGBoost, LightGBM | `src/models/forecasters.py` |
| | Rolling-origin CV, MAE/RMSE/MAPE/sMAPE/MASE, auto best model per series | `src/models/cv.py`, `metrics.py` |
| | 7/14/30/90-day horizon with prediction intervals | `cv.py` (empirical intervals), Forecasting page |
| | SHAP explanation of the tree model | `src/models/explain.py` |
| | Demand clustering (KMeans + hierarchical + rule label) | `src/analytics/clustering.py` |
| D. Prescriptive | Safety stock, ROP, EOQ at 90/95/99 % service level | `src/inventory/policy.py` |
| | Stock-out / overstock / expiry-risk flags, reorder list | `policy.py`, `planner.py` |
| | What-if simulator (price, promo, lead time, surge) | `src/inventory/whatif.py` |
| | Anomaly detection (STL residual z-score + Isolation Forest, explained) | `src/analytics/anomaly.py` |
| | Policy back-test: forecast-driven vs moving-average replenishment | `src/inventory/simulate.py` |
| E. Dashboard | 10 pages, global filters, CSV downloads, `st.cache_data`, tooltips | `app/` |

## Dashboard pages
Overview · Sales Explorer · Seasonality · ABC-XYZ (+ clusters) · Forecasting (+ SHAP) · Inventory Planner ·
What-If Simulator · Anomalies · Model Performance · About/Methodology.

Global sidebar filters: date range and medicine category (ATC class). A *branch* filter appears automatically if the
data has a `branch` column; this dataset is a single sales point, so it is hidden.

## Project layout

```
.
├── config.yaml            data mapping, assumptions, model/CV/inventory settings
├── data/{raw,processed}   raw Kaggle files; cleaned parquet + pharma.db (SQLite)
├── src/{data,features,models,inventory,analytics,viz,reporting}, pipeline.py
├── app/{Home.py,pages/}   Streamlit dashboard
├── notebooks/             01_eda, 02_modeling, 03_inventory (executed, with outputs)
├── tests/                 pytest: cleaning, features, metrics/CV, inventory formulas
├── reports/{results,figures}   every number in the report comes from here
├── docs/                  data_dictionary.md, project_report.md, viva_notes.md
├── Makefile, requirements.txt, pyproject.toml (ruff/black)
```

## Reproducibility and honesty rules
* All numbers in `docs/` are generated from `reports/results/*` by `src/reporting/build_report.py`; none are typed by hand.
* Seeds are fixed (`config.yaml → project.seed`). Training is CPU-only, no paid APIs.
* Time-series CV is rolling-origin; each fold refits using only the past. Rolling features are shifted by one day.
* Model selection and reporting use the same folds, so the reported gain is slightly optimistic (see report, Limitations).

## Screenshots (to be added)
Place images in `reports/figures/` and link them here. Suggested list: see "Screenshot checklist" in
[docs/project_report.md](docs/project_report.md#appendix-screenshot-checklist).

| Page | File |
|---|---|
| Overview | `reports/figures/app_overview.png` |
| Forecasting | `reports/figures/app_forecasting.png` |
| Inventory Planner | `reports/figures/app_inventory.png` |
| Model Performance | `reports/figures/app_model_performance.png` |


cd "D:\Pharma project"
.venv\Scripts\activate
python -m src.pipeline data
python -m src.pipeline train && python -m src.pipeline analyze
python -m streamlit run app/Home.py
python -m pytest -q