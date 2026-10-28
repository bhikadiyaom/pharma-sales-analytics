# Viva Notes: 25 likely questions

Numbers in these answers are generated from the executed pipeline (`reports/results/`). The data is the **real Kaggle Pharma Sales
Data**; prices, costs, lead times, stock and shelf life are **assumptions** (say this early; examiners like honesty).

**Cheat-sheet numbers**

| Fact | Value |
|---|---|
| Data | {{n_days}} days ({{date_start}} to {{date_end}}), {{n_cats}} ATC classes |
| CV | {{cv_folds}} folds × {{cv_horizon}} days, last {{oos_days}} days out-of-sample, selection by {{selection_metric}} |
| Selected models | {{win_counts}} |
| Gain vs seasonal naive | MAE {{mean_mae_improvement_pct}} %, RMSE {{mean_rmse_improvement_pct}} % (mean over series) |
| Back-test at {{bt_sl}} SL | fill rate {{fill_baseline}} % → {{fill_forecast}} %, lost units {{lost_units_reduction_pct}} % lower, inventory value {{inventory_value_change_pct}} % |

---

### A. Problem and data

**1. What problem does the project solve?**
Stock-outs hurt patients and revenue; overstock wastes money and expires. I forecast demand per drug class and convert the forecast and its error into reorder points, safety stock and risk alerts.

**2. Which data did you use? Is any of it synthetic?**
The real Kaggle Pharma Sales Data: quantities of 8 ATC classes, daily/hourly/weekly/monthly, {{date_start}} to {{date_end}}. Nothing is synthetic. The file has no prices, costs, stock, expiry or branch, so those inputs are explicit assumptions in `config.yaml`, and the app labels them.

**3. How did you clean the data and what did you find?**
Schema validation, duplicate removal, calendar completion, imputation, and capping of only extreme outliers. The data was already clean (no gaps or duplicates); {{outliers_total}} points were capped. Daily sums were reconciled with the monthly file ({{recon_within1}} % of pairs within 1 %). Everything is logged in `data_quality_report.json`.

**4. Why a conservative outlier rule?**
Pharmacy demand has real peaks (outbreaks, bulk buying). My first versions either capped genuine R03 peaks or were inflated by the outlier itself, so I floor the robust scale with a trimmed global SD and only cap at |z| > 6. Unit tests lock this behaviour.

### B. Forecasting

**5. Why these models?**
They cover four families: baselines (naive, seasonal naive) that any model must beat; a classical statistical model (SARIMAX); an additive decomposition model (Prophet); and gradient-boosted trees on lag features (XGBoost, LightGBM). Comparing families on identical folds is the fair way to choose.

**6. Why not an LSTM / deep learning?**
Only 8 series with about 2,100 daily points each, small noisy counts, CPU-only laptop. Deep models need much more data (the M5-type setting has thousands of series) and are harder to explain and tune, and they often do not beat tuned statistical/tree models at this scale [6, 7]. It is listed as future scope.

**7. How did you handle seasonality?**
Weekly: seasonal differencing (period 7) in SARIMAX, weekday features and lags (7, 14, 28) in the trees, weekly component in Prophet. Annual: Fourier terms in SARIMAX (period 365 is impractical for a plain seasonal ARIMA), yearly component in Prophet, cyclic day-of-year features for the trees. STL and heatmaps document the patterns.

**8. How do you avoid data leakage?**
(i) Rolling-origin CV: every fold refits on data strictly before the test window. (ii) Rolling features use `shift(1)`; a unit test confirms past features do not change if future values change. (iii) Prediction-interval calibration uses older origins. (iv) The back-test sets sigma from *previous* errors only.

**9. What is rolling-origin CV and why not k-fold?**
Training window expands, the test window is the next 14 days, repeated {{cv_folds}} times. Ordinary k-fold shuffles time and lets the model train on the future, giving over-optimistic scores [10].

**10. Why these metrics, and why select by {{selection_metric}}?**
MAE and RMSE are in units/day; MASE compares with seasonal naive; sMAPE/MAPE are percentages but break near zero. MAE rewards the *median* (≈0 for intermittent series); stock planning needs the *mean*, which RMSE rewards. In a smoke run MAE picked naive for the intermittent series, so I switched.

**11. MAPE is high. Is the model bad?**
Daily counts are small and noisy (single digits for most classes), so percentage errors are large for every model; the selected-model MAPE is {{mean_best_mape}} % vs {{mean_baseline_mape}} % for the baseline. What matters is the relative gain and that the error is quantified and used in safety stock.

**12. What is wrong with sMAPE for N05C?**
N05C is ~68 % zero days. A mean forecast is never exactly 0, so sMAPE penalises it heavily on zero days even though MAE/RMSE/MASE improve. The average sMAPE gain is {{mean_smape_improvement_pct}} % overall and {{smape_gain_regular}} % excluding the intermittent series.

**13. Which model won and is the difference significant?**
{{win_counts}}. The top four models are within a few percent of each other on average RMSE, and I did not run a formal test (Diebold-Mariano) nor nested selection; so I claim "seasonality-aware models clearly beat baselines" but only a modest edge among them.

**14. How are prediction intervals built?**
Empirical, model-agnostic: forecasts from 8 older origins up to 90 days ahead give residuals scaled by √forecast; their 5 %/95 % quantiles per horizon bucket give a {{coverage_nominal}} % interval. Scored on more recent origins the mean coverage was {{coverage_mean}} %, with series-level variation (84 % to 96 %).

**15. What does SHAP show?**
For XGBoost the top features are {{shap_top3}}: recent demand level matters most, then weekday and annual-cycle features. SHAP explains the model, not causality.

**16. How were the clusters and ABC-XYZ used?**
ABC ranks classes by (assumed-price) revenue; XYZ by the CV of weekly demand (real). Together they give a stocking policy per class (for example R03 is AZ: high value but erratic, so more safety stock and frequent review). Clustering (k = {{cluster_k}}, silhouette {{cluster_sil}}) separates stable, seasonal/erratic and intermittent classes, but with 8 series it is only exploratory.

### C. Inventory and business impact

**17. Explain safety stock, reorder point and EOQ.**
SS = z·√(L·σ² + d̄²·σ_L²) covers demand/forecast-error variability and supplier variability during the lead time; ROP = d̄·L + SS triggers an order; EOQ = √(2DS/H) balances order cost and holding cost. σ is the model's out-of-sample daily RMSE, so a better forecast directly lowers safety stock.

**18. Why use forecast error instead of demand standard deviation?**
Demand that is predictable (e.g. weekday pattern) does not need buffer stock; only the *unpredictable* part does. Forecast error measures exactly that.

**19. What is the business impact, and how reliable is it?**
In the replay of the last {{oos_days}} days at {{bt_sl}}: fill rate {{fill_baseline}} % vs {{fill_forecast}} %, stock-out days {{stockout_base}} % vs {{stockout_fc}} %, lost units {{lost_base}} vs {{lost_fc}}, average inventory value {{inv_base}} vs {{inv_fc}} (forecast vs moving average ordering). It is one year of one pharmacy, with assumed lead times, so it indicates direction and rough size, not a guarantee. The forecast policy was better in {{bt_better_n}} of {{bt_n_cats}} classes (worse: {{bt_worse_list}}).

**20. How do you measure "overstock reduction" without expiry data?**
I use average inventory value as a proxy. A true expiry saving needs batch/expiry dates, which the dataset lacks; the planner's expiry-risk flag therefore uses user-entered stock and shelf life and is illustrative.

**21. What does the what-if simulator do and how credible is it?**
It multiplies baseline demand by price-elasticity, promotion and surge factors, changes lead time, and recomputes SS/ROP/EOQ and revenue. Elasticity is an assumption (−0.6 default), so it shows sensitivity, not an estimate.

**22. How does anomaly detection work?**
STL gives the expected value; a Pearson-style residual (÷√expected) is robust-z-scored (MAD, floored); |z| > 5 is flagged, and an Isolation Forest provides a second opinion ("confirmed"). {{anomaly_total}} candidate days overall, almost all spikes. Without an event log, causes are hypotheses.

### D. Engineering and limitations

**23. How is the project reproducible and tested?**
`config.yaml` holds all settings and seeds; `make data / train / app` (or the equivalent `python -m` commands) regenerate everything; a pytest suite covers cleaning, features (leakage), metrics, CV splits, inventory formulas, planner flags, anomaly detection and a headless render test of every dashboard page. Report numbers are generated from result files, not typed.

**24. What are the main limitations?**
Aggregated classes and one sales point; assumed prices/costs/lead times/stock; no price/promo/weather covariates; selection and reporting on the same folds; approximate intervals; a single-year back-test; anomaly thresholds are heuristic.

**25. What would you do next?**
Medicine- and branch-level hierarchical forecasts; Croston/TSB or negative-binomial models for intermittent items; exogenous drivers (weather, epidemics, promotions); significance tests and nested selection; batch/expiry-aware simulation; maybe deep global models if many series become available.

---

### Quick answers to curveballs
* *"Why does the app say assumptions?"* Because the data has quantities only; I will not present invented prices as facts.
* *"Why is the Branch filter missing?"* The dataset has one sales point; the filter appears automatically if a `branch` column exists.
* *"Why did SARIMAX beat XGBoost?"* Likely (I did not test this) because the series are short and noise-dominated while the weekly/annual pattern is regular, so explicit seasonality is hard to beat; XGBoost won on the highest-volume class (N02BE).
* *"Could the results be luck?"* Possibly for small gaps between top models, which is why I do not claim a single winner; the gap to the baselines is consistent across all series.
