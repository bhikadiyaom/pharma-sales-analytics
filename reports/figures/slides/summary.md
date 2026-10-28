# Results at a glance (generated from reports/results)

Data: real Kaggle Pharma Sales, 8 ATC classes, daily 2014-01-02 to 2019-10-08.
Prices, costs, lead times and stock are assumptions (config.yaml).

* Best model per class: M01AB: SARIMAX, M01AE: SARIMAX, N02BA: SARIMAX, N02BE: XGBoost, N05B: SARIMAX, N05C: SARIMAX, R03: SARIMAX, R06: SARIMAX
* Gain over seasonal naive (mean over classes): MAE 26.2%, RMSE 27.3%
* sMAPE gain 16.6% (misleading for intermittent N05C; see report)
* Back-test at 95% service level, last 364 days, forecast vs 28-day moving average:
  fill rate 96.77% -> 98.01%,
  stock-out days 18.1% lower, lost units 38.5% lower,
  avg inventory value -9.8%
* Caveats: one pharmacy, one year, assumed lead times/costs; top models are within a few percent of each other.
