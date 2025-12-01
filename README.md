reconciliation and hybridization

demand forecasting pipeline that brings ts and ml forecasts to same level then merges them

what it does

reconciliation brings ts (time series) and ml (machine learning) forecasts to same granularity level. joins ts_forecast and ml_forecast tables, splits forecasts proportionally to days in period, adds segment names, handles mid-term forecasts separately, outputs reconciled_forecast table

hybridization merges reconciled forecasts into single hybrid forecast using business rules. ml forecast used for promo (not retired), short lifecycle, new assortment. ts forecast used for retired/low volume with low ts values (< 0.01). ensemble is average of ts and ml for everything else. outputs hybrid_forecast table with forecast_source column

usage

```python
from reconciliation import reconciliation
from hybridization import hybridization

config = {
    'IB_HIST_END_DT': datetime(2023, 12, 31),
    'IB_FC_HORIZ': 90,
    'ts_time_lvl': 'MONTH',
    'ml_time_lvl': 'WEEK.2'
}
df_reconciled = reconciliation(df_ts, df_ml, df_segments, config)
df_hybrid = hybridization(df_reconciled)
print(df_hybrid[['PRODUCT_LVL_ID', 'HYBRID_FORECAST_VALUE', 'FORECAST_SOURCE']])
```

input data

for reconciliation need ts_forecast with columns product_lvl_id, location_lvl_id, customer_lvl_id, distr_channel_lvl_id, period_dt, forecast_value. ml_forecast same columns. ts_segments has product_lvl_id, location_lvl_id, customer_lvl_id, distr_channel_lvl_id, segment_name. config is dict with parameters

for hybridization need reconciled_forecast output from reconciliation with ts_forecast_value_rec, ml_forecast_value, segment_name, demand_type, assortment_type

output

reconciled_forecast has all id columns, period_dt, period_end_dt, ts_forecast_value_rec, ml_forecast_value, demand_type, assortment_type, segment_name

hybrid_forecast has same as reconciled plus hybrid_forecast_value, ensemble_forecast_value, forecast_source, ts_forecast_value

test

```bash
cd src
python test_reconciliation.py
python test_hybridization.py
```

both should run without errors and save csv files

config parameters

ib_hist_end_dt is last known date (datetime)

ib_fc_horiz is forecast horizon in days (int)

delays_config_length is delay config length for mid-term split (int)

ts_time_lvl is ts time level like 'DAY', 'WEEK.2', 'MONTH' (str)

ml_time_lvl is ml time level (str)

ib_zero_demand_threshold is threshold for zero demand, default 0.01 (float)

files

src/reconciliation.py has reconciliation logic

src/hybridization.py has hybridization logic

src/test_reconciliation.py has reconciliation tests

src/test_hybridization.py has hybridization tests

notebooks/hybridization.ipynb has example notebook

thats it
