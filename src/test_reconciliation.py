import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from reconciliation import reconciliation


def generate_test_data():
    
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    
    ts_data = []
    ml_data = []
    
    for date in dates:
        for prod in ['P001', 'P002', 'P003']:
            for loc in ['L001', 'L002']:
                ts_data.append({
                    'PRODUCT_LVL_ID': prod,
                    'LOCATION_LVL_ID': loc,
                    'CUSTOMER_LVL_ID': 'C001',
                    'DISTR_CHANNEL_LVL_ID': 'CH1',
                    'PERIOD_DT': date,
                    'PERIOD_END_DT': date,
                    'FORECAST_VALUE': np.random.uniform(50, 150)
                })
                
                ml_data.append({
                    'PRODUCT_LVL_ID': prod,
                    'LOCATION_LVL_ID': loc,
                    'CUSTOMER_LVL_ID': 'C001',
                    'DISTR_CHANNEL_LVL_ID': 'CH1',
                    'PERIOD_DT': date,
                    'PERIOD_END_DT': date,
                    'FORECAST_VALUE': np.random.uniform(60, 140),
                    'DEMAND_TYPE': np.random.choice(['promo', 'regular']),
                    'ASSORTMENT_TYPE': np.random.choice(['new', 'old'])
                })
    
    df_ts = pd.DataFrame(ts_data)
    df_ml = pd.DataFrame(ml_data)
    
    segments_data = []
    for prod in ['P001', 'P002', 'P003']:
        for loc in ['L001', 'L002']:
            segments_data.append({
                'product_lvl_id': prod,
                'location_lvl_id': loc,
                'customer_lvl_id': 'C001',
                'distr_channel_lvl_id': 'CH1',
                'SEGMENT_NAME': np.random.choice(['Regular', 'Short', 'Retired', 'Low Volume'])
            })
    
    df_segments = pd.DataFrame(segments_data)
    
    return df_ts, df_ml, df_segments


def test_reconciliation():
    
    print("test started")
    
    df_ts, df_ml, df_segments = generate_test_data()
    
    config = {
        'IB_HIST_END_DT': datetime(2023, 12, 31),
        'IB_FC_HORIZ': 90
    }
    
    df_result = reconciliation(df_ts, df_ml, df_segments, config)
    
    print(f"\nresults {len(df_result)} records")
    
    print("\nsample output")
    print(df_result.head(10))
    
    print("\ncolumns")
    print(df_result.columns.tolist())
    
    if 'TS_FORECAST_VALUE_REC' in df_result.columns and 'ML_FORECAST_VALUE' in df_result.columns:
        print("\nts and ml forecasts present")
    
    if 'SEGMENT_NAME' in df_result.columns:
        print(f"\nsegment distribution")
        print(df_result['SEGMENT_NAME'].value_counts())
    
    print("\ntest complete")
    
    return df_result


if __name__ == '__main__':
    df_result = test_reconciliation()
    df_result.to_csv('reconciled_forecast_output.csv', index=False)
    print("\nsaved to reconciled_forecast_output.csv")

