import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hybridization import hybridization, IB_ZERO_DEMAND_THRESHOLD


def generate_reconciled_forecast_data(
    start_date: str = '2023-01-01',
    end_date: str = '2023-01-31',
    num_products: int = 5,
    num_locations: int = 3
) -> pd.DataFrame:
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    products = [f'PROD_{i:03d}' for i in range(1, num_products + 1)]
    locations = [f'LOC_{i:03d}' for i in range(1, num_locations + 1)]
    
    records = []
    
    for date in dates:
        for product in products:
            for location in locations:
                record = {
                    'PRODUCT_LVL_ID': product,
                    'LOCATION_LVL_ID': location,
                    'CUSTOMER_LVL_ID': f'CUST_{np.random.randint(1, 100):03d}',
                    'DISTR_CHANNEL_LVL_ID': f'CH_{np.random.randint(1, 5):01d}',
                    'PERIOD_DT': date,
                    'PERIOD_END_DT': date + timedelta(days=1),
                    'TS_FORECAST_VALUE_REC': abs(np.random.normal(100, 30)),
                    'ML_FORECAST_VALUE': abs(np.random.normal(100, 30)),
                    'SEGMENT_NAME': np.random.choice([
                        'Regular', 'Short', 'Retired', 'Low Volume', None
                    ], p=[0.5, 0.15, 0.15, 0.15, 0.05]),
                    'DEMAND_TYPE': np.random.choice(['regular', 'promo'], p=[0.7, 0.3]),
                    'ASSORTMENT_TYPE': np.random.choice(['old', 'new'], p=[0.8, 0.2])
                }
                
                records.append(record)
    
    df = pd.DataFrame(records)
    
    df.loc[0, 'SEGMENT_NAME'] = 'Retired'
    df.loc[0, 'TS_FORECAST_VALUE_REC'] = 0.005
    df.loc[0, 'ML_FORECAST_VALUE'] = 50.0
    
    df.loc[1, 'DEMAND_TYPE'] = 'promo'
    df.loc[1, 'SEGMENT_NAME'] = 'Regular'
    df.loc[1, 'TS_FORECAST_VALUE_REC'] = 80.0
    df.loc[1, 'ML_FORECAST_VALUE'] = 120.0
    
    df.loc[2, 'ASSORTMENT_TYPE'] = 'new'
    df.loc[2, 'TS_FORECAST_VALUE_REC'] = 40.0
    df.loc[2, 'ML_FORECAST_VALUE'] = 90.0
    
    df.loc[3, 'SEGMENT_NAME'] = 'Short'
    df.loc[3, 'TS_FORECAST_VALUE_REC'] = 60.0
    df.loc[3, 'ML_FORECAST_VALUE'] = 75.0
    
    df.loc[4, 'SEGMENT_NAME'] = 'Low Volume'
    df.loc[4, 'TS_FORECAST_VALUE_REC'] = 0.008
    df.loc[4, 'ML_FORECAST_VALUE'] = 30.0
    
    df.loc[5, 'TS_FORECAST_VALUE_REC'] = np.nan
    df.loc[5, 'ML_FORECAST_VALUE'] = 55.0
    
    df.loc[6, 'TS_FORECAST_VALUE_REC'] = 45.0
    df.loc[6, 'ML_FORECAST_VALUE'] = np.nan
    
    return df


def test_hybridization():
    
    print("test started")
    
    df_input = generate_reconciled_forecast_data(
        start_date='2023-01-01',
        end_date='2023-01-10',
        num_products=3,
        num_locations=2
    )
    
    df_output = hybridization(df_input, ib_zero_demand_threshold=IB_ZERO_DEMAND_THRESHOLD)
    
    print("\nforecast sources")
    print(df_output['FORECAST_SOURCE'].value_counts())
    
    print("\nsamples")
    
    for source in ['ml', 'ts', 'ensemble']:
        print(f"\n{source}")
        sample = df_output[df_output['FORECAST_SOURCE'] == source].head(3)
        
        if len(sample) > 0:
            cols_to_show = [
                'PRODUCT_LVL_ID', 'SEGMENT_NAME', 'DEMAND_TYPE', 'ASSORTMENT_TYPE',
                'TS_FORECAST_VALUE', 'ML_FORECAST_VALUE', 
                'HYBRID_FORECAST_VALUE', 'ENSEMBLE_FORECAST_VALUE', 'FORECAST_SOURCE'
            ]
            print(sample[cols_to_show].to_string(index=False))
        else:
            print("no records")
    
    print("\nvalidation")
    
    ml_cases = df_output[
        ((df_output['DEMAND_TYPE'].str.lower() == 'promo') & 
         (df_output['SEGMENT_NAME'].str.lower() != 'retired')) |
        (df_output['SEGMENT_NAME'].str.lower() == 'short') |
        (df_output['ASSORTMENT_TYPE'].str.lower() == 'new')
    ]
    ml_correct = (ml_cases['FORECAST_SOURCE'] == 'ml').sum()
    print(f"rule 1 (ml forecast) {ml_correct}/{len(ml_cases)} cases correctly applied")
    
    ts_cases = df_output[
        ((df_output['SEGMENT_NAME'].str.lower() == 'retired') |
         (df_output['SEGMENT_NAME'].str.lower() == 'low volume')) &
        (df_output['TS_FORECAST_VALUE'] <= IB_ZERO_DEMAND_THRESHOLD)
    ]
    ts_correct = (ts_cases['FORECAST_SOURCE'] == 'ts').sum()
    print(f"rule 2 (ts forecast) {ts_correct}/{len(ts_cases)} cases correctly applied")
    
    ensemble_cases = df_output[df_output['FORECAST_SOURCE'] == 'ensemble']
    ensemble_has_value = ensemble_cases['ENSEMBLE_FORECAST_VALUE'].notna().sum()
    print(f"rule 3 (ensemble) {ensemble_has_value}/{len(ensemble_cases)} cases have ensemble values")
    
    print("\ntest complete")
    
    return df_output


def show_detailed_examples():
    
    print("\ndetailed examples")
    
    test_cases = pd.DataFrame([
        {
            'PRODUCT_LVL_ID': 'P001',
            'LOCATION_LVL_ID': 'L001',
            'CUSTOMER_LVL_ID': 'C001',
            'DISTR_CHANNEL_LVL_ID': 'CH1',
            'PERIOD_DT': datetime(2023, 1, 1),
            'PERIOD_END_DT': datetime(2023, 1, 2),
            'TS_FORECAST_VALUE_REC': 80.0,
            'ML_FORECAST_VALUE': 120.0,
            'SEGMENT_NAME': 'Regular',
            'DEMAND_TYPE': 'promo',
            'ASSORTMENT_TYPE': 'old',
            'DESCRIPTION': 'promo demand (not retired) -> should use ml'
        },
        {
            'PRODUCT_LVL_ID': 'P002',
            'LOCATION_LVL_ID': 'L001',
            'CUSTOMER_LVL_ID': 'C001',
            'DISTR_CHANNEL_LVL_ID': 'CH1',
            'PERIOD_DT': datetime(2023, 1, 1),
            'PERIOD_END_DT': datetime(2023, 1, 2),
            'TS_FORECAST_VALUE_REC': 60.0,
            'ML_FORECAST_VALUE': 95.0,
            'SEGMENT_NAME': 'Short',
            'DEMAND_TYPE': 'regular',
            'ASSORTMENT_TYPE': 'old',
            'DESCRIPTION': 'short lifecycle -> should use ml'
        },
        {
            'PRODUCT_LVL_ID': 'P003',
            'LOCATION_LVL_ID': 'L001',
            'CUSTOMER_LVL_ID': 'C001',
            'DISTR_CHANNEL_LVL_ID': 'CH1',
            'PERIOD_DT': datetime(2023, 1, 1),
            'PERIOD_END_DT': datetime(2023, 1, 2),
            'TS_FORECAST_VALUE_REC': 40.0,
            'ML_FORECAST_VALUE': 110.0,
            'SEGMENT_NAME': 'Regular',
            'DEMAND_TYPE': 'regular',
            'ASSORTMENT_TYPE': 'new',
            'DESCRIPTION': 'new assortment -> should use ml'
        },
        {
            'PRODUCT_LVL_ID': 'P004',
            'LOCATION_LVL_ID': 'L001',
            'CUSTOMER_LVL_ID': 'C001',
            'DISTR_CHANNEL_LVL_ID': 'CH1',
            'PERIOD_DT': datetime(2023, 1, 1),
            'PERIOD_END_DT': datetime(2023, 1, 2),
            'TS_FORECAST_VALUE_REC': 0.005,
            'ML_FORECAST_VALUE': 50.0,
            'SEGMENT_NAME': 'Retired',
            'DEMAND_TYPE': 'regular',
            'ASSORTMENT_TYPE': 'old',
            'DESCRIPTION': 'retired with low ts forecast -> should use ts'
        },
        {
            'PRODUCT_LVL_ID': 'P005',
            'LOCATION_LVL_ID': 'L001',
            'CUSTOMER_LVL_ID': 'C001',
            'DISTR_CHANNEL_LVL_ID': 'CH1',
            'PERIOD_DT': datetime(2023, 1, 1),
            'PERIOD_END_DT': datetime(2023, 1, 2),
            'TS_FORECAST_VALUE_REC': 0.008,
            'ML_FORECAST_VALUE': 35.0,
            'SEGMENT_NAME': 'Low Volume',
            'DEMAND_TYPE': 'regular',
            'ASSORTMENT_TYPE': 'old',
            'DESCRIPTION': 'low volume with low ts forecast -> should use ts'
        },
        {
            'PRODUCT_LVL_ID': 'P006',
            'LOCATION_LVL_ID': 'L001',
            'CUSTOMER_LVL_ID': 'C001',
            'DISTR_CHANNEL_LVL_ID': 'CH1',
            'PERIOD_DT': datetime(2023, 1, 1),
            'PERIOD_END_DT': datetime(2023, 1, 2),
            'TS_FORECAST_VALUE_REC': 75.0,
            'ML_FORECAST_VALUE': 85.0,
            'SEGMENT_NAME': 'Regular',
            'DEMAND_TYPE': 'regular',
            'ASSORTMENT_TYPE': 'old',
            'DESCRIPTION': 'regular case -> should use ensemble (average)'
        }
    ])
    
    result = hybridization(test_cases)
    
    for idx, row in result.iterrows():
        print(f"\ncase {idx + 1} {row['DESCRIPTION']}")
        print(f"  ts forecast {row['TS_FORECAST_VALUE']:.3f}")
        print(f"  ml forecast {row['ML_FORECAST_VALUE']:.3f}")
        print(f"  hybrid forecast {row['HYBRID_FORECAST_VALUE']:.3f}")
        print(f"  forecast source {row['FORECAST_SOURCE']}")
        if pd.notna(row['ENSEMBLE_FORECAST_VALUE']):
            print(f"  ensemble value {row['ENSEMBLE_FORECAST_VALUE']:.3f}")


def test_mid_term_hybrid_forecast():
    
    print("\nmid-term test")
    
    from hybridization import create_mid_term_hybrid_forecast
    
    df_input = pd.DataFrame([
        {
            'PRODUCT_LVL_ID': 'P001',
            'LOCATION_LVL_ID': 'LOC_001',
            'CUSTOMER_LVL_ID': 'CUST_001',
            'DISTR_CHANNEL_LVL_ID': 'CH_1',
            'PERIOD_DT': datetime(2023, 1, 1),
            'PERIOD_END_DT': datetime(2023, 1, 2),
            'TS_FORECAST_VALUE_REC': 100.0,
            'SEGMENT_NAME': 'Regular',
            'DEMAND_TYPE': 'regular',
            'ASSORTMENT_TYPE': 'old'
        },
        {
            'PRODUCT_LVL_ID': 'P002',
            'LOCATION_LVL_ID': 'LOC_002',
            'CUSTOMER_LVL_ID': 'CUST_002',
            'DISTR_CHANNEL_LVL_ID': 'CH_2',
            'PERIOD_DT': datetime(2023, 1, 1),
            'PERIOD_END_DT': datetime(2023, 1, 2),
            'TS_FORECAST_VALUE_REC': 150.0,
            'SEGMENT_NAME': 'Short',
            'DEMAND_TYPE': 'promo',
            'ASSORTMENT_TYPE': 'new'
        },
        {
            'PRODUCT_LVL_ID': 'P003',
            'LOCATION_LVL_ID': 'LOC_003',
            'CUSTOMER_LVL_ID': 'CUST_003',
            'DISTR_CHANNEL_LVL_ID': 'CH_3',
            'PERIOD_DT': datetime(2023, 1, 1),
            'PERIOD_END_DT': datetime(2023, 1, 2),
            'TS_FORECAST_VALUE_REC': 75.0,
            'SEGMENT_NAME': 'Low Volume',
            'DEMAND_TYPE': 'regular',
            'ASSORTMENT_TYPE': 'old'
        }
    ])
    
    df_output = create_mid_term_hybrid_forecast(df_input)
    
    print("validation")
    
    required_columns = ['HYBRID_FORECAST_VALUE', 'TS_FORECAST_VALUE', 'FORECAST_SOURCE', 'ENSEMBLE_FORECAST_VALUE']
    missing_columns = [col for col in required_columns if col not in df_output.columns]
    
    if missing_columns:
        print(f"missing columns {missing_columns}")
    else:
        print(f"all required columns present")
    
    if 'HYBRID_FORECAST_VALUE' in df_output.columns and 'TS_FORECAST_VALUE' in df_output.columns:
        matches = (df_output['HYBRID_FORECAST_VALUE'] == df_output['TS_FORECAST_VALUE']).sum()
        total = len(df_output)
        print(f"hybrid = ts {matches}/{total}")
        
        if matches != total:
            print("mismatch detected")
            print(df_output[['PRODUCT_LVL_ID', 'TS_FORECAST_VALUE', 'HYBRID_FORECAST_VALUE']])
    
    if 'FORECAST_SOURCE' in df_output.columns:
        ts_count = (df_output['FORECAST_SOURCE'] == 'ts').sum()
        total = len(df_output)
        print(f"forecast source is ts {ts_count}/{total}")
        
        if ts_count != total:
            print("some forecast sources are not ts")
            print(df_output['FORECAST_SOURCE'].value_counts())
    
    if 'ENSEMBLE_FORECAST_VALUE' in df_output.columns:
        nan_count = df_output['ENSEMBLE_FORECAST_VALUE'].isna().sum()
        total = len(df_output)
        print(f"ensemble is nan {nan_count}/{total}")
    
    print("\nsample output")
    cols_to_show = ['PRODUCT_LVL_ID', 'TS_FORECAST_VALUE', 'HYBRID_FORECAST_VALUE', 
                    'FORECAST_SOURCE', 'ENSEMBLE_FORECAST_VALUE']
    print(df_output[cols_to_show].to_string(index=False))
    
    print("\nmid-term test complete")
    
    return df_output


if __name__ == '__main__':
    df_result = test_hybridization()
    show_detailed_examples()
    df_mid_term = test_mid_term_hybrid_forecast()
    
    output_file = 'hybrid_forecast_output.csv'
    df_result.to_csv(output_file, index=False)
    print(f"\nresults saved to {output_file}")
    
    output_file_mid = 'mid_term_hybrid_forecast_output.csv'
    df_mid_term.to_csv(output_file_mid, index=False)
    print(f"mid-term saved to {output_file_mid}")

