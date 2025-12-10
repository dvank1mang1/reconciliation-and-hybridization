import pandas as pd
import numpy as np


IB_ZERO_DEMAND_THRESHOLD = 0.01


def hybridization(
    reconciled_forecast: pd.DataFrame,
    ib_zero_demand_threshold: float = IB_ZERO_DEMAND_THRESHOLD
) -> pd.DataFrame:
    
    df = reconciled_forecast.copy()
    
    if 'TS_FORECAST_VALUE_REC' in df.columns and 'ML_FORECAST_VALUE' in df.columns:
        df['TS_FORECAST_VALUE_F'] = df['TS_FORECAST_VALUE_REC'].fillna(df['ML_FORECAST_VALUE'])
    elif 'TS_FORECAST_VALUE_REC' in df.columns:
        df['TS_FORECAST_VALUE_F'] = df['TS_FORECAST_VALUE_REC']
    else:
        df['TS_FORECAST_VALUE_F'] = df.get('ML_FORECAST_VALUE', np.nan)
    
    if 'ML_FORECAST_VALUE' in df.columns and 'TS_FORECAST_VALUE_REC' in df.columns:
        df['ML_FORECAST_VALUE_F'] = df['ML_FORECAST_VALUE'].fillna(df['TS_FORECAST_VALUE_REC'])
    elif 'ML_FORECAST_VALUE' in df.columns:
        df['ML_FORECAST_VALUE_F'] = df['ML_FORECAST_VALUE']
    else:
        df['ML_FORECAST_VALUE_F'] = df.get('TS_FORECAST_VALUE_REC', np.nan)
    
    if 'SEGMENT_NAME' not in df.columns:
        df['SEGMENT_NAME'] = np.nan
    
    if 'DEMAND_TYPE' not in df.columns:
        df['DEMAND_TYPE'] = np.nan
    
    if 'ASSORTMENT_TYPE' not in df.columns:
        df['ASSORTMENT_TYPE'] = np.nan
    
    df['DEMAND_TYPE_LOWER'] = df['DEMAND_TYPE'].fillna('').astype(str).str.lower()
    df['SEGMENT_NAME_LOWER'] = df['SEGMENT_NAME'].fillna('').astype(str).str.lower()
    df['ASSORTMENT_TYPE_LOWER'] = df['ASSORTMENT_TYPE'].fillna('').astype(str).str.lower()
    
    def calculate_hybrid_forecast(row):
        if ((row['DEMAND_TYPE_LOWER'] == 'promo' and row['SEGMENT_NAME_LOWER'] != 'retired') or
            row['SEGMENT_NAME_LOWER'] == 'short' or
            row['ASSORTMENT_TYPE_LOWER'] == 'new'):
            return row['ML_FORECAST_VALUE_F']
        elif ((row['SEGMENT_NAME_LOWER'] == 'retired' or row['SEGMENT_NAME_LOWER'] == 'low volume') and
              row['TS_FORECAST_VALUE_F'] <= ib_zero_demand_threshold):
            return row['TS_FORECAST_VALUE_F']
        else:
            if pd.notna(row['TS_FORECAST_VALUE_F']) and row['TS_FORECAST_VALUE_F'] <= ib_zero_demand_threshold:
                return row['TS_FORECAST_VALUE_F']
            else:
                values = [row['TS_FORECAST_VALUE_F'], row['ML_FORECAST_VALUE_F']]
                valid_values = [v for v in values if pd.notna(v)]
                if len(valid_values) > 0:
                    return np.mean(valid_values)
                else:
                    return np.nan
    
    def calculate_forecast_source(row):
        if ((row['DEMAND_TYPE_LOWER'] == 'promo' and row['SEGMENT_NAME_LOWER'] != 'retired') or
            row['SEGMENT_NAME_LOWER'] == 'short' or
            row['ASSORTMENT_TYPE_LOWER'] == 'new'):
            return 'ml'
        elif ((row['SEGMENT_NAME_LOWER'] == 'retired' or row['SEGMENT_NAME_LOWER'] == 'low volume') and
              row['TS_FORECAST_VALUE_F'] <= ib_zero_demand_threshold):
            return 'ts'
        else:
            return 'ensemble'
    
    def calculate_ensemble_value(row):
        if ((row['DEMAND_TYPE_LOWER'] == 'promo' and row['SEGMENT_NAME_LOWER'] != 'retired') or
            row['SEGMENT_NAME_LOWER'] == 'short' or
            row['ASSORTMENT_TYPE_LOWER'] == 'new'):
            return np.nan
        elif ((row['SEGMENT_NAME_LOWER'] == 'retired' or row['SEGMENT_NAME_LOWER'] == 'low volume') and
              row['TS_FORECAST_VALUE_F'] <= ib_zero_demand_threshold):
            return np.nan
        else:
            values = [row['TS_FORECAST_VALUE_F'], row['ML_FORECAST_VALUE_F']]
            valid_values = [v for v in values if pd.notna(v)]
            if len(valid_values) > 0:
                return np.mean(valid_values)
            else:
                return np.nan
    
    df['HYBRID_FORECAST_VALUE'] = df.apply(calculate_hybrid_forecast, axis=1)
    df['FORECAST_SOURCE'] = df.apply(calculate_forecast_source, axis=1)
    df['ENSEMBLE_FORECAST_VALUE'] = df.apply(calculate_ensemble_value, axis=1)
    
    if 'TS_FORECAST_VALUE_REC' in df.columns:
        df['TS_FORECAST_VALUE'] = df['TS_FORECAST_VALUE_REC']
    
    if 'ML_FORECAST_VALUE' not in df.columns:
        df['ML_FORECAST_VALUE'] = np.nan
    
    df = df.drop(columns=['DEMAND_TYPE_LOWER', 'SEGMENT_NAME_LOWER', 'ASSORTMENT_TYPE_LOWER',
                          'TS_FORECAST_VALUE_F', 'ML_FORECAST_VALUE_F'], errors='ignore')
    
    return df


def create_mid_term_hybrid_forecast(reconciled_forecast: pd.DataFrame) -> pd.DataFrame:
    
    df = reconciled_forecast.copy()
    
    if 'TS_FORECAST_VALUE_REC' in df.columns:
        df['HYBRID_FORECAST_VALUE'] = df['TS_FORECAST_VALUE_REC']
        df['TS_FORECAST_VALUE'] = df['TS_FORECAST_VALUE_REC']
        df['FORECAST_SOURCE'] = 'ts'
        df['ENSEMBLE_FORECAST_VALUE'] = np.nan
    
    if 'ML_FORECAST_VALUE' not in df.columns:
        df['ML_FORECAST_VALUE'] = np.nan
    
    return df