"""
hybrid forecast generation module

merges ts and ml forecasts into single hybrid forecast value

business rules:
- ml forecast: promo, short lifecycle, new assortment
- ts forecast: retired, low volume, near-zero forecasts  
- ensemble: average for all other cases

todo:
    1. adaptive selection instead of simple average
    2. config file for consolidation rules
    3. multi-level reconciliation support
"""

import pandas as pd
import numpy as np
from typing import Optional


# config
# todo: move to external config file
IB_ZERO_DEMAND_THRESHOLD = 0.01


def hybridization(
    reconciled_forecast: pd.DataFrame,
    ib_zero_demand_threshold: float = IB_ZERO_DEMAND_THRESHOLD
) -> pd.DataFrame:
    """
    generate hybrid forecast by consolidating ts and ml forecast values
    
    parameters
    ----------
    reconciled_forecast : pd.dataframe
        input dataframe with reconciled forecasts
        
    ib_zero_demand_threshold : float
        threshold for zero demand, default 0.01
    
    returns
    -------
    pd.dataframe
        hybrid forecast table with hybrid_forecast_value, ensemble_forecast_value, forecast_source
    
    algorithm logic
    ---------------
    1. fill missing values
    2. determine hybrid_forecast_value based on rules:
       - ml: promo (not retired), short segments, new assortment
       - ts: retired/low volume with low ts forecast
       - ensemble: average for all other cases
    3. set forecast_source
    4. calculate ensemble_forecast_value
    """
    
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
        df['SEGMENT_NAME'] = ''
    else:
        df['SEGMENT_NAME'] = df['SEGMENT_NAME'].fillna('')
    
    df['DEMAND_TYPE_LOWER'] = df['DEMAND_TYPE'].str.lower() if 'DEMAND_TYPE' in df.columns else ''
    df['SEGMENT_NAME_LOWER'] = df['SEGMENT_NAME'].str.lower()
    df['ASSORTMENT_TYPE_LOWER'] = df['ASSORTMENT_TYPE'].str.lower() if 'ASSORTMENT_TYPE' in df.columns else ''
    
    def calculate_hybrid_forecast(row):
        """apply business rules to determine hybrid forecast value"""
        if ((row['DEMAND_TYPE_LOWER'] == 'promo' and row['SEGMENT_NAME_LOWER'] != 'retired') or
            row['SEGMENT_NAME_LOWER'] == 'short' or
            row['ASSORTMENT_TYPE_LOWER'] == 'new'):
            return row['ML_FORECAST_VALUE_F']
        
        # TODO: Rule 2 - Use TS forecast for retired/low volume with low forecast
        # ELSE CASE WHEN (SEGMENT_NAME = 'Retired' OR SEGMENT_NAME = 'Low Volume')
        #               TS_FORECAST_VALUE_F <= IB_ZERO_DEMAND_THRESHOLD
        # THEN TS_FORECAST_VALUE_F
        elif ((row['SEGMENT_NAME_LOWER'] == 'retired' or row['SEGMENT_NAME_LOWER'] == 'low volume') and
              row['TS_FORECAST_VALUE_F'] <= ib_zero_demand_threshold):
            return row['TS_FORECAST_VALUE_F']
        
        # TODO: Rule 3 - Use average (ensemble) for all other cases
        # TODO Enhancement #1: Replace with Adaptive Selection method
        # ELSE AVERAGE(TS_FORECAST_VALUE_F, ML_FORECAST_VALUE_F)
        else:
            # Average handles NaN values: AVERAGE(missing, 1) = 1; AVERAGE(missing, missing) = missing
            values = [row['TS_FORECAST_VALUE_F'], row['ML_FORECAST_VALUE_F']]
            valid_values = [v for v in values if pd.notna(v)]
            if len(valid_values) > 0:
                return np.mean(valid_values)
            else:
                return np.nan
    
    def calculate_forecast_source(row):
        """determine which forecast source was used"""
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
        """calculate ensemble value"""
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
    
    df = df.drop(columns=['DEMAND_TYPE_LOWER', 'SEGMENT_NAME_LOWER', 'ASSORTMENT_TYPE_LOWER',
                          'TS_FORECAST_VALUE_F', 'ML_FORECAST_VALUE_F'], errors='ignore')
    
    return df


def create_mid_term_hybrid_forecast(reconciled_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    create mid-term hybrid forecast by selecting ts forecast as hybrid forecast
    
    parameters
    ----------
    reconciled_forecast : pd.dataframe
        input dataframe with mid_reconciled_forecast data
        
    returns
    -------
    pd.dataframe
        dataframe with hybrid_forecast_value set to ts_forecast_value
    """
    df = reconciled_forecast.copy()
    
    if 'TS_FORECAST_VALUE_REC' in df.columns:
        df['HYBRID_FORECAST_VALUE'] = df['TS_FORECAST_VALUE_REC']
        df['TS_FORECAST_VALUE'] = df['TS_FORECAST_VALUE_REC']
        df['FORECAST_SOURCE'] = 'ts'
        df['ENSEMBLE_FORECAST_VALUE'] = np.nan
    
    return df