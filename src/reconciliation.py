"""
reconciliation module

brings ml and ts forecasts to same granularity
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def number_days(time_lvl, period_dt):
    if time_lvl.lower() == 'day':
        return 1
    elif time_lvl.lower().startswith('week'):
        return 7
    elif time_lvl.lower() == 'month':
        next_month = period_dt.replace(day=28) + timedelta(days=4)
        return (next_month.replace(day=1) - period_dt.replace(day=1)).days
    return 1


def reconciliation(
    ts_forecast: pd.DataFrame,
    ml_forecast: pd.DataFrame,
    ts_segments: pd.DataFrame = None,
    config: dict = None
) -> pd.DataFrame:
    """reconcile ts and ml forecasts"""
    
    if config is None:
        config = {}
    
    ib_hist_end_dt = config.get('IB_HIST_END_DT', datetime.now())
    ib_fc_horiz = config.get('IB_FC_HORIZ', 90)
    
    ts_product_lvl = config.get('ts_product_lvl', 7)
    ts_location_lvl = config.get('ts_location_lvl', 1)
    ts_customer_lvl = config.get('ts_customer_lvl', 5)
    ts_distr_channel_lvl = config.get('ts_distr_channel_lvl', 1)
    ts_time_lvl = config.get('ts_time_lvl', 'MONTH')
    
    ml_product_lvl = config.get('ml_product_lvl', 7)
    ml_location_lvl = config.get('ml_location_lvl', 5)
    ml_customer_lvl = config.get('ml_customer_lvl', 4)
    ml_distr_channel_lvl = config.get('ml_distr_channel_lvl', 1)
    ml_time_lvl = config.get('ml_time_lvl', 'WEEK.2')
    
    df_ml = ml_forecast.copy()
    df_ts = ts_forecast.copy()
    
    delays_config_length = config.get('delays_config_length', 0)
    
    mid_reconciled_dfs = []
    if ib_fc_horiz > delays_config_length:
        mask = df_ts['PERIOD_DT'] > ib_hist_end_dt + timedelta(days=delays_config_length)
        df_mid_ts = df_ts[mask].copy()
        df_mid_ts['ML_FORECAST_VALUE'] = np.nan
        df_mid_ts['DEMAND_TYPE'] = 'regular'
        df_mid_ts['ASSORTMENT_TYPE'] = 'old'
        if len(df_mid_ts) > 0:
            mid_reconciled_dfs.append(df_mid_ts)
        df_ts = df_ts[~mask].copy()
    
    if 'PERIOD_END_DT' not in df_ml.columns:
        df_ml['PERIOD_END_DT'] = df_ml.apply(
            lambda x: x['PERIOD_DT'] + timedelta(days=number_days(ml_time_lvl, x['PERIOD_DT']) - 1), 
            axis=1
        )
    
    if 'PERIOD_END_DT' not in df_ts.columns:
        df_ts['PERIOD_END_DT'] = df_ts.apply(
            lambda x: x['PERIOD_DT'] + timedelta(days=number_days(ts_time_lvl, x['PERIOD_DT']) - 1),
            axis=1
        )
    
    df_ml = df_ml[df_ml['PERIOD_DT'] > ib_hist_end_dt].copy()
    df_ts = df_ts[df_ts['PERIOD_DT'] > ib_hist_end_dt].copy()
    
    product_col_ml = f'product_lvl_id' if 'product_lvl_id' in df_ml.columns else 'PRODUCT_LVL_ID'
    location_col_ml = f'location_lvl_id' if 'location_lvl_id' in df_ml.columns else 'LOCATION_LVL_ID'
    customer_col_ml = f'customer_lvl_id' if 'customer_lvl_id' in df_ml.columns else 'CUSTOMER_LVL_ID'
    channel_col_ml = f'distr_channel_lvl_id' if 'distr_channel_lvl_id' in df_ml.columns else 'DISTR_CHANNEL_LVL_ID'
    
    product_col_ts = f'product_lvl_id' if 'product_lvl_id' in df_ts.columns else 'PRODUCT_LVL_ID'
    location_col_ts = f'location_lvl_id' if 'location_lvl_id' in df_ts.columns else 'LOCATION_LVL_ID'
    customer_col_ts = f'customer_lvl_id' if 'customer_lvl_id' in df_ts.columns else 'CUSTOMER_LVL_ID'
    channel_col_ts = f'distr_channel_lvl_id' if 'distr_channel_lvl_id' in df_ts.columns else 'DISTR_CHANNEL_LVL_ID'
    
    df_ts = df_ts.rename(columns={
        'FORECAST_VALUE': 'TS_FORECAST_VALUE',
        product_col_ts: 'product_lvl_id',
        location_col_ts: 'location_lvl_id', 
        customer_col_ts: 'customer_lvl_id',
        channel_col_ts: 'distr_channel_lvl_id'
    })
    
    df_ml = df_ml.rename(columns={
        'FORECAST_VALUE': 'ML_FORECAST_VALUE',
        'FORECAST_VALUE_total': 'ML_FORECAST_VALUE',
        product_col_ml: 'product_lvl_id',
        location_col_ml: 'location_lvl_id',
        customer_col_ml: 'customer_lvl_id', 
        channel_col_ml: 'distr_channel_lvl_id'
    })
    
    df_ml['ml_days'] = df_ml['PERIOD_DT'].apply(lambda x: number_days(ml_time_lvl, x))
    df_ts['ts_days'] = df_ts['PERIOD_DT'].apply(lambda x: number_days(ts_time_lvl, x))
    
    df_ml['ML_FORECAST_VALUE'] = df_ml.apply(
        lambda x: x['ML_FORECAST_VALUE'] * ((x['PERIOD_END_DT'] - x['PERIOD_DT']).days + 1) / x['ml_days']
        if x['ml_days'] > 0 else x['ML_FORECAST_VALUE'],
        axis=1
    )
    
    df_ts['TS_FORECAST_VALUE'] = df_ts.apply(
        lambda x: x['TS_FORECAST_VALUE'] * ((x['PERIOD_END_DT'] - x['PERIOD_DT']).days + 1) / x['ts_days']
        if x['ts_days'] > 0 else x['TS_FORECAST_VALUE'],
        axis=1
    )
    
    df_joined = pd.merge(
        df_ml,
        df_ts,
        on=['product_lvl_id', 'location_lvl_id', 'customer_lvl_id', 
            'distr_channel_lvl_id', 'PERIOD_DT'],
        how='left',
        suffixes=('_ml', '_ts')
    )
    
    df_joined['PERIOD_END_DT'] = df_joined['PERIOD_END_DT_ml'].fillna(df_joined['PERIOD_END_DT_ts'])
    
    df_joined['TS_FORECAST_VALUE'] = df_joined['TS_FORECAST_VALUE'].fillna(0)
    
    group_cols = ['product_lvl_id', 'location_lvl_id', 'customer_lvl_id', 
                  'distr_channel_lvl_id', 'PERIOD_DT']
    
    df_reconciled = df_joined.groupby(group_cols, as_index=False).agg({
        'PERIOD_END_DT': 'min',
        'TS_FORECAST_VALUE': 'sum',
        'ML_FORECAST_VALUE': 'first',
        'DEMAND_TYPE': 'first' if 'DEMAND_TYPE' in df_joined.columns else lambda x: 'regular',
        'ASSORTMENT_TYPE': 'first' if 'ASSORTMENT_TYPE' in df_joined.columns else lambda x: 'old'
    })
    
    if ts_segments is not None:
        df_reconciled = pd.merge(
            df_reconciled,
            ts_segments,
            on=['product_lvl_id', 'location_lvl_id', 'customer_lvl_id', 'distr_channel_lvl_id'],
            how='left'
        )
    
    df_reconciled = df_reconciled.rename(columns={
        'TS_FORECAST_VALUE': 'TS_FORECAST_VALUE_REC'
    })
    
    df_reconciled['PRODUCT_LVL_ID'] = df_reconciled['product_lvl_id']
    df_reconciled['LOCATION_LVL_ID'] = df_reconciled['location_lvl_id']
    df_reconciled['CUSTOMER_LVL_ID'] = df_reconciled['customer_lvl_id']
    df_reconciled['DISTR_CHANNEL_LVL_ID'] = df_reconciled['distr_channel_lvl_id']
    
    if len(mid_reconciled_dfs) > 0:
        for df_mid in mid_reconciled_dfs:
            if 'FORECAST_VALUE' in df_mid.columns:
                df_mid = df_mid.rename(columns={'FORECAST_VALUE': 'TS_FORECAST_VALUE_REC'})
            df_mid['PRODUCT_LVL_ID'] = df_mid.get('product_lvl_id', df_mid.get('PRODUCT_LVL_ID'))
            df_mid['LOCATION_LVL_ID'] = df_mid.get('location_lvl_id', df_mid.get('LOCATION_LVL_ID'))
            df_mid['CUSTOMER_LVL_ID'] = df_mid.get('customer_lvl_id', df_mid.get('CUSTOMER_LVL_ID'))
            df_mid['DISTR_CHANNEL_LVL_ID'] = df_mid.get('distr_channel_lvl_id', df_mid.get('DISTR_CHANNEL_LVL_ID'))
        df_reconciled = pd.concat([df_reconciled] + mid_reconciled_dfs, ignore_index=True)
    
    return df_reconciled

