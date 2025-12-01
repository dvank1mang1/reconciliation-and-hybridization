"""
results visualization

shows forecast results in tables and graphs
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hybridization import hybridization
from test_hybridization import generate_reconciled_forecast_data


def visualize_forecast_comparison(df_hybrid):
    """compare ts, ml, and hybrid forecasts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a1a')
    
    sample = df_hybrid.head(30).copy()
    sample['index'] = range(len(sample))
    
    ax1 = axes[0, 0]
    ax1.plot(sample['index'], sample['TS_FORECAST_VALUE'], 'o-', 
            label='ts forecast', color='#4CAF50', linewidth=2)
    ax1.plot(sample['index'], sample['ML_FORECAST_VALUE'], 's-',
            label='ml forecast', color='#2196F3', linewidth=2)
    ax1.plot(sample['index'], sample['HYBRID_FORECAST_VALUE'], '^-',
            label='hybrid forecast', color='#FF9800', linewidth=2, markersize=8)
    ax1.set_xlabel('sample index', color='white')
    ax1.set_ylabel('forecast value', color='white')
    ax1.set_title('forecast comparison', color='white', fontsize=12)
    ax1.legend(facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
    ax1.set_facecolor('#2a2a2a')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['right'].set_color('white')
    ax1.grid(True, alpha=0.2, color='white')
    
    ax2 = axes[0, 1]
    source_counts = df_hybrid['FORECAST_SOURCE'].value_counts()
    colors = {'ml': '#2196F3', 'ts': '#4CAF50', 'ensemble': '#FF9800'}
    bars = ax2.bar(source_counts.index, source_counts.values,
                   color=[colors.get(x, '#666') for x in source_counts.index])
    ax2.set_xlabel('forecast source', color='white')
    ax2.set_ylabel('count', color='white')
    ax2.set_title('forecast source distribution', color='white', fontsize=12)
    ax2.set_facecolor('#2a2a2a')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['right'].set_color('white')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', color='white')
    
    ax3 = axes[1, 0]
    for source in ['ml', 'ts', 'ensemble']:
        subset = df_hybrid[df_hybrid['FORECAST_SOURCE'] == source]['HYBRID_FORECAST_VALUE'].dropna()
        if len(subset) > 0:
            ax3.hist(subset, alpha=0.5, label=source, bins=20,
                    color=colors.get(source, '#666'))
    ax3.set_xlabel('hybrid forecast value', color='white')
    ax3.set_ylabel('frequency', color='white')
    ax3.set_title('hybrid forecast distribution by source', color='white', fontsize=12)
    ax3.legend(facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
    ax3.set_facecolor('#2a2a2a')
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('white')
    ax3.spines['top'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.spines['right'].set_color('white')
    ax3.grid(True, alpha=0.2, color='white')
    
    ax4 = axes[1, 1]
    ensemble_only = df_hybrid[df_hybrid['FORECAST_SOURCE'] == 'ensemble'].copy()
    if len(ensemble_only) > 0:
        ensemble_only = ensemble_only.head(20)
        x = range(len(ensemble_only))
        width = 0.35
        ax4.bar([i - width/2 for i in x], ensemble_only['TS_FORECAST_VALUE'], 
               width, label='ts', color='#4CAF50', alpha=0.7)
        ax4.bar([i + width/2 for i in x], ensemble_only['ML_FORECAST_VALUE'],
               width, label='ml', color='#2196F3', alpha=0.7)
        ax4.plot(x, ensemble_only['HYBRID_FORECAST_VALUE'], 
                'o-', label='hybrid (avg)', color='#FF9800', linewidth=2, markersize=8)
        ax4.set_xlabel('ensemble cases', color='white')
        ax4.set_ylabel('forecast value', color='white')
        ax4.set_title('ensemble averaging verification', color='white', fontsize=12)
        ax4.legend(facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
        ax4.set_facecolor('#2a2a2a')
        ax4.tick_params(colors='white')
        ax4.spines['bottom'].set_color('white')
        ax4.spines['top'].set_color('white')
        ax4.spines['left'].set_color('white')
        ax4.spines['right'].set_color('white')
        ax4.grid(True, alpha=0.2, color='white')
    
    plt.tight_layout()
    return fig


def create_results_table(df_hybrid):
    """create formatted results table"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    available_cols = [col for col in ['PRODUCT_LVL_ID', 'SEGMENT_NAME', 'DEMAND_TYPE', 
                   'TS_FORECAST_VALUE', 'ML_FORECAST_VALUE',
                   'HYBRID_FORECAST_VALUE', 'FORECAST_SOURCE'] if col in df_hybrid.columns]
    
    if not available_cols:
        available_cols = list(df_hybrid.columns[:7])
    
    sample_data = df_hybrid[available_cols].head(15).copy()
    display_cols = available_cols
    
    for col in ['TS_FORECAST_VALUE', 'ML_FORECAST_VALUE', 'HYBRID_FORECAST_VALUE']:
        if col in sample_data.columns:
            sample_data[col] = sample_data[col].round(2)
    
    cell_text = []
    for idx, row in sample_data.iterrows():
        cell_text.append([str(val) if pd.notna(val) else 'nan' for val in row])
    
    if len(cell_text) == 0:
        cell_text = [['no data'] * len(display_cols)]
    
    table = ax.table(cellText=cell_text,
                    colLabels=display_cols,
                    cellLoc='center',
                    loc='center',
                    colColours=['#444444'] * len(display_cols))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(display_cols)):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#FF9800')
    
    for i in range(1, len(cell_text) + 1):
        for j in range(len(display_cols)):
            cell = table[(i, j)]
            cell.set_text_props(color='white')
            if i % 2 == 0:
                cell.set_facecolor('#2a2a2a')
            else:
                cell.set_facecolor('#1a1a1a')
            cell.set_edgecolor('white')
    
    ax.set_title('hybrid forecast results sample', 
                fontsize=14, color='white', pad=20, weight='bold')
    
    plt.tight_layout()
    return fig


def create_statistics_table(df_hybrid):
    """create statistics summary table"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    stats_data = []
    for source in ['ml', 'ts', 'ensemble']:
        subset = df_hybrid[df_hybrid['FORECAST_SOURCE'] == source]['HYBRID_FORECAST_VALUE']
        if len(subset) > 0:
            stats_data.append([
                source,
                len(subset),
                f"{subset.mean():.2f}",
                f"{subset.std():.2f}",
                f"{subset.min():.2f}",
                f"{subset.max():.2f}"
            ])
    
    col_labels = ['source', 'count', 'mean', 'std', 'min', 'max']
    
    if len(stats_data) == 0:
        stats_data = [['no data', '0', '0', '0', '0', '0']]
    
    table = ax.table(cellText=stats_data,
                    colLabels=col_labels,
                    cellLoc='center',
                    loc='center',
                    colColours=['#444444'] * len(col_labels))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#FF9800')
    
    colors = {'ml': '#2196F3', 'ts': '#4CAF50', 'ensemble': '#FF9800'}
    for i in range(1, len(stats_data) + 1):
        source = stats_data[i-1][0]
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            cell.set_text_props(color='white')
            cell.set_facecolor(colors.get(source, '#2a2a2a'))
            cell.set_edgecolor('white')
            if j == 0:
                cell.set_text_props(weight='bold', color='white')
    
    ax.set_title('forecast statistics by source', 
                fontsize=14, color='white', pad=20, weight='bold')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os
    os.makedirs('../visualizations', exist_ok=True)
    
    print("generating test data")
    df_reconciled = generate_reconciled_forecast_data(num_products=5, num_locations=3)
    
    print("running hybridization")
    df_hybrid = hybridization(df_reconciled)
    
    print("creating visualizations")
    
    fig1 = visualize_forecast_comparison(df_hybrid)
    fig1.savefig('../visualizations/results_comparison.png', dpi=150, facecolor='#1a1a1a')
    print("saved visualizations/results_comparison.png")
    
    fig2 = create_results_table(df_hybrid)
    fig2.savefig('../visualizations/results_table.png', dpi=150, facecolor='#1a1a1a')
    print("saved visualizations/results_table.png")
    
    fig3 = create_statistics_table(df_hybrid)
    fig3.savefig('../visualizations/results_statistics.png', dpi=150, facecolor='#1a1a1a')
    print("saved visualizations/results_statistics.png")
    
    print("\nall result visualizations saved")

