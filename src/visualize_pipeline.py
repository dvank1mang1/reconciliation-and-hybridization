"""
pipeline visualization

shows reconciliation and hybridization steps
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_pipeline():
    """draw pipeline flowchart"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    box_height = 0.8
    box_width = 2.5
    
    steps = [
        (1, 8, 'customer data\nloading', 'purple'),
        (4, 8, 'input data\npreprocessing', 'purple'),
        (4, 6, 'ml models\ntraining', 'blue'),
        (4, 4, 'time series\nmodel training', 'blue'),
        (7, 6, 'ml forecasts\ncalculation', 'blue'),
        (7, 4, 'time series\nforecasts calculation', 'blue'),
        (10, 5, 'forecast\nhybridization', 'orange'),
        (1, 2, 'forecast\ndisaggregation', 'green'),
        (3.5, 2, 'forecast\ndisaccumulation', 'green'),
        (6, 2, 'auto-\ncorrection', 'green'),
        (8.5, 2, 'forecast output\ngeneration', 'green'),
        (11, 2, 'alerts\ncalculation', 'green'),
        (11, 0.5, 'forecast transfer\nto downstream', 'green')
    ]
    
    for x, y, label, color in steps:
        rect = patches.FancyBboxPatch(
            (x, y), box_width, box_height,
            boxstyle='round,pad=0.1',
            edgecolor='white',
            facecolor=color,
            alpha=0.7,
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + box_width/2, y + box_height/2, label,
                ha='center', va='center',
                color='white', fontsize=9, weight='bold')
    
    arrows = [
        (1 + box_width, 8.4, 4, 8.4),
        (4 + box_width/2, 8, 4 + box_width/2, 6.8),
        (4 + box_width/2, 8, 4 + box_width/2, 4.8),
        (4 + box_width, 6.4, 7, 6.4),
        (4 + box_width, 4.4, 7, 4.4),
        (7 + box_width, 6.4, 10, 5.4),
        (7 + box_width, 4.4, 10, 5.4),
        (10 + box_width/2, 5, 10 + box_width/2, 2.8),
        (1.5, 2.8, 1.5, 2.8),
        (3, 2.4, 3.5, 2.4),
        (6, 2.4, 6.5, 2.4),
        (8.5, 2.4, 9, 2.4),
        (10.5, 2.4, 11, 2.4),
        (11 + box_width/2, 2, 11 + box_width/2, 1.3)
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='white',
                                 lw=2, alpha=0.6))
    
    ax.text(10 + box_width/2, 9, 'reconciliation and hybridization steps',
           ha='center', fontsize=14, color='white', weight='bold',
           bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    
    plt.title('e2e demand forecasting pipeline', 
             fontsize=16, color='white', pad=20)
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    plt.tight_layout()
    return fig


def visualize_reconciliation_flow():
    """draw reconciliation step details"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    boxes = [
        (1, 4, 'ts_forecast\n(time series)', 'blue'),
        (1, 2, 'ml_forecast\n(machine learning)', 'blue'),
        (1, 0.5, 'ts_segments\n(segment names)', 'blue'),
        (5, 3, 'reconciliation\nmodule', 'orange'),
        (9, 3, 'reconciled_forecast\n(same granularity)', 'green')
    ]
    
    for x, y, label, color in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), 2, 0.8,
            boxstyle='round,pad=0.1',
            edgecolor='white',
            facecolor=color,
            alpha=0.7,
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + 1, y + 0.4, label,
                ha='center', va='center',
                color='white', fontsize=9, weight='bold')
    
    arrows = [
        (3, 4.4, 5, 3.6),
        (3, 2.4, 5, 3.2),
        (3, 0.9, 5, 3),
        (7, 3.4, 9, 3.4)
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='white',
                                 lw=2, alpha=0.6))
    
    plt.title('reconciliation step', fontsize=14, color='white', pad=20)
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    plt.tight_layout()
    return fig


def visualize_hybridization_flow():
    """draw hybridization step details"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    boxes = [
        (1, 3, 'reconciled_forecast\n(ts + ml)', 'green'),
        (5, 4.5, 'rule 1\nml forecast', 'blue'),
        (5, 3, 'rule 2\nts forecast', 'blue'),
        (5, 1.5, 'rule 3\nensemble', 'blue'),
        (9, 3, 'hybrid_forecast\n(merged)', 'orange')
    ]
    
    for x, y, label, color in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), 2, 0.8,
            boxstyle='round,pad=0.1',
            edgecolor='white',
            facecolor=color,
            alpha=0.7,
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + 1, y + 0.4, label,
                ha='center', va='center',
                color='white', fontsize=9, weight='bold')
    
    arrows = [
        (3, 3.4, 5, 4.7),
        (3, 3.4, 5, 3.4),
        (3, 3.4, 5, 1.9),
        (7, 4.9, 9, 3.6),
        (7, 3.4, 9, 3.4),
        (7, 1.9, 9, 3.2)
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='white',
                                 lw=2, alpha=0.6))
    
    ax.text(5, 5.5, 'promo/short/new', ha='left', fontsize=8, color='white')
    ax.text(5, 3.9, 'retired/low volume', ha='left', fontsize=8, color='white')
    ax.text(5, 2.4, 'everything else', ha='left', fontsize=8, color='white')
    
    plt.title('hybridization step', fontsize=14, color='white', pad=20)
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os
    os.makedirs('../visualizations', exist_ok=True)
    
    fig1 = visualize_pipeline()
    fig1.savefig('../visualizations/pipeline_flow.png', dpi=150, facecolor='#1a1a1a')
    print("saved visualizations/pipeline_flow.png")
    
    fig2 = visualize_reconciliation_flow()
    fig2.savefig('../visualizations/reconciliation_flow.png', dpi=150, facecolor='#1a1a1a')
    print("saved visualizations/reconciliation_flow.png")
    
    fig3 = visualize_hybridization_flow()
    fig3.savefig('../visualizations/hybridization_flow.png', dpi=150, facecolor='#1a1a1a')
    print("saved visualizations/hybridization_flow.png")
    
    print("\nall diagrams saved")

