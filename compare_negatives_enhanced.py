import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.plot_style import (
    FIGURE_SIZES, PRIMARY_COLORS, PLOT_PARAMS, MODEL_COLORS, HEATMAP_CMAP,
    create_figure, format_axis, save_figure, get_color_palette, setup_plotting_style
)

# Apply unified style
setup_plotting_style()

# Create output directory
output_dir = Path('plots/negatives_comparison')
output_dir.mkdir(parents=True, exist_ok=True)

# Load the metrics
df = pd.read_csv('results/metrics/binary_metrics.csv')

# Filter for BCE loss only and with_title=True for fair comparison
df_bce = df[(df['loss_type'] == 'BCE') & (df['with_title'] == True)].copy()

# Separate by number of added negatives
df_500 = df_bce[df_bce['nb_added_negs'] == 500]
df_0 = df_bce[df_bce['nb_added_negs'] == 0]

# Get unique models (excluding Ensemble)
models = sorted(df_500[df_500['model_name'] != 'Ensemble']['model_name'].unique())

# Metrics to compare
metrics_to_compare = ['f1', 'recall', 'precision', 'accuracy', 'roc_auc', 'AP', 'MCC']

# Helper function to abbreviate model names
def abbreviate_model(m):
    return m.replace('BiomedNLP-', '').replace('BiomedBERT-abs', 'BiomedBERT').replace('-ft', '-FT')

print("Creating standalone plots...")
print("="*80)

# ============================================================================
# PLOT 1: BAR PLOT - All models performance comparison for key metrics
# ============================================================================
print("1/10: Creating key metrics bar plot...")
fig, ax = create_figure(figsize='wide')
key_metrics = ['f1', 'roc_auc', 'AP']
x = np.arange(len(models))
width = 0.12

# Use unified color palette
colors_500 = [PRIMARY_COLORS['teal'], PRIMARY_COLORS['purple'], PRIMARY_COLORS['gold']]
colors_0 = [PRIMARY_COLORS['green'], PRIMARY_COLORS['red'], PRIMARY_COLORS['orange']]

for i, metric in enumerate(key_metrics):
    metric_data_500 = []
    metric_data_0 = []

    for model in models:
        vals_500 = df_500[df_500['model_name'] == model][metric]
        vals_0 = df_0[df_0['model_name'] == model][metric]
        metric_data_500.append(vals_500.mean())
        metric_data_0.append(vals_0.mean())

    ax.bar(x + i*width*2 - width*3, metric_data_500, width,
            label=f'{metric.upper()} (500 neg)', color=colors_500[i], alpha=0.85)
    ax.bar(x + i*width*2 - width*2, metric_data_0, width,
            label=f'{metric.upper()} (0 neg)', color=colors_0[i], alpha=0.85)

format_axis(ax,
            xlabel='Model',
            ylabel='Score',
            title='Key Metrics: F1, ROC-AUC, AP - 500 vs 0 Added Negatives',
            grid=True,
            grid_axis='y',
            legend=True,
            legend_kwargs={'loc': 'lower right', 'fontsize': 9, 'ncol': 2})
ax.set_xticks(x)
ax.set_xticklabels([abbreviate_model(m) for m in models], rotation=25, ha='right')
ax.set_ylim([0.8, 1.0])

save_figure(fig, output_dir / '01_key_metrics_barplot.png')
plt.close(fig)
print(f"   ✓ Saved: {output_dir / '01_key_metrics_barplot.png'}")

# ============================================================================
# PLOT 2: BOXPLOT - F1 Score Distribution Across Folds
# ============================================================================
print("2/10: Creating F1 score boxplot...")
fig, ax = create_figure(figsize='wide')
boxplot_data = []
positions = []
colors = []

pos = 0
for i, model in enumerate(models):
    # 500 negatives
    vals_500 = df_500[df_500['model_name'] == model]['f1'].values
    boxplot_data.append(vals_500)
    positions.append(pos)
    colors.append(PRIMARY_COLORS['blue'])
    pos += 1

    # 0 negatives
    vals_0 = df_0[df_0['model_name'] == model]['f1'].values
    boxplot_data.append(vals_0)
    positions.append(pos)
    colors.append(PRIMARY_COLORS['orange'])
    pos += 1.5

bp = ax.boxplot(boxplot_data, positions=positions, widths=0.8, patch_artist=True,
                  showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=5))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(PLOT_PARAMS['alpha'])

# Set x-axis labels
tick_positions = [(positions[i*2] + positions[i*2+1]) / 2 for i in range(len(models))]
ax.set_xticks(tick_positions)
ax.set_xticklabels([abbreviate_model(m) for m in models], rotation=25, ha='right')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=PRIMARY_COLORS['blue'], alpha=PLOT_PARAMS['alpha'], label='500 added neg'),
                   Patch(facecolor=PRIMARY_COLORS['orange'], alpha=PLOT_PARAMS['alpha'], label='0 added neg')]

format_axis(ax,
            ylabel='F1 Score',
            title='F1 Score Distribution Across Folds (5 folds per model)',
            grid=True,
            grid_axis='y',
            legend=False)
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

save_figure(fig, output_dir / '02_f1_boxplot.png')
plt.close(fig)
print(f"   ✓ Saved: {output_dir / '02_f1_boxplot.png'}")

# ============================================================================
# PLOT 3: BOXPLOT - ROC-AUC Distribution Across Folds
# ============================================================================
print("3/10: Creating ROC-AUC boxplot...")
fig, ax = create_figure(figsize='wide')
boxplot_data = []
positions = []
colors = []

pos = 0
for i, model in enumerate(models):
    # 500 negatives
    vals_500 = df_500[df_500['model_name'] == model]['roc_auc'].values
    boxplot_data.append(vals_500)
    positions.append(pos)
    colors.append(PRIMARY_COLORS['blue'])
    pos += 1

    # 0 negatives
    vals_0 = df_0[df_0['model_name'] == model]['roc_auc'].values
    boxplot_data.append(vals_0)
    positions.append(pos)
    colors.append(PRIMARY_COLORS['orange'])
    pos += 1.5

bp = ax.boxplot(boxplot_data, positions=positions, widths=0.8, patch_artist=True,
                  showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=5))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(PLOT_PARAMS['alpha'])

ax.set_xticks(tick_positions)
ax.set_xticklabels([abbreviate_model(m) for m in models], rotation=25, ha='right')

format_axis(ax,
            ylabel='ROC-AUC',
            title='ROC-AUC Distribution Across Folds (5 folds per model)',
            grid=True,
            grid_axis='y',
            legend=False)
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

save_figure(fig, output_dir / '03_roc_auc_boxplot.png')
plt.close(fig)
print(f"   ✓ Saved: {output_dir / '03_roc_auc_boxplot.png'}")

# ============================================================================
# PLOT 4: BOXPLOT - Average Precision (AP) Distribution
# ============================================================================
print("4/10: Creating AP boxplot...")
fig, ax = create_figure(figsize='wide')
boxplot_data = []
positions = []
colors = []

pos = 0
for i, model in enumerate(models):
    # 500 negatives
    vals_500 = df_500[df_500['model_name'] == model]['AP'].values
    boxplot_data.append(vals_500)
    positions.append(pos)
    colors.append(PRIMARY_COLORS['blue'])
    pos += 1

    # 0 negatives
    vals_0 = df_0[df_0['model_name'] == model]['AP'].values
    boxplot_data.append(vals_0)
    positions.append(pos)
    colors.append(PRIMARY_COLORS['orange'])
    pos += 1.5

bp = ax.boxplot(boxplot_data, positions=positions, widths=0.8, patch_artist=True,
                  showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=5))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(PLOT_PARAMS['alpha'])

ax.set_xticks(tick_positions)
ax.set_xticklabels([abbreviate_model(m) for m in models], rotation=25, ha='right')

format_axis(ax,
            ylabel='Average Precision',
            title='AP Distribution Across Folds (5 folds per model)',
            grid=True,
            grid_axis='y',
            legend=False)
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

save_figure(fig, output_dir / '04_ap_boxplot.png')
plt.close(fig)
print(f"   ✓ Saved: {output_dir / '04_ap_boxplot.png'}")

# ============================================================================
# PLOT 5: RADAR CHART - Performance Profile Comparison
# ============================================================================
print("5/10: Creating radar chart...")
fig = plt.figure(figsize=FIGURE_SIZES['square'])
ax = fig.add_subplot(111, projection='polar')

# Select one representative model for radar chart
selected_model = 'roberta-base'
radar_metrics = ['F1', 'Recall', 'Precision', 'Accuracy', 'ROC-AUC', 'AP']

# Get data for selected model
model_500 = df_500[df_500['model_name'] == selected_model][['f1', 'recall', 'precision', 'accuracy', 'roc_auc', 'AP']].mean()
model_0 = df_0[df_0['model_name'] == selected_model][['f1', 'recall', 'precision', 'accuracy', 'roc_auc', 'AP']].mean()

values_500 = model_500.values.tolist()
values_0 = model_0.values.tolist()

# Close the plot
values_500 += values_500[:1]
values_0 += values_0[:1]

# Compute angle for each metric
angles = [n / float(len(radar_metrics)) * 2 * pi for n in range(len(radar_metrics))]
angles += angles[:1]

# Plot
ax.plot(angles, values_500, 'o-', linewidth=PLOT_PARAMS['linewidth'],
         label='500 added neg', color=PRIMARY_COLORS['blue'])
ax.fill(angles, values_500, alpha=0.25, color=PRIMARY_COLORS['blue'])

ax.plot(angles, values_0, 'o-', linewidth=PLOT_PARAMS['linewidth'],
         label='0 added neg', color=PRIMARY_COLORS['orange'])
ax.fill(angles, values_0, alpha=0.25, color=PRIMARY_COLORS['orange'])

# Fix axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics)
ax.set_ylim(0.8, 1.0)
ax.set_title(f'Performance Profile: {selected_model}\nRadar Chart Comparison',
              fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

save_figure(fig, output_dir / '05_radar_chart.png')
plt.close(fig)
print(f"   ✓ Saved: {output_dir / '05_radar_chart.png'}")

# ============================================================================
# PLOT 6: IMPROVEMENT HEATMAP
# ============================================================================
print("6/10: Creating improvement heatmap...")
fig, ax = create_figure(figsize='medium')

# Calculate improvements for each model and metric
improvement_matrix = []
for model in models:
    model_improvements = []
    for metric in ['f1', 'recall', 'precision', 'roc_auc', 'AP', 'MCC']:
        mean_500 = df_500[df_500['model_name'] == model][metric].mean()
        mean_0 = df_0[df_0['model_name'] == model][metric].mean()
        improvement = ((mean_0 - mean_500) / mean_500) * 100  # Percentage improvement
        model_improvements.append(improvement)
    improvement_matrix.append(model_improvements)

improvement_df = pd.DataFrame(
    improvement_matrix,
    index=[abbreviate_model(m) for m in models],
    columns=['F1', 'Recall', 'Precision', 'ROC-AUC', 'AP', 'MCC']
)

sns.heatmap(improvement_df, annot=True, fmt='.2f', cmap=HEATMAP_CMAP, center=0,
            ax=ax, cbar_kws={'label': 'Improvement (%)'}, vmin=-5, vmax=10)
ax.set_title('Performance Improvement (%)\n0 neg vs 500 neg (Green = Better)',
              fontsize=13, fontweight='bold')
ax.set_xlabel('Metric', fontsize=11, fontweight='bold')
ax.set_ylabel('Model', fontsize=11, fontweight='bold')

save_figure(fig, output_dir / '06_improvement_heatmap.png')
plt.close(fig)
print(f"   ✓ Saved: {output_dir / '06_improvement_heatmap.png'}")

# ============================================================================
# PLOT 7: All metrics bar chart
# ============================================================================
print("7/10: Creating all metrics bar chart...")
fig, ax = create_figure(figsize='extra_wide')
x = np.arange(len(metrics_to_compare))
width = 0.08

for i, model in enumerate(models):
    means_500 = []
    means_0 = []
    for metric in metrics_to_compare:
        means_500.append(df_500[df_500['model_name'] == model][metric].mean())
        means_0.append(df_0[df_0['model_name'] == model][metric].mean())

    offset = (i - len(models)/2 + 0.5) * width * 2
    ax.bar(x + offset, means_500, width, label=f'{model.split("-")[0]} (500)', alpha=0.7)
    ax.bar(x + offset + width, means_0, width, label=f'{model.split("-")[0]} (0)', alpha=0.9)

format_axis(ax,
            xlabel='Metrics',
            ylabel='Score',
            title='All Metrics Performance Comparison - BCE Loss: 500 vs 0 Added Negatives',
            grid=True,
            grid_axis='y',
            legend=True,
            legend_kwargs={'bbox_to_anchor': (1.05, 1), 'loc': 'upper left', 'fontsize': 8, 'ncol': 2})
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in metrics_to_compare])

save_figure(fig, output_dir / '07_all_metrics_barplot.png')
plt.close(fig)
print(f"   ✓ Saved: {output_dir / '07_all_metrics_barplot.png'}")

# ============================================================================
# PLOT 8: MCC Boxplot (important metric)
# ============================================================================
print("8/10: Creating MCC boxplot...")
fig, ax = create_figure(figsize='wide')
boxplot_data = []
positions = []
colors = []

pos = 0
for i, model in enumerate(models):
    vals_500 = df_500[df_500['model_name'] == model]['MCC'].values
    boxplot_data.append(vals_500)
    positions.append(pos)
    colors.append(PRIMARY_COLORS['blue'])
    pos += 1

    vals_0 = df_0[df_0['model_name'] == model]['MCC'].values
    boxplot_data.append(vals_0)
    positions.append(pos)
    colors.append(PRIMARY_COLORS['orange'])
    pos += 1.5

bp = ax.boxplot(boxplot_data, positions=positions, widths=0.8, patch_artist=True,
                showmeans=True, meanprops=dict(marker='D', markerfacecolor='red', markersize=5))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(PLOT_PARAMS['alpha'])

ax.set_xticks(tick_positions)
ax.set_xticklabels([abbreviate_model(m) for m in models], rotation=25, ha='right')

format_axis(ax,
            ylabel='Matthews Correlation Coefficient',
            title='MCC Distribution Across Folds (Important for Imbalanced Data)',
            grid=True,
            grid_axis='y',
            legend=False)
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

save_figure(fig, output_dir / '08_mcc_boxplot.png')
plt.close(fig)
print(f"   ✓ Saved: {output_dir / '08_mcc_boxplot.png'}")

# ============================================================================
# PLOT 9: Recall vs Precision Scatter
# ============================================================================
print("9/10: Creating recall vs precision scatter plot...")
fig, ax = create_figure(figsize='medium')

for model in models:
    # 500 negatives
    recall_500 = df_500[df_500['model_name'] == model]['recall'].values
    precision_500 = df_500[df_500['model_name'] == model]['precision'].values
    ax.scatter(recall_500, precision_500, s=100, alpha=0.6, marker='o',
              label=f'{model.split("-")[0]} (500)')

    # 0 negatives
    recall_0 = df_0[df_0['model_name'] == model]['recall'].values
    precision_0 = df_0[df_0['model_name'] == model]['precision'].values
    ax.scatter(recall_0, precision_0, s=100, alpha=0.9, marker='s',
              label=f'{model.split("-")[0]} (0)')

ax.plot([0.8, 1.0], [0.8, 1.0], 'k--', alpha=0.3, label='Perfect Balance')

format_axis(ax,
            xlabel='Recall',
            ylabel='Precision',
            title='Recall vs Precision Trade-off\n(Circle=500 neg, Square=0 neg)',
            grid=True,
            legend=True,
            legend_kwargs={'bbox_to_anchor': (1.05, 1), 'loc': 'upper left', 'fontsize': 8})
ax.set_xlim([0.8, 1.0])
ax.set_ylim([0.8, 1.0])

save_figure(fig, output_dir / '09_recall_precision_scatter.png')
plt.close(fig)
print(f"   ✓ Saved: {output_dir / '09_recall_precision_scatter.png'}")

# ============================================================================
# PLOT 10: Average metrics comparison with error bars
# ============================================================================
print("10/10: Creating average metrics comparison...")
fig, ax = create_figure(figsize='wide')

avg_metrics_500 = []
avg_metrics_0 = []
std_metrics_500 = []
std_metrics_0 = []

for metric in metrics_to_compare:
    avg_500 = df_500[metric].mean()
    avg_0 = df_0[metric].mean()
    std_500 = df_500[metric].std()
    std_0 = df_0[metric].std()

    avg_metrics_500.append(avg_500)
    avg_metrics_0.append(avg_0)
    std_metrics_500.append(std_500)
    std_metrics_0.append(std_0)

x = np.arange(len(metrics_to_compare))
width = 0.35

bars1 = ax.bar(x - width/2, avg_metrics_500, width, yerr=std_metrics_500,
               label='500 added negatives', color=PRIMARY_COLORS['blue'],
               alpha=PLOT_PARAMS['alpha'], capsize=5)
bars2 = ax.bar(x + width/2, avg_metrics_0, width, yerr=std_metrics_0,
               label='0 added negatives', color=PRIMARY_COLORS['orange'],
               alpha=PLOT_PARAMS['alpha'], capsize=5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

format_axis(ax,
            xlabel='Metric',
            ylabel='Average Score',
            title='Average Performance Across All Models (with standard deviation)',
            grid=True,
            grid_axis='y',
            legend=True,
            legend_kwargs={'fontsize': 10})
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in metrics_to_compare])

save_figure(fig, output_dir / '10_average_metrics_comparison.png')
plt.close(fig)
print(f"   ✓ Saved: {output_dir / '10_average_metrics_comparison.png'}")

# Print summary
print("\n" + "="*80)
print("SUMMARY: BCE Loss with Title - 500 vs 0 Added Negatives")
print("="*80)

for model in models:
    print(f"\n{model}:")
    print("-" * 60)
    for metric in metrics_to_compare:
        mean_500 = df_500[df_500['model_name'] == model][metric].mean()
        std_500 = df_500[df_500['model_name'] == model][metric].std()
        mean_0 = df_0[df_0['model_name'] == model][metric].mean()
        std_0 = df_0[df_0['model_name'] == model][metric].std()
        diff = mean_0 - mean_500
        pct_change = (diff / mean_500) * 100 if mean_500 > 0 else 0
        sign = "+" if diff > 0 else ""
        print(f"  {metric:12s}: 500_neg={mean_500:.4f}±{std_500:.4f}  |  "
              f"0_neg={mean_0:.4f}±{std_0:.4f}  |  diff={sign}{diff:.4f} ({sign}{pct_change:.2f}%)")

print("\n" + "="*80)
print(f"All plots saved to: {output_dir}")
print("="*80)
