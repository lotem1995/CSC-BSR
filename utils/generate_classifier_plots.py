import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# Use boardstate-light style for documentation
style_path = Path(__file__).parent / 'styles' / 'boardstate-light.mplstyle'
if style_path.exists():
    plt.style.use(str(style_path))
else:
    print(f"Warning: Style file not found at {style_path}, using default style")

# Create data from the CSV
data = {
    'Prediction': ['knn', 'knn', 'mahalanobis', 'knn', 'mahalanobis', 'knn', 'mahalanobis', 'mahalanobis', 'softmax', 'softmax', 'softmax', 'softmax', 'knn', 'mahalanobis', 'softmax'],
    'OOD_Method': ['binary_ood_model', 'softmax', 'binary_ood_model', 'ensemble', 'softmax', 'knn', 'ensemble', 'knn', 'binary_ood_model', 'softmax', 'ensemble', 'knn', 'mahalanobis', 'mahalanobis', 'mahalanobis'],
    'Overall Acc': [0.9541, 0.9502, 0.9448, 0.9429, 0.9418, 0.9384, 0.9338, 0.9305, 0.8640, 0.8612, 0.8527, 0.8511, 0.7459, 0.7459, 0.7430],
    'OOD Recall': [0.9910, 0.4000, 0.9910, 0.0776, 0.4000, 0.1015, 0.0776, 0.1015, 0.9910, 0.4000, 0.0776, 0.1015, 0.9403, 0.9403, 0.9403],
    'False Rejection': [0.0172, 0.0047, 0.0172, 0.0012, 0.0047, 0.0133, 0.0012, 0.0133, 0.0172, 0.0047, 0.0012, 0.0133, 0.2587, 0.2587, 0.2587],
    'Clean Acc': [0.9697, 0.9709, 0.9600, 0.9695, 0.9622, 0.9759, 0.9600, 0.9676, 0.8753, 0.8789, 0.8764, 0.8848, 0.9986, 0.9986, 0.9946]
}

df = pd.DataFrame(data)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# Define color palette - using boardstate-light palette
colors_pred = {'knn': '#5B46D6', 'softmax': '#EB5757', 'mahalanobis': '#F2994A'}
colors_ood = {'binary_ood_model': '#219653', 'softmax': '#EB5757', 'knn': '#2D9CDB', 
              'ensemble': '#F2C94C', 'mahalanobis': '#9B51E0'}

# 1. Overall Accuracy vs OOD Recall (scatter plot - the key trade-off)
ax1 = plt.subplot(2, 3, 1)
for pred_method in df['Prediction'].unique():
    mask = df['Prediction'] == pred_method
    data_subset = df[mask]
    ax1.scatter(data_subset['OOD Recall'], data_subset['Overall Acc'], 
               s=200, alpha=0.7, label=pred_method, color=colors_pred[pred_method])

# Highlight the winner
winner = df[(df['Prediction'] == 'knn') & (df['OOD_Method'] == 'binary_ood_model')]
ax1.scatter(winner['OOD Recall'].values, winner['Overall Acc'].values, 
           s=500, color='#219653', marker='*', edgecolor='#0E1220', linewidth=2, zorder=5, label='Winner')

ax1.set_xlabel('OOD Recall', fontsize=11, fontweight='bold')
ax1.set_ylabel('Overall Accuracy', fontsize=11, fontweight='bold')
ax1.set_title('Key Trade-off: OOD Recall vs Overall Accuracy', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(0.7, 1.0)

# 2. Overall Accuracy by Prediction Method
ax2 = plt.subplot(2, 3, 2)
methods = df.groupby('Prediction')['Overall Acc'].apply(list).to_dict()
pred_methods = list(methods.keys())
positions = np.arange(len(pred_methods))
boxplot_data = [methods[m] for m in pred_methods]
bp = ax2.boxplot(boxplot_data, tick_labels=pred_methods, patch_artist=True)
for patch, method in zip(bp['boxes'], pred_methods):
    patch.set_facecolor(colors_pred[method])
    patch.set_alpha(0.7)
ax2.set_ylabel('Overall Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Accuracy Distribution by Prediction Method', fontsize=12, fontweight='bold')
ax2.set_ylim(0.7, 1.0)

# 3. OOD Recall by OOD Method
ax3 = plt.subplot(2, 3, 3)
ood_methods = df['OOD_Method'].unique()
ood_recall_by_method = {}
for method in ood_methods:
    ood_recall_by_method[method] = df[df['OOD_Method'] == method]['OOD Recall'].mean()

sorted_methods = sorted(ood_recall_by_method.items(), key=lambda x: x[1], reverse=True)
methods_sorted, recalls = zip(*sorted_methods)
bars = ax3.barh(methods_sorted, recalls, color=[colors_ood.get(m, '#5B667A') for m in methods_sorted], alpha=0.8)
ax3.set_xlabel('Average OOD Recall', fontsize=11, fontweight='bold')
ax3.set_title('OOD Detection Performance by Method', fontsize=12, fontweight='bold')
for i, (bar, val) in enumerate(zip(bars, recalls)):
    ax3.text(val + 0.02, i, f'{val:.2%}', va='center', fontsize=10, fontweight='bold')

# 4. False Rejection Rate (lower is better)
ax4 = plt.subplot(2, 3, 4)
false_reject_by_method = {}
for method in ood_methods:
    false_reject_by_method[method] = df[df['OOD_Method'] == method]['False Rejection'].mean()

sorted_fr = sorted(false_reject_by_method.items(), key=lambda x: x[1])
methods_fr, fr_rates = zip(*sorted_fr)
bars = ax4.barh(methods_fr, fr_rates, color=[colors_ood.get(m, '#5B667A') for m in methods_fr], alpha=0.8)
ax4.set_xlabel('Average False Rejection Rate', fontsize=11, fontweight='bold')
ax4.set_title('False Rejection Rate by OOD Method (Lower is Better)', fontsize=12, fontweight='bold')
for i, (bar, val) in enumerate(zip(bars, fr_rates)):
    ax4.text(val + 0.002, i, f'{val:.2%}', va='center', fontsize=10, fontweight='bold')

# 5. Clean Accuracy (on valid, non-OOD tiles)
ax5 = plt.subplot(2, 3, 5)
top_configs = df.nlargest(10, 'Clean Acc')
config_labels = [f"{row['Prediction']}\n+{row['OOD_Method'][:4]}" for _, row in top_configs.iterrows()]
colors_bars = [colors_pred[row['Prediction']] for _, row in top_configs.iterrows()]
bars = ax5.barh(config_labels, top_configs['Clean Acc'].values, color=colors_bars, alpha=0.8)
ax5.set_xlabel('Clean Accuracy (non-OOD samples)', fontsize=11, fontweight='bold')
ax5.set_title('Top 10 Configurations: Clean Accuracy', fontsize=12, fontweight='bold')
ax5.set_xlim(0.85, 1.0)
for i, (bar, val) in enumerate(zip(bars, top_configs['Clean Acc'].values)):
    ax5.text(val - 0.01, i, f'{val:.2%}', va='center', ha='right', fontsize=9, fontweight='bold', color='#141824')

# 6. Performance Heatmap (Top configurations)
ax6 = plt.subplot(2, 3, 6)
top_8 = df.nlargest(8, 'Overall Acc').copy()
top_8['Config'] = top_8.apply(lambda x: f"{x['Prediction']}\n+{x['OOD_Method'][:6]}", axis=1)
metrics = ['Overall Acc', 'OOD Recall', 'Clean Acc']
metric_data = top_8[metrics].values
norm_data = (metric_data - metric_data.min(axis=0)) / (metric_data.max(axis=0) - metric_data.min(axis=0))

im = ax6.imshow(norm_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax6.set_xticks(np.arange(len(metrics)))
ax6.set_yticks(np.arange(len(top_8)))
ax6.set_xticklabels(metrics, fontsize=10)
ax6.set_yticklabels(top_8['Config'].values, fontsize=9)
ax6.set_title('Top 8 Configurations: Normalized Metrics', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(len(top_8)):
    for j in range(len(metrics)):
        text = ax6.text(j, i, f'{metric_data[i, j]:.3f}',
                       ha="center", va="center", color="#141824", fontsize=8, fontweight='bold')

plt.colorbar(im, ax=ax6, label='Normalized Score')

plt.tight_layout()

# Save to docs/assets with relative path handling
output_path = Path(__file__).parent.parent / 'docs' / 'assets' / 'classifier_performance.png'
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path.relative_to(Path(__file__).parent.parent)}")

# Print summary statistics
print("\n" + "="*70)
print("CLASSIFIER PERFORMANCE SUMMARY")
print("="*70)

print("\n📊 Best Overall Configuration:")
best = df.loc[df['Overall Acc'].idxmax()]
print(f"  Prediction Method: {best['Prediction'].upper()}")
print(f"  OOD Method: {best['OOD_Method'].upper()}")
print(f"  Overall Accuracy: {best['Overall Acc']:.2%}")
print(f"  OOD Recall: {best['OOD Recall']:.2%}")
print(f"  False Rejection: {best['False Rejection']:.2%}")
print(f"  Clean Accuracy: {best['Clean Acc']:.2%}")

print("\n📈 Top 5 Configurations by Overall Accuracy:")
for idx, (_, row) in enumerate(df.nlargest(5, 'Overall Acc').iterrows(), 1):
    print(f"  {idx}. {row['Prediction'].upper()} + {row['OOD_Method'].upper()}")
    print(f"     Acc: {row['Overall Acc']:.2%} | OOD Recall: {row['OOD Recall']:.2%} | Clean: {row['Clean Acc']:.2%}")

print("\n⚠️  Worst Configurations (by Overall Accuracy):")
for idx, (_, row) in enumerate(df.nsmallest(3, 'Overall Acc').iterrows(), 1):
    print(f"  {idx}. {row['Prediction'].upper()} + {row['OOD_Method'].upper()}: {row['Overall Acc']:.2%}")

print("\n" + "="*70)
