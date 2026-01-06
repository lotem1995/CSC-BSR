import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import numpy as np

from VAE.model_evaluation import get_multicycle_scores


def find_optimal_cycles(model, id_dataset, ood_dataset, max_cycles=10, device="cuda"):
    # 1. Get scores for both datasets
    print("Processing ID data...")
    id_scores = get_multicycle_scores(model, id_dataset, max_cycles, device)

    print("Processing OOD data...")
    ood_scores = get_multicycle_scores(model, ood_dataset, max_cycles, device)

    # 2. Calculate means and plot
    cycles = range(1, max_cycles + 1)
    id_means = [np.mean(id_scores[c]) for c in cycles]
    ood_means = [np.mean(ood_scores[c]) for c in cycles]

    # Calculate the "Gap" ratio (higher is better)
    gap_ratios = [ood_m / id_m for ood_m, id_m in zip(ood_means, id_means)]

    # 3. Plotting
    plt.figure(figsize=(12, 5))

    # Subplot 1: Raw Scores
    plt.subplot(1, 2, 1)
    plt.plot(cycles, id_means, 'b-o', label='ID (Normal)')
    plt.plot(cycles, ood_means, 'r-o', label='OOD (Anomaly)')
    plt.xlabel('Cycle Number')
    plt.ylabel('Average Drift Score')
    plt.title('Drift Score vs. Cycles')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Separation Quality
    plt.subplot(1, 2, 2)
    plt.plot(cycles, gap_ratios, 'g-x')
    plt.xlabel('Cycle Number')
    plt.ylabel('Ratio (OOD / ID)')
    plt.title('Separation Quality (Higher is Better)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    best_cycle = np.argmax(gap_ratios) + 1
    print(f"Recommended Cycle Count: {best_cycle} (Ratio: {gap_ratios[best_cycle - 1]:.2f})")
    return best_cycle


# Usage (assuming you have your datasets ready):
# best_k = find_optimal_cycles(model, test_loader.dataset, ood_dataset)

def analyze_ood_threshold(scores_normal, scores_ood, plot=True):
    """
    Calculates the optimal OOD detection threshold using ROC analysis
    and optionally plots the distribution and ROC curve.

    Args:
        scores_normal (array): Consistency scores for In-Distribution data.
        scores_ood (array): Consistency scores for Out-of-Distribution data.
        plot (bool): Whether to generate the Histogram and ROC plots.

    Returns:
        float: The mathematically optimal threshold (Youden's J statistic).
    """
    # 1. Prepare Data for Scikit-Learn
    # Label 0 = Normal, Label 1 = OOD
    y_true = np.concatenate([np.zeros(len(scores_normal)), np.ones(len(scores_ood))])
    y_scores = np.concatenate([scores_normal, scores_ood])

    # 2. Compute ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 3. Find Optimal Threshold (Maximize TPR - FPR)
    # J = Sensitivity + Specificity - 1
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]

    # metrics at this specific threshold
    best_tpr = tpr[ix]
    best_fpr = fpr[ix]

    print(f"\n--- OOD THRESHOLD ANALYSIS ---")
    print(f"Optimal Threshold: {best_thresh:.6f}")
    print(f"AUC Score:         {roc_auc:.4f}")
    print(f"Recall (TPR):      {best_tpr * 100:.1f}%  (Percentage of OODs caught)")
    print(f"False Alarm (FPR): {best_fpr * 100:.1f}%  (Percentage of Normal flagged incorrectly)")

    if plot:
        # Plot 1: Histograms
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(scores_normal, bins=50, alpha=0.6, label='Normal', density=True, color='blue')
        plt.hist(scores_ood, bins=50, alpha=0.6, label='OOD (Anomalies)', density=True, color='red')
        plt.axvline(best_thresh, color='k', linestyle='--', label=f'Threshold: {best_thresh:.2f}')
        plt.title("Score Distributions")
        plt.xlabel("Reconstruction Error")
        plt.legend()

        # Plot 2: ROC Curve
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})', color='darkorange')
        plt.scatter(best_fpr, best_tpr, marker='o', color='black', label='Optimal Point')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return best_thresh