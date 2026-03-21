"""
Generate Precision-Recall Curve for Imbalanced Anomaly Detection
"""
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score, auc

def generate_pr_curve(csv_file, output_file="pr_curve.png"):
    """Generate Precision-Recall curve from evaluation data"""
    
    # Load data
    y_true = []
    y_scores = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            actual = 1 if row['actual_anomaly'].lower() == 'true' else 0
            error = float(row['error'])
            
            y_true.append(actual)
            y_scores.append(error)
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Compute average precision (area under PR curve)
    avg_precision = average_precision_score(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR curve (AP={avg_precision:.3f}, AUC={pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (Entity-Drift Mode)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    # Add baseline (random classifier for imbalanced data)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline (Random): {baseline:.3f}')
    plt.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curve saved to: {output_file}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("PRECISION-RECALL ANALYSIS")
    print("=" * 70)
    print(f"Average Precision (AP):  {avg_precision:.4f}")
    print(f"PR AUC:                  {pr_auc:.4f}")
    print(f"Baseline (Random):       {baseline:.4f}")
    print(f"Improvement over Random: {(avg_precision - baseline) / baseline * 100:.2f}%")
    
    # Find optimal threshold (max F1 score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
    
    print(f"\nOptimal Threshold:       {optimal_threshold:.6f}")
    print(f"  Precision at optimal:  {precision[optimal_idx]:.4f}")
    print(f"  Recall at optimal:     {recall[optimal_idx]:.4f}")
    print(f"  F1 at optimal:         {f1_scores[optimal_idx]:.4f}")
    print("=" * 70)
    
    return {
        'average_precision': avg_precision,
        'pr_auc': pr_auc,
        'baseline': baseline,
        'optimal_threshold': optimal_threshold,
        'optimal_precision': precision[optimal_idx],
        'optimal_recall': recall[optimal_idx],
        'optimal_f1': f1_scores[optimal_idx]
    }

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "evaluation_drift.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "pr_curve.png"
    
    if not Path(csv_file).exists():
        print(f"Error: File {csv_file} not found")
        sys.exit(1)
    
    generate_pr_curve(csv_file, output_file)
