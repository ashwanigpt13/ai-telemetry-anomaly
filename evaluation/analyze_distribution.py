"""Analyze error distribution in evaluation data"""
import csv
import statistics

data_path = "../data/evaluation/evaluation.csv"

normal_errors = []
anomaly_errors = []

with open(data_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        error = float(row['error'])
        if row['actual_anomaly'] == 'True':
            anomaly_errors.append(error)
        else:
            normal_errors.append(error)

print(f"Normal samples: n={len(normal_errors)}")
if normal_errors:
    print(f"  min={min(normal_errors):.3f}, max={max(normal_errors):.3f}")
    print(f"  mean={statistics.mean(normal_errors):.3f}, std={statistics.stdev(normal_errors):.3f}")

print(f"\nAnomaly samples: n={len(anomaly_errors)}")
if anomaly_errors:
    print(f"  min={min(anomaly_errors):.3f}, max={max(anomaly_errors):.3f}")
    print(f"  mean={statistics.mean(anomaly_errors):.3f}, std={statistics.stdev(anomaly_errors):.3f}")

# Find optimal threshold
if normal_errors and anomaly_errors:
    normal_mean = statistics.mean(normal_errors)
    normal_std = statistics.stdev(normal_errors)
    anomaly_mean = statistics.mean(anomaly_errors)
    
    # Try different thresholds
    print("\nThreshold analysis:")
    for threshold in [1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]:
        tp = sum(1 for e in anomaly_errors if e > threshold)
        fn = sum(1 for e in anomaly_errors if e <= threshold)
        fp = sum(1 for e in normal_errors if e > threshold)
        tn = sum(1 for e in normal_errors if e <= threshold)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  T={threshold:.2f}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (TP={tp}, FP={fp}, FN={fn})")
