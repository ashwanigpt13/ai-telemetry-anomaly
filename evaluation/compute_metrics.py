import csv
import sys

csv_file = sys.argv[1]
mode_name = sys.argv[2]

tp = fp = fn = tn = 0
latencies = []

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        actual = row['actual_anomaly'].lower() == 'true'
        predicted = row['predicted_anomaly'].lower() == 'true'
        
        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif actual and not predicted:
            fn += 1
        else:
            tn += 1
        
        latencies.append(float(row['latency_ms']))

total = tp + fp + fn + tn
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
avg_latency = sum(latencies) / len(latencies) if latencies else 0

# Write to output file
output_file = f"c:\\ai-telemetry-anomaly\\evaluation\\metrics_{mode_name}.txt"
with open(output_file, 'w') as f:
    f.write(f"{mode_name.upper()} RESULTS:\n")
    f.write(f"Total samples: {total}\n")
    f.write(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Average Latency: {avg_latency:.2f} ms\n")

print(f"Results written to {output_file}")
