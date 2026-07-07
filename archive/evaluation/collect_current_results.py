"""
Simple script to collect baseline data directly from containers
"""
import subprocess
import csv
import json

def collect_and_analyze(mode_name, container_name="ingestion-service"):
    """Collect evaluation data from container and analyze"""
    print(f"\n{'='*70}")
    print(f"Collecting {mode_name} results...")
    print(f"{'='*70}")
    
    # Docker copy command
    cmd = ["docker", "cp", f"{container_name}:/data/evaluation.csv", f"evaluation_{mode_name}.csv"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        
        # Analyze
        tp = fp = fn = tn = 0
        total = 0
        
        with open(f"evaluation_{mode_name}.csv", 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                actual = row['actual_anomaly'].lower() == 'true'
                predicted = row['predicted_anomaly'].lower() == 'true'
                
                if actual and predicted: tp += 1
                elif not actual and predicted: fp += 1
                elif actual and not predicted: fn += 1
                else: tn += 1
                total += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            "mode": mode_name,
            "total_samples": total,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
        
        print(f"✓ Collected {total} samples")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Collect current experiment (global_dynamic should be running)
    result = collect_and_analyze("global_dynamic")
    
    if result:
        # Save
        with open("global_dynamic_results.json", 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Results saved to global_dynamic_results.json")
