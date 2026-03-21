"""
Quick Baseline Runner - Simplified for global_dynamic and entity_dynamic
"""
import subprocess
import time
import csv
import json
from pathlib import Path

def run_baseline(mode, duration=120):
    """Run a single baseline experiment"""
    print(f"\n{'='*70}")
    print(f"Running {mode} baseline...")
    print(f"{'='*70}")
    
    # 1. Update docker-compose
    print(f"  [1/6] Updating threshold mode to {mode}...")
    compose_file = Path(__file__).parent.parent / "docker-compose.yml"
    with open(compose_file, 'r') as f:
        content = f.read()
    
    # Replace THRESHOLD_MODE
    import re
    content = re.sub(r'(THRESHOLD_MODE=)[a-z_]+', f'\\1{mode}', content)
    
    with open(compose_file, 'w') as f:
        f.write(content)
    print("  ✓ Updated")
    
    # 2. Stop services
    print("  [2/6] Stopping services...")
    subprocess.run(["docker-compose", "down", "-v"], 
                   capture_output=True, 
                   cwd=compose_file.parent)
    print("  ✓ Stopped")
    
    # 3. Start services
    print("  [3/6] Starting services...")
    result = subprocess.run(["docker-compose", "up", "-d"],
                           capture_output=True,
                           text=True,
                           cwd=compose_file.parent)
    
    if result.returncode != 0:
        print(f"  ✗ Failed: {result.stderr}")
        return None
   
    print("  ✓ Started")
    
    # 4. Wait for healthy
    print("  [4/6] Waiting for services to be healthy (30s)...")
    time.sleep(30)
    print("  ✓ Ready")
    
    # 5. Run experiment
    print(f"  [5/6] Running experiment for {duration} seconds...")
    time.sleep(duration)
    print("  ✓ Complete")
    
    # 6. Collect data 
    print("  [6/6] Collecting results...")
    time.sleep(5)  # Wait for buffer flush
    
    csv_file = Path(__file__).parent / f"evaluation_{mode}.csv"
    result = subprocess.run(
        ["docker", "exec", "ingestion-service", "cat", "/data/evaluation.csv"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    if result.returncode == 0 and result.stdout:
        with open(csv_file, 'w') as f:
            f.write(result.stdout)
        print(f"  ✓ Saved to {csv_file.name}")
    else:
        print(f"  ✗ No data collected: {result.stderr}")
        return None
    
    # Analyze
    try:
        tp = fp = fn = tn = 0
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                actual = row['actual_anomaly'].lower() == 'true'
                predicted = row['predicted_anomaly'].lower() == 'true'
                
                if actual and predicted: tp += 1
                elif not actual and predicted: fp += 1
                elif actual and not predicted: fn += 1
                else: tn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            "mode": mode,
            "total_samples": tp + fp + fn + tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
        
        print(f"\n  Results:")
        print(f"    Samples: {results['total_samples']}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1: {f1:.4f}")
        print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        
        return results
        
    except Exception as e:
        print(f"  ✗ Analysis failed: {e}")
        return None

if __name__ == "__main__":
    print("="*70)
    print("BASELINE COMPARISON RUNNER")
    print("="*70)
    
    modes = ["global_dynamic", "entity_dynamic"]
    all_results = []
    
    for mode in modes:
        result = run_baseline(mode, duration=120)
        if result:
            all_results.append(result)
        time.sleep(5)  # Pause between experiments
    
    # Save results
    if all_results:
        output_file = Path(__file__).parent / "baseline_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'='*70}")
        print(f"All results saved to: {output_file}")
        print(f"{'='*70}")
        
        # Print summary table
        print(f"\n{'='*70}")
        print("SUMMARY TABLE")
        print(f"{'='*70}")
        print(f"{'Mode':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Samples':>10}")
        print("-"*70)
        for r in all_results:
            print(f"{r['mode']:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1_score']:>10.4f} {r['total_samples']:>10}")
    else:
        print("\n✗ No results collected")
