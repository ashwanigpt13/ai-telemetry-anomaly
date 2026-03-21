"""
Comprehensive Baseline Comparison Experiment Runner

Runs experiments for all threshold modes and generates comparison table.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Import metrics module
sys.path.insert(0, str(Path(__file__).parent))
from metrics import load_evaluation_data, compute_classification_metrics

# Available threshold modes
THRESHOLD_MODES = [
    "global_static",
    "global_dynamic", 
    "entity_dynamic",
    "entity_drift"
]

def run_comprehensive_comparison(duration=300, output_dir="results"):
    """Run all baseline comparisons"""
    
    results = []
    
    print("=" * 80)
    print("COMPREHENSIVE BASELINE COMPARISON")
    print("=" * 80)
    print(f"Duration: {duration} seconds per experiment")
    print(f"Modes: {', '.join(THRESHOLD_MODES)}")
    print()
    
    for mode in THRESHOLD_MODES:
        print(f"\n{'='*80}")
        print(f"Running experiment: {mode}")
        print(f"{'='*80}")
        
        # Clear previous data
        print("  [1/6] Clearing previous evaluation data...")
        subprocess.run(["docker-compose", "down", "-v"], 
                      capture_output=True, cwd=Path(__file__).parent.parent)
        
        # Update threshold mode in docker-compose.yml
        print(f"  [2/6] Setting threshold mode to: {mode}")
        update_threshold_mode(mode)
        
        # Start services
        print("  [3/6] Starting services...")
        result = subprocess.run(
            ["docker-compose", "up", "-d"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode != 0:
            print(f"  ❌ Failed to start services: {result.stderr}")
            results.append({
                "mode": mode,
                "error": "Failed to start services"
            })
            continue
        
        # Wait for services to be healthy
        print("  [4/6] Waiting for services to be healthy...")
        if not wait_for_services(timeout=60):
            print(f"  ❌ Services not healthy")
            results.append({
                "mode": mode,
                "error": "Services not healthy"
            })
            subprocess.run(["docker-compose", "down"], 
                          capture_output=True, cwd=Path(__file__).parent.parent)
            continue
        
        # Run experiment
        print(f"  [5/6] Running {duration}-second experiment...")
        time.sleep(duration)
        
        # Wait for data flush
        print("  [6/6] Waiting for data flush...")
        time.sleep(10)
        
        # Collect results
        print("  Collecting evaluation data...")
        metrics = collect_results(mode)
        
        if metrics:
            results.append(metrics)
            print(f"  ✅ {mode}: Precision={metrics['classification']['precision']:.4f}, "
                  f"Recall={metrics['classification']['recall']:.4f}, "
                  f"F1={metrics['classification']['f1_score']:.4f}")
        else:
            print(f"  ❌ Failed to collect results for {mode}")
            results.append({
                "mode": mode,
                "error": "Failed to collect results"
            })
        
        # Stop services
        subprocess.run(["docker-compose", "down"], 
                      capture_output=True, cwd=Path(__file__).parent.parent)
        
        # Small delay between experiments
        time.sleep(5)
    
    # Save results
    output_path = Path(output_dir) / "baseline_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")
    
    # Print comparison table
    print_comparison_table(results)
    
    return results

def update_threshold_mode(mode):
    """Update THRESHOLD_MODE in docker-compose.yml"""
    compose_file = Path(__file__).parent.parent / "docker-compose.yml"
    
    with open(compose_file, 'r') as f:
        content = f.read()
    
    # Replace THRESHOLD_MODE value
    import re
    pattern = r'(THRESHOLD_MODE=)[a-z_]+'
    replacement = f'\\1{mode}'
    content = re.sub(pattern, replacement, content)
    
    with open(compose_file, 'w') as f:
        f.write(content)

def wait_for_services(timeout=60):
    """Wait for services to be healthy"""
    start = time.time()
    
    while time.time() - start < timeout:
        result = subprocess.run(
            ["docker-compose", "ps"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if "unhealthy" in result.stdout.lower():
            time.sleep(5)
            continue
        
        # Check if key services are running
        if all(svc in result.stdout for svc in ["ingestion", "model", "threshold"]):
            time.sleep(10)  # Extra stabilization time
            return True
        
        time.sleep(5)
    
    return False

def collect_results(mode):
    """Collect and analyze results for a mode"""
    try:
        # Copy evaluation data from container
        result = subprocess.run(
            ["docker", "cp", "ingestion-service:/data/evaluation.csv", 
             f"evaluation_{mode}.csv"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode != 0:
            print(f"    Warning: Failed to copy evaluation data: {result.stderr}")
            return None
        
        # Load and compute metrics
        csv_file = Path(__file__).parent / f"evaluation_{mode}.csv"
        
        if not csv_file.exists() or csv_file.stat().st_size == 0:
            print(f"    Warning: Evaluation file is empty or missing")
            return None
        
        df = load_evaluation_data(str(csv_file))
        
        if df is None or len(df) == 0:
            print(f"    Warning: No data in evaluation file")
            return None
        
        metrics = compute_classification_metrics(df)
        
        return {
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "classification": metrics
        }
        
    except Exception as e:
        print(f"    Error collecting results: {e}")
        return None

def print_comparison_table(results):
    """Print formatted comparison table"""
    
    print("\nBASELINE COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Method':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'TP':>8} {'FP':>8} {'FN':>8}")
    print("-" * 90)
    
    for result in results:
        if "error" in result:
            print(f"{result['mode']:<20} {'ERROR':>12} {'---':>12} {'---':>12} {'---':>8} {'---':>8} {'---':>8}")
        else:
            cls = result['classification']
            print(f"{result['mode']:<20} "
                  f"{cls['precision']:>12.4f} "
                  f"{cls['recall']:>12.4f} "
                  f"{cls['f1_score']:>12.4f} "
                  f"{cls['true_positives']:>8} "
                  f"{cls['false_positives']:>8} "
                  f"{cls['false_negatives']:>8}")
    
    print("=" * 90)
    
    # Find best performer
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['classification']['f1_score'])
        print(f"\n🔥 Best F1-Score: {best['mode']} ({best['classification']['f1_score']:.4f})")
        
        # Compare to baseline
        global_static = next((r for r in valid_results if r['mode'] == 'global_static'), None)
        if global_static and best['mode'] != 'global_static':
            improvement = ((best['classification']['f1_score'] - 
                          global_static['classification']['f1_score']) / 
                          global_static['classification']['f1_score'] * 100)
            print(f"   Improvement over global_static: {improvement:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive baseline comparison")
    parser.add_argument("--duration", type=int, default=120, 
                       help="Duration of each experiment in seconds")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    run_comprehensive_comparison(duration=args.duration, output_dir=args.output)
