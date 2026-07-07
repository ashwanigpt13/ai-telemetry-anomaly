"""
Scenario-Based Experiment Runner

Runs experiments with different anomaly scenarios:
- spike_only: Only spike anomalies
- drift_only: Only drift anomalies  
- noise_only: Only noise anomalies
- mixed: All anomaly types (default)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Import metrics module
sys.path.insert(0, str(Path(__file__).parent))
from metrics import load_evaluation_data, compute_classification_metrics

# Available scenarios
SCENARIOS = ["spike_only", "drift_only", "noise_only", "mixed"]

def run_scenario_experiments(duration=180, output_dir="results"):
    """Run experiments for all scenarios"""
    
    results = []
    
    print("=" * 80)
    print("SCENARIO-BASED EXPERIMENTAL EVALUATION")
    print("=" * 80)
    print(f"Duration: {duration} seconds per experiment")
    print(f"Scenarios: {', '.join(SCENARIOS)}")
    print()
    
    for scenario in SCENARIOS:
        print(f"\n{'='*80}")
        print(f"Running scenario: {scenario.upper()}")
        print(f"{'='*80}")
        
        # Clear previous data
        print("  [1/6] Clearing previous evaluation data...")
        subprocess.run(["docker-compose", "down", "-v"], 
                      capture_output=True, cwd=Path(__file__).parent.parent)
        
        # Update simulation mode in docker-compose.yml
        print(f"  [2/6] Setting simulation mode to: {scenario}")
        update_simulation_mode(scenario)
        
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
                "scenario": scenario,
                "error": "Failed to start services"
            })
            continue
        
        # Wait for services to be healthy
        print("  [4/6] Waiting for services to be healthy...")
        if not wait_for_services(timeout=60):
            print(f"  ❌ Services not healthy")
            results.append({
                "scenario": scenario,
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
        metrics = collect_results(scenario)
        
        if metrics:
            results.append(metrics)
            print(f"  ✅ {scenario}: Precision={metrics['classification']['precision']:.4f}, "
                  f"Recall={metrics['classification']['recall']:.4f}, "
                  f"F1={metrics['classification']['f1_score']:.4f}")
        else:
            print(f"  ❌ Failed to collect results for {scenario}")
            results.append({
                "scenario": scenario,
                "error": "Failed to collect results"
            })
        
        # Stop services
        subprocess.run(["docker-compose", "down"], 
                      capture_output=True, cwd=Path(__file__).parent.parent)
        
        # Small delay between experiments
        time.sleep(5)
    
    # Save results
    output_path = Path(output_dir) / "scenario_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")
    
    # Print comparison table
    print_scenario_table(results)
    
    return results

def update_simulation_mode(scenario):
    """Update SIMULATION_MODE in docker-compose.yml"""
    compose_file = Path(__file__).parent.parent / "docker-compose.yml"
    
    with open(compose_file, 'r') as f:
        lines = f.readlines()
    
    updated = False
    for i, line in enumerate(lines):
        if 'SIMULATION_MODE=' in line or ('- SIMULATION_MODE:' in line):
            # Update existing line
            if '=' in line:
                lines[i] = re.sub(r'(SIMULATION_MODE=)\w+', f'\\1{scenario}', line)
            updated = True
            break
    
    if not updated:
        # Add SIMULATION_MODE to simulator service environment
        for i, line in enumerate(lines):
            if 'simulator:' in line:
                # Find environment section
                for j in range(i, min(i+30, len(lines))):
                    if 'environment:' in lines[j]:
                        # Add after environment line
                        indent = '      '
                        lines.insert(j+1, f'{indent}- SIMULATION_MODE={scenario}\n')
                        break
                break
    
    with open(compose_file, 'w') as f:
        f.writelines(lines)

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

def collect_results(scenario):
    """Collect and analyze results for a scenario"""
    try:
        # Copy evaluation data from container
        result = subprocess.run(
            ["docker", "cp", "ingestion-service:/data/evaluation.csv", 
             f"evaluation_scenario_{scenario}.csv"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode != 0:
            print(f"    Warning: Failed to copy evaluation data: {result.stderr}")
            return None
        
        # Load and compute metrics
        csv_file = Path(__file__).parent / f"evaluation_scenario_{scenario}.csv"
        
        if not csv_file.exists() or csv_file.stat().st_size == 0:
            print(f"    Warning: Evaluation file is empty or missing")
            return None
        
        df = load_evaluation_data(str(csv_file))
        
        if df is None or len(df) == 0:
            print(f"    Warning: No data in evaluation file")
            return None
        
        metrics = compute_classification_metrics(df)
        
        # Analyze anomaly type distribution
        anomaly_types = df['anomaly_type'].value_counts().to_dict()
        
        return {
            "scenario": scenario,
            "timestamp": datetime.now().isoformat(),
            "total_records": len(df),
            "anomaly_types": anomaly_types,
            "classification": metrics
        }
        
    except Exception as e:
        print(f"    Error collecting results: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_scenario_table(results):
    """Print formatted scenario comparison table"""
    
    print("\nSCENARIO-BASED PERFORMANCE COMPARISON")
    print("=" * 100)
    print(f"{'Scenario':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'TP':>8} {'FP':>8} {'FN':>8} {'TN':>8}")
    print("-" * 100)
    
    for result in results:
        if "error" in result:
            print(f"{result['scenario']:<15} {'ERROR':>12} {'---':>12} {'---':>12} {'---':>8} {'---':>8} {'---':>8} {'---':>8}")
        else:
            cls = result['classification']
            print(f"{result['scenario']:<15} "
                  f"{cls['precision']:>12.4f} "
                  f"{cls['recall']:>12.4f} "
                  f"{cls['f1_score']:>12.4f} "
                  f"{cls['true_positives']:>8} "
                  f"{cls['false_positives']:>8} "
                  f"{cls['false_negatives']:>8} "
                  f"{cls['true_negatives']:>8}")
    
    print("=" * 100)
    
    # Print anomaly type distribution for each scenario
    print("\nANOMALY TYPE DISTRIBUTION BY SCENARIO")
    print("=" * 80)
    for result in results:
        if "error" not in result and "anomaly_types" in result:
            print(f"\n{result['scenario'].upper()}:")
            for anom_type, count in result['anomaly_types'].items():
                print(f"  {anom_type}: {count}")
    
    # Find best performer
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['classification']['f1_score'])
        worst = min(valid_results, key=lambda x: x['classification']['f1_score'])
        print(f"\n🔥 Best F1-Score: {best['scenario']} ({best['classification']['f1_score']:.4f})")
        print(f"   Worst F1-Score: {worst['scenario']} ({worst['classification']['f1_score']:.4f})")
        
        if len(valid_results) >= 2:
            f1_variance = np.var([r['classification']['f1_score'] for r in valid_results])
            print(f"   F1-Score Variance: {f1_variance:.6f}")
            print(f"   → {'Low variance indicates robustness across scenarios' if f1_variance < 0.01 else 'High variance indicates scenario-dependent performance'}")

if __name__ == "__main__":
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Run scenario-based experiments")
    parser.add_argument("--duration", type=int, default=120, 
                       help="Duration of each experiment in seconds")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--scenarios", type=str, default=",".join(SCENARIOS),
                       help="Comma-separated list of scenarios to run")
    
    args = parser.parse_args()
    
    # Override SCENARIOS if specified
    if args.scenarios:
        SCENARIOS = [s.strip() for s in args.scenarios.split(",")]
    
    run_scenario_experiments(duration=args.duration, output_dir=args.output)
