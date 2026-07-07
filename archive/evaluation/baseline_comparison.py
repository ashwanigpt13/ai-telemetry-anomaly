"""
Run baseline comparison across all threshold modes
"""
import subprocess
import time
import csv
import os

MODES = ["global_static", "global_dynamic", "entity_dynamic", "entity_drift"]
DURATION = 90
WARMUP = 30
RESULTS = {}

def run_command(cmd, timeout=None):
    """Run a command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result

def update_threshold_mode(mode):
    """Update THRESHOLD_MODE in docker-compose.yml"""
    compose_path = "../docker-compose.yml"
    with open(compose_path, 'r') as f:
        content = f.read()
    
    # Update threshold service THRESHOLD_MODE
    import re
    content = re.sub(
        r'THRESHOLD_MODE=[a-z_]+',
        f'THRESHOLD_MODE={mode}',
        content
    )
    
    with open(compose_path, 'w') as f:
        f.write(content)
    print(f"  Updated THRESHOLD_MODE to {mode}")

def restart_services():
    """Restart all services"""
    print("  Stopping services...")
    run_command("docker-compose down", timeout=60)
    print("  Starting services...")
    result = run_command("docker-compose up -d", timeout=180)
    print("  Waiting for services to be healthy...")
    time.sleep(70)  # Wait for health checks
    return True

def clear_evaluation_data():
    """Clear evaluation data in container"""
    run_command('docker exec ingestion-service sh -c "truncate -s 0 /data/evaluation.csv 2>/dev/null || true"')
    run_command('docker exec ingestion-service sh -c "rm -f /data/evaluation.csv 2>/dev/null || true"')
    print("  Cleared evaluation data")

def collect_data():
    """Collect evaluation data"""
    print(f"  Warmup period: {WARMUP}s...")
    time.sleep(WARMUP)
    
    print(f"  Collecting data for {DURATION}s...")
    time.sleep(DURATION)
    
    print("  Cooldown period: 10s...")
    time.sleep(10)

def compute_metrics():
    """Compute metrics from evaluation data"""
    # Copy data from container
    result = run_command('docker exec ingestion-service cat /data/evaluation.csv')
    if not result.stdout.strip():
        return None
    
    # Parse data
    lines = result.stdout.strip().split('\n')
    
    tp = fp = fn = tn = 0
    latencies = []
    
    # Parse without header (first line might be data or header)
    for line in lines:
        parts = line.split(',')
        if len(parts) >= 8 and parts[0] != 'entity_id':
            try:
                actual = parts[2].lower() == 'true'
                predicted = parts[4].lower() == 'true'
                latency = float(parts[8]) if len(parts) > 8 else 0
                
                if actual and predicted: tp += 1
                elif not actual and predicted: fp += 1
                elif actual and not predicted: fn += 1
                else: tn += 1
                
                latencies.append(latency)
            except (ValueError, IndexError):
                continue
    
    if tp + fp == 0 or tp + fn == 0:
        return None
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    latencies.sort()
    p95_idx = int(len(latencies) * 0.95)
    latency_p95 = latencies[p95_idx] if latencies else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'tn': tn,
        'latency_p95': latency_p95,
        'total_records': len(latencies)
    }

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('..')  # Move to project root
    
    print("=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    
    for mode in MODES:
        print(f"\n[{MODES.index(mode)+1}/{len(MODES)}] Testing mode: {mode}")
        print("-" * 40)
        
        # Update config
        update_threshold_mode(mode)
        
        # Restart services
        if not restart_services():
            print(f"  ERROR: Failed to start services for {mode}")
            continue
        
        # Clear data
        clear_evaluation_data()
        
        # Collect data
        collect_data()
        
        # Compute metrics
        metrics = compute_metrics()
        
        if metrics:
            RESULTS[mode] = metrics
            print(f"  Results: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        else:
            print(f"  ERROR: No data collected for {mode}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Mode':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>8} {'FN':>8} {'Latency p95':>12}")
    print("-" * 80)
    
    for mode in MODES:
        if mode in RESULTS:
            r = RESULTS[mode]
            print(f"{mode:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['fp']:>8} {r['fn']:>8} {r['latency_p95']:>10.0f}ms")
        else:
            print(f"{mode:<20} {'ERROR':>10}")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
