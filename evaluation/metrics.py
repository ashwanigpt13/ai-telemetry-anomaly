"""
Evaluation Metrics Script

Computes:
- Precision, Recall, F1-score
- False positives, False negatives
- Drift recovery time measurement
- Latency statistics

Input: CSV file with columns:
  entity_id, timestamp, actual_anomaly, anomaly_type, 
  predicted_anomaly, error, threshold, drift, latency_ms

Output:
- Prints metrics to console
- Saves results.json
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
import statistics


def load_evaluation_data(csv_path: str) -> List[Dict]:
    """Load evaluation data from CSV file"""
    records = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert types
            row['actual_anomaly'] = row['actual_anomaly'].lower() == 'true'
            row['predicted_anomaly'] = row['predicted_anomaly'].lower() == 'true'
            row['drift'] = row['drift'].lower() == 'true'
            row['timestamp'] = int(row['timestamp'])
            row['error'] = float(row['error'])
            row['threshold'] = float(row['threshold'])
            row['latency_ms'] = float(row.get('latency_ms', 0))
            records.append(row)
    return records


def compute_classification_metrics(records: List[Dict]) -> Dict:
    """
    Compute precision, recall, F1-score
    
    Returns dict with:
        precision, recall, f1_score, true_positives, false_positives,
        true_negatives, false_negatives, total_samples
    """
    tp = fp = tn = fn = 0
    
    for r in records:
        actual = r['actual_anomaly']
        predicted = r['predicted_anomaly']
        
        if actual and predicted:
            tp += 1
        elif not actual and predicted:
            fp += 1
        elif actual and not predicted:
            fn += 1
        else:
            tn += 1
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_samples": len(records)
    }


def compute_per_entity_metrics(records: List[Dict]) -> Dict[str, Dict]:
    """Compute classification metrics per entity"""
    entity_records = defaultdict(list)
    
    for r in records:
        entity_records[r['entity_id']].append(r)
    
    entity_metrics = {}
    for entity_id, recs in entity_records.items():
        entity_metrics[entity_id] = compute_classification_metrics(recs)
    
    return entity_metrics


def compute_drift_recovery_time(records: List[Dict]) -> Dict[str, Dict]:
    """
    Measure drift recovery time per entity
    
    Drift recovery is measured as the time from drift start until:
    - anomaly rate decreases (stabilization)
    - threshold adapts (returns to baseline)
    
    Returns dict with entity_id -> {
        drift_start, drift_end, recovery_time_ms, 
        anomaly_rate_before, anomaly_rate_during, anomaly_rate_after
    }
    """
    entity_records = defaultdict(list)
    for r in records:
        entity_records[r['entity_id']].append(r)
    
    drift_metrics = {}
    
    for entity_id, recs in entity_records.items():
        # Sort by timestamp
        recs = sorted(recs, key=lambda x: x['timestamp'])
        
        # Find drift periods
        drift_periods = []
        drift_start = None
        
        for i, r in enumerate(recs):
            # Detect drift start (actual anomaly type is drift)
            if r['anomaly_type'] == 'drift' and drift_start is None:
                drift_start = r['timestamp']
            
            # Detect drift end (anomaly type returns to normal or different type)
            if drift_start is not None and r['anomaly_type'] != 'drift':
                drift_end = r['timestamp']
                
                # Measure stabilization: how long until prediction rate normalizes
                # Look for period where false positive rate drops
                stabilization_idx = i
                for j in range(i, min(i + 50, len(recs))):  # Look ahead 50 samples
                    if not recs[j]['predicted_anomaly']:
                        stabilization_idx = j
                        break
                
                recovery_time_ms = recs[stabilization_idx]['timestamp'] - drift_start if stabilization_idx < len(recs) else 0
                
                drift_periods.append({
                    "drift_start": drift_start,
                    "drift_end": drift_end,
                    "recovery_time_ms": recovery_time_ms
                })
                drift_start = None
        
        if drift_periods:
            avg_recovery = statistics.mean([p['recovery_time_ms'] for p in drift_periods])
            drift_metrics[entity_id] = {
                "drift_events": len(drift_periods),
                "avg_recovery_time_ms": round(avg_recovery, 2),
                "min_recovery_time_ms": min(p['recovery_time_ms'] for p in drift_periods),
                "max_recovery_time_ms": max(p['recovery_time_ms'] for p in drift_periods),
                "drift_periods": drift_periods
            }
    
    return drift_metrics


def compute_latency_metrics(records: List[Dict]) -> Dict:
    """
    Compute latency statistics
    
    Returns dict with:
        average_latency_ms, p50_latency_ms, p95_latency_ms, 
        p99_latency_ms, min_latency_ms, max_latency_ms
    """
    latencies = [r['latency_ms'] for r in records if r['latency_ms'] > 0]
    
    if not latencies:
        return {
            "average_latency_ms": 0,
            "p50_latency_ms": 0,
            "p95_latency_ms": 0,
            "p99_latency_ms": 0,
            "min_latency_ms": 0,
            "max_latency_ms": 0,
            "sample_count": 0
        }
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return {
        "average_latency_ms": round(statistics.mean(latencies), 2),
        "p50_latency_ms": round(sorted_latencies[int(n * 0.5)], 2),
        "p95_latency_ms": round(sorted_latencies[int(n * 0.95)], 2),
        "p99_latency_ms": round(sorted_latencies[int(n * 0.99)], 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "sample_count": n
    }


def compute_all_metrics(records: List[Dict]) -> Dict:
    """Compute all metrics"""
    return {
        "classification": compute_classification_metrics(records),
        "per_entity": compute_per_entity_metrics(records),
        "drift_recovery": compute_drift_recovery_time(records),
        "latency": compute_latency_metrics(records)
    }


def print_metrics(metrics: Dict):
    """Print metrics in formatted table"""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    
    # Classification metrics
    clf = metrics['classification']
    print("\n📊 Classification Metrics:")
    print(f"  {'Metric':<20} {'Value':<15}")
    print(f"  {'-' * 35}")
    print(f"  {'Precision':<20} {clf['precision']:<15.4f}")
    print(f"  {'Recall':<20} {clf['recall']:<15.4f}")
    print(f"  {'F1-Score':<20} {clf['f1_score']:<15.4f}")
    print(f"  {'True Positives':<20} {clf['true_positives']:<15}")
    print(f"  {'False Positives':<20} {clf['false_positives']:<15}")
    print(f"  {'True Negatives':<20} {clf['true_negatives']:<15}")
    print(f"  {'False Negatives':<20} {clf['false_negatives']:<15}")
    print(f"  {'Total Samples':<20} {clf['total_samples']:<15}")
    
    # Per-entity metrics
    print("\n📈 Per-Entity Metrics:")
    print(f"  {'Entity':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Samples':<10}")
    print(f"  {'-' * 61}")
    for entity_id, em in metrics['per_entity'].items():
        print(f"  {entity_id:<15} {em['precision']:<12.4f} {em['recall']:<12.4f} {em['f1_score']:<12.4f} {em['total_samples']:<10}")
    
    # Drift recovery
    if metrics['drift_recovery']:
        print("\n🔄 Drift Recovery Time:")
        print(f"  {'Entity':<15} {'Events':<10} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print(f"  {'-' * 61}")
        for entity_id, dm in metrics['drift_recovery'].items():
            print(f"  {entity_id:<15} {dm['drift_events']:<10} {dm['avg_recovery_time_ms']:<12.1f} {dm['min_recovery_time_ms']:<12} {dm['max_recovery_time_ms']:<12}")
    else:
        print("\n🔄 Drift Recovery: No drift events detected")
    
    # Latency
    lat = metrics['latency']
    print("\n⏱️  Latency Metrics:")
    print(f"  {'Metric':<20} {'Value (ms)':<15}")
    print(f"  {'-' * 35}")
    print(f"  {'Average':<20} {lat['average_latency_ms']:<15.2f}")
    print(f"  {'P50 (Median)':<20} {lat['p50_latency_ms']:<15.2f}")
    print(f"  {'P95':<20} {lat['p95_latency_ms']:<15.2f}")
    print(f"  {'P99':<20} {lat['p99_latency_ms']:<15.2f}")
    print(f"  {'Min':<20} {lat['min_latency_ms']:<15.2f}")
    print(f"  {'Max':<20} {lat['max_latency_ms']:<15.2f}")
    
    print("\n" + "=" * 60)


def save_results(metrics: Dict, output_path: str, mode: str = None):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Add mode field for compatibility with experiment runner
    output = {
        "mode": mode or "unknown",
        "timestamp": datetime.now().isoformat(),
        "classification": metrics["classification"],
        "per_entity": metrics["per_entity"],
        "drift_recovery": metrics["drift_recovery"],
        "latency": metrics["latency"]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute evaluation metrics for anomaly detection')
    parser.add_argument('--input', '-i', type=str, default='/data/evaluation.csv',
                        help='Path to evaluation CSV file')
    parser.add_argument('--output', '-o', type=str, default='results/results.json',
                        help='Path to output JSON file')
    parser.add_argument('--mode', '-m', type=str, default=None,
                        help='Threshold mode name (for result labeling)')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    # Load data
    print(f"📂 Loading data from: {args.input}")
    records = load_evaluation_data(args.input)
    print(f"   Loaded {len(records)} records")
    
    # Compute metrics
    print("🔄 Computing metrics...")
    metrics = compute_all_metrics(records)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results
    save_results(metrics, args.output, args.mode)
    
    return 0


if __name__ == '__main__':
    exit(main())
