"""
Class Distribution Analysis for Imbalanced Dataset Evaluation
"""
import csv
import sys
from pathlib import Path

def analyze_class_distribution(csv_file):
    """Analyze class distribution in evaluation data"""
    
    total_samples = 0
    anomalies = 0
    normal = 0
    
    # Count by entity
    entity_stats = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entity_id = row['entity_id']
            actual = row['actual_anomaly'].lower() == 'true'
            
            if entity_id not in entity_stats:
                entity_stats[entity_id] = {'total': 0, 'anomalies': 0, 'normal': 0}
            
            total_samples += 1
            entity_stats[entity_id]['total'] += 1
            
            if actual:
                anomalies += 1
                entity_stats[entity_id]['anomalies'] += 1
            else:
                normal += 1
                entity_stats[entity_id]['normal'] += 1
    
    # Overall statistics
    anomaly_pct = (anomalies / total_samples * 100) if total_samples > 0 else 0
    normal_pct = (normal / total_samples * 100) if total_samples > 0 else 0
    
    print("=" * 70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total Samples:      {total_samples:,}")
    print(f"  Anomalous Samples:  {anomalies:,} ({anomaly_pct:.2f}%)")
    print(f"  Normal Samples:     {normal:,} ({normal_pct:.2f}%)")
    print(f"  Imbalance Ratio:    1:{normal/anomalies:.2f}" if anomalies > 0 else "  Imbalance Ratio:    N/A")
    
    print(f"\n{'Entity':<15} {'Total':>8} {'Anomalies':>12} {'Normal':>10} {'Anomaly %':>12}")
    print("-" * 70)
    
    for entity_id in sorted(entity_stats.keys()):
        stats = entity_stats[entity_id]
        entity_anom_pct = (stats['anomalies'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{entity_id:<15} {stats['total']:>8} {stats['anomalies']:>12} {stats['normal']:>10} {entity_anom_pct:>11.2f}%")
    
    print("=" * 70)
    
    # Return statistics for use in other scripts
    return {
        'total_samples': total_samples,
        'anomalies': anomalies,
        'normal': normal,
        'anomaly_percentage': anomaly_pct,
        'imbalance_ratio': normal / anomalies if anomalies > 0 else 0,
        'per_entity': entity_stats
    }

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "evaluation_drift.csv"
    
    if not Path(csv_file).exists():
        print(f"Error: File {csv_file} not found")
        sys.exit(1)
    
    analyze_class_distribution(csv_file)
