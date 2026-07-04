"""Compute latent-space distance statistics between sample groups.

Outputs: evaluation/latent_analysis/latent_distance_report.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

INPUT_CSV = Path('evaluation/latent_vectors/latent_vectors_combined.csv')
OUTPUT_DIR = Path('evaluation/latent_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUTPUT_DIR / 'latent_distance_report.json'

try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f'Input CSV not found: {INPUT_CSV}; aborting.')
    raise SystemExit(1)

latent_cols = [c for c in df.columns if c.startswith('latent_')]
if len(latent_cols) == 0:
    print('No latent columns found; aborting.')
    raise SystemExit(1)

X = df[latent_cols].values
labels = df['label'].fillna('unknown').astype(str).values

# Group indices
normal_idx = np.where(labels == 'normal')[0]
anomaly_idx = np.where(labels == 'anomalous')[0]

# Helper to compute average pairwise Euclidean distance for a pair of groups

def avg_pairwise_distance(idx_a, idx_b):
    if len(idx_a) == 0 or len(idx_b) == 0:
        return None
    if len(idx_a) == 1 and len(idx_b) == 1:
        return float(np.linalg.norm(X[idx_a[0]] - X[idx_b[0]]))

    # Sample a manageable number for large groups; use all pairs if small
    if len(idx_a) * len(idx_b) <= 50000:
        pairs = []
        for i in idx_a:
            for j in idx_b:
                pairs.append(np.linalg.norm(X[i] - X[j]))
        return float(np.mean(pairs))

    # For very large groups, sample 10k random pairs
    rng = np.random.default_rng(42)
    a_sample = rng.choice(idx_a, size=min(200, len(idx_a)), replace=False)
    b_sample = rng.choice(idx_b, size=min(200, len(idx_b)), replace=False)
    dists = []
    for i in a_sample:
        for j in b_sample:
            dists.append(np.linalg.norm(X[i] - X[j]))
    return float(np.mean(dists))

report = {
    'normal_to_normal': {
        'average_distance': avg_pairwise_distance(normal_idx, normal_idx),
        'n_samples': int(len(normal_idx)),
    },
    'normal_to_anomaly': {
        'average_distance': avg_pairwise_distance(normal_idx, anomaly_idx),
        'n_samples_normal': int(len(normal_idx)),
        'n_samples_anomaly': int(len(anomaly_idx)),
    },
    'anomaly_to_anomaly': {
        'average_distance': avg_pairwise_distance(anomaly_idx, anomaly_idx),
        'n_samples': int(len(anomaly_idx)),
    },
}

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)

print(f'Saved latent distance report to {OUT_JSON}')
