"""Compute statistics over latent vectors and save to JSON.

Outputs: evaluation/latent_analysis/latent_statistics.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

INPUT_CSV = Path('evaluation/latent_vectors/latent_vectors_combined.csv')
OUTPUT_DIR = Path('evaluation/latent_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUTPUT_DIR / 'latent_statistics.json'

# Load
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f'Input CSV not found: {INPUT_CSV}; aborting.')
    raise SystemExit(1)

latent_cols = [c for c in df.columns if c.startswith('latent_')]
if len(latent_cols) == 0:
    print('No latent columns found; aborting.')
    raise SystemExit(1)

X = df[latent_cols].values  # shape (n_samples, n_latent)

stats = {
    'n_samples': int(X.shape[0]),
    'n_latent_dims': int(X.shape[1]),
    'mean': np.mean(X, axis=0).tolist(),
    'std': np.std(X, axis=0, ddof=0).tolist(),
    'min': np.min(X, axis=0).tolist(),
    'max': np.max(X, axis=0).tolist(),
    'average_latent_norm': float(np.linalg.norm(X, axis=1).mean()),
    'covariance_matrix': np.cov(X, rowvar=False).tolist(),
}

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2)

print(f'Saved latent statistics to {OUT_JSON}')
