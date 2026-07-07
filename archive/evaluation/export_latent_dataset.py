"""Export a compact latent dataset for clustering, explainability, and quantum experiments.

Saves: evaluation/latent_analysis/latent_dataset.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

INPUT_CSV = Path('evaluation/latent_vectors/latent_vectors_combined.csv')
OUTPUT_DIR = Path('evaluation/latent_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTPUT_DIR / 'latent_dataset.csv'

try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f'Input CSV not found: {INPUT_CSV}; aborting.')
    raise SystemExit(1)

latent_cols = [c for c in df.columns if c.startswith('latent_')]
if len(latent_cols) == 0:
    print('No latent columns found; aborting.')
    raise SystemExit(1)

latents = df[latent_cols].values
latent_norms = np.linalg.norm(latents, axis=1)
reconstruction_error = latent_norms ** 2

export_df = pd.DataFrame({
    'engine_id': df['engine_id'],
    'cycle': df['cycle'],
    'label': df['label'],
    'reconstruction_error': reconstruction_error,
    'latent_norm': latent_norms,
    **{col: df[col].values for col in latent_cols},
})

export_df.to_csv(OUT_CSV, index=False)
print(f'Saved latent dataset to {OUT_CSV}')
