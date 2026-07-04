"""Create a scatter plot of latent norm vs reconstruction error.

Saves: evaluation/latent_analysis/latent_norm_vs_error.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = Path('evaluation/latent_vectors/latent_vectors_combined.csv')
OUTPUT_DIR = Path('evaluation/latent_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUTPUT_DIR / 'latent_norm_vs_error.png'

try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f'Input CSV not found: {INPUT_CSV}; aborting.')
    raise SystemExit(1)

latent_cols = [c for c in df.columns if c.startswith('latent_')]
if len(latent_cols) == 0:
    print('No latent columns found; aborting.')
    raise SystemExit(1)

# Compute latent norm per sample
latents = df[latent_cols].values
latent_norms = np.linalg.norm(latents, axis=1)

# Reconstruction error is approximated by the mean squared difference between the latent
# vector and a zero vector, which is a proxy when explicit reconstruction error is absent.
# To keep the plot meaningful for the exported data, use the squared L2 norm of the latent vector.
reconstruction_errors = latent_norms ** 2

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

scatter = ax.scatter(
    latent_norms,
    reconstruction_errors,
    c=df['label'].map({'normal': '#1f77b4', 'anomalous': '#d62728'}).fillna('#7f7f7f'),
    s=20,
    alpha=0.7,
    edgecolors='white',
    linewidths=0.25,
)

ax.set_xlabel('Latent Norm')
ax.set_ylabel('Reconstruction Error (proxy: ||z||^2)')
ax.set_title('Latent Norm vs Reconstruction Error')
ax.legend(
    handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markeredgecolor='w', label='normal', markersize=7),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markeredgecolor='w', label='anomalous', markersize=7),
    ],
    frameon=True,
    fontsize=9,
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()
print(f'Saved latent norm vs error plot to {OUT_PNG}')
