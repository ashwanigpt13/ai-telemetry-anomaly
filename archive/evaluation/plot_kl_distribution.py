"""Plot per-sample KL divergence histogram and save a publication-quality figure.

Saves: evaluation/latent_analysis/kl_distribution.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = Path('evaluation/latent_vectors/latent_vectors_combined.csv')
OUTPUT_DIR = Path('evaluation/latent_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUTPUT_DIR / 'kl_distribution.png'

# Load
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f'Input CSV not found: {INPUT_CSV}; aborting.')
    raise SystemExit(1)

mean_cols = [c for c in df.columns if c.startswith('latent_')]
logvar_cols = [c for c in df.columns if c.startswith('logvar_')]
if len(mean_cols) == 0 or len(logvar_cols) == 0:
    print('Required latent or logvar columns not found; aborting.')
    raise SystemExit(1)

means = df[mean_cols].values
logvars = df[logvar_cols].values

# KL per sample for diagonal Gaussian prior N(0, I):
# KL = 0.5 * sum(mean^2 + exp(logvar) - logvar - 1)
kl_per_sample = 0.5 * np.sum(means ** 2 + np.exp(logvars) - logvars - 1.0, axis=1)

kl_mean = float(np.mean(kl_per_sample))
kl_std = float(np.std(kl_per_sample, ddof=0))

# Plot
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

ax.hist(kl_per_sample, bins=50, color='#4c72b0', edgecolor='black', alpha=0.85)
ax.axvline(kl_mean, color='#d62728', linestyle='--', linewidth=1.5, label=f'Mean = {kl_mean:.3f}')

ax.set_xlabel('KL divergence (per sample)')
ax.set_ylabel('Count')
ax.set_title('KL Divergence Distribution Across Samples')

# Annotate mean and std
text_x = 0.98
text_y = 0.95
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(
    text_x,
    text_y,
    f'Mean = {kl_mean:.4f}\nStd = {kl_std:.4f}',
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=props,
)

ax.legend(frameon=True, fontsize=9)
ax.tick_params(axis='both', which='major', labelsize=9)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()
print(f'Saved KL distribution plot to {OUT_PNG}')
