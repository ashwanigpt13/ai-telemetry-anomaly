"""Plot UMAP of latent vectors and save a publication-quality figure.

Saves: evaluation/latent_analysis/latent_umap.png
If UMAP (umap-learn) is not installed, script prints a message and exits 0.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = Path('evaluation/latent_vectors/latent_vectors_combined.csv')
OUTPUT_DIR = Path('evaluation/latent_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUTPUT_DIR / 'latent_umap.png'

# Load
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Input CSV not found: {INPUT_CSV}; skipping UMAP.")
    raise SystemExit(0)

latent_cols = [c for c in df.columns if c.startswith('latent_')]
if len(latent_cols) == 0:
    print('No latent columns found; skipping UMAP.')
    raise SystemExit(0)

X = df[latent_cols].values
labels = df['label'].values

# Try to import UMAP
try:
    from umap import UMAP
except Exception:
    try:
        import umap.umap_ as umap_mod
        UMAP = umap_mod.UMAP
    except Exception:
        print('UMAP is not installed; skipping UMAP plot.')
        raise SystemExit(0)

# Run UMAP
umap = UMAP(n_components=2, random_state=42)
X2 = umap.fit_transform(X)

# Plot
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

color_map = {'normal': '#1f77b4', 'anomalous': '#d62728'}
marker_map = {'normal': 'o', 'anomalous': 'X'}

for lab in np.unique(labels):
    mask = labels == lab
    ax.scatter(
        X2[mask, 0], X2[mask, 1],
        c=color_map.get(lab, '#7f7f7f'),
        marker=marker_map.get(lab, 'o'),
        s=18,
        alpha=0.8,
        linewidths=0.25,
        edgecolors='w',
        label=f"{lab} ({mask.sum():,})"
    )

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_title('UMAP of VAE Latent Means (random_state=42)')
ax.legend(frameon=True, fontsize=9)
ax.tick_params(axis='both', which='major', labelsize=9)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()
print(f"Saved UMAP plot to {OUT_PNG}")
