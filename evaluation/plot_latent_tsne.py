"""Plot t-SNE of latent vectors and save a publication-quality figure.

Saves: evaluation/latent_analysis/latent_tsne.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

INPUT_CSV = Path('evaluation/latent_vectors/latent_vectors_combined.csv')
OUTPUT_DIR = Path('evaluation/latent_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUTPUT_DIR / 'latent_tsne.png'

# Load
df = pd.read_csv(INPUT_CSV)
latent_cols = [c for c in df.columns if c.startswith('latent_')]
X = df[latent_cols].values
labels = df['label'].values

# t-SNE
tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=30)
X2 = tsne.fit_transform(X)

# Prepare plot
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

# Colors and markers
color_map = {'normal': '#1f77b4', 'anomalous': '#d62728'}
marker_map = {'normal': 'o', 'anomalous': 'X'}

for lab in np.unique(labels):
    mask = labels == lab
    ax.scatter(
        X2[mask, 0], X2[mask, 1],
        c=color_map.get(lab, '#7f7f7f'),
        marker=marker_map.get(lab, 'o'),
        s=18,
        alpha=0.75,
        linewidths=0.25,
        edgecolors='w',
        label=f"{lab} ({mask.sum():,})"
    )

# Aesthetics
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('t-SNE of VAE Latent Means (random_state=42)')
ax.legend(frameon=True, fontsize=9)
ax.tick_params(axis='both', which='major', labelsize=9)

# Save
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.close()
print(f"Saved t-SNE plot to {OUT_PNG}")
