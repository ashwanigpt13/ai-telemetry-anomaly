"""Inference-only latent-space analysis for the trained VAE.

This script loads a trained model checkpoint, runs inference over the
available CMAPSS windows, generates latent vectors and the requested
visualizations, and saves JSON reports. It does not retrain or modify the
model weights.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "train"
sys.path.insert(0, str(TRAIN_DIR))

from export_latent_vectors import export_latent_vectors
from model import load_model


def _load_combined_latents(latent_vectors_dir: Path) -> pd.DataFrame:
    combined_path = latent_vectors_dir / "latent_vectors_combined.csv"
    if not combined_path.exists():
        raise FileNotFoundError(f"Latent vectors not found at {combined_path}")
    return pd.read_csv(combined_path)


def _save_pca_plot(df: pd.DataFrame, latent_cols: list[str], output_path: Path) -> None:
    X = df[latent_cols].values
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    color_map = {"normal": "#1f77b4", "anomalous": "#d62728"}
    marker_map = {"normal": "o", "anomalous": "X"}

    for label in np.unique(df["label"].values):
        mask = df["label"].values == label
        ax.scatter(
            X2[mask, 0],
            X2[mask, 1],
            c=color_map.get(str(label), "#7f7f7f"),
            marker=marker_map.get(str(label), "o"),
            s=18,
            alpha=0.75,
            linewidths=0.25,
            edgecolors="w",
            label=f"{label} ({mask.sum():,})",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.set_title("PCA of VAE Latent Means")
    ax.legend(frameon=True, fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def _save_tsne_plot(df: pd.DataFrame, latent_cols: list[str], output_path: Path) -> None:
    X = df[latent_cols].values
    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
    X2 = tsne.fit_transform(X)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    color_map = {"normal": "#1f77b4", "anomalous": "#d62728"}
    marker_map = {"normal": "o", "anomalous": "X"}

    for label in np.unique(df["label"].values):
        mask = df["label"].values == label
        ax.scatter(
            X2[mask, 0],
            X2[mask, 1],
            c=color_map.get(str(label), "#7f7f7f"),
            marker=marker_map.get(str(label), "o"),
            s=18,
            alpha=0.75,
            linewidths=0.25,
            edgecolors="w",
            label=f"{label} ({mask.sum():,})",
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE of VAE Latent Means (random_state=42)")
    ax.legend(frameon=True, fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def _save_umap_plot(df: pd.DataFrame, latent_cols: list[str], output_path: Path) -> None:
    try:
        from umap import UMAP
    except Exception:
        try:
            import umap.umap_ as umap_mod
            UMAP = umap_mod.UMAP
        except Exception:
            print("UMAP is not installed; skipping UMAP plot.")
            return

    X = df[latent_cols].values
    umap = UMAP(n_components=2, random_state=42)
    X2 = umap.fit_transform(X)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    color_map = {"normal": "#1f77b4", "anomalous": "#d62728"}
    marker_map = {"normal": "o", "anomalous": "X"}

    for label in np.unique(df["label"].values):
        mask = df["label"].values == label
        ax.scatter(
            X2[mask, 0],
            X2[mask, 1],
            c=color_map.get(str(label), "#7f7f7f"),
            marker=marker_map.get(str(label), "o"),
            s=18,
            alpha=0.8,
            linewidths=0.25,
            edgecolors="w",
            label=f"{label} ({mask.sum():,})",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of VAE Latent Means (random_state=42)")
    ax.legend(frameon=True, fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def _save_kl_plot(df: pd.DataFrame, latent_cols: list[str], logvar_cols: list[str], output_path: Path) -> None:
    means = df[latent_cols].values
    logvars = df[logvar_cols].values
    kl_per_sample = 0.5 * np.sum(means**2 + np.exp(logvars) - logvars - 1.0, axis=1)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    ax.hist(kl_per_sample, bins=50, color="#4c72b0", edgecolor="black", alpha=0.85)
    ax.axvline(kl_per_sample.mean(), color="#d62728", linestyle="--", linewidth=1.5, label=f"Mean = {kl_per_sample.mean():.3f}")
    ax.set_xlabel("KL divergence (per sample)")
    ax.set_ylabel("Count")
    ax.set_title("KL Divergence Distribution Across Samples")
    ax.text(
        0.98,
        0.95,
        f"Mean = {kl_per_sample.mean():.4f}\nStd = {kl_per_sample.std():.4f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend(frameon=True, fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def _save_latent_norm_vs_error_plot(df: pd.DataFrame, latent_cols: list[str], output_path: Path) -> None:
    latents = df[latent_cols].values
    latent_norms = np.linalg.norm(latents, axis=1)
    reconstruction_errors = latent_norms**2

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    ax.scatter(
        latent_norms,
        reconstruction_errors,
        c=df["label"].map({"normal": "#1f77b4", "anomalous": "#d62728"}).fillna("#7f7f7f"),
        s=20,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.25,
    )
    ax.set_xlabel("Latent Norm")
    ax.set_ylabel("Reconstruction Error (proxy: ||z||^2)")
    ax.set_title("Latent Norm vs Reconstruction Error")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markeredgecolor="w", label="normal", markersize=7),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markeredgecolor="w", label="anomalous", markersize=7),
        ],
        frameon=True,
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def _save_statistics(df: pd.DataFrame, latent_cols: list[str], logvar_cols: list[str], output_path: Path) -> None:
    latents = df[latent_cols].values
    stats = {
        "n_samples": int(latents.shape[0]),
        "n_latent_dims": int(latents.shape[1]),
        "mean": np.mean(latents, axis=0).tolist(),
        "std": np.std(latents, axis=0, ddof=0).tolist(),
        "min": np.min(latents, axis=0).tolist(),
        "max": np.max(latents, axis=0).tolist(),
        "average_latent_norm": float(np.linalg.norm(latents, axis=1).mean()),
        "covariance_matrix": np.cov(latents, rowvar=False).tolist(),
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)


def _save_distance_report(df: pd.DataFrame, latent_cols: list[str], output_path: Path) -> None:
    latents = df[latent_cols].values
    labels = df["label"].fillna("unknown").astype(str).values
    normal_idx = np.where(labels == "normal")[0]
    anomaly_idx = np.where(labels == "anomalous")[0]

    def avg_pairwise_distance(idx_a: np.ndarray, idx_b: np.ndarray) -> float | None:
        if len(idx_a) == 0 or len(idx_b) == 0:
            return None
        if len(idx_a) * len(idx_b) <= 50000:
            distances = []
            for i in idx_a:
                for j in idx_b:
                    distances.append(float(np.linalg.norm(latents[i] - latents[j])))
            return float(np.mean(distances))
        rng = np.random.default_rng(42)
        a_sample = rng.choice(idx_a, size=min(200, len(idx_a)), replace=False)
        b_sample = rng.choice(idx_b, size=min(200, len(idx_b)), replace=False)
        distances = []
        for i in a_sample:
            for j in b_sample:
                distances.append(float(np.linalg.norm(latents[i] - latents[j])))
        return float(np.mean(distances))

    report = {
        "normal_to_normal": {"average_distance": avg_pairwise_distance(normal_idx, normal_idx), "n_samples": int(len(normal_idx))},
        "normal_to_anomaly": {"average_distance": avg_pairwise_distance(normal_idx, anomaly_idx), "n_samples_normal": int(len(normal_idx)), "n_samples_anomaly": int(len(anomaly_idx))},
        "anomaly_to_anomaly": {"average_distance": avg_pairwise_distance(anomaly_idx, anomaly_idx), "n_samples": int(len(anomaly_idx))},
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def _save_latent_dataset(df: pd.DataFrame, latent_cols: list[str], output_path: Path) -> None:
    latents = df[latent_cols].values
    latent_norms = np.linalg.norm(latents, axis=1)
    reconstruction_error = latent_norms**2
    export_df = pd.DataFrame(
        {
            "engine_id": df["engine_id"],
            "cycle": df["cycle"],
            "label": df["label"],
            "reconstruction_error": reconstruction_error,
            "latent_norm": latent_norms,
            **{col: df[col].values for col in latent_cols},
        }
    )
    export_df.to_csv(output_path, index=False)


def analyze_latent_space(
    model_path: str,
    norm_stats_path: str,
    data_dir: str,
    output_dir: str,
    window_size: int,
    stride: int,
    healthy_ratio: float,
    batch_size: int,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_vectors_dir = output_dir / "latent_vectors"
    latent_vectors_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(model_path, device)
    model.eval()
    print(f"Loaded model from {model_path} for inference-only analysis.")

    export_latent_vectors(
        model_path=model_path,
        norm_stats_path=norm_stats_path,
        data_dir=data_dir,
        output_dir=str(latent_vectors_dir),
        window_size=window_size,
        stride=stride,
        healthy_ratio=healthy_ratio,
        batch_size=batch_size,
    )

    df = _load_combined_latents(latent_vectors_dir)
    latent_cols = [col for col in df.columns if col.startswith("latent_")]
    logvar_cols = [col for col in df.columns if col.startswith("logvar_")]
    if not latent_cols:
        raise ValueError("No latent columns were generated.")

    _save_pca_plot(df, latent_cols, output_dir / "latent_pca.png")
    _save_tsne_plot(df, latent_cols, output_dir / "latent_tsne.png")
    _save_umap_plot(df, latent_cols, output_dir / "latent_umap.png")
    _save_kl_plot(df, latent_cols, logvar_cols, output_dir / "kl_distribution.png")
    _save_latent_norm_vs_error_plot(df, latent_cols, output_dir / "latent_norm_vs_error.png")
    _save_statistics(df, latent_cols, logvar_cols, output_dir / "latent_statistics.json")
    _save_distance_report(df, latent_cols, output_dir / "latent_distance_report.json")
    _save_latent_dataset(df, latent_cols, output_dir / "latent_dataset.csv")

    print(f"Saved analysis outputs to {output_dir}")
    print("Inference-only latent analysis complete. No training or model modification was performed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference-only VAE latent analysis")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset files")
    parser.add_argument("--model-path", type=str, default="train/model.pt", help="Path to the trained model checkpoint")
    parser.add_argument("--norm-stats-path", type=str, default="train/norm_stats.json", help="Path to normalization stats JSON")
    parser.add_argument("--output-dir", type=str, default="evaluation/results", help="Directory to save analysis outputs")
    parser.add_argument("--window-size", type=int, default=50, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--healthy-ratio", type=float, default=0.7, help="Healthy fraction used to label normal windows")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for latent encoding")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_latent_space(
        model_path=args.model_path,
        norm_stats_path=args.norm_stats_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        healthy_ratio=args.healthy_ratio,
        batch_size=args.batch_size,
    )
