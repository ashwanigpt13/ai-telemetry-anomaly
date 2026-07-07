#!/usr/bin/env python3
"""Export publication-quality figures from frozen benchmark artifacts."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "paper_figures"
EXPORT_FORMATS = ["png", "pdf", "svg"]

FONT_SIZES = {
    "title": 14,
    "label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 11,
}

FIGURE_SPECS = [
    {
        "slug": "roc_curve",
        "description": "Receiver operating characteristic curves for the classical and hybrid models.",
        "section": "Benchmark Results",
        "figure_number": "Figure X",
    },
    {
        "slug": "pr_curve",
        "description": "Precision-recall curves for the classical and hybrid models.",
        "section": "Benchmark Results",
        "figure_number": "Figure X",
    },
    {
        "slug": "training_curves",
        "description": "Training and validation loss curves from the frozen training run.",
        "section": "Training Procedure",
        "figure_number": "Figure X",
    },
    {
        "slug": "parameter_comparison",
        "description": "Total and quantum parameter comparison for the two models.",
        "section": "Efficiency Analysis",
        "figure_number": "Figure X",
    },
    {
        "slug": "training_time",
        "description": "Training time comparison for the classical and hybrid models.",
        "section": "Efficiency Analysis",
        "figure_number": "Figure X",
    },
    {
        "slug": "confusion_matrix_classical",
        "description": "Confusion matrix for the classical model on the aligned evaluation set.",
        "section": "Error Analysis",
        "figure_number": "Figure X",
    },
    {
        "slug": "confusion_matrix_hybrid",
        "description": "Confusion matrix for the hybrid quantum model on the aligned evaluation set.",
        "section": "Error Analysis",
        "figure_number": "Figure X",
    },
    {
        "slug": "score_distribution_classical",
        "description": "Classical model reconstruction-error distribution by ground-truth class.",
        "section": "Error Analysis",
        "figure_number": "Figure X",
    },
    {
        "slug": "score_distribution_hybrid",
        "description": "Hybrid model reconstruction-error distribution by ground-truth class.",
        "section": "Error Analysis",
        "figure_number": "Figure X",
    },
    {
        "slug": "pca_classical",
        "description": "PCA projection of classical model latent embeddings.",
        "section": "Latent Space Analysis",
        "figure_number": "Figure X",
    },
    {
        "slug": "pca_hybrid",
        "description": "PCA projection of hybrid model latent embeddings.",
        "section": "Latent Space Analysis",
        "figure_number": "Figure X",
    },
    {
        "slug": "tsne_classical",
        "description": "t-SNE projection of classical model latent embeddings.",
        "section": "Latent Space Analysis",
        "figure_number": "Figure X",
    },
    {
        "slug": "tsne_hybrid",
        "description": "t-SNE projection of hybrid model latent embeddings.",
        "section": "Latent Space Analysis",
        "figure_number": "Figure X",
    },
]


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": FONT_SIZES["tick"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["label"],
        "xtick.labelsize": FONT_SIZES["tick"],
        "ytick.labelsize": FONT_SIZES["tick"],
        "legend.fontsize": FONT_SIZES["legend"],
        "figure.titlesize": FONT_SIZES["title"],
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, slug: str) -> List[str]:
    filenames = []
    for extension in EXPORT_FORMATS:
        filename = f"{slug}.{extension}"
        fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
        filenames.append(filename)
    plt.close(fig)
    return filenames


def plot_roc_curve(results: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for model_key, label in [("classical", "Classical"), ("hybrid", "Hybrid Quantum")]:
        metrics = results["models"][model_key]["metrics"]
        curve = metrics["roc_curve"]
        ax.plot(curve["fpr"], curve["tpr"], linewidth=2.0, label=f"{label} (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1.2, label="Chance")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "roc_curve")


def plot_pr_curve(results: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for model_key, label in [("classical", "Classical"), ("hybrid", "Hybrid Quantum")]:
        metrics = results["models"][model_key]["metrics"]
        curve = metrics["pr_curve"]
        ax.plot(curve["recall"], curve["precision"], linewidth=2.0, label=f"{label} (AP={metrics['pr_auc']:.3f})")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    save_figure(fig, "pr_curve")


def plot_training_curves() -> None:
    rows = read_csv(REPO_ROOT / "archive" / "loss.csv")
    epochs = np.asarray([int(row["epoch"]) for row in rows])
    train_loss = np.asarray([float(row["train_loss"]) for row in rows])
    val_loss = np.asarray([float(row["validation_total_loss"]) for row in rows])
    train_reconstruction = np.asarray([float(row["train_reconstruction_loss"]) for row in rows])
    val_reconstruction = np.asarray([float(row["validation_reconstruction_loss"]) for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
    axes[0].plot(epochs, train_loss, linewidth=2.0, label="Train")
    axes[0].plot(epochs, val_loss, linewidth=2.0, label="Validation")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, train_reconstruction, linewidth=2.0, label="Train")
    axes[1].plot(epochs, val_reconstruction, linewidth=2.0, label="Validation")
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("Training Curves")
    fig.tight_layout()
    save_figure(fig, "training_curves")


def plot_parameter_comparison(comparison_rows: List[Dict[str, str]]) -> None:
    names = [row["model"] for row in comparison_rows]
    total = np.asarray([float(row["total_parameters"]) for row in comparison_rows])
    quantum = np.asarray([float(row["quantum_parameters"]) for row in comparison_rows])
    x = np.arange(len(names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.bar(x - width / 2, total, width, label="Total Parameters")
    ax.bar(x + width / 2, quantum, width, label="Quantum Parameters")
    ax.set_title("Parameter Comparison")
    ax.set_ylabel("Parameter Count")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "parameter_comparison")


def plot_training_time(comparison_rows: List[Dict[str, str]]) -> None:
    names = [row["model"] for row in comparison_rows]
    values = [float(row["training_time_seconds"]) for row in comparison_rows]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    bars = ax.bar(names, values, color=["#4C78A8", "#F58518"])
    ax.set_title("Training Time")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=FONT_SIZES["annotation"])
    save_figure(fig, "training_time")


def plot_confusion_matrix(slug: str, title: str, matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Normal", "Anomaly"])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=FONT_SIZES["annotation"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, slug)


def plot_confusion_matrices(results: Dict[str, Any]) -> None:
    for model_key, title in [("classical", "Confusion Matrix: Classical"), ("hybrid", "Confusion Matrix: Hybrid Quantum")]:
        matrix = np.asarray(results["models"][model_key]["metrics"]["confusion_matrix"]["matrix"], dtype=int)
        plot_confusion_matrix(f"confusion_matrix_{model_key}", title, matrix)


def plot_score_distribution(slug: str, title: str, csv_path: Path) -> None:
    rows = read_csv(csv_path)
    normal_scores = [float(row["anomaly_score"]) for row in rows if int(row["ground_truth"]) == 0]
    anomaly_scores = [float(row["anomaly_score"]) for row in rows if int(row["ground_truth"]) == 1]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.hist(normal_scores, bins=60, density=True, alpha=0.68, label="Normal", color="#4C78A8")
    ax.hist(anomaly_scores, bins=60, density=True, alpha=0.68, label="Anomaly", color="#F58518")
    ax.set_title(title)
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, slug)


def pca_projection(latent: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=42).fit_transform(latent)


def tsne_projection(latent: np.ndarray) -> np.ndarray:
    return TSNE(
        n_components=2,
        perplexity=35,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(latent)


def plot_embedding(slug: str, title: str, embedding: np.ndarray, labels: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    normal = labels == 0
    anomaly = labels == 1
    ax.scatter(embedding[normal, 0], embedding[normal, 1], s=8, alpha=0.55, label="Normal", color="#4C78A8", linewidths=0)
    ax.scatter(embedding[anomaly, 0], embedding[anomaly, 1], s=12, alpha=0.75, label="Anomaly", color="#F58518", linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False, markerscale=2)
    save_figure(fig, slug)


def plot_latent_figures() -> None:
    latent_dir = REPO_ROOT / "results" / "latent_analysis"
    labels = np.load(latent_dir / "labels.npy")
    for model_key, label in [("classical", "Classical"), ("hybrid", "Hybrid Quantum")]:
        latent = np.load(latent_dir / f"{model_key}_latent.npy")
        plot_embedding(f"pca_{model_key}", f"Latent PCA: {label}", pca_projection(latent), labels)
        plot_embedding(f"tsne_{model_key}", f"Latent t-SNE: {label}", tsne_projection(latent), labels)


def write_manifest() -> None:
    entries = []
    for spec in FIGURE_SPECS:
        for extension in EXPORT_FORMATS:
            filename = f"{spec['slug']}.{extension}"
            entries.append({
                "filename": filename,
                "description": spec["description"],
                "paper_section": spec["section"],
                "figure_number_placeholder": spec["figure_number"],
            })
    with open(OUTPUT_DIR / "figure_manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"figures": entries}, handle, indent=2)


def verify_exports() -> None:
    missing = []
    for spec in FIGURE_SPECS:
        for extension in EXPORT_FORMATS:
            path = OUTPUT_DIR / f"{spec['slug']}.{extension}"
            if not path.exists() or path.stat().st_size == 0:
                missing.append(str(path))
    manifest = OUTPUT_DIR / "figure_manifest.json"
    if not manifest.exists() or manifest.stat().st_size == 0:
        missing.append(str(manifest))
    if missing:
        raise RuntimeError("Missing exported files:\n" + "\n".join(missing))


def main() -> int:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = read_json(REPO_ROOT / "results" / "FinalResults" / "results.json")
    comparison_rows = read_csv(REPO_ROOT / "results" / "FinalResults" / "comparison_table.csv")

    plot_roc_curve(results)
    plot_pr_curve(results)
    plot_training_curves()
    plot_parameter_comparison(comparison_rows)
    plot_training_time(comparison_rows)
    plot_confusion_matrices(results)
    plot_score_distribution(
        "score_distribution_classical",
        "Score Distribution: Classical",
        REPO_ROOT / "results" / "error_analysis" / "classical_sample_errors.csv",
    )
    plot_score_distribution(
        "score_distribution_hybrid",
        "Score Distribution: Hybrid Quantum",
        REPO_ROOT / "results" / "error_analysis" / "hybrid_sample_errors.csv",
    )
    plot_latent_figures()
    write_manifest()
    verify_exports()

    exported_count = len(FIGURE_SPECS) * len(EXPORT_FORMATS)
    print(f"Exported {exported_count} figure files to {OUTPUT_DIR}")
    print(f"Manifest: {OUTPUT_DIR / 'figure_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
