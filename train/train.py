"""
Training Script for LSTM Autoencoder

Trains the anomaly detection model on NASA Turbofan dataset.
Outputs: model.pt, norm_stats.json, global_stats.json, loss.csv, training_curve.png
"""
import os
import sys
import json
import argparse
import time
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Tuple, Optional
import csv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import load_and_prepare_data
from preprocess import preprocess_data
from window import create_windows_by_engine, split_windows
from model import create_model, save_model, LSTMAutoencoder, compute_kl_loss
from quantum_layer import QuantumLatentLayer

# Optional: matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception as exc:
    HAS_MATPLOTLIB = False
    print(f"Warning: matplotlib not available ({exc}), training curve plot will not be generated")


# Configuration
class Config:
    # Data
    DATA_DIR = "data"
    DATASET = "FD001"
    HEALTHY_RATIO = 0.7  # Use first 70% of cycles as healthy
    
    # Windowing
    WINDOW_SIZE = 50
    STRIDE = 1
    
    # Model - IMPROVED
    HIDDEN_DIM = 128      # Increased from 64
    LATENT_DIM = 64       # Increased from 32
    NUM_LAYERS = 2        # Increased from 1
    DROPOUT = 0.2         # Increased from 0.1
    
    # Training - RESEARCH
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 0.001
    USE_QUANTUM = False
    NUM_QUBITS = 4
    NUM_QUANTUM_LAYERS = 2
    RANDOM_SEED = 42
    SMOKE_TEST = False
    SMOKE_TEST_EPOCHS = 3
    SMOKE_TEST_BATCH_SIZE = 8
    SMOKE_TEST_WINDOW_FRACTION = 0.1
    WEIGHT_DECAY = 1e-5
    TRAIN_RATIO = 0.8
    BETA = 0.001         # Weight for KL divergence in VAE loss
    MAX_BETA = 0.001     # Maximum KL weight for beta scheduling
    BETA_SCHEDULER = "linear"  # Scheduler type for future extensions
    CONTRASTIVE_WEIGHT = 0.1  # Weight for contrastive loss
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 1e-4
    
    # Output
    RESULTS_DIR = "results"
    MODEL_PATH = os.path.join("train", "classical_model.pt")
    CLASSICAL_MODEL_PATH = os.path.join("train", "classical_model.pt")
    HYBRID_MODEL_PATH = os.path.join("train", "hybrid_model.pt")
    MODEL_METADATA_PATH = os.path.join("train", "classical_model_metadata.json")
    CLASSICAL_MODEL_METADATA_PATH = os.path.join("train", "classical_model_metadata.json")
    HYBRID_MODEL_METADATA_PATH = os.path.join("train", "hybrid_model_metadata.json")
    NORM_STATS_PATH = "norm_stats.json"
    GLOBAL_STATS_PATH = "global_stats.json"
    EXPERIMENT_CONFIG_PATH = os.path.join(RESULTS_DIR, "experiment_config.json")
    MODEL_STATISTICS_PATH = os.path.join(RESULTS_DIR, "model_statistics.json")
    TRAINING_METRICS_PATH = os.path.join(RESULTS_DIR, "training_metrics.json")
    TRAINING_SUMMARY_PATH = os.path.join(RESULTS_DIR, "training_summary.json")
    SMOKE_TEST_MODEL_PATH = os.path.join("train", "model_smoke_test.pt")
    SMOKE_TEST_NORM_STATS_PATH = "norm_stats_smoke_test.json"
    SMOKE_TEST_GLOBAL_STATS_PATH = "global_stats_smoke_test.json"
    
    # Reproducibility
    SEED = 42
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 1e-4


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore_weights(self, model: nn.Module) -> None:
        """Restore model to best weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def get_beta(epoch: int, max_epochs: int, max_beta: float) -> float:
    """Linearly ramp beta from 0 up to the requested maximum across training epochs."""
    if max_epochs <= 1:
        return max_beta
    progress = min(max(epoch, 0) / float(max_epochs), 1.0)
    return min(max_beta * progress, max_beta)


def compute_vae_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    beta: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the VAE loss as reconstruction loss + beta * KL divergence."""
    reconstruction_loss = nn.functional.mse_loss(reconstructed, original)
    kl_loss = compute_kl_loss(mean, logvar)
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def count_positive_negative_pairs(engine_ids: torch.Tensor) -> Tuple[int, int]:
    """Count same-engine and different-engine pairs within a batch."""
    if engine_ids.numel() < 2:
        return 0, 0

    engine_list = engine_ids.detach().cpu().tolist()
    positive_pairs = 0
    negative_pairs = 0
    for i in range(len(engine_list)):
        for j in range(i + 1, len(engine_list)):
            if engine_list[i] == engine_list[j]:
                positive_pairs += 1
            else:
                negative_pairs += 1
    return positive_pairs, negative_pairs


def compute_contrastive_loss(
    latent_vectors: torch.Tensor,
    engine_ids: torch.Tensor,
    temperature: float = 0.2
) -> Tuple[torch.Tensor, bool, int, int]:
    """Lightweight InfoNCE-style contrastive loss using cosine similarity in mini-batches."""
    positive_pairs, negative_pairs = count_positive_negative_pairs(engine_ids)
    if latent_vectors.size(0) < 2 or positive_pairs == 0:
        return latent_vectors.new_zeros(()), False, positive_pairs, negative_pairs

    embeddings = torch.nn.functional.normalize(latent_vectors, dim=1)
    batch_size = embeddings.size(0)

    positive_indices = []
    for i in range(batch_size):
        same_engine = torch.nonzero(engine_ids == engine_ids[i], as_tuple=False).flatten()
        same_engine = same_engine[same_engine != i]
        if same_engine.numel() == 0:
            positive_indices.append(-1)
        else:
            positive_indices.append(int(same_engine[0].item()))

    valid_indices = [i for i, idx in enumerate(positive_indices) if idx >= 0]
    if not valid_indices:
        return latent_vectors.new_zeros(()), False, positive_pairs, negative_pairs

    anchor_embeddings = embeddings[valid_indices]
    similarity = torch.matmul(anchor_embeddings, embeddings.t()) / temperature

    # Mask self-similarity so each anchor does not target itself.
    self_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
    similarity = similarity.masked_fill(self_mask[valid_indices], -1e9)

    targets = torch.tensor([positive_indices[i] for i in valid_indices], device=embeddings.device, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(similarity, targets)
    return loss, True, positive_pairs, negative_pairs


class EngineGroupedBatchSampler:
    """Create mini-batches that include multiple windows from the same engine when possible."""

    def __init__(self, engine_ids: np.ndarray, batch_size: int, shuffle: bool = True, seed: Optional[int] = None):
        self.engine_ids = np.asarray(engine_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        grouped_indices = {}
        for index, engine_id in enumerate(self.engine_ids):
            grouped_indices.setdefault(int(engine_id), []).append(index)

        if self.batch_size <= 1:
            for index in range(0, len(self.engine_ids), self.batch_size or 1):
                yield list(range(index, min(index + (self.batch_size or 1), len(self.engine_ids))))
            return

        engine_ids = list(grouped_indices.keys())
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(engine_ids)

        while True:
            available_engines = [engine_id for engine_id in engine_ids if grouped_indices[engine_id]]
            if not available_engines:
                break

            if self.shuffle:
                rng.shuffle(available_engines)

            anchor_engine = available_engines[0]
            primary_count = min(max(2, self.batch_size // 2), len(grouped_indices[anchor_engine]))
            batch = []

            for _ in range(primary_count):
                if not grouped_indices[anchor_engine]:
                    break
                batch.append(grouped_indices[anchor_engine].pop(0))

            remaining_slots = self.batch_size - len(batch)
            while remaining_slots > 0:
                candidates = [engine_id for engine_id in available_engines if grouped_indices[engine_id]]
                if not candidates:
                    break
                if self.shuffle:
                    rng.shuffle(candidates)
                candidate_engine = candidates[0]
                if not grouped_indices[candidate_engine]:
                    continue
                batch.append(grouped_indices[candidate_engine].pop(0))
                remaining_slots -= 1

            if batch:
                yield batch

    def __len__(self) -> int:
        return max(1, (len(self.engine_ids) + self.batch_size - 1) // self.batch_size)


def create_dataloader(
    windows: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    engine_ids: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> DataLoader:
    """Create PyTorch DataLoader from windows and optional per-window engine IDs."""
    tensor = torch.FloatTensor(windows)
    if engine_ids is None:
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    engine_tensor = torch.as_tensor(engine_ids, dtype=torch.long)
    dataset = TensorDataset(tensor, engine_tensor)
    sampler = EngineGroupedBatchSampler(engine_ids, batch_size=batch_size, shuffle=shuffle, seed=seed)
    loader = DataLoader(dataset, batch_sampler=sampler)
    return loader


def train_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    beta: float,
    contrastive_weight: float = 0.1
) -> Tuple[float, float, float, float, float]:
    """Train for one epoch and track total, reconstruction, KL, contrastive losses, and batch timing."""
    model.train()
    total_loss = 0.0
    reconstruction_loss = 0.0
    kl_loss = 0.0
    contrastive_loss = 0.0
    num_batches = 0
    batch_time_total = 0.0

    for batch_idx, batch in enumerate(loader):
        batch_start = time.time()
        x = batch[0].to(device)

        # Forward pass
        reconstructed, mean, logvar, _ = model(x)
        loss, recon_loss, kld_loss = compute_vae_loss(x, reconstructed, mean, logvar, beta)

        # Contrastive loss over latent means (same engine positives, different engine negatives)
        latent_vectors = mean
        batch_engine_ids = batch[1].to(device) if len(batch) > 1 else torch.arange(x.size(0), device=device)
        contrastive, has_positive_pairs, positive_pairs, negative_pairs = compute_contrastive_loss(
            latent_vectors,
            batch_engine_ids,
        )

        if batch_idx == 0:
            unique_engines = torch.unique(batch_engine_ids).numel()
            print(f"Unique Engines : {unique_engines}")
            print(f"Positive Pairs : {positive_pairs}")
            print(f"Negative Pairs : {negative_pairs}")

        if not has_positive_pairs:
            print("Warning: batch has zero positive pairs; skipping contrastive loss for this batch.")
            weighted_contrastive = 0.0
            total_loss_value = loss
        else:
            weighted_contrastive = contrastive_weight * contrastive
            total_loss_value = loss + weighted_contrastive

            if batch_idx == 0:
                print(f"Contrastive Raw = {contrastive.item():.6f}")
                print(f"Raw: {contrastive.item():.6f}")
                print(f"Weight: {contrastive_weight:.6f}")
                print(f"Weighted: {weighted_contrastive.item():.6f}")
                print(
                    f"Total Loss = Recon + Beta x KL + ContrastiveWeight x Contrastive = "
                    f"{recon_loss.item():.6f} + {beta * kld_loss.item():.6f} + {weighted_contrastive.item():.6f}"
                )

        # Backward pass
        optimizer.zero_grad()
        total_loss_value.backward()

        encoder_grad_norm = torch.sqrt(sum(
            p.grad.pow(2).sum() for p in model.encoder.parameters() if p.grad is not None
        )).item() if any(p.grad is not None for p in model.encoder.parameters()) else 0.0
        if batch_idx == 0:
            print(f"Gradient Norm = {encoder_grad_norm:.6f}")

        optimizer.step()

        batch_time_total += time.time() - batch_start
        total_loss += total_loss_value.item()
        reconstruction_loss += recon_loss.item()
        kl_loss += kld_loss.item()
        contrastive_loss += contrastive.item()
        num_batches += 1

    return total_loss / num_batches, reconstruction_loss / num_batches, kl_loss / num_batches, contrastive_loss / num_batches, batch_time_total


def validate(
    model: LSTMAutoencoder,
    loader: DataLoader,
    device: str,
    beta: float,
    contrastive_weight: float = 0.1
) -> Tuple[float, float, float, float]:
    """Validate the model and track total, reconstruction, KL, and contrastive losses."""
    model.eval()
    total_loss = 0.0
    reconstruction_loss = 0.0
    kl_loss = 0.0
    contrastive_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            reconstructed, mean, logvar, _ = model(x)
            loss, recon_loss, kld_loss = compute_vae_loss(x, reconstructed, mean, logvar, beta)
            batch_engine_ids = batch[1].to(device) if len(batch) > 1 else torch.arange(x.size(0), device=device)
            contrastive, has_positive_pairs, _, _ = compute_contrastive_loss(mean, batch_engine_ids)
            if has_positive_pairs:
                total_loss += (loss + contrastive_weight * contrastive).item()
                contrastive_loss += contrastive.item()
            else:
                total_loss += loss.item()
            reconstruction_loss += recon_loss.item()
            kl_loss += kld_loss.item()
            num_batches += 1

    return total_loss / num_batches, reconstruction_loss / num_batches, kl_loss / num_batches, contrastive_loss / num_batches


def compute_reconstruction_errors(
    model: LSTMAutoencoder,
    windows: np.ndarray,
    device: str,
    batch_size: int = 256
) -> np.ndarray:
    """Compute reconstruction error for all windows."""
    model.eval()
    errors = []
    
    loader = create_dataloader(windows, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            reconstructed, _, _, _ = model(x)
            
            # MSE per sample (average over sequence and features)
            batch_errors = ((reconstructed - x) ** 2).mean(dim=(1, 2))
            errors.extend(batch_errors.cpu().numpy().tolist())
    
    return np.array(errors)


def compute_global_stats(errors: np.ndarray) -> dict:
    """
    Compute global statistics from reconstruction errors.
    
    These are used for cold-start thresholding.
    """
    stats = {
        "global_mean": float(np.mean(errors)),
        "global_std": float(np.std(errors)),
        "num_samples": len(errors),
        "min_error": float(np.min(errors)),
        "max_error": float(np.max(errors)),
        "percentiles": {
            "p50": float(np.percentile(errors, 50)),
            "p90": float(np.percentile(errors, 90)),
            "p95": float(np.percentile(errors, 95)),
            "p99": float(np.percentile(errors, 99))
        },
        "metadata": {
            "description": "Global reconstruction error statistics for cold-start threshold",
            "dataset": "NASA Turbofan FD001"
        }
    }
    
    return stats


def save_global_stats(stats: dict, filepath: str) -> None:
    """Save global statistics to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved global stats to {filepath}")


def save_json(data: dict, filepath: str) -> None:
    """Save a generic dictionary to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON file to {filepath}")


def parse_bool(value) -> bool:
    """Parse boolean command-line values such as True/False."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: True or False")


def configure_model_outputs(config: Config) -> None:
    """Route model artifacts to classical or hybrid filenames."""
    if config.USE_QUANTUM:
        config.MODEL_PATH = config.HYBRID_MODEL_PATH
        config.MODEL_METADATA_PATH = config.HYBRID_MODEL_METADATA_PATH
    else:
        config.MODEL_PATH = config.CLASSICAL_MODEL_PATH
        config.MODEL_METADATA_PATH = config.CLASSICAL_MODEL_METADATA_PATH


def verify_checkpoint_strict(filepath: str, device: str) -> None:
    """Reload a saved checkpoint with strict state-dict validation."""
    checkpoint = torch.load(filepath, map_location=device)
    model_config = checkpoint.get("model_config", {})
    verification_model = LSTMAutoencoder(
        input_dim=checkpoint.get("input_dim", model_config.get("input_dim", 16)),
        hidden_dim=checkpoint.get("hidden_dim", model_config.get("hidden_dim", 64)),
        latent_dim=checkpoint.get("latent_dim", model_config.get("latent_dim", 32)),
        num_layers=checkpoint.get("num_layers", model_config.get("num_layers", 1)),
        dropout=checkpoint.get("dropout", model_config.get("dropout", 0.1)),
        use_quantum=checkpoint.get("use_quantum", model_config.get("use_quantum", False)),
    ).to(device)

    missing, unexpected = verification_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    print(f"Missing keys : {len(missing)}")
    print(f"Unexpected keys : {len(unexpected)}")

    model_type = "Hybrid" if verification_model.use_quantum else "Classical"
    print(f"{model_type} checkpoint verified successfully")


def save_training_history(history: dict, filepath: str) -> None:
    """Save training history to CSV file."""
    headers = [
        'epoch',
        'train_loss',
        'train_reconstruction_loss',
        'train_kl_loss',
        'train_contrastive_loss',
        'validation_total_loss',
        'validation_reconstruction_loss',
        'validation_kl_loss',
        'validation_contrastive_loss',
        'beta',
        'learning_rate'
    ]
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        num_rows = len(history.get('train_loss', []))
        for i in range(num_rows):
            writer.writerow([
                i + 1,
                history.get('train_loss', [None] * num_rows)[i],
                history.get('train_reconstruction_loss', [None] * num_rows)[i],
                history.get('train_kl_loss', [None] * num_rows)[i],
                history.get('train_contrastive_loss', [None] * num_rows)[i],
                history.get('val_loss', [None] * num_rows)[i],
                history.get('val_reconstruction_loss', [None] * num_rows)[i],
                history.get('val_kl_loss', [None] * num_rows)[i],
                history.get('val_contrastive_loss', [None] * num_rows)[i],
                history.get('beta', [None] * num_rows)[i],
                history.get('learning_rate', [None] * num_rows)[i]
            ])
    print(f"Saved training history to {filepath}")


def plot_training_curve(history: dict, filepath: str) -> None:
    """Plot and save training curves, contrastive loss, and beta schedule."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not available)")
        return

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Total losses
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Total')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Total')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction losses
    axes[0, 1].plot(epochs, history['train_reconstruction_loss'], 'b-', label='Train Reconstruction')
    axes[0, 1].plot(epochs, history['val_reconstruction_loss'], 'r-', label='Val Reconstruction')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Contrastive losses
    axes[1, 0].plot(epochs, history.get('train_contrastive_loss', []), 'b-', label='Train Contrastive')
    axes[1, 0].plot(epochs, history.get('val_contrastive_loss', []), 'r-', label='Val Contrastive')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Contrastive Loss')
    axes[1, 0].set_title('Contrastive Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # KL losses
    axes[1, 1].plot(epochs, history['train_kl_loss'], 'b-', label='Train KL')
    axes[1, 1].plot(epochs, history['val_kl_loss'], 'r-', label='Val KL')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('KL Loss')
    axes[1, 1].set_title('KL Divergence Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved training curve to {filepath}")

    contrastive_path = filepath.replace('training_curve.png', 'contrastive_loss.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history.get('train_contrastive_loss', []), 'b-', label='Train Contrastive')
    ax.plot(epochs, history.get('val_contrastive_loss', []), 'r-', label='Val Contrastive')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Contrastive Loss')
    ax.set_title('Contrastive Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(contrastive_path, dpi=150)
    plt.close()
    print(f"Saved contrastive loss plot to {contrastive_path}")

    # Separate beta schedule plot
    beta_path = filepath.replace('training_curve.png', 'beta_schedule.png')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, history.get('beta', []), color='tab:green', marker='o', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Beta')
    ax.set_title('Beta Schedule')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(beta_path, dpi=150)
    plt.close()
    print(f"Saved beta schedule plot to {beta_path}")


def subset_windows(windows: np.ndarray, engine_ids: np.ndarray, fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce the dataset to a deterministic fraction of windows for smoke tests."""
    if len(windows) <= 1:
        return windows, engine_ids

    subset_size = max(1, int(len(windows) * fraction))
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(windows), size=subset_size, replace=False)
    return windows[indices], engine_ids[indices]


def train(config: Config) -> None:
    """Main training function."""
    configure_model_outputs(config)
    model_type = "hybrid" if config.USE_QUANTUM else "classical"

    print("=" * 60)
    print("LSTM Autoencoder Training")
    print("=" * 60)
    
    # Ensure result directories exist early
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Print config values for reproducibility
    print("\nExperiment Configuration:")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"  Hidden dimension: {config.HIDDEN_DIM}")
    print(f"  Latent dimension: {config.LATENT_DIM}")
    print(f"  Num layers: {config.NUM_LAYERS}")
    print(f"  Dropout: {config.DROPOUT}")
    print(f"  Use quantum: {config.USE_QUANTUM}")
    print(f"  Num qubits: {config.NUM_QUBITS}")
    print(f"  Num quantum layers: {config.NUM_QUANTUM_LAYERS}")
    print(f"  Random seed: {config.RANDOM_SEED}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Beta: {config.BETA}")
    print(f"  Max beta: {config.MAX_BETA}")
    print(f"  Contrastive weight: {config.CONTRASTIVE_WEIGHT}")
    print(f"  Dataset: {config.DATASET}")
    print(f"  Window size: {config.WINDOW_SIZE}")
    print(f"  Stride: {config.STRIDE}")
    print(f"  Train ratio: {config.TRAIN_RATIO}")
    print(f"  Model path: {config.MODEL_PATH}")
    print(f"  Metadata path: {config.MODEL_METADATA_PATH}")

    # Set seed
    set_seed(config.RANDOM_SEED)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    df, features = load_and_prepare_data(
        data_dir=config.DATA_DIR,
        dataset=config.DATASET,
        healthy_ratio=config.HEALTHY_RATIO
    )
    num_features = len(features)
    print(f"Features: {features}")
    
    # Step 2: Preprocess
    print("\n[Step 2] Preprocessing...")
    from preprocess import FeatureNormalizer
    normalizer = FeatureNormalizer()
    df_sorted, normalized, _ = preprocess_data(df, features, None)
    normalizer.fit(df_sorted, features)
    normalized = normalizer.transform(df_sorted)
    
    # Save normalization stats
    normalizer.save_stats(config.NORM_STATS_PATH)
    
    # Step 3: Create windows
    print("\n[Step 3] Creating windows...")
    windows, engine_ids = create_windows_by_engine(
        df_sorted, normalized,
        window_size=config.WINDOW_SIZE,
        stride=config.STRIDE
    )
    
    if config.SMOKE_TEST:
        print("\n[Step 3.5] Running smoke-test subset...")
        windows, engine_ids = subset_windows(
            windows,
            engine_ids,
            fraction=config.SMOKE_TEST_WINDOW_FRACTION,
            seed=config.SEED,
        )
        print(f"Using {len(windows)} windows for smoke test ({config.SMOKE_TEST_WINDOW_FRACTION * 100:.1f}% of available windows)")

    # Step 4: Split data
    print("\n[Step 4] Splitting data...")
    train_windows, val_windows, train_ids, val_ids = split_windows(
        windows, engine_ids,
        train_ratio=config.TRAIN_RATIO,
        seed=config.SEED
    )
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_windows,
        config.BATCH_SIZE,
        shuffle=True,
        engine_ids=train_ids
    )
    val_loader = create_dataloader(
        val_windows,
        config.BATCH_SIZE,
        shuffle=False,
        engine_ids=val_ids
    )
    
    if config.SMOKE_TEST:
        config.BATCH_SIZE = config.SMOKE_TEST_BATCH_SIZE
        config.EPOCHS = config.SMOKE_TEST_EPOCHS
        config.EARLY_STOPPING_PATIENCE = 1
        config.MODEL_PATH = config.SMOKE_TEST_MODEL_PATH
        config.NORM_STATS_PATH = config.SMOKE_TEST_NORM_STATS_PATH
        config.GLOBAL_STATS_PATH = config.SMOKE_TEST_GLOBAL_STATS_PATH
        print("Smoke test mode enabled")
        print(f"  Epochs: {config.EPOCHS}")
        print(f"  Batch size: {config.BATCH_SIZE}")
        print(f"  Windows used: {len(windows)}")

    experiment_metadata = {
        "experiment_name": f"{model_type}_vae_full_run",
        "model_type": model_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "use_quantum": config.USE_QUANTUM,
        "quantum_enabled": config.USE_QUANTUM,
        "num_qubits": config.NUM_QUBITS,
        "quantum_layers": config.NUM_QUANTUM_LAYERS,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "beta": config.BETA,
        "contrastive_weight": config.CONTRASTIVE_WEIGHT,
        "random_seed": config.RANDOM_SEED,
        "dataset": config.DATASET,
        "window_size": config.WINDOW_SIZE,
        "stride": config.STRIDE,
        "train_ratio": config.TRAIN_RATIO,
        "dataset_size": len(windows),
        "training_device": device,
        "smoke_test": config.SMOKE_TEST
    }
    save_json(experiment_metadata, config.EXPERIMENT_CONFIG_PATH)

    # Step 5: Create model
    print("\n[Step 5] Creating model...")
    model = create_model(
        input_dim=num_features,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        use_quantum=config.USE_QUANTUM,
        device=device
    )

    if config.USE_QUANTUM and hasattr(model, 'quantum_layer') and model.quantum_layer is not None:
        print("Configuring quantum layer to research settings")
        model.quantum_layer = QuantumLatentLayer(
            num_qubits=config.NUM_QUBITS,
            num_layers=config.NUM_QUANTUM_LAYERS,
            input_dim=model.quantum_layer.input_dim,
            output_dim=model.quantum_layer.output_dim,
            device_name=model.quantum_layer.device_name,
        )

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum_parameters = model.count_quantum_parameters() if hasattr(model, 'count_quantum_parameters') else 0
    classical_parameters = trainable_parameters - quantum_parameters

    model_statistics = {
        "model_type": model_type,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "quantum_parameters": quantum_parameters,
        "classical_parameters": classical_parameters,
        "use_quantum": config.USE_QUANTUM,
        "quantum_enabled": config.USE_QUANTUM,
        "num_qubits": config.NUM_QUBITS,
        "quantum_layers": config.NUM_QUANTUM_LAYERS
    }
    print("Model Statistics:")
    print(f"  Total parameters: {total_parameters}")
    print(f"  Trainable parameters: {trainable_parameters}")
    print(f"  Quantum parameters: {quantum_parameters}")
    print(f"  Classical parameters: {classical_parameters}")
    save_json(model_statistics, config.MODEL_STATISTICS_PATH)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        restore_best_weights=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_reconstruction_loss': [],
        'train_kl_loss': [],
        'train_contrastive_loss': [],
        'val_loss': [],
        'val_reconstruction_loss': [],
        'val_kl_loss': [],
        'val_contrastive_loss': [],
        'beta': [],
        'learning_rate': []
    }
    
    training_start_time = time.time()
    epoch_durations = []
    total_batch_time = 0.0
    total_batches = 0

    # Step 6: Training...
    print("\n[Step 6] Training...")
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(config.EPOCHS):
        epoch_start = time.time()
        beta = get_beta(epoch, config.EPOCHS, config.MAX_BETA)

        train_loader = create_dataloader(
            train_windows,
            config.BATCH_SIZE,
            shuffle=True,
            engine_ids=train_ids,
            seed=config.SEED + epoch
        )
        val_loader = create_dataloader(
            val_windows,
            config.BATCH_SIZE,
            shuffle=False,
            engine_ids=val_ids,
            seed=config.SEED + epoch
        )

        train_loss, train_recon_loss, train_kl_loss, train_contrastive_loss, batch_time = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta,
            contrastive_weight=config.CONTRASTIVE_WEIGHT
        )
        val_loss, val_recon_loss, val_kl_loss, val_contrastive_loss = validate(
            model,
            val_loader,
            device,
            beta,
            contrastive_weight=config.CONTRASTIVE_WEIGHT
        )
        
        scheduler.step(val_loss)
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_reconstruction_loss'].append(train_recon_loss)
        history['train_kl_loss'].append(train_kl_loss)
        history['train_contrastive_loss'].append(train_contrastive_loss)
        history['val_loss'].append(val_loss)
        history['val_reconstruction_loss'].append(val_recon_loss)
        history['val_kl_loss'].append(val_kl_loss)
        history['val_contrastive_loss'].append(val_contrastive_loss)
        history['beta'].append(beta)
        history['learning_rate'].append(current_lr)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            marker = " *"
        else:
            marker = ""

        epoch_duration = time.time() - epoch_start
        epoch_durations.append(epoch_duration)
        total_batch_time += batch_time
        total_batches += int(np.ceil(len(train_windows) / float(config.BATCH_SIZE)))
        
        print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
              f"Beta: {beta:.6f} | "
              f"Train Reconstruction: {train_recon_loss:.6f} | "
              f"Train KL: {train_kl_loss:.6f} | "
              f"Train Contrastive: {train_contrastive_loss:.6f} | "
              f"Train Total: {train_loss:.6f} | "
              f"Validation Reconstruction: {val_recon_loss:.6f} | "
              f"Validation KL: {val_kl_loss:.6f} | "
              f"Validation Contrastive: {val_contrastive_loss:.6f} | "
              f"Validation Total: {val_loss:.6f}{marker}")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {early_stopping.best_loss:.6f}")
            break
    
    # Restore best model weights
    early_stopping.restore_weights(model)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    
    # Save training history
    print("\n[Step 7] Saving training history...")
    save_training_history(history, "loss.csv")
    plot_training_curve(history, "training_curve.png")

    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    average_epoch_duration = sum(epoch_durations) / len(epoch_durations) if epoch_durations else 0.0
    average_batch_duration = total_batch_time / total_batches if total_batches else 0.0

    training_metrics = {
        "training_start_time": datetime.fromtimestamp(training_start_time, timezone.utc).isoformat(),
        "training_end_time": datetime.fromtimestamp(training_end_time, timezone.utc).isoformat(),
        "total_duration_seconds": training_duration,
        "average_epoch_duration_seconds": average_epoch_duration,
        "average_batch_duration_seconds": average_batch_duration,
        "epochs_completed": len(epoch_durations),
        "total_batches": total_batches,
        "device": device
    }
    save_json(training_metrics, config.TRAINING_METRICS_PATH)

    # Step 8: Compute reconstruction errors on training data
    print("\n[Step 8] Computing reconstruction errors...")
    all_windows = windows  # Use all windows for global stats
    errors = compute_reconstruction_errors(model, all_windows, device)
    
    # Compute global stats
    global_stats = compute_global_stats(errors)
    print(f"Global Mean: {global_stats['global_mean']:.6f}")
    print(f"Global Std: {global_stats['global_std']:.6f}")
    print(f"95th percentile: {global_stats['percentiles']['p95']:.6f}")
    
    # Save global stats
    save_global_stats(global_stats, config.GLOBAL_STATS_PATH)
    
    # Step 9: Save model
    print("\n[Step 9] Saving model...")
    norm_stats = normalizer.get_stats()
    save_model(
        model,
        config.MODEL_PATH,
        feature_mean=norm_stats["feature_mean"],
        feature_std=norm_stats["feature_std"]
    )
    verify_checkpoint_strict(config.MODEL_PATH, device)

    model_metadata = {
        "model_type": model_type,
        "training_timestamp": datetime.fromtimestamp(training_end_time, timezone.utc).isoformat(),
        "dataset": config.DATASET,
        "window_size": config.WINDOW_SIZE,
        "feature_names": features,
        "hidden_dim": config.HIDDEN_DIM,
        "latent_dim": config.LATENT_DIM,
        "num_layers": config.NUM_LAYERS,
        "dropout": config.DROPOUT,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "random_seed": config.RANDOM_SEED,
        "best_validation_loss": best_val_loss,
        "best_epoch": best_epoch,
        "training_time_seconds": training_duration,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "quantum_parameters": quantum_parameters,
        "classical_parameters": classical_parameters,
        "use_quantum": config.USE_QUANTUM,
        "quantum_enabled": config.USE_QUANTUM,
        "num_qubits": config.NUM_QUBITS,
        "num_quantum_layers": config.NUM_QUANTUM_LAYERS
    }
    save_json(model_metadata, config.MODEL_METADATA_PATH)

    training_summary = {
        "model_type": model_type,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "epochs_completed": len(epoch_durations),
        "early_stopped": early_stopping.early_stop,
        "training_time_seconds": training_duration,
        "use_quantum": config.USE_QUANTUM,
        "quantum_enabled": config.USE_QUANTUM
    }
    save_json(training_summary, config.TRAINING_SUMMARY_PATH)
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Outputs:")
    print(f"  - {config.MODEL_PATH}")
    print(f"  - {config.NORM_STATS_PATH}")
    print(f"  - {config.GLOBAL_STATS_PATH}")
    print(f"  - loss.csv")
    print(f"  - training_curve.png")
    print(f"\nModel performance:")
    print(f"  - Best validation loss: {best_val_loss:.6f}")
    print(f"  - Global reconstruction error: {global_stats['global_mean']:.6f} ± {global_stats['global_std']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=Config.HIDDEN_DIM, help="LSTM hidden dimension")
    parser.add_argument("--latent-dim", type=int, default=Config.LATENT_DIM, help="Latent dimension")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--seed", type=int, default=Config.RANDOM_SEED, help="Random seed")
    parser.add_argument("--early-stopping-patience", type=int, default=Config.EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4, help="Early stopping min delta")
    parser.add_argument("--beta", type=float, default=Config.BETA, help="Weight for KL divergence in VAE loss")
    parser.add_argument("--use-quantum", type=parse_bool, default=Config.USE_QUANTUM, help="Enable the hybrid quantum layer: True or False")
    parser.add_argument("--smoke-test", action="store_true", help="Train a tiny smoke test run")
    args = parser.parse_args()
    
    # Update config
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.HIDDEN_DIM = args.hidden_dim
    config.LATENT_DIM = args.latent_dim
    config.DATA_DIR = args.data_dir
    config.SEED = args.seed
    config.RANDOM_SEED = args.seed
    config.EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    config.EARLY_STOPPING_MIN_DELTA = args.early_stopping_min_delta
    config.BETA = args.beta
    config.MAX_BETA = args.beta
    config.USE_QUANTUM = args.use_quantum
    config.SMOKE_TEST = args.smoke_test
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
