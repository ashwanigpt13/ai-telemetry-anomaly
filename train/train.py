"""
Training Script for LSTM Autoencoder

Trains the anomaly detection model on NASA Turbofan dataset.
Outputs: model.pt, norm_stats.json, global_stats.json
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import load_and_prepare_data
from preprocess import preprocess_data
from window import create_windows_by_engine, split_windows
from model import create_model, save_model, LSTMAutoencoder


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
    
    # Training - IMPROVED
    BATCH_SIZE = 32
    EPOCHS = 100          # Increased from 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    TRAIN_RATIO = 0.8
    
    # Output
    MODEL_PATH = "model.pt"
    NORM_STATS_PATH = "norm_stats.json"
    GLOBAL_STATS_PATH = "global_stats.json"
    
    # Reproducibility
    SEED = 42


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


def create_dataloader(
    windows: np.ndarray,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """Create PyTorch DataLoader from windows."""
    tensor = torch.FloatTensor(windows)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def train_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in loader:
        x = batch[0].to(device)
        
        # Forward pass
        reconstructed = model(x)
        loss = nn.functional.mse_loss(reconstructed, x)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(
    model: LSTMAutoencoder,
    loader: DataLoader,
    device: str
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            reconstructed = model(x)
            loss = nn.functional.mse_loss(reconstructed, x)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


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
            reconstructed = model(x)
            
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


def train(config: Config) -> None:
    """Main training function."""
    print("=" * 60)
    print("LSTM Autoencoder Training")
    print("=" * 60)
    
    # Set seed
    set_seed(config.SEED)
    
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
    
    # Step 4: Split data
    print("\n[Step 4] Splitting data...")
    train_windows, val_windows, _, _ = split_windows(
        windows, engine_ids,
        train_ratio=config.TRAIN_RATIO,
        seed=config.SEED
    )
    
    # Create dataloaders
    train_loader = create_dataloader(train_windows, config.BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_windows, config.BATCH_SIZE, shuffle=False)
    
    # Step 5: Create model
    print("\n[Step 5] Creating model...")
    model = create_model(
        input_dim=num_features,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        device=device
    )
    
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
    
    # Step 6: Training loop
    print("\n[Step 6] Training...")
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            marker = " *"
        else:
            marker = ""
        
        print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}{marker}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    
    # Step 7: Compute reconstruction errors on training data
    print("\n[Step 7] Computing reconstruction errors...")
    all_windows = windows  # Use all windows for global stats
    errors = compute_reconstruction_errors(model, all_windows, device)
    
    # Compute global stats
    global_stats = compute_global_stats(errors)
    print(f"Global Mean: {global_stats['global_mean']:.6f}")
    print(f"Global Std: {global_stats['global_std']:.6f}")
    print(f"95th percentile: {global_stats['percentiles']['p95']:.6f}")
    
    # Save global stats
    save_global_stats(global_stats, config.GLOBAL_STATS_PATH)
    
    # Step 8: Save model
    print("\n[Step 8] Saving model...")
    norm_stats = normalizer.get_stats()
    save_model(
        model,
        config.MODEL_PATH,
        feature_mean=norm_stats["feature_mean"],
        feature_std=norm_stats["feature_std"]
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Outputs:")
    print(f"  - {config.MODEL_PATH}")
    print(f"  - {config.NORM_STATS_PATH}")
    print(f"  - {config.GLOBAL_STATS_PATH}")
    print(f"\nModel performance:")
    print(f"  - Best validation loss: {best_val_loss:.6f}")
    print(f"  - Global reconstruction error: {global_stats['global_mean']:.6f} ± {global_stats['global_std']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="LSTM hidden dimension")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
