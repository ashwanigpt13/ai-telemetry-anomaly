"""Export latent vectors for train, validation, and NASA test windows.

This script loads a trained VAE model and normalization stats, then runs
inference over train/val windows from healthy training data and full NASA
test windows. It stores engine_id, cycle number, label, and latent mean
vectors for every window.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT_DIR / "train"
sys.path.insert(0, str(TRAIN_DIR))

from dataset import (
    clean_data,
    download_dataset,
    load_and_prepare_data,
    load_test_data,
)
from model import load_model
from preprocess import FeatureNormalizer, sort_data
from window import create_sliding_windows, create_windows_by_engine, split_windows


def create_windows_with_cycles(
    df: pd.DataFrame,
    normalized_features: np.ndarray,
    window_size: int,
    stride: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows and return engine_id and cycle metadata."""
    all_windows = []
    all_engine_ids = []
    all_end_cycles = []

    for engine_id in df['engine_id'].unique():
        mask = df['engine_id'] == engine_id
        engine_features = normalized_features[mask]
        engine_cycles = df.loc[mask, 'cycle'].values

        if len(engine_features) < window_size:
            continue

        windows = create_sliding_windows(engine_features, window_size, stride)
        if windows.size == 0:
            continue

        num_windows = windows.shape[0]
        end_indices = np.arange(num_windows) * stride + window_size - 1
        end_cycles = engine_cycles[end_indices]

        all_windows.append(windows)
        all_engine_ids.extend([engine_id] * num_windows)
        all_end_cycles.extend(end_cycles.tolist())

    if len(all_windows) == 0:
        raise ValueError(f"No windows created. Check that sequences are at least {window_size} long.")

    windows = np.concatenate(all_windows, axis=0)
    engine_ids = np.array(all_engine_ids, dtype=np.int32)
    end_cycles = np.array(all_end_cycles, dtype=np.int32)

    return windows, engine_ids, end_cycles


def label_windows_by_healthy_ratio(
    df: pd.DataFrame,
    window_end_cycles: np.ndarray,
    engine_ids: np.ndarray,
    healthy_ratio: float
) -> np.ndarray:
    """Label windows as normal/anomalous by engine healthy threshold."""
    labels = []

    for engine_id in np.unique(engine_ids):
        engine_mask = engine_ids == engine_id
        engine_cycles = df[df['engine_id'] == engine_id]['cycle'].values
        healthy_cycles = int(engine_cycles.max() * healthy_ratio)

        labels.extend((window_end_cycles[engine_mask] <= healthy_cycles).astype(int).tolist())

    return np.array(labels, dtype=np.int32)


def compute_latent_features(
    model,
    windows: np.ndarray,
    device: str,
    batch_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute latent mean, log variance, and attention weights for windows."""
    model.eval()
    latent_means = []
    latent_logvars = []
    attention_weights = []

    loader = DataLoader(TensorDataset(torch.FloatTensor(windows)), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            encoder_output = model.encoder(x)
            latent_means.append(encoder_output.mean.cpu().numpy())
            latent_logvars.append(encoder_output.logvar.cpu().numpy())
            attention_weights.append(encoder_output.attention_weights.cpu().numpy())

    latent_means = np.vstack(latent_means)
    latent_logvars = np.vstack(latent_logvars)
    attention_weights = np.vstack(attention_weights)

    return latent_means, latent_logvars, attention_weights


def build_partition_dataframe(
    split_name: str,
    engine_ids: np.ndarray,
    end_cycles: np.ndarray,
    labels: np.ndarray,
    latent_means: np.ndarray,
    latent_logvars: np.ndarray,
    attention_weights: np.ndarray
) -> pd.DataFrame:
    """Create a DataFrame holding window metadata and latent vectors."""
    data = {
        'split': [split_name] * len(engine_ids),
        'engine_id': engine_ids,
        'cycle': end_cycles,
        'label': np.where(labels == 1, 'normal', 'anomalous'),
    }

    for idx in range(latent_means.shape[1]):
        data[f'latent_{idx}'] = latent_means[:, idx]
    for idx in range(latent_logvars.shape[1]):
        data[f'logvar_{idx}'] = latent_logvars[:, idx]
    data['attention_mean'] = attention_weights.mean(axis=1)

    return pd.DataFrame(data)


def export_latent_vectors(
    model_path: str,
    norm_stats_path: str,
    data_dir: str,
    output_dir: str,
    window_size: int,
    stride: int,
    healthy_ratio: float,
    batch_size: int
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(model_path, device)

    print("Loading training data and preprocessing...")
    df_train, features = load_and_prepare_data(
        data_dir=data_dir,
        dataset='FD001',
        healthy_ratio=healthy_ratio
    )
    df_train_sorted = sort_data(df_train)

    normalizer = FeatureNormalizer()
    normalizer.load_stats(norm_stats_path)
    normalized_train = normalizer.transform(df_train_sorted)

    windows, engine_ids, end_cycles = create_windows_with_cycles(
        df_train_sorted,
        normalized_train,
        window_size=window_size,
        stride=stride
    )

    train_windows, val_windows, train_ids, val_ids = split_windows(windows, engine_ids, train_ratio=0.8, seed=42)

    # Split metadata by window index mask
    train_mask = np.isin(engine_ids, train_ids)
    val_mask = np.isin(engine_ids, val_ids)

    train_end_cycles = end_cycles[train_mask]
    val_end_cycles = end_cycles[val_mask]

    train_labels = np.ones(len(train_end_cycles), dtype=np.int32)
    val_labels = np.ones(len(val_end_cycles), dtype=np.int32)

    print("Computing train latent vectors...")
    train_latent_means, train_logvars, train_attention = compute_latent_features(model, train_windows, device, batch_size)
    print("Computing validation latent vectors...")
    val_latent_means, val_logvars, val_attention = compute_latent_features(model, val_windows, device, batch_size)

    print("Exporting test windows...")
    df_test = pd.DataFrame()
    test_count = 0

    try:
        test_df, rul_df = load_test_data(data_dir, 'FD001')
        test_df = clean_data(test_df)
        test_df_sorted = sort_data(test_df)
        normalized_test = normalizer.transform(test_df_sorted)

        test_windows, test_engine_ids, test_end_cycles = create_windows_with_cycles(
            test_df_sorted,
            normalized_test,
            window_size=window_size,
            stride=stride
        )
        test_labels = label_windows_by_healthy_ratio(test_df_sorted, test_end_cycles, test_engine_ids, healthy_ratio)

        test_latent_means, test_logvars, test_attention = compute_latent_features(model, test_windows, device, batch_size)

        df_test = build_partition_dataframe(
            'test', test_engine_ids, test_end_cycles, test_labels, test_latent_means, test_logvars, test_attention
        )
        test_count = len(df_test)
    except FileNotFoundError as exc:
        print(f"Warning: NASA test data unavailable: {exc}")
        print("Skipping test partition export.")

    df_train = build_partition_dataframe(
        'train', train_ids, train_end_cycles, train_labels, train_latent_means, train_logvars, train_attention
    )
    df_val = build_partition_dataframe(
        'val', val_ids, val_end_cycles, val_labels, val_latent_means, val_logvars, val_attention
    )

    combined = pd.concat([df_train, df_val] + ([df_test] if not df_test.empty else []), ignore_index=True)
    combined.to_csv(Path(output_dir) / 'latent_vectors_combined.csv', index=False)
    df_train.to_csv(Path(output_dir) / 'latent_vectors_train.csv', index=False)
    df_val.to_csv(Path(output_dir) / 'latent_vectors_val.csv', index=False)
    if not df_test.empty:
        df_test.to_csv(Path(output_dir) / 'latent_vectors_test.csv', index=False)

    print(f"Saved latent vectors to {output_dir}")
    print(f"Train windows: {len(df_train)}, Val windows: {len(df_val)}, Test windows: {test_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export latent vectors for train/val/test windows')
    parser.add_argument('--data-dir', type=str, default='data', help='Dataset directory')
    parser.add_argument('--model-path', type=str, default='train/model.pt', help='Trained model checkpoint path')
    parser.add_argument('--norm-stats-path', type=str, default='train/norm_stats.json', help='Normalization stats path')
    parser.add_argument('--output-dir', type=str, default='evaluation/latent_vectors', help='Directory to save latent vector tables')
    parser.add_argument('--window-size', type=int, default=50, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=1, help='Sliding window stride')
    parser.add_argument('--healthy-ratio', type=float, default=0.7, help='Healthy fraction threshold for labeling')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for inference')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    export_latent_vectors(
        model_path=args.model_path,
        norm_stats_path=args.norm_stats_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        healthy_ratio=args.healthy_ratio,
        batch_size=args.batch_size,
    )
