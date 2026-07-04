"""
Sliding Window Generator for Time Series Data

Creates sliding windows from time series data grouped by engine_id.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Generator


def create_sliding_windows(
    data: np.ndarray,
    window_size: int = 50,
    stride: int = 1
) -> np.ndarray:
    """
    Create sliding windows from a single sequence.
    
    Args:
        data: Input array of shape (sequence_length, num_features)
        window_size: Size of each window
        stride: Step size between windows
        
    Returns:
        Array of shape (num_windows, window_size, num_features)
    """
    if len(data) < window_size:
        return np.array([])
    
    num_windows = (len(data) - window_size) // stride + 1
    windows = []
    
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windows.append(data[start:end])
    
    return np.array(windows)


def create_windows_by_engine(
    df: pd.DataFrame,
    normalized_features: np.ndarray,
    window_size: int = 50,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows grouped by engine_id.
    
    Args:
        df: DataFrame with engine_id column (sorted by engine_id and cycle)
        normalized_features: Normalized feature array aligned with df
        window_size: Size of each window
        stride: Step size between windows
        
    Returns:
        Tuple of (windows, engine_ids) where:
            - windows: shape (num_windows, window_size, num_features)
            - engine_ids: engine_id for each window
    """
    all_windows = []
    all_engine_ids = []
    
    engine_ids = df['engine_id'].unique()
    
    for engine_id in engine_ids:
        # Get indices for this engine
        mask = df['engine_id'] == engine_id
        engine_features = normalized_features[mask]
        
        # Skip if sequence too short
        if len(engine_features) < window_size:
            continue
        
        # Create windows for this engine
        windows = create_sliding_windows(engine_features, window_size, stride)
        
        if len(windows) > 0:
            all_windows.append(windows)
            all_engine_ids.extend([engine_id] * len(windows))
    
    if len(all_windows) == 0:
        raise ValueError(f"No windows created. Check if sequences are at least {window_size} long.")
    
    # Concatenate all windows
    windows = np.concatenate(all_windows, axis=0)
    engine_ids = np.array(all_engine_ids)
    
    print(f"Created {len(windows)} windows of size {window_size}")
    print(f"  From {len(df['engine_id'].unique())} engines")
    print(f"  Window shape: {windows.shape}")
    
    return windows, engine_ids


def window_generator(
    df: pd.DataFrame,
    normalized_features: np.ndarray,
    window_size: int = 50,
    stride: int = 1,
    batch_size: int = 32
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator that yields batches of windows.
    
    Args:
        df: DataFrame with engine_id column
        normalized_features: Normalized feature array
        window_size: Size of each window
        stride: Step size between windows
        batch_size: Number of windows per batch
        
    Yields:
        Tuple of (batch_windows, batch_engine_ids)
    """
    windows, engine_ids = create_windows_by_engine(
        df, normalized_features, window_size, stride
    )
    
    num_batches = len(windows) // batch_size
    
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        yield windows[start:end], engine_ids[start:end]
    
    # Yield remaining windows
    if len(windows) % batch_size != 0:
        yield windows[num_batches * batch_size:], engine_ids[num_batches * batch_size:]


def split_windows(
    windows: np.ndarray,
    engine_ids: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split windows into train and validation sets by engine_id to prevent data leakage.
    
    Args:
        windows: Array of windows
        engine_ids: Engine ID for each window
        train_ratio: Fraction of engines for training
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_windows, val_windows, train_ids, val_ids)
    """
    np.random.seed(seed)
    
    # Get unique engine IDs
    unique_engines = np.unique(engine_ids)
    num_engines = len(unique_engines)
    
    # Shuffle engine IDs
    shuffled_engines = np.random.permutation(unique_engines)
    
    # Split engines (not windows) into train/val
    split_idx = int(num_engines * train_ratio)
    train_engines = set(shuffled_engines[:split_idx])
    val_engines = set(shuffled_engines[split_idx:])
    
    # Create masks for train/val windows
    train_mask = np.isin(engine_ids, list(train_engines))
    val_mask = np.isin(engine_ids, list(val_engines))
    
    train_windows = windows[train_mask]
    val_windows = windows[val_mask]
    train_ids = engine_ids[train_mask]
    val_ids = engine_ids[val_mask]
    
    # Verify no engine appears in both sets
    train_unique = set(np.unique(train_ids))
    val_unique = set(np.unique(val_ids))
    overlap = train_unique & val_unique
    if overlap:
        raise ValueError(f"Data leakage detected! Engines in both train and val: {overlap}")
    
    print(f"Split data: {len(train_windows)} train windows ({len(train_unique)} engines), "
          f"{len(val_windows)} validation windows ({len(val_unique)} engines)")
    
    return train_windows, val_windows, train_ids, val_ids


def get_window_stats(windows: np.ndarray) -> dict:
    """
    Get statistics about the windows.
    
    Args:
        windows: Array of shape (num_windows, window_size, num_features)
        
    Returns:
        Dictionary with statistics
    """
    return {
        "num_windows": len(windows),
        "window_size": windows.shape[1],
        "num_features": windows.shape[2],
        "mean": float(windows.mean()),
        "std": float(windows.std()),
        "min": float(windows.min()),
        "max": float(windows.max())
    }


if __name__ == "__main__":
    from dataset import load_and_prepare_data
    from preprocess import preprocess_data
    
    # Load and preprocess data
    df, features = load_and_prepare_data(
        data_dir="data",
        healthy_ratio=0.7
    )
    df_sorted, normalized, normalizer = preprocess_data(df, features)
    
    # Create windows
    WINDOW_SIZE = 50
    windows, engine_ids = create_windows_by_engine(
        df_sorted, normalized, window_size=WINDOW_SIZE
    )
    
    # Split
    train_windows, val_windows, train_ids, val_ids = split_windows(windows, engine_ids)
    
    # Print stats
    print("\nWindow statistics:")
    print(f"  Train: {get_window_stats(train_windows)}")
    print(f"  Val: {get_window_stats(val_windows)}")
