"""
Preprocessing Module for NASA Turbofan Dataset

Normalizes features using StandardScaler and saves normalization statistics.
"""
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler


class FeatureNormalizer:
    """
    Normalizes features using StandardScaler and saves/loads statistics.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: list = []
        self.is_fitted: bool = False
        
    def fit(self, df: pd.DataFrame, feature_columns: list) -> 'FeatureNormalizer':
        """
        Fit the scaler on the training data.
        
        Args:
            df: Training DataFrame
            feature_columns: List of feature column names
            
        Returns:
            self
        """
        self.feature_names = feature_columns
        features = df[feature_columns].values
        
        self.scaler.fit(features)
        self.is_fitted = True
        
        print(f"Fitted normalizer on {len(features)} samples, {len(feature_columns)} features")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform features using the fitted scaler.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Normalized feature array
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        features = df[self.feature_names].values
        normalized = self.scaler.transform(features)
        
        return normalized
    
    def fit_transform(self, df: pd.DataFrame, feature_columns: list) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            df: Training DataFrame
            feature_columns: List of feature column names
            
        Returns:
            Normalized feature array
        """
        self.fit(df, feature_columns)
        return self.transform(df)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get normalization statistics.
        
        Returns:
            Dictionary with mean and std for each feature
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")
        
        return {
            "feature_names": self.feature_names,
            "feature_mean": self.scaler.mean_.tolist(),
            "feature_std": self.scaler.scale_.tolist(),
            "num_features": len(self.feature_names)
        }
    
    def save_stats(self, filepath: str) -> None:
        """
        Save normalization statistics to JSON file.
        
        Args:
            filepath: Path to save the statistics
        """
        stats = self.get_stats()
        stats["metadata"] = {
            "description": "Normalization statistics for LSTM Autoencoder",
            "dataset": "NASA Turbofan FD001"
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved normalization stats to {filepath}")
    
    def load_stats(self, filepath: str) -> 'FeatureNormalizer':
        """
        Load normalization statistics from JSON file.
        
        Args:
            filepath: Path to load the statistics from
            
        Returns:
            self
        """
        with open(filepath, 'r') as f:
            stats = json.load(f)
        
        self.feature_names = stats["feature_names"]
        self.scaler.mean_ = np.array(stats["feature_mean"])
        self.scaler.scale_ = np.array(stats["feature_std"])
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = len(self.feature_names)
        self.is_fitted = True
        
        print(f"Loaded normalization stats from {filepath}")
        
        return self


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort DataFrame by engine_id and cycle.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Sorted DataFrame
    """
    df_sorted = df.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    print(f"Sorted {len(df_sorted)} records by engine_id and cycle")
    return df_sorted


def preprocess_data(
    df: pd.DataFrame,
    feature_columns: list,
    normalizer: Optional[FeatureNormalizer] = None
) -> Tuple[pd.DataFrame, np.ndarray, FeatureNormalizer]:
    """
    Full preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature columns to normalize
        normalizer: Optional pre-fitted normalizer (for test data)
        
    Returns:
        Tuple of (sorted_df, normalized_features, normalizer)
    """
    # Sort by engine_id and cycle
    df_sorted = sort_data(df)
    
    # Normalize features
    if normalizer is None:
        normalizer = FeatureNormalizer()
        normalized = normalizer.fit_transform(df_sorted, feature_columns)
    else:
        normalized = normalizer.transform(df_sorted)
    
    print(f"Preprocessing complete: {normalized.shape}")
    
    return df_sorted, normalized, normalizer


def create_feature_dataframe(
    df: pd.DataFrame,
    normalized_features: np.ndarray,
    feature_columns: list
) -> pd.DataFrame:
    """
    Create DataFrame with normalized features.
    
    Args:
        df: Original sorted DataFrame
        normalized_features: Normalized feature array
        feature_columns: Feature column names
        
    Returns:
        DataFrame with engine_id, cycle, and normalized features
    """
    # Create new DataFrame
    result = pd.DataFrame(normalized_features, columns=feature_columns)
    result['engine_id'] = df['engine_id'].values
    result['cycle'] = df['cycle'].values
    
    # Reorder columns
    cols = ['engine_id', 'cycle'] + feature_columns
    result = result[cols]
    
    return result


if __name__ == "__main__":
    from dataset import load_and_prepare_data
    
    # Load data
    df, features = load_and_prepare_data(
        data_dir="data",
        healthy_ratio=0.7
    )
    
    # Preprocess
    df_sorted, normalized, normalizer = preprocess_data(df, features)
    
    # Print stats
    stats = normalizer.get_stats()
    print("\nNormalization statistics:")
    for i, name in enumerate(stats["feature_names"]):
        print(f"  {name}: mean={stats['feature_mean'][i]:.4f}, std={stats['feature_std'][i]:.4f}")
    
    # Save stats
    normalizer.save_stats("norm_stats.json")
    
    # Verify normalized data
    print(f"\nNormalized data shape: {normalized.shape}")
    print(f"Mean (should be ~0): {normalized.mean(axis=0)}")
    print(f"Std (should be ~1): {normalized.std(axis=0)}")
