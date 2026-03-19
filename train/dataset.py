"""
NASA Turbofan Dataset Loader

Loads the CMAPSS FD001 dataset for training the anomaly detection model.
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import urllib.request
import zipfile


# Column names for the dataset
COLUMN_NAMES = [
    'engine_id', 'cycle',
    'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

# Sensor columns to use as features (excluding low variance sensors)
FEATURE_COLUMNS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
    'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]

# Columns to drop (low variance or constant)
DROP_COLUMNS = [
    'setting_3',  # Often constant
    'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
    'sensor_16', 'sensor_18', 'sensor_19'  # Low variance
]

# Dataset URLs (multiple fallbacks)
DATASET_URLS = [
    "https://ti.arc.nasa.gov/c/6/",  # NASA TI archive
    "https://github.com/kbacsa-ethz/phm08-cmapss/raw/main/data/CMAPSSData.zip",  # GitHub mirror
]
DATASET_FILENAME = "CMAPSSData.zip"


def generate_synthetic_data(data_dir: str = "data", num_engines: int = 100) -> str:
    """
    Generate synthetic turbofan sensor data that mimics the CMAPSS structure.
    Used as fallback when NASA data is unavailable.
    
    Args:
        data_dir: Directory to store the synthetic data
        num_engines: Number of engines to simulate
        
    Returns:
        Path to the data directory
    """
    print("Generating synthetic turbofan data...")
    np.random.seed(42)
    
    data = []
    for engine_id in range(1, num_engines + 1):
        # Random engine lifecycle length (150-300 cycles)
        max_cycles = np.random.randint(150, 301)
        
        for cycle in range(1, max_cycles + 1):
            # Operating settings (3 continuous values)
            op_setting_1 = np.random.uniform(-0.0015, 0.0015)
            op_setting_2 = np.random.uniform(-0.0006, 0.0006)
            op_setting_3 = np.random.uniform(100.0, 100.0)  # Constant
            
            # Degradation factor based on cycle progression
            degradation = (cycle / max_cycles) ** 1.5
            noise_scale = 0.02
            
            # Sensor readings with degradation patterns (similar to CMAPSS)
            sensors = []
            # Each sensor has base value, degradation trend, and noise
            sensor_params = [
                (518.67, 0, 0.5),    # sensor_2: T24 (total temp at fan inlet)
                (642.44, 2.5, 0.8),   # sensor_3: T30 (total temp at HPC outlet)
                (1580.0, 50, 5.0),    # sensor_4: T50 (total temp at LPT outlet)
                (14.62, -0.8, 0.15),  # sensor_7: Ps30 (static pressure at HPC outlet)
                (21.61, 1.2, 0.2),    # sensor_8: phi (fuel flow / Ps30 ratio)
                (554.36, 3.5, 1.0),   # sensor_9: NRf (corrected fan speed)
                (2388.0, 15, 3.0),    # sensor_11: Nc (corrected core speed)
                (8138.0, 80, 15.0),   # sensor_12: BPR (bypass ratio)
                (8.44, 0.5, 0.1),     # sensor_13: htBleed (bleed enthalpy)
                (392.0, 8, 2.0),      # sensor_14: NRc (corrected core speed)
                (2388.0, 15, 3.0),    # sensor_15: W31 (HPT coolant bleed)
                (38.86, 2.0, 0.5),    # sensor_17: W32 (LPT coolant bleed)
                (23.42, 1.5, 0.3),    # sensor_20: BPR2 (bypass ratio 2)
                (23.36, 1.8, 0.35),   # sensor_21: T48 (total temp at HPT outlet)
            ]
            
            for base, trend, noise_std in sensor_params:
                value = base + trend * degradation + np.random.normal(0, noise_std * (1 + noise_scale * degradation))
                sensors.append(value)
            
            # Constant/low-variance sensors (removed in preprocessing)
            const_sensors = [100.0, 25.0, 8.4195, 0.03, 306.0, 2388.0, 100.0]  # 7 sensors
            
            row = [engine_id, cycle, op_setting_1, op_setting_2, op_setting_3] + sensors[:3] + const_sensors[0:1] + sensors[3:4] + const_sensors[1:2] + sensors[4:6] + const_sensors[2:4] + sensors[6:8] + const_sensors[4:5] + sensors[8:10] + const_sensors[5:7] + sensors[10:]
            # Reorder to match original column structure
            row = [engine_id, cycle, op_setting_1, op_setting_2, op_setting_3,
                   100.0, sensors[0], sensors[1], sensors[2], 25.0, sensors[3], 8.4195,
                   sensors[4], sensors[5], 0.03, 306.0, sensors[6], sensors[7], 2388.0,
                   sensors[8], sensors[9], 100.0, sensors[10], sensors[11], sensors[12], sensors[13]]
            data.append(row)
    
    # Save as train_FD001.txt format
    os.makedirs(data_dir, exist_ok=True)
    train_file = os.path.join(data_dir, "train_FD001.txt")
    
    with open(train_file, 'w') as f:
        for row in data:
            line = ' '.join([f'{v:.4f}' if isinstance(v, float) else str(v) for v in row])
            f.write(line + '\n')
    
    print(f"Generated synthetic data: {len(data)} records for {num_engines} engines")
    print(f"Saved to: {train_file}")
    
    return data_dir


def download_dataset(data_dir: str = "data") -> str:
    """
    Download the NASA Turbofan dataset if not already present.
    Falls back to synthetic data generation if download fails.
    
    Args:
        data_dir: Directory to store the dataset
        
    Returns:
        Path to the dataset directory
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, DATASET_FILENAME)
    
    # Check if already extracted
    train_file = os.path.join(data_dir, "train_FD001.txt")
    if os.path.exists(train_file):
        print(f"Dataset already exists at {data_dir}")
        return data_dir
    
    # Try downloading from multiple sources
    downloaded = False
    for url in DATASET_URLS:
        print(f"Trying to download from: {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print(f"Downloaded to {zip_path}")
            downloaded = True
            break
        except Exception as e:
            print(f"Failed: {e}")
            continue
    
    if downloaded:
        # Extract
        print(f"Extracting dataset...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Extracted to {data_dir}")
            return data_dir
        except Exception as e:
            print(f"Extraction failed: {e}")
    
    # Fallback: generate synthetic data
    print("\nCould not download NASA dataset. Generating synthetic data...")
    print("Note: For production use, download the real CMAPSS dataset from:")
    print("https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6")
    return generate_synthetic_data(data_dir)


def load_train_data(data_dir: str = "data", dataset: str = "FD001") -> pd.DataFrame:
    """
    Load the training dataset.
    
    Args:
        data_dir: Directory containing the dataset
        dataset: Which dataset to use (FD001, FD002, FD003, FD004)
        
    Returns:
        DataFrame with the training data
    """
    train_file = os.path.join(data_dir, f"train_{dataset}.txt")
    
    if not os.path.exists(train_file):
        # Try to download/generate
        print(f"Dataset not found at {train_file}, attempting to download...")
        download_dataset(data_dir)
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Dataset file not found: {train_file}\n"
            f"Please download from: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6"
        )
    
    # Load data (space-separated, no header)
    df = pd.read_csv(
        train_file,
        sep=r'\s+',
        header=None,
        names=COLUMN_NAMES
    )
    
    print(f"Loaded {len(df)} records from {train_file}")
    print(f"Unique engines: {df['engine_id'].nunique()}")
    
    return df


def load_test_data(data_dir: str = "data", dataset: str = "FD001") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the test dataset and RUL values.
    
    Args:
        data_dir: Directory containing the dataset
        dataset: Which dataset to use
        
    Returns:
        Tuple of (test_df, rul_df)
    """
    test_file = os.path.join(data_dir, f"test_{dataset}.txt")
    rul_file = os.path.join(data_dir, f"RUL_{dataset}.txt")
    
    # Load test data
    test_df = pd.read_csv(
        test_file,
        sep=r'\s+',
        header=None,
        names=COLUMN_NAMES
    )
    
    # Load RUL values
    rul_df = pd.read_csv(
        rul_file,
        sep=r'\s+',
        header=None,
        names=['rul']
    )
    
    print(f"Loaded {len(test_df)} test records")
    
    return test_df, rul_df


def filter_healthy_data(
    df: pd.DataFrame,
    healthy_ratio: float = 0.7
) -> pd.DataFrame:
    """
    Filter to keep only healthy data (early cycles before failure).
    
    For each engine, keep only the first `healthy_ratio` portion of cycles.
    This ensures we train only on healthy, normal operation data.
    
    Args:
        df: Input DataFrame
        healthy_ratio: Fraction of cycles to keep (from the start)
        
    Returns:
        Filtered DataFrame with only healthy data
    """
    healthy_dfs = []
    
    for engine_id in df['engine_id'].unique():
        engine_df = df[df['engine_id'] == engine_id].copy()
        max_cycle = engine_df['cycle'].max()
        
        # Keep only early cycles (healthy portion)
        healthy_cycles = int(max_cycle * healthy_ratio)
        healthy_df = engine_df[engine_df['cycle'] <= healthy_cycles]
        healthy_dfs.append(healthy_df)
    
    result = pd.concat(healthy_dfs, ignore_index=True)
    print(f"Filtered to {len(result)} healthy records ({healthy_ratio*100:.0f}% of cycles)")
    
    return result


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing unnecessary columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Drop low-variance columns
    columns_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    df_clean = df.drop(columns=columns_to_drop)
    
    print(f"Removed {len(columns_to_drop)} low-variance columns")
    print(f"Remaining columns: {list(df_clean.columns)}")
    
    return df_clean


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get the feature column names from the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of feature column names
    """
    # Exclude engine_id and cycle
    exclude = ['engine_id', 'cycle']
    features = [col for col in df.columns if col not in exclude]
    return features


def load_and_prepare_data(
    data_dir: str = "data",
    dataset: str = "FD001",
    healthy_ratio: float = 0.7,
    download: bool = True
) -> Tuple[pd.DataFrame, list]:
    """
    Load and prepare the dataset for training.
    
    Args:
        data_dir: Directory containing/to store the dataset
        dataset: Which dataset to use
        healthy_ratio: Fraction of cycles to keep as healthy
        download: Whether to download if not present
        
    Returns:
        Tuple of (prepared DataFrame, list of feature columns)
    """
    # Download if needed
    if download:
        try:
            download_dataset(data_dir)
        except Exception as e:
            print(f"Warning: Could not download dataset: {e}")
    
    # Load training data
    df = load_train_data(data_dir, dataset)
    
    # Filter to healthy data only
    df = filter_healthy_data(df, healthy_ratio)
    
    # Clean data
    df = clean_data(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    
    print(f"\nDataset prepared:")
    print(f"  Records: {len(df)}")
    print(f"  Engines: {df['engine_id'].nunique()}")
    print(f"  Features: {len(feature_cols)}")
    
    return df, feature_cols


if __name__ == "__main__":
    # Test the dataset loader
    df, features = load_and_prepare_data(
        data_dir="data",
        dataset="FD001",
        healthy_ratio=0.7
    )
    
    print(f"\nSample data:")
    print(df.head())
    print(f"\nFeature columns: {features}")
    print(f"\nData statistics:")
    print(df[features].describe())
