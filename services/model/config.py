"""
Configuration management for Model Service
"""
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    SERVICE_NAME: str = "model"
    SERVICE_PORT: int = 8002
    
    # Model configuration
    MODEL_PATH: str = "model.pt"
    NORM_STATS_PATH: str = "norm_stats.json"
    INPUT_DIM: int = 16
    WINDOW_SIZE: int = 50
    HIDDEN_DIM: int = 128      # Increased from 64
    LATENT_DIM: int = 64       # Increased from 32
    NUM_LAYERS: int = 2        # Increased from 1
    DROPOUT: float = 0.2       # Added
    HIDDEN_DIMS: List[int] = [16, 8]  # Legacy - not used for LSTM
    
    # Normalization
    EPSILON: float = 1e-6
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
