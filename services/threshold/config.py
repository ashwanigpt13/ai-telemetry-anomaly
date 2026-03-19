"""
Configuration management for Threshold Engine Service
"""
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    SERVICE_NAME: str = "threshold"
    SERVICE_PORT: int = 8003
    
    # Redis configuration
    REDIS_URL: str = "redis://localhost:6379"
    
    # MQTT configuration
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    MQTT_TOPIC_ANOMALY: str = "anomalies/"
    
    # Window configuration
    ERROR_WINDOW: int = 100
    
    # Computed window sizes
    @property
    def N_RECENT(self) -> int:
        return self.ERROR_WINDOW // 2
    
    @property
    def N_PAST(self) -> int:
        return self.ERROR_WINDOW // 2
    
    # Threshold parameters
    K: float = 3.0
    ALPHA: float = 0.5
    DRIFT_THRESHOLD: float = 0.05
    EPSILON: float = 1e-6
    
    # Precision improvement parameters
    ANOMALY_PERSISTENCE: int = 1  # Require N consecutive anomalies
    ERROR_SMOOTH_WINDOW: int = 1  # Moving average window for errors
    
    # Global statistics for cold start
    GLOBAL_MEAN: float = 0.02
    GLOBAL_STD: float = 0.01
    
    # Threshold mode: global_static, global_dynamic, entity_dynamic, entity_drift
    THRESHOLD_MODE: str = "entity_drift"
    
    # Prediction logging
    PREDICTION_LOG_ENABLED: bool = True
    PREDICTION_LOG_PATH: str = "/data/predictions.csv"
    PREDICTION_BUFFER_SIZE: int = 1000  # Flush to file after this many records
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
