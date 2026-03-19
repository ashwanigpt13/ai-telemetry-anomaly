"""
Configuration management for Ingestion Service
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    SERVICE_NAME: str = "ingestion"
    SERVICE_PORT: int = 8001
    
    # MQTT configuration
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    MQTT_TOPIC_TELEMETRY: str = "telemetry/+"
    MQTT_SHARE_GROUP: str = "ingestion-group"
    
    # Window configuration
    WINDOW_SIZE: int = 50
    WINDOW_STRIDE: int = 25  # Process window every N messages (stride)
    ENTITY_TIMEOUT_SEC: int = 300
    
    # Service URLs
    MODEL_SERVICE_URL: str = "http://localhost:8002"
    THRESHOLD_SERVICE_URL: str = "http://localhost:8003"
    
    # HTTP configuration
    HTTP_TIMEOUT: float = 5.0
    
    # Evaluation logging
    EVALUATION_LOG_ENABLED: bool = True
    EVALUATION_LOG_PATH: str = "/data/evaluation.csv"
    EVALUATION_BUFFER_SIZE: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
