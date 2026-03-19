"""
Configuration management for API Gateway Service
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    SERVICE_NAME: str = "api"
    SERVICE_PORT: int = 8080
    
    # Service URLs
    MODEL_SERVICE_URL: str = "http://localhost:8002"
    THRESHOLD_SERVICE_URL: str = "http://localhost:8003"
    
    # Redis configuration
    REDIS_URL: str = "redis://localhost:6379"
    
    # HTTP configuration
    HTTP_TIMEOUT: float = 10.0
    
    # Window configuration
    ERROR_WINDOW: int = 100
    
    # Computed window sizes
    @property
    def N_RECENT(self) -> int:
        return self.ERROR_WINDOW // 2
    
    @property
    def N_PAST(self) -> int:
        return self.ERROR_WINDOW // 2
    
    # Threshold parameters (for entity state computation)
    K: float = 3.0
    ALPHA: float = 0.5
    DRIFT_THRESHOLD: float = 0.05
    EPSILON: float = 1e-6
    
    # Global statistics for cold start
    GLOBAL_MEAN: float = 0.02
    GLOBAL_STD: float = 0.01
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
