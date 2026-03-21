"""
Configuration management for Simulator Service
"""
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Service configuration
    SERVICE_NAME: str = "simulator"
    SERVICE_PORT: int = 8000
    
    # MQTT configuration
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    MQTT_TOPIC_TELEMETRY_BASE: str = "telemetry/"
    
    # Simulation configuration
    DEFAULT_RATE: float = 10.0  # messages/sec total
    DEFAULT_ENTITIES: List[str] = ["engine_1", "engine_2", "turbine_1", "pump_1", "sensor_1"]
    
    # Feature configuration - matches trained LSTM model (16 turbofan features)
    NUM_FEATURES: int = 16
    # Feature names for reference:
    # setting_1, setting_2, sensor_2 (T24), sensor_3 (T30), sensor_4 (T50), 
    # sensor_7 (Ps30), sensor_8 (phi), sensor_9 (NRf), sensor_11 (Nc), 
    # sensor_12 (BPR), sensor_13 (htBleed), sensor_14 (NRc), sensor_15 (W31),
    # sensor_17 (W32), sensor_20 (BPR2), sensor_21 (T48)
    
    # Anomaly configuration
    ANOMALY_PROBABILITY: float = 0.05
    SPIKE_MAGNITUDE: float = 3.0
    NOISE_MAGNITUDE: float = 1.0
    DRIFT_MAGNITUDE: float = 0.5
    DRIFT_DURATION: float = 300.0  # seconds
    NOISE_IS_ANOMALY: bool = False  # Whether noise injection counts as anomaly
    
    # Scenario mode: "mixed", "spike_only", "drift_only", "noise_only"
    SIMULATION_MODE: str = "mixed"  # Controls which anomaly types to inject
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
