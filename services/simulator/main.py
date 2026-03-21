"""
Simulator Service - Generate synthetic telemetry data with anomaly injection
"""
import asyncio
import json
import logging
import random
import time
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional

import numpy as np
import paho.mqtt.client as mqtt
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Gauge, generate_latest
from pydantic import BaseModel

from config import settings


# Configure structured logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "simulator",
            "message": record.getMessage(),
        }
        if hasattr(record, "entity_id"):
            log_obj["entity_id"] = record.entity_id
        if hasattr(record, "event"):
            log_obj["event"] = record.event
        return json.dumps(log_obj)


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Prometheus metrics
messages_published = Counter("simulator_messages_published_total", "Total messages published", ["entity_id"])
messages_failed = Counter("simulator_messages_failed_total", "Total failed messages", ["entity_id"])
anomalies_injected = Counter("simulator_anomalies_injected_total", "Total anomalies injected", ["entity_id", "type"])
publish_rate = Gauge("simulator_publish_rate", "Current publish rate (msgs/sec)")
active_entities = Gauge("simulator_active_entities", "Number of active entities")


# Anomaly types
class AnomalyType(str, Enum):
    SPIKE = "spike"
    DRIFT = "drift"
    NOISE = "noise"
    NONE = "none"


class SimulatorConfig(BaseModel):
    """Simulator runtime configuration"""
    running: bool = True
    rate: float = 10.0  # messages per second (total across all entities)
    entities: List[str] = ["engine_1", "engine_2", "turbine_1"]
    anomaly_probability: float = 0.05


# Global state
simulator_config = SimulatorConfig()
mqtt_client: Optional[mqtt.Client] = None
entity_states: Dict[str, dict] = {}


def on_mqtt_connect(client, userdata, flags, rc):
    """MQTT on_connect callback"""
    if rc == 0:
        logger.info({"event": "mqtt_connected", "message": "Connected to MQTT broker"})
    else:
        logger.error({"event": "mqtt_connection_failed", "rc": rc})


def on_mqtt_disconnect(client, userdata, rc):
    """MQTT on_disconnect callback"""
    logger.warning({"event": "mqtt_disconnected", "rc": rc})
    if rc != 0:
        logger.info({"event": "mqtt_reconnecting", "message": "Unexpected disconnect, will auto-reconnect"})


def initialize_entity_state(entity_id: str) -> dict:
    """Initialize state for an entity with NASA Turbofan-like sensor values"""
    # NASA Turbofan FD001 baseline values (from norm_stats)
    # These represent typical sensor readings for healthy engines
    nasa_baseline_mean = np.array([
        0.0,        # setting_1
        0.0,        # setting_2
        518.67,     # sensor_2 (T24) - Temperature °C
        643.03,     # sensor_3 (T30) - Temperature °C
        1591.76,    # sensor_4 (T50) - Temperature °C
        8.42,       # sensor_7 (Ps30) - Pressure
        21.89,      # sensor_8 (phi) - Fuel flow ratio
        555.18,     # sensor_9 (NRf) - Speed
        306.0,      # sensor_11 (Nc) - Corrected speed
        2391.56,    # sensor_12 (BPR) - Bypass ratio
        8156.83,    # sensor_13 (htBleed) - Bleed enthalpy
        2388.0,     # sensor_14 (NRc) - Corrected speed
        8.56,       # sensor_15 (W31) - Bypass flow
        100.0,      # sensor_17 (W32) - LPC flow
        23.78,      # sensor_20 (BPR2) - Bypass ratio
        23.78       # sensor_21 (T48) - Temperature
    ])
    
    # Standard deviations for normal variation (from norm_stats)
    nasa_baseline_std = np.array([
        0.001,      # setting_1
        0.0003,     # setting_2
        0.5,        # sensor_2 (T24)
        0.9,        # sensor_3 (T30)
        10.2,       # sensor_4 (T50)
        1.0,        # sensor_7 (Ps30)
        0.3,        # sensor_8 (phi)
        1.2,        # sensor_9 (NRf)
        1.0,        # sensor_11 (Nc)
        4.0,        # sensor_12 (BPR)
        20.5,       # sensor_13 (htBleed)
        1.0,        # sensor_14 (NRc)
        0.13,       # sensor_15 (W31)
        1.0,        # sensor_17 (W32)
        0.4,        # sensor_20 (BPR2)
        0.4         # sensor_21 (T48)
    ])
    
    # Add small entity-specific offset (simulate different engines)
    entity_offset = np.random.randn(settings.NUM_FEATURES) * nasa_baseline_std * 0.1
    
    return {
        "entity_id": entity_id,
        "base_mean": nasa_baseline_mean + entity_offset,
        "base_std": nasa_baseline_std,
        "drift_offset": np.zeros(settings.NUM_FEATURES),
        "drift_active": False,
        "drift_start_time": None,
        "message_count": 0
    }


def generate_normal_features(state: dict) -> np.ndarray:
    """Generate normal (non-anomalous) features"""
    features = np.random.randn(settings.NUM_FEATURES) * state["base_std"] + state["base_mean"]
    
    # Apply drift if active
    if state["drift_active"]:
        features += state["drift_offset"]
    
    return features


def inject_spike_anomaly(features: np.ndarray, state: dict) -> np.ndarray:
    """Inject spike anomaly - sudden large values (scaled to each feature's normal range)"""
    # Spike magnitude is relative to each feature's std
    spike_magnitude = settings.SPIKE_MAGNITUDE  # Number of std deviations
    spike_offsets = np.random.choice([-1, 1], size=settings.NUM_FEATURES) * state["base_std"] * spike_magnitude
    spike_features = features + spike_offsets
    return spike_features


def inject_noise_anomaly(features: np.ndarray, state: dict) -> np.ndarray:
    """Inject noise anomaly - increased variance (scaled to each feature's normal range)"""
    noise_magnitude = settings.NOISE_MAGNITUDE  # Multiplier for std
    noise_offsets = np.random.randn(settings.NUM_FEATURES) * state["base_std"] * noise_magnitude
    noise_features = features + noise_offsets
    return noise_features


def initiate_drift(state: dict):
    """Initiate concept drift for an entity (scaled to each feature's normal range)"""
    state["drift_active"] = True
    state["drift_start_time"] = time.time()
    # Gradual shift in mean - drift magnitude is relative to std
    state["drift_offset"] = np.random.randn(settings.NUM_FEATURES) * state["base_std"] * settings.DRIFT_MAGNITUDE
    logger.info({
        "entity_id": state["entity_id"],
        "event": "drift_initiated",
        "drift_offset": state["drift_offset"].tolist()
    })


def update_drift(state: dict):
    """Update drift state (drift can decay or persist)"""
    if state["drift_active"]:
        elapsed = time.time() - state["drift_start_time"]
        # Drift persists for DRIFT_DURATION seconds
        if elapsed > settings.DRIFT_DURATION:
            state["drift_active"] = False
            state["drift_offset"] = np.zeros(settings.NUM_FEATURES)
            logger.info({
                "entity_id": state["entity_id"],
                "event": "drift_ended"
            })


def decide_anomaly_type() -> AnomalyType:
    """Randomly decide what type of anomaly to inject based on SIMULATION_MODE"""
    if random.random() > simulator_config.anomaly_probability:
        return AnomalyType.NONE
    
    # Choose based on simulation mode
    mode = settings.SIMULATION_MODE.lower()
    
    if mode == "spike_only":
        return AnomalyType.SPIKE
    elif mode == "drift_only":
        return AnomalyType.DRIFT
    elif mode == "noise_only":
        return AnomalyType.NOISE
    else:  # "mixed" or default
        # If anomaly, choose type with weights
        anomaly_types = [AnomalyType.SPIKE, AnomalyType.NOISE, AnomalyType.DRIFT]
        weights = [0.4, 0.3, 0.3]  # Spike more common than drift/noise
        return random.choices(anomaly_types, weights=weights)[0]


def generate_telemetry_message(entity_id: str) -> dict:
    """Generate a single telemetry message for an entity with ground truth labels"""
    # Get or initialize entity state
    if entity_id not in entity_states:
        entity_states[entity_id] = initialize_entity_state(entity_id)
    
    state = entity_states[entity_id]
    state["message_count"] += 1
    
    # Update drift state
    update_drift(state)
    
    # Generate base features
    features = generate_normal_features(state)
    
    # Track ground truth
    is_anomaly = False
    anomaly_type = "normal"
    
    # Decide if anomaly should be injected
    injected_anomaly = decide_anomaly_type()
    
    if injected_anomaly == AnomalyType.SPIKE:
        features = inject_spike_anomaly(features, state)
        anomalies_injected.labels(entity_id=entity_id, type="spike").inc()
        is_anomaly = True
        anomaly_type = "spike"
    elif injected_anomaly == AnomalyType.NOISE:
        features = inject_noise_anomaly(features, state)
        anomalies_injected.labels(entity_id=entity_id, type="noise").inc()
        is_anomaly = settings.NOISE_IS_ANOMALY  # Configurable
        anomaly_type = "noise"
    elif injected_anomaly == AnomalyType.DRIFT and not state["drift_active"]:
        initiate_drift(state)
        anomalies_injected.labels(entity_id=entity_id, type="drift").inc()
        is_anomaly = True
        anomaly_type = "drift"
    
    # If drift is active, mark as anomaly
    if state["drift_active"]:
        is_anomaly = True
        if anomaly_type == "normal":
            anomaly_type = "drift"
    
    # Create message with ground truth
    message = {
        "entity_id": entity_id,
        "timestamp": int(time.time() * 1000),  # Unix epoch in milliseconds
        "features": features.tolist(),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type
    }
    
    return message


def publish_message(entity_id: str, message: dict):
    """Publish message to MQTT"""
    try:
        topic = f"{settings.MQTT_TOPIC_TELEMETRY_BASE}{entity_id}"
        payload = json.dumps(message)
        
        result = mqtt_client.publish(topic, payload, qos=1)
        
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            messages_published.labels(entity_id=entity_id).inc()
            logger.debug({
                "entity_id": entity_id,
                "event": "message_published",
                "topic": topic
            })
        else:
            messages_failed.labels(entity_id=entity_id).inc()
            logger.error({
                "entity_id": entity_id,
                "event": "publish_failed",
                "rc": result.rc
            })
    except Exception as e:
        messages_failed.labels(entity_id=entity_id).inc()
        logger.error({
            "entity_id": entity_id,
            "event": "publish_error",
            "error": str(e)
        })


async def simulation_loop():
    """Main simulation loop"""
    logger.info({"event": "simulation_started"})
    
    message_interval = 1.0 / simulator_config.rate
    entity_cycle_index = 0
    
    while True:
        try:
            if not simulator_config.running:
                await asyncio.sleep(1)
                continue
            
            # Round-robin through entities
            entities = simulator_config.entities
            if not entities:
                await asyncio.sleep(1)
                continue
            
            entity_id = entities[entity_cycle_index % len(entities)]
            entity_cycle_index += 1
            
            # Generate and publish message
            message = generate_telemetry_message(entity_id)
            publish_message(entity_id, message)
            
            # Update metrics
            active_entities.set(len(entities))
            publish_rate.set(simulator_config.rate)
            
            # Sleep to maintain rate
            await asyncio.sleep(message_interval)
            
        except Exception as e:
            logger.error({"event": "simulation_loop_error", "error": str(e)})
            await asyncio.sleep(1)


# FastAPI app
app = FastAPI(title="Simulator Service")


@app.on_event("startup")
async def startup():
    """Initialize resources on startup"""
    global mqtt_client
    
    logger.info({"event": "service_starting", "service": "simulator"})
    
    # Initialize MQTT client
    mqtt_client = mqtt.Client(client_id=f"simulator-{int(time.time())}")
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_disconnect = on_mqtt_disconnect
    
    # Connect to MQTT broker
    logger.info({"event": "mqtt_connecting", "broker": settings.MQTT_BROKER_HOST})
    mqtt_client.connect(settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT, keepalive=60)
    mqtt_client.loop_start()
    
    # Initialize default entities
    simulator_config.entities = settings.DEFAULT_ENTITIES
    simulator_config.rate = settings.DEFAULT_RATE
    
    # Start simulation loop
    asyncio.create_task(simulation_loop())
    
    logger.info({"event": "service_started", "service": "simulator"})


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info({"event": "service_stopping", "service": "simulator"})
    
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    
    logger.info({"event": "service_stopped", "service": "simulator"})


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "simulator",
        "running": simulator_config.running,
        "rate": simulator_config.rate,
        "entities": simulator_config.entities,
        "mqtt_connected": mqtt_client.is_connected() if mqtt_client else False
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/config")
async def get_config():
    """Get current simulator configuration"""
    return simulator_config.dict()


@app.post("/config")
async def update_config(config: SimulatorConfig):
    """Update simulator configuration"""
    global simulator_config
    
    simulator_config.running = config.running
    simulator_config.rate = config.rate
    simulator_config.entities = config.entities
    simulator_config.anomaly_probability = config.anomaly_probability
    
    logger.info({
        "event": "config_updated",
        "running": config.running,
        "rate": config.rate,
        "entities": config.entities,
        "anomaly_probability": config.anomaly_probability
    })
    
    return {"status": "updated", "config": simulator_config.dict()}


@app.post("/start")
async def start_simulation():
    """Start the simulation"""
    simulator_config.running = True
    logger.info({"event": "simulation_started_via_api"})
    return {"status": "started"}


@app.post("/stop")
async def stop_simulation():
    """Stop the simulation"""
    simulator_config.running = False
    logger.info({"event": "simulation_stopped_via_api"})
    return {"status": "stopped"}


@app.get("/entities")
async def get_entities():
    """Get entity states"""
    return {
        "entities": simulator_config.entities,
        "states": {
            entity_id: {
                "message_count": state["message_count"],
                "drift_active": state["drift_active"]
            }
            for entity_id, state in entity_states.items()
        }
    }


@app.post("/inject-anomaly/{entity_id}")
async def inject_anomaly(entity_id: str, anomaly_type: AnomalyType):
    """Manually inject an anomaly for an entity"""
    if entity_id not in simulator_config.entities:
        return {"error": f"Entity {entity_id} not found"}
    
    if entity_id not in entity_states:
        entity_states[entity_id] = initialize_entity_state(entity_id)
    
    state = entity_states[entity_id]
    
    if anomaly_type == AnomalyType.DRIFT:
        initiate_drift(state)
    
    logger.info({
        "entity_id": entity_id,
        "event": "manual_anomaly_injection",
        "type": anomaly_type
    })
    
    return {"status": "injected", "entity_id": entity_id, "type": anomaly_type}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
