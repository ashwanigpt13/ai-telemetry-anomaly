"""
Threshold Engine Service - Adaptive threshold computation with drift detection
"""
import csv
import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional, List, Dict
import threading

import numpy as np
import paho.mqtt.client as mqtt
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from pydantic import BaseModel, Field

from config import settings


# Prediction logging buffer
prediction_buffer: List[Dict] = []
prediction_lock = threading.Lock()

# Global error sliding window (for global_dynamic mode)
global_errors: deque = deque(maxlen=settings.ERROR_WINDOW)

# Anomaly persistence tracking: consecutive anomaly counts per entity
entity_anomaly_streak: Dict[str, int] = {}

# Error smoothing: recent errors per entity for moving average
entity_recent_errors: Dict[str, deque] = {}


# Configure structured logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "threshold",
            "message": record.getMessage(),
        }
        if hasattr(record, "trace_id"):
            log_obj["trace_id"] = record.trace_id
        if hasattr(record, "entity_id"):
            log_obj["entity_id"] = record.entity_id
        if hasattr(record, "event"):
            log_obj["event"] = record.event
        if hasattr(record, "latency_ms"):
            log_obj["latency_ms"] = record.latency_ms
        return json.dumps(log_obj)


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Prometheus metrics
update_requests = Counter("threshold_update_requests_total", "Total threshold update requests")
update_failures = Counter("threshold_update_failures_total", "Total update failures", ["reason"])
update_latency = Histogram("threshold_update_seconds", "Update latency")
anomalies_detected = Counter("threshold_anomalies_detected_total", "Total anomalies detected", ["entity_id"])
drift_detected = Counter("threshold_drift_detected_total", "Total drift events detected", ["entity_id"])
cold_start_count = Counter("threshold_cold_start_total", "Total cold start detections", ["entity_id"])
threshold_values = Histogram("threshold_values", "Threshold value distribution",
                             buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
active_entities = Gauge("threshold_active_entities", "Number of entities with data in Redis")


# Request/Response models
class ThresholdUpdateRequest(BaseModel):
    """Threshold update request"""
    entity_id: str = Field(..., description="Entity identifier")
    timestamp: int = Field(..., description="Unix timestamp in milliseconds")
    error: float = Field(..., description="Reconstruction error from model")


class ThresholdUpdateResponse(BaseModel):
    """Threshold update response"""
    anomaly: bool = Field(..., description="Whether error exceeds threshold")
    error: float = Field(..., description="Reconstruction error")
    threshold: float = Field(..., description="Current adaptive threshold")
    drift: bool = Field(..., description="Whether drift was detected")


# Global resources
redis_client: Optional[aioredis.Redis] = None
mqtt_client: Optional[mqtt.Client] = None


# Lua script for atomic Redis updates
LUA_UPDATE_SCRIPT = """
local recent_key = KEYS[1]
local past_key = KEYS[2]
local error = tonumber(ARGV[1])
local n_recent = tonumber(ARGV[2])
local n_past = tonumber(ARGV[3])

-- Add new error to recent list
redis.call('LPUSH', recent_key, error)

-- If recent list exceeds size, move oldest to past
local recent_len = redis.call('LLEN', recent_key)
if recent_len > n_recent then
    redis.call('RPOPLPUSH', recent_key, past_key)
end

-- Trim both lists to their max sizes
redis.call('LTRIM', recent_key, 0, n_recent - 1)
redis.call('LTRIM', past_key, 0, n_past - 1)

return {recent_len, redis.call('LLEN', past_key)}
"""


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


async def get_error_lists(entity_id: str):
    """Retrieve recent and past error lists from Redis"""
    recent_key = f"entity:{entity_id}:recent_errors"
    past_key = f"entity:{entity_id}:past_errors"
    
    # Get all errors from both lists
    recent_errors_raw = await redis_client.lrange(recent_key, 0, -1)
    past_errors_raw = await redis_client.lrange(past_key, 0, -1)
    
    # Convert from bytes to float
    recent_errors = [float(e) for e in recent_errors_raw]
    past_errors = [float(e) for e in past_errors_raw]
    
    return recent_errors, past_errors


async def update_error_lists(entity_id: str, error: float):
    """Update error lists atomically using Lua script"""
    recent_key = f"entity:{entity_id}:recent_errors"
    past_key = f"entity:{entity_id}:past_errors"
    
    # Execute Lua script for atomic update
    await redis_client.eval(
        LUA_UPDATE_SCRIPT,
        2,  # number of keys
        recent_key,
        past_key,
        error,
        settings.N_RECENT,
        settings.N_PAST
    )


async def update_stats_hash(entity_id: str, mean: float, std: float, threshold: float, drift: bool):
    """Update entity stats in Redis HASH"""
    stats_key = f"entity:{entity_id}:stats"
    
    # Store current stats in Redis HASH
    await redis_client.hset(
        stats_key,
        mapping={
            "mean": str(mean),
            "std": str(std),
            "threshold": str(threshold),
            "drift_flag": str(int(drift))
        }
    )


def compute_statistics(recent_errors: list, past_errors: list):
    """Compute mean and std for recent and past errors"""
    stats = {
        "mean_recent": 0.0,
        "std_recent": settings.EPSILON,
        "mean_past": 0.0,
        "total_errors": len(recent_errors) + len(past_errors)
    }
    
    if len(recent_errors) > 0:
        recent_array = np.array(recent_errors)
        stats["mean_recent"] = float(np.mean(recent_array))
        stats["std_recent"] = max(float(np.std(recent_array)), settings.EPSILON)
    
    if len(past_errors) > 0:
        past_array = np.array(past_errors)
        stats["mean_past"] = float(np.mean(past_array))
    
    return stats


def compute_threshold(stats: dict, drift: bool) -> float:
    """Compute adaptive threshold (entity_drift mode - default)"""
    threshold = stats["mean_recent"] + settings.K * stats["std_recent"]
    
    # Apply drift adjustment
    if drift:
        threshold += settings.ALPHA * stats["std_recent"]
    
    return threshold


def compute_threshold_by_mode(stats: dict, drift: bool, error: float) -> float:
    """
    Compute threshold based on configured mode:
    - global_static: GLOBAL_MEAN + K * GLOBAL_STD
    - global_dynamic: sliding window over all entities
    - entity_dynamic: mean_recent + K * std_recent (no drift adjustment)
    - entity_drift: current implementation with drift adjustment
    """
    mode = settings.THRESHOLD_MODE
    
    if mode == "global_static":
        return settings.GLOBAL_MEAN + settings.K * settings.GLOBAL_STD
    
    elif mode == "global_dynamic":
        # Use global sliding window
        global_errors.append(error)
        if len(global_errors) > 0:
            global_mean = float(np.mean(list(global_errors)))
            global_std = max(float(np.std(list(global_errors))), settings.EPSILON)
            return global_mean + settings.K * global_std
        else:
            return settings.GLOBAL_MEAN + settings.K * settings.GLOBAL_STD
    
    elif mode == "entity_dynamic":
        # Entity-specific but no drift adjustment
        return stats["mean_recent"] + settings.K * stats["std_recent"]
    
    else:  # entity_drift (default)
        return compute_threshold(stats, drift)


def log_prediction(entity_id: str, timestamp: int, predicted_anomaly: bool,
                   threshold: float, error: float, drift: bool):
    """Log prediction for evaluation"""
    if not settings.PREDICTION_LOG_ENABLED:
        return
    
    record = {
        "entity_id": entity_id,
        "timestamp": timestamp,
        "predicted_anomaly": predicted_anomaly,
        "threshold": threshold,
        "error": error,
        "drift": drift
    }
    
    with prediction_lock:
        prediction_buffer.append(record)
        
        # Flush to file if buffer is full
        if len(prediction_buffer) >= settings.PREDICTION_BUFFER_SIZE:
            flush_predictions()


def flush_predictions():
    """Flush prediction buffer to CSV file"""
    global prediction_buffer
    
    if not prediction_buffer:
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(settings.PREDICTION_LOG_PATH), exist_ok=True)
    
    file_exists = os.path.exists(settings.PREDICTION_LOG_PATH)
    
    try:
        with open(settings.PREDICTION_LOG_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "entity_id", "timestamp", "predicted_anomaly", 
                "threshold", "error", "drift"
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerows(prediction_buffer)
        
        logger.info({
            "event": "predictions_flushed",
            "count": len(prediction_buffer),
            "path": settings.PREDICTION_LOG_PATH
        })
        prediction_buffer = []
    except Exception as e:
        logger.error({
            "event": "prediction_flush_error",
            "error": str(e)
        })


def detect_drift(recent_errors: list, past_errors: list, stats: dict) -> bool:
    """Detect concept drift"""
    # Need sufficient data in both windows
    if len(recent_errors) < settings.N_RECENT or len(past_errors) < settings.N_PAST:
        return False
    
    # Check if mean shift exceeds threshold
    mean_diff = abs(stats["mean_recent"] - stats["mean_past"])
    drift = mean_diff > settings.DRIFT_THRESHOLD
    
    return drift


def is_cold_start(total_errors: int) -> bool:
    """Check if entity is in cold start phase"""
    return total_errors < settings.ERROR_WINDOW


async def publish_anomaly(entity_id: str, timestamp: int, anomaly: bool, error: float, 
                         threshold: float, drift: bool):
    """Publish anomaly result to MQTT"""
    try:
        message = {
            "entity_id": entity_id,
            "timestamp": timestamp,
            "anomaly": anomaly,
            "error": error,
            "threshold": threshold,
            "drift": drift
        }
        
        topic = f"{settings.MQTT_TOPIC_ANOMALY}{entity_id}"
        mqtt_client.publish(topic, json.dumps(message), qos=1)
        
        logger.info({
            "entity_id": entity_id,
            "event": "anomaly_published",
            "topic": topic,
            "anomaly": anomaly
        })
        
    except Exception as e:
        logger.error({
            "entity_id": entity_id,
            "event": "mqtt_publish_error",
            "error": str(e)
        })


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global redis_client, mqtt_client
    
    # Startup
    logger.info({"event": "service_starting", "service": "threshold"})
    
    # Initialize Redis client
    redis_client = aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=False
    )
    
    # Test Redis connection
    try:
        await redis_client.ping()
        logger.info({"event": "redis_connected", "url": settings.REDIS_URL})
    except Exception as e:
        logger.error({"event": "redis_connection_failed", "error": str(e)})
        raise
    
    # Initialize MQTT client
    mqtt_client = mqtt.Client(client_id=f"threshold-{uuid.uuid4().hex[:8]}")
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_disconnect = on_mqtt_disconnect
    
    # Connect to MQTT broker
    logger.info({"event": "mqtt_connecting", "broker": settings.MQTT_BROKER_HOST})
    mqtt_client.connect(settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT, keepalive=60)
    mqtt_client.loop_start()
    
    logger.info({"event": "service_started", "service": "threshold"})
    
    yield
    
    # Shutdown
    logger.info({"event": "service_stopping", "service": "threshold"})
    
    # Flush remaining predictions
    with prediction_lock:
        flush_predictions()
    
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    await redis_client.close()
    
    logger.info({"event": "service_stopped", "service": "threshold"})


# FastAPI app
app = FastAPI(title="Threshold Engine Service", lifespan=lifespan)


@app.post("/update", response_model=ThresholdUpdateResponse)
async def update_threshold(request: ThresholdUpdateRequest):
    """
    Update threshold and detect anomalies with drift awareness
    
    - **entity_id**: Entity identifier
    - **timestamp**: Unix timestamp in milliseconds
    - **error**: Reconstruction error from model
    
    Returns anomaly detection result with threshold and drift status
    """
    trace_id = str(uuid.uuid4())
    start_time = time.time()
    
    update_requests.inc()
    
    try:
        entity_id = request.entity_id
        error = request.error
        timestamp = request.timestamp
        
        logger.debug({
            "trace_id": trace_id,
            "entity_id": entity_id,
            "event": "update_started",
            "error": error
        })
        
        # Step 1: Update error lists atomically
        await update_error_lists(entity_id, error)
        
        # Step 1.5: Apply error smoothing (moving average)
        if entity_id not in entity_recent_errors:
            entity_recent_errors[entity_id] = deque(maxlen=settings.ERROR_SMOOTH_WINDOW)
        entity_recent_errors[entity_id].append(error)
        smoothed_error = float(np.mean(list(entity_recent_errors[entity_id])))
        
        # Step 2: Retrieve updated lists
        recent_errors, past_errors = await get_error_lists(entity_id)
        
        # Step 3: Compute statistics
        stats = compute_statistics(recent_errors, past_errors)
        
        # Step 4: Check for cold start
        if is_cold_start(stats["total_errors"]):
            # Use global statistics
            threshold = settings.GLOBAL_MEAN + settings.K * settings.GLOBAL_STD
            drift = False
            raw_anomaly = smoothed_error > threshold
            
            cold_start_count.labels(entity_id=entity_id).inc()
            
            logger.info({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "cold_start",
                "total_errors": stats["total_errors"],
                "threshold": threshold
            })
        else:
            # Step 5: Detect drift
            drift = detect_drift(recent_errors, past_errors, stats)
            
            if drift:
                drift_detected.labels(entity_id=entity_id).inc()
                logger.info({
                    "trace_id": trace_id,
                    "entity_id": entity_id,
                    "event": "drift_detected",
                    "mean_recent": stats["mean_recent"],
                    "mean_past": stats["mean_past"],
                    "diff": abs(stats["mean_recent"] - stats["mean_past"])
                })
            
            # Step 6: Compute threshold based on mode
            threshold = compute_threshold_by_mode(stats, drift, smoothed_error)
            
            # Step 7: Determine raw anomaly (before persistence)
            raw_anomaly = smoothed_error > threshold
        
        # Step 7.5: Apply anomaly persistence filter
        if entity_id not in entity_anomaly_streak:
            entity_anomaly_streak[entity_id] = 0
        
        if raw_anomaly:
            entity_anomaly_streak[entity_id] += 1
        else:
            entity_anomaly_streak[entity_id] = 0
        
        # Only mark as anomaly if consecutive count >= ANOMALY_PERSISTENCE
        anomaly = entity_anomaly_streak[entity_id] >= settings.ANOMALY_PERSISTENCE
        
        # Step 8: Update stats in Redis HASH
        await update_stats_hash(
            entity_id=entity_id,
            mean=stats["mean_recent"],
            std=stats["std_recent"],
            threshold=threshold,
            drift=drift
        )
        
        # Record metrics
        threshold_values.observe(threshold)
        if anomaly:
            anomalies_detected.labels(entity_id=entity_id).inc()
        
        # Step 9: Log prediction for evaluation
        log_prediction(entity_id, timestamp, anomaly, threshold, error, drift)
        
        # Step 10: Publish to MQTT
        await publish_anomaly(entity_id, timestamp, anomaly, error, threshold, drift)
        
        latency = time.time() - start_time
        update_latency.observe(latency)
        
        logger.info({
            "trace_id": trace_id,
            "entity_id": entity_id,
            "event": "update_complete",
            "anomaly": anomaly,
            "error": error,
            "smoothed_error": smoothed_error,
            "threshold": threshold,
            "drift": drift,
            "anomaly_streak": entity_anomaly_streak[entity_id],
            "latency_ms": latency * 1000
        })
        
        return ThresholdUpdateResponse(
            anomaly=anomaly,
            error=error,
            threshold=threshold,
            drift=drift
        )
    
    except Exception as e:
        logger.error({
            "trace_id": trace_id,
            "entity_id": request.entity_id,
            "event": "update_error",
            "error": str(e)
        })
        update_failures.labels(reason="unknown").inc()
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@app.get("/entity/{entity_id}")
async def get_entity_stats(entity_id: str):
    """
    Get current statistics for an entity
    
    Returns error history, statistics, and current threshold
    """
    try:
        # Retrieve error lists
        recent_errors, past_errors = await get_error_lists(entity_id)
        
        # Compute statistics
        stats = compute_statistics(recent_errors, past_errors)
        
        # Check cold start
        if is_cold_start(stats["total_errors"]):
            threshold = settings.GLOBAL_MEAN + settings.K * settings.GLOBAL_STD
            drift = False
        else:
            drift = detect_drift(recent_errors, past_errors, stats)
            threshold = compute_threshold(stats, drift)
        
        return {
            "entity_id": entity_id,
            "total_errors": stats["total_errors"],
            "recent_count": len(recent_errors),
            "past_count": len(past_errors),
            "mean_recent": stats["mean_recent"],
            "std_recent": stats["std_recent"],
            "mean_past": stats["mean_past"],
            "threshold": threshold,
            "drift": drift,
            "cold_start": is_cold_start(stats["total_errors"])
        }
    
    except Exception as e:
        logger.error({
            "entity_id": entity_id,
            "event": "get_stats_error",
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    redis_healthy = False
    
    try:
        await redis_client.ping()
        redis_healthy = True
    except:
        pass
    
    return {
        "status": "healthy" if redis_healthy else "unhealthy",
        "service": "threshold",
        "redis_connected": redis_healthy,
        "mqtt_connected": mqtt_client.is_connected() if mqtt_client else False
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/stats")
async def stats():
    """Service statistics"""
    # Count entities with data
    try:
        keys = await redis_client.keys("entity:*:recent_errors")
        entity_count = len(keys)
        active_entities.set(entity_count)
        
        return {
            "active_entities": entity_count,
            "entity_ids": [key.decode().split(":")[1] for key in keys]
        }
    except Exception as e:
        logger.error({"event": "stats_error", "error": str(e)})
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
