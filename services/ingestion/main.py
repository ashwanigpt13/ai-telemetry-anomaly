"""
Ingestion Service - Handles telemetry ingestion, windowing, and orchestration
"""
import asyncio
import csv
import json
import logging
import os
import time
import uuid
from collections import deque
from typing import Dict, Tuple, Optional, List
from contextlib import asynccontextmanager
import threading

import httpx
import paho.mqtt.client as mqtt
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

from config import settings


# Global event loop reference for thread-safe coroutine scheduling
main_loop: Optional[asyncio.AbstractEventLoop] = None

# Evaluation logging buffer
evaluation_buffer: List[Dict] = []
evaluation_lock = threading.Lock()

# Ground truth tracking per entity (most recent values)
entity_ground_truth: Dict[str, Dict] = {}


# Configure structured logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "ingestion",
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
messages_received = Counter("ingestion_messages_received_total", "Total messages received", ["entity_id"])
messages_processed = Counter("ingestion_messages_processed_total", "Total messages processed")
messages_failed = Counter("ingestion_messages_failed_total", "Total messages failed", ["reason"])
window_processing_latency = Histogram("ingestion_window_processing_seconds", "Window processing latency")
active_buffers = Gauge("ingestion_active_buffers", "Number of active entity buffers")
buffers_cleaned = Counter("ingestion_buffers_cleaned_total", "Total buffers cleaned due to timeout")
model_request_latency = Histogram("ingestion_model_request_seconds", "Model service request latency")
threshold_request_latency = Histogram("ingestion_threshold_request_seconds", "Threshold service request latency")


# Entity buffers: Dict[entity_id, deque[(timestamp, features)]]
entity_buffers: Dict[str, deque] = {}
entity_last_seen: Dict[str, float] = {}
entity_msg_count: Dict[str, int] = {}  # Message counter for stride-based windowing

# HTTP client for async requests
http_client: httpx.AsyncClient = None

# MQTT client
mqtt_client: mqtt.Client = None


def log_evaluation_record(entity_id: str, timestamp: int, predicted_anomaly: bool,
                          error: float, threshold: float, drift: bool):
    """Log evaluation record combining ground truth + predictions"""
    if not settings.EVALUATION_LOG_ENABLED:
        return
    
    # Get ground truth for this entity
    gt = entity_ground_truth.get(entity_id, {})
    actual_anomaly = gt.get("is_anomaly", False)
    anomaly_type = gt.get("anomaly_type", "normal")
    ingestion_time = gt.get("ingestion_time", timestamp)
    
    # Compute end-to-end latency
    current_time = int(time.time() * 1000)
    latency_ms = current_time - ingestion_time
    
    record = {
        "entity_id": entity_id,
        "timestamp": timestamp,
        "actual_anomaly": actual_anomaly,
        "anomaly_type": anomaly_type,
        "predicted_anomaly": predicted_anomaly,
        "error": error,
        "threshold": threshold,
        "drift": drift,
        "latency_ms": latency_ms
    }
    
    with evaluation_lock:
        evaluation_buffer.append(record)
        
        # Flush to file if buffer is full
        if len(evaluation_buffer) >= settings.EVALUATION_BUFFER_SIZE:
            flush_evaluation()


def flush_evaluation():
    """Flush evaluation buffer to CSV file"""
    global evaluation_buffer
    
    if not evaluation_buffer:
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(settings.EVALUATION_LOG_PATH), exist_ok=True)
    
    file_exists = os.path.exists(settings.EVALUATION_LOG_PATH)
    
    try:
        with open(settings.EVALUATION_LOG_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "entity_id", "timestamp", "actual_anomaly", "anomaly_type",
                "predicted_anomaly", "error", "threshold", "drift", "latency_ms"
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerows(evaluation_buffer)
        
        logger.info({
            "event": "evaluation_flushed",
            "count": len(evaluation_buffer),
            "path": settings.EVALUATION_LOG_PATH
        })
        evaluation_buffer = []
    except Exception as e:
        logger.error({
            "event": "evaluation_flush_error",
            "error": str(e)
        })


def on_connect(client, userdata, flags, rc):
    """MQTT on_connect callback"""
    if rc == 0:
        logger.info({"event": "mqtt_connected", "message": "Connected to MQTT broker"})
        # Subscribe using shared subscription for load balancing
        topic = f"$share/{settings.MQTT_SHARE_GROUP}/{settings.MQTT_TOPIC_TELEMETRY}"
        client.subscribe(topic, qos=1)
        logger.info({"event": "mqtt_subscribed", "topic": topic})
    else:
        logger.error({"event": "mqtt_connection_failed", "rc": rc})


def on_disconnect(client, userdata, rc):
    """MQTT on_disconnect callback"""
    logger.warning({"event": "mqtt_disconnected", "rc": rc})
    if rc != 0:
        logger.info({"event": "mqtt_reconnecting", "message": "Unexpected disconnect, will auto-reconnect"})


def on_message(client, userdata, msg):
    """MQTT on_message callback"""
    global main_loop
    try:
        # Parse message
        payload = json.loads(msg.payload.decode())
        entity_id = payload.get("entity_id")
        timestamp = payload.get("timestamp")
        features = payload.get("features")
        
        # Extract ground truth (optional fields)
        is_anomaly = payload.get("is_anomaly", False)
        anomaly_type = payload.get("anomaly_type", "normal")
        ingestion_time = int(time.time() * 1000)  # For latency tracking
        
        if not entity_id or timestamp is None or features is None:
            logger.error({"event": "invalid_message", "message": "Missing required fields"})
            messages_failed.labels(reason="invalid_format").inc()
            return
        
        messages_received.labels(entity_id=entity_id).inc()
        
        # Store ground truth for latest message
        entity_ground_truth[entity_id] = {
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "timestamp": timestamp,
            "ingestion_time": ingestion_time
        }
        
        # Schedule async processing in the main event loop (thread-safe)
        if main_loop is not None and main_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                process_telemetry(entity_id, timestamp, features),
                main_loop
            )
        else:
            logger.warning({"event": "no_event_loop", "message": "Main event loop not available"})
        
    except json.JSONDecodeError as e:
        logger.error({"event": "json_decode_error", "error": str(e)})
        messages_failed.labels(reason="json_decode").inc()
    except Exception as e:
        logger.error({"event": "message_processing_error", "error": str(e)})
        messages_failed.labels(reason="unknown").inc()


async def process_telemetry(entity_id: str, timestamp: int, features: list):
    """Process incoming telemetry message"""
    trace_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Update last seen time
        entity_last_seen[entity_id] = time.time()
        
        # Initialize buffer if needed
        if entity_id not in entity_buffers:
            entity_buffers[entity_id] = deque(maxlen=settings.WINDOW_SIZE)
            entity_msg_count[entity_id] = 0
            active_buffers.set(len(entity_buffers))
            logger.info({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "buffer_created"
            })
        
        # Add to buffer
        entity_buffers[entity_id].append((timestamp, features))
        entity_msg_count[entity_id] += 1
        
        # Check if window is ready (buffer full AND stride reached)
        if len(entity_buffers[entity_id]) >= settings.WINDOW_SIZE:
            if entity_msg_count[entity_id] >= settings.WINDOW_STRIDE:
                await process_window(entity_id, trace_id)
                entity_msg_count[entity_id] = 0  # Reset stride counter
                messages_processed.inc()
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug({
            "trace_id": trace_id,
            "entity_id": entity_id,
            "event": "telemetry_buffered",
            "buffer_size": len(entity_buffers[entity_id]),
            "latency_ms": latency_ms
        })
        
    except Exception as e:
        logger.error({
            "trace_id": trace_id,
            "entity_id": entity_id,
            "event": "processing_error",
            "error": str(e)
        })
        messages_failed.labels(reason="processing_error").inc()


async def process_window(entity_id: str, trace_id: str):
    """Process a complete window for an entity"""
    start_time = time.time()
    
    try:
        # Extract window data
        buffer = entity_buffers[entity_id]
        window_data = [features for (ts, features) in buffer]
        last_timestamp = buffer[-1][0]
        
        logger.info({
            "trace_id": trace_id,
            "entity_id": entity_id,
            "event": "window_ready",
            "window_size": len(window_data)
        })
        
        # Step 1: Call Model Service
        model_start = time.time()
        try:
            model_payload = {
                "entity_id": entity_id,
                "window": window_data
            }
            model_response = await http_client.post(
                f"{settings.MODEL_SERVICE_URL}/infer",
                json=model_payload,
                timeout=settings.HTTP_TIMEOUT
            )
            model_response.raise_for_status()
            model_result = model_response.json()
            reconstruction_error = model_result["reconstruction_error"]
            
            model_latency = (time.time() - model_start) * 1000
            model_request_latency.observe(time.time() - model_start)
            
            logger.info({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "model_inference_complete",
                "error": reconstruction_error,
                "latency_ms": model_latency
            })
            
        except httpx.TimeoutException:
            logger.error({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "model_timeout"
            })
            messages_failed.labels(reason="model_timeout").inc()
            return
        except httpx.HTTPStatusError as e:
            logger.error({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "model_http_error",
                "status_code": e.response.status_code
            })
            messages_failed.labels(reason="model_http_error").inc()
            return
        except Exception as e:
            logger.error({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "model_error",
                "error": str(e)
            })
            messages_failed.labels(reason="model_error").inc()
            return
        
        # Step 2: Call Threshold Engine
        threshold_start = time.time()
        try:
            threshold_payload = {
                "entity_id": entity_id,
                "timestamp": last_timestamp,
                "error": reconstruction_error
            }
            threshold_response = await http_client.post(
                f"{settings.THRESHOLD_SERVICE_URL}/update",
                json=threshold_payload,
                timeout=settings.HTTP_TIMEOUT
            )
            threshold_response.raise_for_status()
            threshold_result = threshold_response.json()
            
            threshold_latency = (time.time() - threshold_start) * 1000
            threshold_request_latency.observe(time.time() - threshold_start)
            
            logger.info({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "threshold_update_complete",
                "anomaly": threshold_result.get("anomaly"),
                "threshold": threshold_result.get("threshold"),
                "drift": threshold_result.get("drift"),
                "latency_ms": threshold_latency
            })
            
            # Log evaluation record combining ground truth + predictions
            log_evaluation_record(
                entity_id=entity_id,
                timestamp=last_timestamp,
                predicted_anomaly=threshold_result.get("anomaly", False),
                error=reconstruction_error,
                threshold=threshold_result.get("threshold", 0),
                drift=threshold_result.get("drift", False)
            )
            
        except httpx.TimeoutException:
            logger.error({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "threshold_timeout"
            })
            messages_failed.labels(reason="threshold_timeout").inc()
            return
        except httpx.HTTPStatusError as e:
            logger.error({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "threshold_http_error",
                "status_code": e.response.status_code
            })
            messages_failed.labels(reason="threshold_http_error").inc()
            return
        except Exception as e:
            logger.error({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "threshold_error",
                "error": str(e)
            })
            messages_failed.labels(reason="threshold_error").inc()
            return
        
        total_latency = (time.time() - start_time) * 1000
        window_processing_latency.observe(time.time() - start_time)
        
        logger.info({
            "trace_id": trace_id,
            "entity_id": entity_id,
            "event": "window_processed",
            "latency_ms": total_latency
        })
        
    except Exception as e:
        logger.error({
            "trace_id": trace_id,
            "entity_id": entity_id,
            "event": "window_processing_error",
            "error": str(e)
        })
        messages_failed.labels(reason="window_processing_error").inc()


async def cleanup_stale_buffers():
    """Remove buffers for entities that haven't sent data in ENTITY_TIMEOUT_SEC"""
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            current_time = time.time()
            stale_entities = []
            
            for entity_id, last_seen in entity_last_seen.items():
                if current_time - last_seen > settings.ENTITY_TIMEOUT_SEC:
                    stale_entities.append(entity_id)
            
            for entity_id in stale_entities:
                if entity_id in entity_buffers:
                    del entity_buffers[entity_id]
                if entity_id in entity_last_seen:
                    del entity_last_seen[entity_id]
                if entity_id in entity_msg_count:
                    del entity_msg_count[entity_id]
                buffers_cleaned.inc()
                logger.info({
                    "entity_id": entity_id,
                    "event": "buffer_removed",
                    "reason": "timeout"
                })
            
            if stale_entities:
                active_buffers.set(len(entity_buffers))
                logger.info({
                    "event": "cleanup_complete",
                    "buffers_removed": len(stale_entities),
                    "active_buffers": len(entity_buffers)
                })
                
        except Exception as e:
            logger.error({"event": "cleanup_error", "error": str(e)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global http_client, mqtt_client, main_loop
    
    # Startup
    logger.info({"event": "service_starting", "service": "ingestion"})
    
    # Store reference to the main event loop for thread-safe scheduling
    main_loop = asyncio.get_running_loop()
    
    # Initialize HTTP client with proper timeout configuration
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.HTTP_TIMEOUT, connect=5.0)
    )
    
    # Initialize MQTT client
    mqtt_client = mqtt.Client(client_id=f"ingestion-{uuid.uuid4().hex[:8]}")
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect
    mqtt_client.on_message = on_message
    
    # Connect to MQTT broker
    logger.info({"event": "mqtt_connecting", "broker": settings.MQTT_BROKER_HOST})
    mqtt_client.connect(settings.MQTT_BROKER_HOST, settings.MQTT_BROKER_PORT, keepalive=60)
    mqtt_client.loop_start()
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_stale_buffers())
    logger.info({
        "event": "cleanup_task_started",
        "check_interval_sec": 60,
        "entity_timeout_sec": settings.ENTITY_TIMEOUT_SEC
    })
    
    logger.info({"event": "service_started", "service": "ingestion"})
    
    yield
    
    # Shutdown
    logger.info({"event": "service_stopping", "service": "ingestion"})
    
    # Flush remaining evaluation records
    with evaluation_lock:
        flush_evaluation()
    
    cleanup_task.cancel()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    await http_client.aclose()
    main_loop = None
    
    logger.info({"event": "service_stopped", "service": "ingestion"})


# FastAPI app
app = FastAPI(title="Ingestion Service", lifespan=lifespan)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ingestion",
        "active_buffers": len(entity_buffers),
        "mqtt_connected": mqtt_client.is_connected() if mqtt_client else False
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/stats")
async def stats():
    """Service statistics endpoint"""
    return {
        "active_buffers": len(entity_buffers),
        "entities": list(entity_buffers.keys()),
        "buffer_sizes": {entity_id: len(buffer) for entity_id, buffer in entity_buffers.items()}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
