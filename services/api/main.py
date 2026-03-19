"""
API Gateway Service - Unified interface for anomaly detection
"""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
import numpy as np
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from config import settings


# Configure structured logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "service": "api",
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
predict_requests = Counter("api_predict_requests_total", "Total predict requests")
predict_failures = Counter("api_predict_failures_total", "Total predict failures", ["reason"])
predict_latency = Histogram("api_predict_seconds", "Predict endpoint latency")
entity_requests = Counter("api_entity_requests_total", "Total entity requests")
entity_failures = Counter("api_entity_failures_total", "Total entity failures", ["reason"])


# Request/Response models
class PredictRequest(BaseModel):
    """Prediction request"""
    entity_id: str = Field(..., description="Entity identifier")
    window: List[List[float]] = Field(..., description="Feature window (WINDOW_SIZE x num_features)")


class PredictResponse(BaseModel):
    """Prediction response"""
    anomaly: bool = Field(..., description="Whether error exceeds threshold")
    error: float = Field(..., description="Reconstruction error from model")
    threshold: float = Field(..., description="Current adaptive threshold")
    drift: bool = Field(..., description="Whether drift was detected")


class EntityStateResponse(BaseModel):
    """Entity state response"""
    entity_id: str
    total_errors: int
    recent_count: int
    past_count: int
    mean_recent: float
    std_recent: float
    mean_past: float
    threshold: float
    drift: bool
    cold_start: bool


# Global resources
http_client: Optional[httpx.AsyncClient] = None
redis_client: Optional[aioredis.Redis] = None


async def get_error_lists(entity_id: str):
    """Retrieve recent and past error lists from Redis"""
    recent_key = f"entity:{entity_id}:recent_errors"
    past_key = f"entity:{entity_id}:past_errors"
    
    # Get all errors from both lists
    recent_errors_raw = await redis_client.lrange(recent_key, 0, -1)
    past_errors_raw = await redis_client.lrange(past_key, 0, -1)
    
    # Convert from bytes to float
    recent_errors = [float(e) for e in recent_errors_raw] if recent_errors_raw else []
    past_errors = [float(e) for e in past_errors_raw] if past_errors_raw else []
    
    return recent_errors, past_errors


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
    """Compute adaptive threshold"""
    threshold = stats["mean_recent"] + settings.K * stats["std_recent"]
    
    # Apply drift adjustment
    if drift:
        threshold += settings.ALPHA * stats["std_recent"]
    
    return threshold


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global http_client, redis_client
    
    # Startup
    logger.info({"event": "service_starting", "service": "api"})
    
    # Initialize HTTP client
    http_client = httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT)
    
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
    
    logger.info({"event": "service_started", "service": "api"})
    
    yield
    
    # Shutdown
    logger.info({"event": "service_stopping", "service": "api"})
    
    await http_client.aclose()
    await redis_client.close()
    
    logger.info({"event": "service_stopped", "service": "api"})


# FastAPI app
app = FastAPI(
    title="API Gateway Service",
    description="Unified interface for anomaly detection system",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict anomaly for a feature window
    
    Orchestrates calls to:
    1. Model Service - Get reconstruction error
    2. Threshold Engine - Compute threshold and detect anomaly
    
    - **entity_id**: Entity identifier
    - **window**: Feature window of shape (WINDOW_SIZE, num_features)
    
    Returns anomaly detection result
    """
    trace_id = str(uuid.uuid4())
    start_time = time.time()
    
    predict_requests.inc()
    
    try:
        entity_id = request.entity_id
        window = request.window
        
        logger.info({
            "trace_id": trace_id,
            "entity_id": entity_id,
            "event": "predict_started",
            "window_shape": [len(window), len(window[0]) if window else 0]
        })
        
        # Step 1: Call Model Service
        model_start = time.time()
        try:
            model_payload = {
                "entity_id": entity_id,
                "window": window
            }
            model_response = await http_client.post(
                f"{settings.MODEL_SERVICE_URL}/infer",
                json=model_payload
            )
            model_response.raise_for_status()
            model_result = model_response.json()
            reconstruction_error = model_result["reconstruction_error"]
            
            model_latency = (time.time() - model_start) * 1000
            
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
            predict_failures.labels(reason="model_timeout").inc()
            raise HTTPException(status_code=504, detail="Model service timeout")
        except httpx.HTTPStatusError as e:
            logger.error({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "model_http_error",
                "status_code": e.response.status_code
            })
            predict_failures.labels(reason="model_error").inc()
            raise HTTPException(status_code=502, detail=f"Model service error: {e.response.status_code}")
        except Exception as e:
            logger.error({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "model_error",
                "error": str(e)
            })
            predict_failures.labels(reason="model_error").inc()
            raise HTTPException(status_code=502, detail=f"Model service error: {str(e)}")
        
        # Step 2: Call Threshold Engine
        threshold_start = time.time()
        try:
            # Get current timestamp in milliseconds
            timestamp = int(time.time() * 1000)
            
            threshold_payload = {
                "entity_id": entity_id,
                "timestamp": timestamp,
                "error": reconstruction_error
            }
            threshold_response = await http_client.post(
                f"{settings.THRESHOLD_SERVICE_URL}/update",
                json=threshold_payload
            )
            threshold_response.raise_for_status()
            threshold_result = threshold_response.json()
            
            threshold_latency = (time.time() - threshold_start) * 1000
            
            logger.info({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "threshold_update_complete",
                "anomaly": threshold_result.get("anomaly"),
                "threshold": threshold_result.get("threshold"),
                "drift": threshold_result.get("drift"),
                "latency_ms": threshold_latency
            })
            
        except httpx.TimeoutException:
            logger.error({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "threshold_timeout"
            })
            predict_failures.labels(reason="threshold_timeout").inc()
            raise HTTPException(status_code=504, detail="Threshold service timeout")
        except httpx.HTTPStatusError as e:
            logger.error({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "threshold_http_error",
                "status_code": e.response.status_code
            })
            predict_failures.labels(reason="threshold_error").inc()
            raise HTTPException(status_code=502, detail=f"Threshold service error: {e.response.status_code}")
        except Exception as e:
            logger.error({
                "trace_id": trace_id,
                "entity_id": entity_id,
                "event": "threshold_error",
                "error": str(e)
            })
            predict_failures.labels(reason="threshold_error").inc()
            raise HTTPException(status_code=502, detail=f"Threshold service error: {str(e)}")
        
        total_latency = (time.time() - start_time) * 1000
        predict_latency.observe(time.time() - start_time)
        
        logger.info({
            "trace_id": trace_id,
            "entity_id": entity_id,
            "event": "predict_complete",
            "anomaly": threshold_result["anomaly"],
            "latency_ms": total_latency
        })
        
        return PredictResponse(
            anomaly=threshold_result["anomaly"],
            error=threshold_result["error"],
            threshold=threshold_result["threshold"],
            drift=threshold_result["drift"]
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error({
            "trace_id": trace_id,
            "entity_id": request.entity_id,
            "event": "predict_error",
            "error": str(e)
        })
        predict_failures.labels(reason="unknown").inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/entity/{entity_id}", response_model=EntityStateResponse)
async def get_entity_state(entity_id: str):
    """
    Get current state and statistics for an entity
    
    Reads directly from Redis to retrieve:
    - Error history (recent and past)
    - Statistics (mean, std)
    - Current threshold
    - Drift status
    - Cold start status
    
    - **entity_id**: Entity identifier
    
    Returns entity state information
    """
    entity_requests.inc()
    
    try:
        logger.info({
            "entity_id": entity_id,
            "event": "entity_state_requested"
        })
        
        # Retrieve error lists from Redis
        recent_errors, past_errors = await get_error_lists(entity_id)
        
        # Compute statistics
        stats = compute_statistics(recent_errors, past_errors)
        
        # Check cold start
        cold_start = is_cold_start(stats["total_errors"])
        
        if cold_start:
            threshold = settings.GLOBAL_MEAN + settings.K * settings.GLOBAL_STD
            drift = False
        else:
            drift = detect_drift(recent_errors, past_errors, stats)
            threshold = compute_threshold(stats, drift)
        
        logger.info({
            "entity_id": entity_id,
            "event": "entity_state_retrieved",
            "total_errors": stats["total_errors"],
            "cold_start": cold_start
        })
        
        return EntityStateResponse(
            entity_id=entity_id,
            total_errors=stats["total_errors"],
            recent_count=len(recent_errors),
            past_count=len(past_errors),
            mean_recent=stats["mean_recent"],
            std_recent=stats["std_recent"],
            mean_past=stats["mean_past"],
            threshold=threshold,
            drift=drift,
            cold_start=cold_start
        )
    
    except Exception as e:
        logger.error({
            "entity_id": entity_id,
            "event": "entity_state_error",
            "error": str(e)
        })
        entity_failures.labels(reason="unknown").inc()
        raise HTTPException(status_code=500, detail=f"Failed to get entity state: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    redis_healthy = False
    model_healthy = False
    threshold_healthy = False
    
    try:
        await redis_client.ping()
        redis_healthy = True
    except:
        pass
    
    try:
        response = await http_client.get(f"{settings.MODEL_SERVICE_URL}/health", timeout=2.0)
        model_healthy = response.status_code == 200
    except:
        pass
    
    try:
        response = await http_client.get(f"{settings.THRESHOLD_SERVICE_URL}/health", timeout=2.0)
        threshold_healthy = response.status_code == 200
    except:
        pass
    
    all_healthy = redis_healthy and model_healthy and threshold_healthy
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "service": "api",
        "dependencies": {
            "redis": "healthy" if redis_healthy else "unhealthy",
            "model": "healthy" if model_healthy else "unhealthy",
            "threshold": "healthy" if threshold_healthy else "unhealthy"
        }
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
