"""
Model Service - PyTorch autoencoder inference service
"""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import torch
import torch.nn as nn
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
            "service": "model",
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
inference_requests = Counter("model_inference_requests_total", "Total inference requests")
inference_failures = Counter("model_inference_failures_total", "Total inference failures", ["reason"])
inference_latency = Histogram("model_inference_seconds", "Inference latency")
reconstruction_errors = Histogram("model_reconstruction_errors", "Reconstruction error distribution", 
                                  buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])


# Request/Response models
class InferenceRequest(BaseModel):
    """Model inference request"""
    entity_id: str = Field(..., description="Entity identifier")
    window: List[List[float]] = Field(..., description="Feature window (WINDOW_SIZE x num_features)")


class InferenceResponse(BaseModel):
    """Model inference response"""
    reconstruction_error: float = Field(..., description="Mean squared reconstruction error")


class TemporalAttention(nn.Module):
    """Differentiable temporal attention over LSTM outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs: torch.Tensor):
        scores = self.attention(lstm_outputs).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)
        return weights, context


# Autoencoder model definition
class LSTMEncoder(nn.Module):
    """LSTM-based encoder for sequence data"""
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = TemporalAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        _, context = self.attention(lstm_out)
        latent = self.fc(context)
        return latent, (h_n, c_n)

    def get_attention_weights(self, x):
        """Return attention weights for the provided input sequence."""
        lstm_out, _ = self.lstm(x)
        attention_weights, _ = self.attention(lstm_out)
        return attention_weights


class LSTMDecoder(nn.Module):
    """LSTM-based decoder for sequence reconstruction"""
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, latent):
        # latent: (batch, latent_dim)
        hidden = self.fc(latent)
        # Repeat for sequence length
        hidden_seq = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        # Decode through LSTM
        lstm_out, _ = self.lstm(hidden_seq)
        # Project to output dimension
        output = self.output_layer(lstm_out)
        return output


class Autoencoder(nn.Module):
    """LSTM Autoencoder for anomaly detection"""
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 32, 
                 seq_len: int = 50, num_layers: int = 1, dropout: float = 0.0, hidden_dims: List[int] = None):
        super().__init__()
        # hidden_dims parameter is kept for backwards compatibility but ignored
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        latent, _ = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


# Global model and normalization stats
model: nn.Module = None
feature_mean: np.ndarray = None
feature_std: np.ndarray = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load PyTorch model and normalization statistics"""
    global model, feature_mean, feature_std
    
    logger.info({"event": "loading_model", "path": settings.MODEL_PATH})
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(settings.MODEL_PATH, map_location=device, weights_only=False)
        
        # Extract model architecture parameters for LSTM Autoencoder
        # Use values from checkpoint if available, otherwise use settings
        input_dim = checkpoint.get("input_dim", settings.INPUT_DIM)
        hidden_dim = checkpoint.get("hidden_dim", settings.HIDDEN_DIM)
        latent_dim = checkpoint.get("latent_dim", settings.LATENT_DIM)
        seq_len = checkpoint.get("window_size", settings.WINDOW_SIZE)
        num_layers = checkpoint.get("num_layers", settings.NUM_LAYERS)
        dropout = checkpoint.get("dropout", getattr(settings, "DROPOUT", 0.0))
        
        # Initialize LSTM Autoencoder model
        model = Autoencoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            latent_dim=latent_dim,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout
        )
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing or unexpected:
            logger.info({
                "event": "checkpoint_state_dict_compatibility",
                "missing": missing,
                "unexpected": unexpected,
            })
        model.to(device)
        model.eval()
        
        logger.info({
            "event": "model_loaded",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "num_layers": num_layers,
            "dropout": dropout
        })
        
        # Try to load normalization statistics from norm_stats.json (MDF spec)
        try:
            import os
            norm_stats_path = settings.NORM_STATS_PATH
            if os.path.exists(norm_stats_path):
                with open(norm_stats_path, 'r') as f:
                    norm_stats = json.load(f)
                
                feature_mean = np.array(norm_stats["feature_mean"], dtype=np.float32)
                feature_std = np.array(norm_stats["feature_std"], dtype=np.float32)
                
                logger.info({
                    "event": "norm_stats_loaded",
                    "source": "norm_stats.json",
                    "mean_shape": feature_mean.shape,
                    "std_shape": feature_std.shape
                })
            else:
                # Fallback: Load from model checkpoint (backward compatibility)
                feature_mean = checkpoint.get("feature_mean", np.zeros(input_dim))
                feature_std = checkpoint.get("feature_std", np.ones(input_dim))
                
                # Ensure numpy arrays
                if isinstance(feature_mean, torch.Tensor):
                    feature_mean = feature_mean.cpu().numpy()
                if isinstance(feature_std, torch.Tensor):
                    feature_std = feature_std.cpu().numpy()
                
                logger.info({
                    "event": "norm_stats_loaded",
                    "source": "model_checkpoint",
                    "message": "norm_stats.json not found, using embedded stats"
                })
        except Exception as e:
            logger.warning({
                "event": "norm_stats_load_warning",
                "error": str(e),
                "message": "Using embedded stats from model checkpoint"
            })
            # Fallback to checkpoint
            feature_mean = checkpoint.get("feature_mean", np.zeros(input_dim))
            feature_std = checkpoint.get("feature_std", np.ones(input_dim))
            
            # Ensure numpy arrays
            if isinstance(feature_mean, torch.Tensor):
                feature_mean = feature_mean.cpu().numpy()
            if isinstance(feature_std, torch.Tensor):
                feature_std = feature_std.cpu().numpy()
        
        logger.info({
            "event": "model_loaded",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "seq_len": seq_len,
            "device": str(device)
        })
        
    except FileNotFoundError:
        logger.warning({
            "event": "model_file_not_found",
            "path": settings.MODEL_PATH,
            "message": "Using dummy model for development"
        })
        # Create a dummy model for development/testing
        model = Autoencoder(input_dim=settings.INPUT_DIM, hidden_dims=settings.HIDDEN_DIMS)
        model.to(device)
        model.eval()
        feature_mean = np.zeros(settings.INPUT_DIM)
        feature_std = np.ones(settings.INPUT_DIM)
        logger.info({"event": "dummy_model_created", "input_dim": settings.INPUT_DIM})
    
    except Exception as e:
        logger.error({"event": "model_load_error", "error": str(e)})
        raise


def normalize_features(window: np.ndarray) -> np.ndarray:
    """Normalize features using training statistics"""
    return (window - feature_mean) / (feature_std + settings.EPSILON)


def compute_reconstruction_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute mean squared error between original and reconstructed"""
    mse = torch.mean((original - reconstructed) ** 2).item()
    return mse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info({"event": "service_starting", "service": "model"})
    load_model()
    logger.info({"event": "service_started", "service": "model"})
    
    yield
    
    # Shutdown
    logger.info({"event": "service_stopping", "service": "model"})
    logger.info({"event": "service_stopped", "service": "model"})


# FastAPI app
app = FastAPI(title="Model Service", lifespan=lifespan)


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Perform inference on a window of features
    
    - **entity_id**: Entity identifier
    - **window**: Feature window of shape (WINDOW_SIZE, num_features)
    
    Returns reconstruction error (MSE)
    """
    trace_id = str(uuid.uuid4())
    start_time = time.time()
    
    inference_requests.inc()
    
    try:
        # Validate input shape
        window_array = np.array(request.window, dtype=np.float32)
        
        if window_array.ndim != 2:
            logger.error({
                "trace_id": trace_id,
                "entity_id": request.entity_id,
                "event": "invalid_shape",
                "shape": window_array.shape
            })
            inference_failures.labels(reason="invalid_shape").inc()
            raise HTTPException(status_code=400, detail="Window must be 2D array")
        
        window_size, num_features = window_array.shape
        
        if num_features != len(feature_mean):
            logger.error({
                "trace_id": trace_id,
                "entity_id": request.entity_id,
                "event": "feature_mismatch",
                "expected": len(feature_mean),
                "got": num_features
            })
            inference_failures.labels(reason="feature_mismatch").inc()
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(feature_mean)} features, got {num_features}"
            )
        
        logger.debug({
            "trace_id": trace_id,
            "entity_id": request.entity_id,
            "event": "inference_started",
            "window_shape": window_array.shape
        })
        
        # Normalize features
        normalized_window = normalize_features(window_array)
        
        # Convert to torch tensor and add batch dimension for LSTM
        # Input shape: (seq_len, num_features) -> (1, seq_len, num_features)
        input_tensor = torch.tensor(normalized_window, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Perform inference
        with torch.no_grad():
            reconstructed = model(input_tensor)
        
        # Compute reconstruction error (remove batch dimension for MSE calculation)
        reconstruction_error = compute_reconstruction_error(input_tensor.squeeze(0), reconstructed.squeeze(0))
        
        # Record metrics
        reconstruction_errors.observe(reconstruction_error)
        latency = time.time() - start_time
        inference_latency.observe(latency)
        
        logger.info({
            "trace_id": trace_id,
            "entity_id": request.entity_id,
            "event": "inference_complete",
            "error": reconstruction_error,
            "latency_ms": latency * 1000
        })
        
        return InferenceResponse(reconstruction_error=reconstruction_error)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error({
            "trace_id": trace_id,
            "entity_id": request.entity_id,
            "event": "inference_error",
            "error": str(e)
        })
        inference_failures.labels(reason="unknown").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "model",
        "device": str(device),
        "model_loaded": model is not None
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/info")
async def info():
    """Model information endpoint"""
    return {
        "input_dim": len(feature_mean) if feature_mean is not None else None,
        "device": str(device),
        "model_type": model.__class__.__name__ if model else None,
        "feature_mean": feature_mean.tolist() if feature_mean is not None else None,
        "feature_std": feature_std.tolist() if feature_std is not None else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
