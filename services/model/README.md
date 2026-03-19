# Model Service

PyTorch autoencoder inference service for anomaly detection via reconstruction error.

## Architecture

- **Input**: Feature window of shape (WINDOW_SIZE, num_features)
- **Processing**: Normalize → Autoencoder inference → MSE calculation
- **Output**: Reconstruction error (float)

## Features

- ✅ PyTorch autoencoder model
- ✅ Feature normalization using training statistics
- ✅ GPU support (CUDA if available)
- ✅ Reconstruction error computation (MSE)
- ✅ Structured JSON logging with trace IDs
- ✅ Prometheus metrics
- ✅ FastAPI async endpoint
- ✅ Input validation

## Endpoints

### `POST /infer`
Perform inference on a feature window.

**Request:**
```json
{
  "entity_id": "engine_1",
  "window": [[0.52, 0.11, 0.93], ...]
}
```

**Response:**
```json
{
  "reconstruction_error": 0.023
}
```

### `GET /health`
Service health check

### `GET /metrics`
Prometheus metrics endpoint

### `GET /info`
Model information (architecture, dimensions, normalization stats)

## Model File Format

The service expects a PyTorch checkpoint file (`model.pt`) with the following structure:

```python
{
    "model_state_dict": model.state_dict(),
    "input_dim": int,                    # Number of features
    "hidden_dims": [16, 8],              # Encoder architecture
    "feature_mean": np.ndarray,          # Shape: (input_dim,)
    "feature_std": np.ndarray,           # Shape: (input_dim,)
}
```

If the model file is not found, a dummy model is created for development/testing.

## Configuration

See `.env.example` for configuration options.

Key settings:
- `MODEL_PATH=model.pt` - Path to PyTorch model file
- `INPUT_DIM=3` - Number of input features
- `EPSILON=1e-6` - Small constant for numerical stability

## Running

### Local Development
```bash
pip install -r requirements.txt
python main.py
```

### Docker
```bash
docker build -t model-service .
docker run -p 8002:8002 -v $(pwd)/model.pt:/app/model.pt model-service
```

## Normalization

Features are normalized using training statistics:
```python
normalized = (features - mean) / (std + epsilon)
```

## Metrics

- `model_inference_requests_total` - Total inference requests
- `model_inference_failures_total` - Failed inference requests by reason
- `model_inference_seconds` - Inference latency histogram
- `model_reconstruction_errors` - Reconstruction error distribution

## Error Handling

The service handles:
- Invalid input shapes
- Feature dimension mismatches
- Model loading errors
- Inference failures

## Logging

Structured JSON logs include:
- `trace_id` - Request tracing
- `entity_id` - Entity identifier
- `event` - Event type
- `latency_ms` - Operation latency
- `error` - Reconstruction error value

## Performance

- GPU support via CUDA (if available)
- Batch processing per window
- No-gradient computation for inference
- Efficient numpy/torch operations
