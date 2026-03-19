# Threshold Engine Service

Adaptive threshold computation service with drift detection and Redis-based state management.

## Architecture

- **Input**: Reconstruction errors via `POST /update`
- **Processing**: Adaptive threshold computation with drift detection
- **Output**: Anomaly result → MQTT (`anomalies/{entity_id}`)
- **State**: Redis (entity-specific error history)

## Features

- ✅ **Adaptive per-entity thresholds** - `threshold = mean + K*std`
- ✅ **Drift detection** - Compares recent vs past error distributions
- ✅ **Drift adjustment** - Increases threshold during drift
- ✅ **Atomic Redis updates** - Lua script for consistency
- ✅ **Cold start handling** - Uses global statistics
- ✅ **MQTT publishing** - Sends anomaly results
- ✅ **Structured JSON logging** - With trace IDs
- ✅ **Prometheus metrics** - Monitoring and alerting

## Endpoints

### `POST /update`
Update threshold and detect anomalies.

**Request:**
```json
{
  "entity_id": "engine_1",
  "timestamp": 1710000000000,
  "error": 0.023
}
```

**Response:**
```json
{
  "anomaly": true,
  "error": 0.023,
  "threshold": 0.018,
  "drift": true
}
```

### `GET /entity/{id}`
Get current statistics for an entity.

**Response:**
```json
{
  "entity_id": "engine_1",
  "total_errors": 150,
  "recent_count": 50,
  "past_count": 50,
  "mean_recent": 0.025,
  "std_recent": 0.008,
  "mean_past": 0.015,
  "threshold": 0.042,
  "drift": true,
  "cold_start": false
}
```

### `GET /health`
Service health check

### `GET /metrics`
Prometheus metrics endpoint

### `GET /stats`
Service statistics (active entities)

## Redis Schema

Per entity:
- `entity:{id}:recent_errors` → LIST (N_RECENT errors)
- `entity:{id}:past_errors` → LIST (N_PAST errors)

## Algorithm

### 1. Update Phase (Atomic via Lua)
```lua
LPUSH entity:{id}:recent_errors error
IF LLEN(recent_errors) > N_RECENT:
    RPOPLPUSH recent_errors → past_errors
LTRIM recent_errors to N_RECENT
LTRIM past_errors to N_PAST
```

### 2. Statistics Computation
```python
mean_recent = mean(recent_errors)
std_recent = max(std(recent_errors), EPSILON)
mean_past = mean(past_errors)
```

### 3. Drift Detection
```python
if len(recent) >= N_RECENT and len(past) >= N_PAST:
    drift = |mean_recent - mean_past| > DRIFT_THRESHOLD
else:
    drift = False
```

### 4. Threshold Computation
```python
threshold = mean_recent + K * std_recent
if drift:
    threshold += ALPHA * std_recent
```

### 5. Anomaly Decision
```python
anomaly = error > threshold
```

### 6. Cold Start
```python
if total_errors < ERROR_WINDOW:
    threshold = GLOBAL_MEAN + K * GLOBAL_STD
    drift = False
```

## Configuration

See `.env.example` for all configuration options.

Key parameters:
- `ERROR_WINDOW=100` - Total error history (N_RECENT + N_PAST)
- `K=3` - Threshold multiplier (standard deviations)
- `ALPHA=0.5` - Drift adjustment factor
- `DRIFT_THRESHOLD=0.05` - Mean difference threshold for drift
- `EPSILON=1e-6` - Numerical stability constant

## Running

### Local Development
```bash
pip install -r requirements.txt
python main.py
```

### Docker
```bash
docker build -t threshold-service .
docker run -p 8003:8003 --env-file .env threshold-service
```

## Metrics

- `threshold_update_requests_total` - Total update requests
- `threshold_update_failures_total` - Failed updates by reason
- `threshold_update_seconds` - Update latency histogram
- `threshold_anomalies_detected_total` - Anomalies detected per entity
- `threshold_drift_detected_total` - Drift events per entity
- `threshold_cold_start_total` - Cold start detections per entity
- `threshold_values` - Threshold value distribution
- `threshold_active_entities` - Number of entities with data

## Logging

Structured JSON logs include:
- `trace_id` - Request tracing
- `entity_id` - Entity identifier
- `event` - Event type
- `anomaly` - Detection result
- `threshold` - Computed threshold
- `drift` - Drift status
- `latency_ms` - Operation latency

## MQTT Output

Published to `anomalies/{entity_id}`:
```json
{
  "entity_id": "engine_1",
  "timestamp": 1710000000000,
  "anomaly": true,
  "error": 0.023,
  "threshold": 0.018,
  "drift": true
}
```

## Error Handling

The service handles:
- Redis connection failures
- MQTT publish errors
- Invalid input data
- Statistical edge cases (empty lists, etc.)
