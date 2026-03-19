# API Gateway Service

Unified REST API interface for the anomaly detection system. Orchestrates calls to model and threshold services.

## Architecture

- **Role**: Frontend API for external clients
- **Dependencies**: Model Service, Threshold Engine, Redis
- **Endpoints**: Predict, Entity State

## Features

- ✅ **POST /predict** - End-to-end anomaly prediction
- ✅ **GET /entity/{id}** - Direct Redis state queries
- ✅ **Service orchestration** - Model → Threshold pipeline
- ✅ **Error handling** - Timeouts and retries
- ✅ **Health checks** - Dependency monitoring
- ✅ **Structured logging** - With trace IDs
- ✅ **Prometheus metrics** - Request monitoring

## Endpoints

### `POST /predict`
Perform anomaly prediction on a feature window.

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
  "anomaly": true,
  "error": 0.023,
  "threshold": 0.018,
  "drift": true
}
```

**Flow:**
1. Call Model Service (`POST /infer`) → Get reconstruction error
2. Call Threshold Engine (`POST /update`) → Get anomaly decision
3. Return combined result

**Status Codes:**
- `200` - Success
- `400` - Invalid request
- `502` - Downstream service error
- `504` - Service timeout

### `GET /entity/{entity_id}`
Get current state and statistics for an entity.

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

**Data Source:** Reads directly from Redis
- `entity:{id}:recent_errors`
- `entity:{id}:past_errors`

**Status Codes:**
- `200` - Success
- `500` - Redis connection error

### `GET /health`
Service health check with dependency status.

**Response:**
```json
{
  "status": "healthy",
  "service": "api",
  "dependencies": {
    "redis": "healthy",
    "model": "healthy",
    "threshold": "healthy"
  }
}
```

### `GET /metrics`
Prometheus metrics endpoint

## Configuration

See `.env.example` for all configuration options.

Key settings:
- `MODEL_SERVICE_URL` - Model service endpoint
- `THRESHOLD_SERVICE_URL` - Threshold service endpoint
- `REDIS_URL` - Redis connection string
- `HTTP_TIMEOUT=10.0` - Request timeout (seconds)

## Running

### Local Development
```bash
pip install -r requirements.txt
python main.py
```

### Docker
```bash
docker build -t api-gateway .
docker run -p 8080:8080 --env-file .env api-gateway
```

## Usage Examples

### Predict anomaly
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "engine_1",
    "window": [[0.5, 0.1, 0.9], [0.4, 0.2, 0.8], ...]
  }'
```

### Get entity state
```bash
curl http://localhost:8080/entity/engine_1
```

### Health check
```bash
curl http://localhost:8080/health
```

## Metrics

- `api_predict_requests_total` - Total prediction requests
- `api_predict_failures_total` - Failed predictions by reason
- `api_predict_seconds` - Prediction latency histogram
- `api_entity_requests_total` - Total entity state requests
- `api_entity_failures_total` - Failed entity requests by reason

## Logging

Structured JSON logs include:
- `trace_id` - Request tracing across services
- `entity_id` - Entity identifier
- `event` - Event type
- `latency_ms` - Operation latency
- `service` - Service name (api)

## Error Handling

The service handles:
- **Model service errors** - HTTP errors, timeouts
- **Threshold service errors** - HTTP errors, timeouts
- **Redis errors** - Connection failures
- **Invalid input** - Validation errors
- **Timeout management** - Configurable per request

## Data Flow

### Predict Endpoint
```
Client → API Gateway → Model Service (reconstruction error)
                    → Threshold Engine (anomaly decision)
                    → Client (result)
```

### Entity State Endpoint
```
Client → API Gateway → Redis (read error history)
                    → Client (computed state)
```

## Dependencies

- **Model Service** - Required for predictions
- **Threshold Engine** - Required for predictions
- **Redis** - Required for entity state queries

## Integration

This service acts as the system's public API:
- External applications call `/predict`
- Monitoring dashboards query `/entity/{id}`
- Load balancers check `/health`

## Performance

- Async I/O for concurrent requests
- HTTP connection pooling via httpx
- Direct Redis reads for entity state
- Configurable timeouts

## Security Considerations

For production:
- Add authentication (API keys, OAuth)
- Enable HTTPS/TLS
- Rate limiting
- Input validation
- CORS configuration
