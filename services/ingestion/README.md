# Ingestion Service

Real-time telemetry ingestion service that buffers incoming messages and orchestrates anomaly detection pipeline.

## Architecture

- **Input**: MQTT messages from `telemetry/{entity_id}` topic
- **Processing**: Buffers messages per entity until window is complete
- **Output**: Calls Model Service → Threshold Engine

## Features

- ✅ Shared MQTT subscription for load balancing
- ✅ Per-entity buffering with deque
- ✅ Async HTTP calls to downstream services
- ✅ Automatic stale buffer cleanup (300s timeout)
- ✅ Structured JSON logging with trace IDs
- ✅ Prometheus metrics
- ✅ Health check endpoint

## Endpoints

- `GET /health` - Service health status
- `GET /metrics` - Prometheus metrics
- `GET /stats` - Buffer statistics

## Configuration

See `.env.example` for all configuration options.

Key settings:
- `WINDOW_SIZE=50` - Number of samples before processing
- `ENTITY_TIMEOUT_SEC=300` - Buffer cleanup timeout
- `MQTT_SHARE_GROUP=ingestion-group` - Load balancing group name

## Running

### Local Development
```bash
pip install -r requirements.txt
python main.py
```

### Docker
```bash
docker build -t ingestion-service .
docker run -p 8001:8001 --env-file .env ingestion-service
```

## Metrics

- `ingestion_messages_received_total` - Total messages received per entity
- `ingestion_messages_processed_total` - Total windows processed
- `ingestion_messages_failed_total` - Failed messages by reason
- `ingestion_window_processing_seconds` - Window processing latency
- `ingestion_active_buffers` - Current number of entity buffers
- `ingestion_model_request_seconds` - Model service request latency
- `ingestion_threshold_request_seconds` - Threshold service request latency

## Logging

Structured JSON logs include:
- `trace_id` - Request tracing
- `entity_id` - Entity identifier
- `event` - Event type
- `latency_ms` - Operation latency
- `service` - Service name (ingestion)

## Error Handling

The service handles:
- Invalid JSON messages
- Missing required fields
- HTTP timeouts (model/threshold services)
- HTTP errors from downstream services
- MQTT disconnections (auto-reconnect)
- Stale buffer cleanup
