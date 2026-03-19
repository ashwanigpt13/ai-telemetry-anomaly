# Simulator Service

Synthetic telemetry data generator with configurable anomaly injection for testing and benchmarking.

## Architecture

- **Output**: MQTT messages to `telemetry/{entity_id}`
- **Features**: Generates multi-dimensional time series data
- **Anomalies**: Spike, drift, and noise injection

## Features

- ✅ **Multi-entity support** - Simulate multiple IoT devices
- ✅ **Configurable rate** - Total messages/sec across all entities
- ✅ **Round-robin publishing** - Fair distribution across entities
- ✅ **Anomaly injection** - Spike, drift, and noise
- ✅ **REST API control** - Start/stop, configure, manual injection
- ✅ **Structured logging** - JSON logs
- ✅ **Prometheus metrics** - Publishing rate, anomalies injected
- ✅ **Drift simulation** - Gradual mean shift with duration

## Endpoints

### `GET /health`
Service health check

### `GET /metrics`
Prometheus metrics endpoint

### `GET /config`
Get current simulator configuration

### `POST /config`
Update simulator configuration

**Request:**
```json
{
  "running": true,
  "rate": 10.0,
  "entities": ["engine_1", "engine_2"],
  "anomaly_probability": 0.05
}
```

### `POST /start`
Start the simulation

### `POST /stop`
Stop the simulation

### `GET /entities`
Get entity states and statistics

### `POST /inject-anomaly/{entity_id}`
Manually inject an anomaly

**Request:**
```json
{
  "anomaly_type": "spike"
}
```

Query parameter: `anomaly_type` (spike, drift, noise)

## Telemetry Message Format

Published to `telemetry/{entity_id}`:
```json
{
  "entity_id": "engine_1",
  "timestamp": 1710000000000,
  "features": [0.52, 0.11, 0.93]
}
```

- `timestamp`: Unix epoch in milliseconds (UTC)
- `features`: Array of floating-point values (default: 3 features)

## Anomaly Types

### 1. Spike Anomaly
Sudden large deviation in feature values.
```python
features = normal_features + random_sign * SPIKE_MAGNITUDE
```
- **Probability**: 40% of anomalies
- **Magnitude**: 2.0 (configurable)
- **Duration**: Single message

### 2. Noise Anomaly
Increased variance/noise in features.
```python
features = normal_features + randn() * NOISE_MAGNITUDE
```
- **Probability**: 30% of anomalies
- **Magnitude**: 0.5 (configurable)
- **Duration**: Single message

### 3. Drift Anomaly
Gradual shift in mean (concept drift).
```python
features = normal_features + drift_offset
```
- **Probability**: 30% of anomalies
- **Magnitude**: 0.3 (configurable)
- **Duration**: 300 seconds (configurable)
- **Effect**: Persistent shift until duration expires

## Configuration

See `.env.example` for all configuration options.

Key parameters:
- `DEFAULT_RATE=10.0` - Total messages per second
- `NUM_FEATURES=3` - Number of features per message
- `ANOMALY_PROBABILITY=0.05` - 5% of messages have anomalies
- `SPIKE_MAGNITUDE=2.0` - Spike intensity
- `NOISE_MAGNITUDE=0.5` - Noise intensity
- `DRIFT_MAGNITUDE=0.3` - Drift offset
- `DRIFT_DURATION=300.0` - Drift persistence (seconds)

## Running

### Local Development
```bash
pip install -r requirements.txt
python main.py
```

### Docker
```bash
docker build -t simulator-service .
docker run -p 8000:8000 --env-file .env simulator-service
```

## Usage Examples

### Start simulation with custom rate
```bash
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{
    "running": true,
    "rate": 100.0,
    "entities": ["engine_1", "engine_2", "turbine_1"],
    "anomaly_probability": 0.1
  }'
```

### Manually inject drift
```bash
curl -X POST "http://localhost:8000/inject-anomaly/engine_1?anomaly_type=drift"
```

### Stop simulation
```bash
curl -X POST http://localhost:8000/stop
```

### Get entity statistics
```bash
curl http://localhost:8000/entities
```

## Metrics

- `simulator_messages_published_total` - Total messages published per entity
- `simulator_messages_failed_total` - Failed messages per entity
- `simulator_anomalies_injected_total` - Anomalies injected per entity and type
- `simulator_publish_rate` - Current publish rate (msgs/sec)
- `simulator_active_entities` - Number of active entities

## Logging

Structured JSON logs include:
- `service` - Service name (simulator)
- `entity_id` - Entity identifier
- `event` - Event type
- `timestamp` - ISO timestamp

## Data Generation

### Normal Features
```python
features ~ N(base_mean, base_std)
base_mean ~ N(0, 0.2)
base_std = 0.1
```

### Entity State
Each entity maintains:
- `base_mean` - Normal distribution mean
- `base_std` - Normal distribution std
- `drift_offset` - Current drift offset
- `drift_active` - Whether drift is active
- `message_count` - Total messages published

## Testing Strategy

1. **Baseline**: Run without anomalies (`anomaly_probability=0`)
2. **Spike test**: High spike probability for rapid detection testing
3. **Drift test**: Force drift injection via API, test recovery
4. **Load test**: Increase rate to test system capacity
5. **Multi-entity**: Scale entities to test partitioning

## Performance

- Supports 1K+ msgs/sec on standard hardware
- Round-robin scheduling ensures fair entity distribution
- Async I/O for non-blocking MQTT publishing
- Configurable via REST API without restart
