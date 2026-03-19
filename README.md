# AI Telemetry Anomaly Detection System

## Quick Start with Docker Compose

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 1.29+
- 4GB RAM minimum
- 10GB disk space

### Build and Start All Services

```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps
```

### Services and Ports

| Service | Port | Description |
|---------|------|-------------|
| **MQTT Broker** | 1883 | Mosquitto MQTT broker |
| **Redis** | 6379 | State storage |
| **Simulator** | 8000 | Telemetry data generator |
| **Ingestion** | 8001 | Message processing |
| **Model** | 8002 | ML inference |
| **Threshold** | 8003 | Anomaly detection |
| **API Gateway** | 8080 | REST API |

### Service Startup Order

1. **mqtt** - MQTT broker
2. **redis** - State store
3. **simulator** - Starts publishing telemetry
4. **model** - ML inference service
5. **threshold** - Anomaly detection engine
6. **ingestion** - Telemetry processing
7. **api** - API gateway

### Testing the System

```bash
# Check API Gateway health
curl http://localhost:8080/health

# Start simulation
curl -X POST http://localhost:8000/start

# Make a prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "entity_id": "engine_1",
    "window": [[0.5, 0.1, 0.9], [0.4, 0.2, 0.8]]
  }'

# Get entity state
curl http://localhost:8080/entity/engine_1

# View simulator config
curl http://localhost:8000/config
```

### Monitoring

Access service metrics:
```bash
curl http://localhost:8000/metrics  # Simulator
curl http://localhost:8001/metrics  # Ingestion
curl http://localhost:8002/metrics  # Model
curl http://localhost:8003/metrics  # Threshold
curl http://localhost:8080/metrics  # API
```

### Scaling Services

Scale ingestion service for higher throughput:
```bash
docker-compose up -d --scale ingestion=3
```

### Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (data loss!)
docker-compose down -v
```

### Configuration

Environment variables can be customized in `docker-compose.yml`:

**Simulation:**
- `DEFAULT_RATE` - Messages per second (default: 10.0)
- `ANOMALY_PROBABILITY` - Anomaly injection rate (default: 0.05)

**Model:**
- `INPUT_DIM` - Number of features (default: 3)

**Threshold:**
- `K` - Threshold multiplier (default: 3)
- `DRIFT_THRESHOLD` - Drift detection sensitivity (default: 0.05)
- `ERROR_WINDOW` - Error history size (default: 100)

### Troubleshooting

**Services not starting:**
```bash
# Check logs
docker-compose logs [service-name]

# Restart service
docker-compose restart [service-name]
```

**MQTT connection issues:**
```bash
# Test MQTT broker
docker exec mqtt-broker mosquitto_sub -t 'telemetry/#' -v
```

**Redis connection issues:**
```bash
# Test Redis
docker exec redis redis-cli ping
```

**Model file missing:**
```bash
# Generate dummy model
cd services/model
python train_dummy_model.py
```

### Development

Run individual services locally:
```bash
# Start only dependencies
docker-compose up -d mqtt redis

# Run service locally
cd services/ingestion
pip install -r requirements.txt
python main.py
```

### Data Persistence

Volumes are created for:
- `mqtt-data` - MQTT persistence
- `mqtt-logs` - MQTT logs
- `redis-data` - Redis persistence

### Network

All services communicate via `anomaly-detection-network` bridge network.

### Health Checks

All services include health checks:
- MQTT: Subscribes to $SYS topics
- Redis: Ping command
- Services: HTTP health endpoints

Status check:
```bash
docker-compose ps
```

### Performance Tuning

For high throughput (1K+ msgs/sec):

1. Scale ingestion:
   ```bash
   docker-compose up -d --scale ingestion=5
   ```

2. Increase Redis memory:
   ```yaml
   redis:
     command: redis-server --appendonly yes --maxmemory 2gb
   ```

3. Adjust simulator rate:
   ```bash
   curl -X POST http://localhost:8000/config \
     -H "Content-Type: application/json" \
     -d '{"rate": 100.0, "running": true}'
   ```

### Production Considerations

For production deployment:

1. **Security:**
   - Enable MQTT authentication
   - Add API authentication
   - Use TLS/SSL
   - Restrict network access

2. **Monitoring:**
   - Add Prometheus + Grafana
   - Configure alerting
   - Log aggregation

3. **Reliability:**
   - Use external Redis cluster
   - Add load balancer
   - Configure backups
   - Set resource limits

4. **Scaling:**
   - Use Kubernetes (see deployment specs)
   - Configure auto-scaling
   - Optimize resource allocation
