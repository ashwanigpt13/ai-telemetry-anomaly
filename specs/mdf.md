# 📘 MACHINE DESIGN FILE (FINAL SPEC)

## Project: Scalable Drift-Aware Entity-Specific Anomaly Detection Framework (MQTT + AI + Streaming)

---

# 1. 🎯 OBJECTIVE

Design a real-time anomaly detection system for industrial IoT telemetry that:

* Supports **multi-entity streaming (1K–10K msgs/sec total)**
* Maintains **independent adaptive thresholds per entity**
* Handles **concept drift in real-time**
* Achieves **<100ms latency at p95 (publish → anomaly decision)**

---

# 2. 🧱 TECH STACK

* Language: Python 3.11
* API Framework: FastAPI (async)
* ML Framework: PyTorch
* MQTT Client: paho-mqtt (non-blocking)
* Data Processing: NumPy
* State Store: Redis (LIST + HASH)
* Containerization: Docker + docker-compose
* Orchestration: Kubernetes (Rancher Desktop)
* Logging: JSON structured logs (with trace_id)
* Metrics: Prometheus (/metrics per service)

---

# 3. 📊 DATA MODELS

## Telemetry Message

```json
{
  "entity_id": "engine_1",
  "timestamp": 1710000000000,
  "features": [0.52, 0.11, 0.93]
}
```

* timestamp: Unix epoch (milliseconds, UTC)
* features: fixed-order numeric array

---

## Model Input Window

* Shape: (WINDOW_SIZE, num_features)
* Window type: count-based
* Window timestamp = last element timestamp

---

## Model Output

```json
{
  "reconstruction_error": 0.023
}
```

---

## Anomaly Output (MQTT)

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

---

# 4. 🏗️ SYSTEM ARCHITECTURE

Simulator → MQTT → Ingestion → Model → Threshold → MQTT (anomalies)

* Ingestion calls Model Service
* Ingestion calls Threshold Engine
* Threshold Engine publishes anomaly

---

# 5. 📡 MQTT CONFIG

## Topics

* telemetry/{entity_id}
* anomalies/{entity_id}

## Subscription & Publish

* Ingestion subscribes using:
  `$share/group/` + MQTT_TOPIC_TELEMETRY

* Threshold publishes to:
  `MQTT_TOPIC_ANOMALY + entity_id`

---

# 6. 📥 INGESTION SERVICE

## Data Structure

```python
entity_buffers: Dict[str, deque[(timestamp, features)]]
```

## Rules

* Buffer until WINDOW_SIZE
* If no messages for an entity for ENTITY_TIMEOUT_SEC → delete buffer
* Partition by entity_id (via shared subscription)
* On full window:

  * call Model Service
  * call Threshold Engine

---

# 7. 🧠 MODEL SERVICE

POST /infer

* Input: window
* Output: reconstruction_error

## Notes

* LSTM Autoencoder
* Normalization uses training mean/std
* Model loaded from `model.pt`

---

# 8. 🧩 THRESHOLD ENGINE

POST /update

## Redis Schema

* entity:{id}:recent_errors → LIST
* entity:{id}:past_errors → LIST
* entity:{id}:stats → HASH

---

## Window Sizes

* N_RECENT = ERROR_WINDOW / 2
* N_PAST = ERROR_WINDOW / 2

---

## Redis Atomic Update (Lua)

```
LPUSH entity:{id}:recent_errors error

IF LLEN recent_errors > N_RECENT:
    RPOPLPUSH recent_errors → past_errors

LTRIM recent_errors to N_RECENT
LTRIM past_errors to N_PAST
```

(All above operations must run inside a single Lua script for atomicity)

---

## Statistics

```python
mean_recent = mean(recent_errors)
std_recent = max(std(recent_errors), EPSILON)

mean_past = mean(past_errors)

total_errors = len(recent_errors) + len(past_errors)
```

(Values fetched from Redis LISTs and converted to arrays)

---

## Threshold

```python
threshold = mean_recent + K * std_recent
```

---

## Drift Detection

```python
if len(recent_errors) >= N_RECENT and len(past_errors) >= N_PAST:
    drift = abs(mean_recent - mean_past) > DRIFT_THRESHOLD
else:
    drift = False
```

---

## Drift Adjustment

```python
if drift:
    threshold += ALPHA * std_recent
```

---

## Decision

```python
anomaly = error > threshold
```

---

## Cold Start

```python
if total_errors < ERROR_WINDOW:
    threshold = GLOBAL_MEAN + K * GLOBAL_STD
    drift = False
```

---

## State Update

After computation, update Redis HASH:

* mean = mean_recent
* std = std_recent
* threshold = threshold
* drift_flag = drift

---

## Notes

* GLOBAL stats loaded from `global_stats.json`
* DRIFT_THRESHOLD and ALPHA are dataset-dependent

---

# 9. 🌐 API GATEWAY

## POST /predict

```json
{
  "entity_id": "engine_1",
  "window": [[...]]
}
```

### Response

```json
{
  "anomaly": true,
  "error": 0.023,
  "threshold": 0.018,
  "drift": true
}
```

* Calls Model → Threshold

---

## GET /entity/{id}

* Reads state from Redis

---

# 10. ⚙️ CONFIGURATION

* WINDOW_SIZE=50

* ERROR_WINDOW=100

* K=3

* ALPHA=0.5

* DRIFT_THRESHOLD=0.05

* EPSILON=1e-6

* ENTITY_TIMEOUT_SEC=300

* MQTT_TOPIC_TELEMETRY=telemetry/+

* MQTT_TOPIC_ANOMALY=anomalies/

* REDIS_URL=redis://localhost:6379

---

# 11. 🧠 TRAINING PIPELINE

## Dataset

* NASA Turbofan

## Data Usage

* Training uses only healthy data
* No anomalies in training
* Test = held-out engines + synthetic anomalies

## Features

* Fixed feature list defined in `features.yaml`

## Training Steps

* Normalize data
* Create sliding windows
* Train LSTM autoencoder
* Compute reconstruction errors

## Outputs

* model.pt
* norm_stats.json
* global_stats.json

## Global Stats

* Computed from reconstruction_error on training windows

## Synthetic Anomalies

* Spike
* Drift
* Noise

(Used only during evaluation)

## Reproducibility

* Fixed random seeds

---

# 12. 🧪 EXPERIMENT DESIGN

## Baselines

1. Global static
   threshold = GLOBAL_MEAN + K * GLOBAL_STD

2. Global dynamic
   threshold = mean(global_recent_errors) + K * std(global_recent_errors)
   window size = ERROR_WINDOW across all entities

3. Entity dynamic
   threshold = mean_recent + K * std_recent

4. Proposed
   entity + drift-aware

---

## Metrics

* Precision / Recall / F1
* False positives per entity
* Drift recovery time
* Latency (p95)

---

# 13. 🚀 DEPLOYMENT

## Docker Compose (Dev)

* mqtt
* redis
* simulator
* ingestion
* model
* threshold
* api

## Kubernetes (Scaling)

* Same services as pods
* Horizontal scaling enabled

---

# 14. 📊 LOGGING

```json
{
  "timestamp": "...",
  "level": "INFO",
  "trace_id": "...",
  "service": "threshold",
  "entity_id": "engine_1",
  "event": "anomaly_detected",
  "error": 0.023,
  "threshold": 0.018,
  "latency_ms": 12
}
```

---

# 15. 📈 METRICS

* Each service exposes `/metrics`

---

# 16. 🧭 IMPLEMENTATION PLAN

1. ingestion
2. model
3. threshold
4. redis integration
5. api
6. benchmarking

---

# 17. 🔮 FUTURE

* advanced drift detection
* transformer models
* kubernetes optimization

---

# 18. 🛠️ DEVELOPMENT STRATEGY

Phase 1: docker-compose (build + debug)
Phase 2: Kubernetes (scale + benchmark)

---
