# AI Telemetry Anomaly Detection System

## Complete Technical Documentation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Service Components](#3-service-components)
4. [Data Flow Pipeline](#4-data-flow-pipeline)
5. [Machine Learning Model](#5-machine-learning-model)
6. [Threshold Engine & Algorithms](#6-threshold-engine--algorithms)
7. [Anomaly Detection Modes](#7-anomaly-detection-modes)
8. [Configuration Reference](#8-configuration-reference)
9. [Technology Stack](#9-technology-stack)
10. [Deployment Guide](#10-deployment-guide)
11. [Evaluation & Metrics](#11-evaluation--metrics)
12. [API Reference](#12-api-reference)

---

## 1. Executive Summary

### Project Overview

This project implements a **Scalable Drift-Aware Entity-Specific Anomaly Detection Framework** for Industrial IoT telemetry data. The system is designed to detect anomalies in real-time streaming data from multiple entities (e.g., industrial engines, turbines) while adapting to concept drift.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Real-time Processing** | <100ms latency at p95 (publish → anomaly decision) |
| **Multi-entity Support** | Independent adaptive thresholds per entity |
| **High Throughput** | 1K–10K messages/sec total capacity |
| **Concept Drift Handling** | Automatic drift detection and threshold adjustment |
| **Scalable Architecture** | Microservices with horizontal scaling support |

### Problem Statement

Industrial IoT systems generate continuous telemetry data that requires real-time monitoring for anomalies. Traditional static threshold approaches fail because:
- Sensor characteristics vary across entities
- Operating conditions cause concept drift over time
- Global thresholds miss entity-specific anomalies

### Solution Approach

This system implements an **LSTM Autoencoder** for anomaly detection combined with **adaptive entity-specific thresholds** that automatically adjust to concept drift. The key innovation is the drift-aware threshold computation that:

1. Maintains separate error histories for each entity
2. Detects concept drift by comparing recent vs. past error distributions
3. Dynamically adjusts thresholds when drift is detected

---

## 2. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  Simulator   │────▶│    MQTT      │────▶│  Ingestion   │────▶│    Model     │
  │   Service    │     │   Broker     │     │   Service    │     │   Service    │
  │  (Port 8000) │     │ (Port 1883)  │     │ (Port 8001)  │     │ (Port 8002)  │
  └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
         │                    │                    │                    │
         │                    │                    │                    │
         │                    ▼                    ▼                    │
         │             ┌──────────────┐     ┌──────────────┐           │
         │             │   Redis      │◀────│  Threshold   │◀──────────┘
         │             │(State Store) │     │   Engine     │
         │             │ (Port 6379)  │     │ (Port 8003)  │
         │             └──────────────┘     └──────────────┘
         │                                        │
         │                                        │ Anomaly Results
         │                                        ▼
         │                                 ┌──────────────┐
         └────────────────────────────────▶│ API Gateway  │
                                           │ (Port 8080)  │
                                           └──────────────┘
```

### Communication Patterns

| Pattern | Usage |
|---------|-------|
| **MQTT Pub/Sub** | Telemetry streaming, anomaly notifications |
| **HTTP REST** | Model inference, threshold computation, external API |
| **Redis Lists** | Entity error history storage |
| **Shared Subscriptions** | Load balancing across ingestion instances |

### Message Flow

```
1. Simulator → MQTT [telemetry/{entity_id}]
2. Ingestion ← MQTT (shared subscription)
3. Ingestion → Model Service (HTTP POST /infer)
4. Ingestion → Threshold Engine (HTTP POST /update)
5. Threshold Engine → MQTT [anomalies/{entity_id}]
6. Threshold Engine → Redis (state update)
```

---

## 3. Service Components

### 3.1 Simulator Service (Port 8000)

**Purpose:** Generate synthetic telemetry data with anomaly injection for testing and evaluation.

**Features:**
- NASA Turbofan-like sensor data simulation (16 features)
- Multiple anomaly types: spike, drift, noise
- Configurable anomaly probability
- Multi-entity support

**Simulated Sensor Features:**

| Index | Sensor | Description | Baseline Value |
|-------|--------|-------------|----------------|
| 0-1 | Settings | Operational settings | ~0 |
| 2 | T24 | Temperature (°C) | 518.67 |
| 3 | T30 | Temperature (°C) | 643.03 |
| 4 | T50 | Temperature (°C) | 1591.76 |
| 5 | Ps30 | Pressure | 8.42 |
| 6 | phi | Fuel flow ratio | 21.89 |
| 7 | NRf | Speed | 555.18 |
| ... | ... | ... | ... |

**Message Format:**
```json
{
  "entity_id": "engine_1",
  "timestamp": 1710000000000,
  "features": [0.52, 0.11, 518.5, 643.1, ...],
  "is_anomaly": false,
  "anomaly_type": "normal"
}
```

---

### 3.2 Ingestion Service (Port 8001)

**Purpose:** Orchestrate telemetry processing, windowing, and service coordination.

**Key Responsibilities:**
1. Subscribe to MQTT telemetry topics
2. Maintain sliding window buffers per entity
3. Call Model Service for inference
4. Call Threshold Engine for anomaly detection
5. Log evaluation data

**Windowing Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WINDOW_SIZE` | 50 | Number of samples per window |
| `WINDOW_STRIDE` | 25 | Samples between windows (50% overlap) |
| `ENTITY_TIMEOUT_SEC` | 300 | Entity buffer timeout |

**Data Structure:**
```python
entity_buffers: Dict[str, deque[(timestamp, features)]]  # Sliding windows
entity_last_seen: Dict[str, float]                       # Activity tracking
```

---

### 3.3 Model Service (Port 8002)

**Purpose:** Run LSTM Autoencoder inference and compute reconstruction error.

**Architecture:**
- **Encoder:** LSTM layers → Latent space compression
- **Decoder:** Latent space → LSTM layers → Reconstruction

**Model Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `INPUT_DIM` | 16 | Number of input features |
| `HIDDEN_DIM` | 128 | LSTM hidden dimension |
| `LATENT_DIM` | 64 | Latent space dimension |
| `NUM_LAYERS` | 2 | Number of LSTM layers |
| `DROPOUT` | 0.2 | Dropout rate |

**Inference Pipeline:**
```
Input Window → Normalize → LSTM Encode → Latent → LSTM Decode → Reconstruct → MSE Loss
     (50×16)                                                           → Error
```

---

### 3.4 Threshold Engine (Port 8003)

**Purpose:** Compute adaptive thresholds with drift detection and anomaly classification.

**Core Algorithm:**

```python
# 1. Update error lists (Lua script for atomicity)
LPUSH entity:{id}:recent_errors error
IF LLEN recent_errors > N_RECENT:
    RPOPLPUSH recent_errors → past_errors

# 2. Compute statistics
mean_recent = mean(recent_errors)
std_recent = max(std(recent_errors), EPSILON)
mean_past = mean(past_errors)

# 3. Detect drift
drift = abs(mean_recent - mean_past) > DRIFT_THRESHOLD

# 4. Compute threshold
threshold = mean_recent + K * std_recent
if drift:
    threshold += ALPHA * std_recent

# 5. Classify anomaly
anomaly = error > threshold
```

**Advanced Features:**

| Feature | Parameter | Description |
|---------|-----------|-------------|
| **Persistence Filter** | `ANOMALY_PERSISTENCE` | Require N consecutive anomalies |
| **Error Smoothing** | `ERROR_SMOOTH_WINDOW` | Moving average over recent errors |

---

### 3.5 API Gateway (Port 8080)

**Purpose:** External REST API for predictions, entity state queries, and management.

**Endpoints:**
- `GET /health` - Service health check
- `POST /predict` - Direct prediction request
- `GET /entity/{id}` - Get entity state
- `GET /metrics` - Prometheus metrics

---

## 4. Data Flow Pipeline

### End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

 STEP 1: Data Generation
 ┌─────────────────────────────────────────────────────────┐
 │  Simulator generates telemetry with optional anomalies  │
 │  • NASA Turbofan baseline + noise                       │
 │  • Anomaly injection (spike/drift/noise) at 5% rate    │
 └──────────────────────────┬──────────────────────────────┘
                            ▼
 STEP 2: Message Publishing (MQTT)
 ┌─────────────────────────────────────────────────────────┐
 │  telemetry/{entity_id}                                  │
 │  • QoS 1 (at least once delivery)                      │
 │  • Shared subscription for load balancing              │
 └──────────────────────────┬──────────────────────────────┘
                            ▼
 STEP 3: Windowing & Buffering (Ingestion)
 ┌─────────────────────────────────────────────────────────┐
 │  Sliding Window Buffer (per entity)                     │
 │  • WINDOW_SIZE = 50 samples                            │
 │  • WINDOW_STRIDE = 25 (50% overlap)                    │
 │  • Timeout cleanup for inactive entities               │
 └──────────────────────────┬──────────────────────────────┘
                            ▼
 STEP 4: Feature Normalization & Inference (Model)
 ┌─────────────────────────────────────────────────────────┐
 │  LSTM Autoencoder                                       │
 │  • Z-score normalization: (x - mean) / std             │
 │  • Encode → Latent (64-dim) → Decode                   │
 │  • Output: reconstruction_error (MSE)                  │
 └──────────────────────────┬──────────────────────────────┘
                            ▼
 STEP 5: Threshold & Anomaly Classification (Threshold)
 ┌─────────────────────────────────────────────────────────┐
 │  Adaptive Threshold Engine                              │
 │  • Entity-specific error history in Redis              │
 │  • Drift detection: |mean_recent - mean_past| > τ      │
 │  • Threshold: μ + K×σ (+ α×σ if drift)                 │
 │  • Decision: anomaly = (error > threshold)             │
 └──────────────────────────┬──────────────────────────────┘
                            ▼
 STEP 6: Output & Notification
 ┌─────────────────────────────────────────────────────────┐
 │  Anomaly message published to anomalies/{entity_id}     │
 │  {anomaly: true, error: 0.023, threshold: 0.018, ...}  │
 └─────────────────────────────────────────────────────────┘
```

### Latency Budget

| Stage | Target | Description |
|-------|--------|-------------|
| MQTT → Ingestion | <10ms | Message delivery |
| Ingestion → Model | <30ms | HTTP + inference |
| Ingestion → Threshold | <20ms | HTTP + computation |
| Total E2E | <100ms | p95 latency goal |

---

## 5. Machine Learning Model

### LSTM Autoencoder Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LSTM AUTOENCODER ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         INPUT WINDOW                │
                    │      (50 timesteps × 16 features)   │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │              ENCODER                │
                    │  ┌────────────────────────────────┐ │
                    │  │   LSTM Layer 1 (hidden=128)    │ │
                    │  │             ▼                  │ │
                    │  │   LSTM Layer 2 (hidden=128)    │ │
                    │  │             ▼                  │ │
                    │  │   Dense (128 → 64)             │ │
                    │  └────────────────────────────────┘ │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │          LATENT SPACE               │
                    │         (64-dimensional)            │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │              DECODER                │
                    │  ┌────────────────────────────────┐ │
                    │  │   Dense (64 → 128)             │ │
                    │  │             ▼                  │ │
                    │  │   Repeat (50 timesteps)        │ │
                    │  │             ▼                  │ │
                    │  │   LSTM Layer 1 (hidden=128)    │ │
                    │  │             ▼                  │ │
                    │  │   LSTM Layer 2 (hidden=128)    │ │
                    │  │             ▼                  │ │
                    │  │   Dense (128 → 16)             │ │
                    │  └────────────────────────────────┘ │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │            OUTPUT                   │
                    │    Reconstructed Window (50×16)     │
                    │                                     │
                    │    Error = MSE(input, output)       │
                    └─────────────────────────────────────┘
```

### Why LSTM Autoencoder?

| Advantage | Explanation |
|-----------|-------------|
| **Temporal Patterns** | Captures time-series dependencies |
| **Unsupervised** | Learns normal behavior without labeled anomalies |
| **Generalization** | Reconstruction error naturally higher for anomalies |
| **Entity-Agnostic** | Single model works across all entities |

### Model Files

- `model.pt` - PyTorch model checkpoint
- `norm_stats.json` - Feature normalization statistics (mean, std)

---

## 6. Threshold Engine & Algorithms

### Adaptive Threshold Formula

```
threshold = μ_recent + K × σ_recent + (α × σ_recent if drift)
```

Where:
- `μ_recent` = Mean of recent N errors
- `σ_recent` = Standard deviation of recent N errors  
- `K` = Sensitivity multiplier (default: 1.0)
- `α` = Drift adjustment factor (default: 0.1)

### Drift Detection Algorithm

```python
def detect_drift(recent_errors, past_errors):
    """
    Detect concept drift by comparing error distributions
    """
    mean_recent = np.mean(recent_errors)
    mean_past = np.mean(past_errors)
    
    # Drift detected if mean shift exceeds threshold
    drift = abs(mean_recent - mean_past) > DRIFT_THRESHOLD
    
    return drift
```

### Redis State Management

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REDIS DATA SCHEMA                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  entity:{id}:recent_errors  →  LIST [e1, e2, ..., eN]  (N = ERROR_WINDOW/2)
  entity:{id}:past_errors    →  LIST [e1, e2, ..., eN]  (N = ERROR_WINDOW/2)
  entity:{id}:stats          →  HASH {mean, std, threshold, drift_flag}

  ┌─────────────────────────────────────┐
  │        Error Window Flow            │
  │                                     │
  │   New Error → [Recent] → [Past]     │
  │                 │          │        │
  │                 N/2       N/2       │
  │                 ◄───────────►       │
  │              Total Window: N        │
  └─────────────────────────────────────┘
```

### Lua Script for Atomicity

```lua
-- Atomic update of error lists
LPUSH entity:{id}:recent_errors error

IF LLEN recent_errors > N_RECENT:
    RPOPLPUSH recent_errors → past_errors

LTRIM recent_errors to N_RECENT
LTRIM past_errors to N_PAST
```

---

## 7. Anomaly Detection Modes

The system supports four threshold computation modes for experimental comparison:

### Mode Comparison

| Mode | Formula | Use Case |
|------|---------|----------|
| `global_static` | `GLOBAL_MEAN + K × GLOBAL_STD` | Baseline (no adaptation) |
| `global_dynamic` | Sliding window across all entities | Global trend tracking |
| `entity_dynamic` | `μ_entity + K × σ_entity` | Per-entity, no drift |
| `entity_drift` | `μ_entity + K × σ_entity + α×σ (if drift)` | **Recommended** |

### Detailed Mode Descriptions

#### 1. Global Static (`global_static`)
- Uses pre-computed global statistics
- No adaptation to entity behavior or drift
- **Pros:** Simple, no cold start
- **Cons:** High false positive/negative rates

#### 2. Global Dynamic (`global_dynamic`)
- Maintains sliding window of all entity errors
- Threshold adapts to global system trends
- **Pros:** Adapts to system-wide changes
- **Cons:** Misses entity-specific anomalies

#### 3. Entity Dynamic (`entity_dynamic`)
- Per-entity error history
- No drift adjustment
- **Pros:** Entity-specific sensitivity
- **Cons:** May miss drift-related anomalies

#### 4. Entity Drift (`entity_drift`) ✓ Recommended
- Per-entity error history
- Active drift detection and threshold adjustment
- **Pros:** Best precision-recall balance
- **Cons:** Requires warmup period

---

## 8. Configuration Reference

### Docker Compose Environment Variables

#### Simulator Service

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_BROKER_HOST` | mqtt | MQTT broker hostname |
| `MQTT_BROKER_PORT` | 1883 | MQTT broker port |
| `DEFAULT_RATE` | 2.0 | Messages per second |
| `NUM_FEATURES` | 16 | Feature dimensions |
| `ANOMALY_PROBABILITY` | 0.05 | Anomaly injection rate |
| `SPIKE_MAGNITUDE` | 3.0 | Spike anomaly magnitude |
| `NOISE_MAGNITUDE` | 1.0 | Noise anomaly magnitude |
| `DRIFT_MAGNITUDE` | 0.5 | Drift anomaly magnitude |
| `DRIFT_DURATION` | 300.0 | Drift duration (seconds) |

#### Model Service

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | model.pt | Model checkpoint path |
| `NORM_STATS_PATH` | norm_stats.json | Normalization stats path |
| `INPUT_DIM` | 16 | Number of input features |
| `HIDDEN_DIM` | 128 | LSTM hidden dimension |
| `LATENT_DIM` | 64 | Latent space dimension |
| `NUM_LAYERS` | 2 | Number of LSTM layers |
| `DROPOUT` | 0.2 | Dropout rate |
| `EPSILON` | 1e-6 | Numerical stability |

#### Threshold Engine

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | redis://redis:6379 | Redis connection URL |
| `ERROR_WINDOW` | 20 | Total error window size |
| `K` | 1.0 | Threshold sensitivity multiplier |
| `ALPHA` | 0.1 | Drift adjustment factor |
| `ANOMALY_PERSISTENCE` | 1 | Consecutive anomalies required |
| `ERROR_SMOOTH_WINDOW` | 1 | Error smoothing window |
| `DRIFT_THRESHOLD` | 0.05 | Drift detection threshold |
| `THRESHOLD_MODE` | entity_drift | Threshold computation mode |
| `GLOBAL_MEAN` | 0.9 | Global mean for static mode |
| `GLOBAL_STD` | 0.1 | Global std for static mode |

#### Ingestion Service

| Variable | Default | Description |
|----------|---------|-------------|
| `WINDOW_SIZE` | 50 | Samples per window |
| `WINDOW_STRIDE` | 25 | Samples between windows |
| `ENTITY_TIMEOUT_SEC` | 300 | Buffer timeout |
| `HTTP_TIMEOUT` | 15.0 | HTTP request timeout |
| `MODEL_SERVICE_URL` | http://model:8002 | Model service URL |
| `THRESHOLD_SERVICE_URL` | http://threshold:8003 | Threshold service URL |

### Parameter Tuning Guide

| Parameter | Effect of Increase | Typical Range |
|-----------|-------------------|---------------|
| `K` | Fewer anomalies detected (↑ Precision, ↓ Recall) | 1.0 - 3.0 |
| `ALPHA` | Larger drift adjustment | 0.05 - 0.2 |
| `ERROR_WINDOW` | Smoother threshold, slower adaptation | 10 - 50 |
| `ANOMALY_PERSISTENCE` | Fewer false positives | 1 - 3 |
| `DRIFT_THRESHOLD` | Less frequent drift detection | 0.01 - 0.1 |

---

## 9. Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.11 | Primary development language |
| **API Framework** | FastAPI (async) | High-performance REST APIs |
| **ML Framework** | PyTorch | Neural network implementation |
| **MQTT Client** | paho-mqtt | MQTT message handling |
| **Data Processing** | NumPy | Numerical operations |
| **State Store** | Redis 7 | Entity state and error history |
| **Message Broker** | Eclipse Mosquitto 2.0 | MQTT broker |
| **Containerization** | Docker + docker-compose | Service deployment |

### Monitoring & Observability

| Aspect | Implementation |
|--------|---------------|
| **Metrics** | Prometheus `/metrics` endpoint per service |
| **Logging** | JSON structured logs with trace_id |
| **Health Checks** | HTTP `/health` endpoint per service |

### Service Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                    SERVICE DEPENDENCY GRAPH                      │
└─────────────────────────────────────────────────────────────────┘

                           mqtt ──────────────────┐
                             │                    │
                          redis                   │
                             │                    │
                           model                  │
                             │                    │
                         threshold                │
                             │                    │
                ┌────────────┴────────────┐       │
                │                         │       │
            ingestion ◀───────────────────┴───────┘
                │
              api
```

---

## 10. Deployment Guide

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 1.29+
- 4GB RAM minimum
- 10GB disk space

### Quick Start

```bash
# Clone repository (if not already done)
cd ai-telemetry-anomaly

# Build all services
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps
```

### Service Ports

| Service | Port | Health Endpoint |
|---------|------|-----------------|
| MQTT Broker | 1883 | N/A |
| Redis | 6379 | N/A |
| Simulator | 8000 | `/health` |
| Ingestion | 8001 | `/health` |
| Model | 8002 | `/health` |
| Threshold | 8003 | `/health` |
| API Gateway | 8080 | `/health` |

### Scaling for Production

```bash
# Scale ingestion service for higher throughput
docker-compose up -d --scale ingestion=3
```

### Common Operations

```bash
# Stop all services
docker-compose down

# Stop and remove data volumes
docker-compose down -v

# Rebuild specific service
docker-compose build threshold
docker-compose up -d threshold

# View service logs
docker-compose logs -f threshold
```

### Windows PowerShell Scripts

```powershell
# Start services
.\scripts\start.ps1

# Stop services
.\scripts\stop.ps1

# Run tests
.\scripts\test.ps1
```

---

## 11. Evaluation & Metrics

### Classification Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Precision** | TP / (TP + FP) | Accuracy of positive predictions |
| **Recall** | TP / (TP + FN) | Coverage of actual positives |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of P and R |

### Confusion Matrix

```
                      Predicted
                   Positive  Negative
              ┌──────────┬──────────┐
    Actual    │    TP    │    FN    │  Positive
              ├──────────┼──────────┤
              │    FP    │    TN    │  Negative
              └──────────┴──────────┘
```

### Running Experiments

```bash
cd evaluation

# Run single mode experiment
python experiment_runner.py --single entity_drift --duration 60

# Run all modes for comparison
python experiment_runner.py --duration 300 --output results/

# Generate comparison table
python generate_results_table.py
```

### Evaluation Data Format

```csv
entity_id,timestamp,actual_anomaly,anomaly_type,predicted_anomaly,error,threshold,drift,latency_ms
engine_1,1710000000000,False,normal,False,0.015,0.023,False,45
engine_1,1710000025000,True,spike,True,0.089,0.023,False,52
```

### Expected Performance (entity_drift mode)

| Metric | Target | Notes |
|--------|--------|-------|
| Precision | ≥0.75 | Minimize false alarms |
| Recall | ≥0.90 | Catch most anomalies |
| F1 Score | ≥0.80 | Balanced performance |
| Latency (p95) | <100ms | Real-time requirement |

---

## 12. API Reference

### Simulator Service

#### Start Simulation
```http
POST /start
```

#### Stop Simulation
```http
POST /stop
```

#### Get Configuration
```http
GET /config
```

### Model Service

#### Inference Request
```http
POST /infer
Content-Type: application/json

{
  "entity_id": "engine_1",
  "window": [[0.5, 0.1, ...], [0.4, 0.2, ...], ...]
}
```

**Response:**
```json
{
  "reconstruction_error": 0.023
}
```

### Threshold Engine

#### Update Threshold
```http
POST /update
Content-Type: application/json

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
  "drift": false
}
```

### API Gateway

#### Predict
```http
POST /predict
Content-Type: application/json

{
  "entity_id": "engine_1",
  "window": [[0.5, 0.1, 0.9], [0.4, 0.2, 0.8], ...]
}
```

**Response:**
```json
{
  "anomaly": true,
  "error": 0.023,
  "threshold": 0.018,
  "drift": false
}
```

#### Get Entity State
```http
GET /entity/{entity_id}
```

**Response:**
```json
{
  "entity_id": "engine_1",
  "mean": 0.015,
  "std": 0.003,
  "threshold": 0.024,
  "drift_flag": false,
  "recent_errors_count": 10,
  "past_errors_count": 10
}
```

---

## Appendix A: File Structure

```
ai-telemetry-anomaly/
├── docker-compose.yml       # Service orchestration
├── Makefile                 # Build automation
├── README.md                # Quick start guide
├── DOCUMENTATION.md         # This file
├── docker/
│   └── mosquitto.conf       # MQTT broker configuration
├── scripts/
│   ├── start.ps1            # Windows start script
│   ├── stop.ps1             # Windows stop script
│   └── test.ps1             # Windows test script
├── services/
│   ├── api/                 # API Gateway service
│   ├── ingestion/           # Ingestion service
│   ├── model/               # Model inference service
│   ├── simulator/           # Data simulator service
│   └── threshold/           # Threshold engine service
├── evaluation/
│   ├── experiment_runner.py # Experiment automation
│   ├── metrics.py           # Metrics computation
│   └── baseline_comparison.py
├── specs/
│   └── mdf.md               # Machine Design File
├── train/                   # Model training scripts
└── results/                 # Experiment results
```

---

## Appendix B: Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Services not healthy | Dependencies not ready | Wait for health checks or increase timeout |
| High false positive rate | K too low | Increase K (e.g., 1.5 → 2.0) |
| Low recall | K too high | Decrease K |
| Redis connection failed | Redis not running | Check `docker-compose ps` |
| Model inference slow | Large batch size | Reduce window size or enable GPU |

### Debugging Commands

```bash
# Check service logs
docker-compose logs -f [service-name]

# Check Redis state
docker exec -it redis redis-cli
> KEYS entity:*
> LRANGE entity:engine_1:recent_errors 0 -1

# Check MQTT messages
docker exec -it mqtt-broker mosquitto_sub -t "#" -v

# Test model endpoint
curl -X POST http://localhost:8002/infer \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "test", "window": [[0.1]*16]*50}'
```

---

## Appendix C: Research Context

### Related Work

This system implements concepts from several research domains:

1. **LSTM Autoencoders for Anomaly Detection**
   - Malhotra, P., et al. (2016). "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"

2. **Adaptive Thresholding**
   - Control chart methods (Shewhart, CUSUM, EWMA)
   - Dynamic threshold adjustment

3. **Concept Drift Detection**
   - Page-Hinkley test
   - Distribution comparison methods

### Novel Contributions

1. **Entity-Specific Drift-Aware Thresholds**: Combines per-entity adaptation with drift detection
2. **Real-Time Microservices Architecture**: Sub-100ms latency at scale
3. **Persistence Filter**: Reduces false positives through consecutive window validation

---

*Document Version: 1.0*  
*Last Updated: March 2026*
