# 📘 TRAINING SPEC (PHASE 0)

## Objective

Train an LSTM Autoencoder for anomaly detection on NASA Turbofan dataset.

---

## Dataset

* NASA CMAPSS Turbofan dataset (FD001)
* Multivariate time series per engine_id

---

## Data Characteristics

Each row contains:

* engine_id
* cycle
* operational settings
* multiple sensor readings

---

## Data Usage

* Use only **healthy data for training**
* Exclude late cycles near failure (or use early cycles only)
* No anomalies in training data

---

## Preprocessing

1. Sort by engine_id and cycle
2. Normalize features using StandardScaler
3. Create sliding windows

---

## Windowing

* WINDOW_SIZE = 50
* Input shape: (WINDOW_SIZE, num_features)

---

## Model

LSTM Autoencoder:

Encoder:

* LSTM → Dense

Decoder:

* LSTM → Output

Loss:

* MSE

---

## Training

* Train on windows
* Use validation split
* Save best model

---

## Outputs (Artifacts)

* model.pt
* norm_stats.json (feature mean/std)
* global_stats.json (mean/std of reconstruction errors)

---

## Global Stats Computation

* Run inference on training data
* Compute reconstruction_error per window
* Compute:

  * GLOBAL_MEAN
  * GLOBAL_STD

---

## Reproducibility

* Set random seed
* Save config

---
