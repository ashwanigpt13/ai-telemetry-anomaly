# Experiment Card

This card records the frozen experiment configuration used to reproduce the classical and hybrid quantum telemetry anomaly detection results.

## Dataset

- Dataset family: NASA CMAPSS turbofan degradation data.
- Benchmark split: `FD001`.
- Active data directory: `data/`.
- Training files: `data/train_FD001.txt`.
- Evaluation files: `data/test_FD001.txt`, `data/RUL_FD001.txt`.
- Evaluation labels: generated from remaining useful life with anomaly threshold `RUL <= 30.0`.
- Healthy training ratio: `0.7`.
- Window size: `50`.
- Window stride: `1`.
- Feature names:
  - `setting_1`
  - `setting_2`
  - `sensor_2`
  - `sensor_3`
  - `sensor_4`
  - `sensor_7`
  - `sensor_8`
  - `sensor_9`
  - `sensor_11`
  - `sensor_12`
  - `sensor_13`
  - `sensor_14`
  - `sensor_15`
  - `sensor_17`
  - `sensor_20`
  - `sensor_21`

## Hardware

- Hostname: `DESKTOP-VTD3G9H`.
- Platform: `Windows-11-10.0.26200-SP0`.
- Machine: `AMD64`.
- Processor string: `Intel64 Family 6 Model 140 Stepping 1, GenuineIntel`.
- Logical CPU count visible to Python: `8`.
- GPU/CUDA: unavailable for benchmark execution.
- Benchmark device: `cpu`.
- Physical memory: unavailable from the sandboxed environment.

## Software Versions

- Python: `3.12.3`.
- PyTorch: `2.9.1+cpu`.
- CUDA runtime reported by PyTorch: `None`.
- PennyLane: `0.45.1`.
- NumPy: `2.3.5`.
- pandas: `2.3.3`.
- scikit-learn: `1.9.0`.
- Matplotlib: `3.10.7`.
- openpyxl: `3.1.5`.
- Operating system: Windows 11.

## Repository State

- Repository commit: `5d5b140c8bf37cbebabeecfae5f478b50ded4d4a`.
- Working tree state at card generation: dirty, containing public-release cleanup moves and generated paper assets.
- Structure document: `repository_structure.md`.

## Random Seed

- Primary training and final benchmark seed: `42`.
- Statistical validation seeds: `42`, `123`, `456`.

## Training Configuration

- Training entry point: `train/train.py`.
- No retraining was performed for paper-asset generation.
- Epochs: `20`.
- Batch size: `8`.
- Learning rate: `0.001`.
- Optimizer: Adam.
- Weight decay: `1e-5`.
- Train ratio: `0.8`.
- Early stopping patience: `5`.
- Early stopping minimum delta: `1e-4`.
- VAE KL beta: `0.001`.
- Maximum beta: `0.001`.
- Beta scheduler: linear.
- Contrastive loss weight: `0.1`.
- Random seed: `42`.

## Model Configuration

Shared architecture:

- Model family: LSTM variational autoencoder with temporal attention and latent bottleneck.
- Input dimension: `16`.
- Hidden dimension: `128`.
- Latent dimension: `64`.
- Number of LSTM layers: `2`.
- Dropout: `0.2`.
- Window size: `50`.

Classical model:

- Checkpoint: `train/classical_model.pt`.
- Model type: `classical`.
- Quantum enabled: `false`.
- Total parameters: `500473`.
- Trainable parameters: `500473`.
- Quantum parameters: `0`.
- Classical parameters: `500473`.
- Best validation loss: `0.7426523895017685`.
- Best epoch: `6`.
- Training time: `242.86706495285034` seconds.

Hybrid quantum model:

- Checkpoint: `train/hybrid_model.pt`.
- Model type: `hybrid`.
- Quantum enabled: `true`.
- Number of qubits: `4`.
- Number of quantum layers: `2`.
- Total parameters: `500573`.
- Trainable parameters: `500573`.
- Quantum parameters: `24`.
- Classical parameters: `500549`.
- Best validation loss: `0.8593889075139213`.
- Best epoch: `13`.
- Training time: `3854.864205121994` seconds.

## Evaluation Configuration

- Final benchmark script: `evaluation/benchmark.py`.
- Frozen benchmark result: `results/FinalResults/results.json`.
- Benchmark generated at: `2026-07-07T20:19:25Z`.
- Evaluation device: `cpu`.
- Evaluation seed: `42`.
- Dataset: `FD001`.
- Window size: `50`.
- Stride: `1`.
- RUL anomaly threshold: `30.0`.
- Classical threshold source: `global_stats.json.percentiles.p95`.
- Classical threshold: `0.6131619691848755`.
- Hybrid threshold source: `train/hybrid_global_stats.json.percentiles.p95`.
- Hybrid frozen-benchmark compatibility path: `train/global_stats.json.percentiles.p95`.
- Hybrid threshold: `0.49388029724359506`.
- Metadata compatibility fields checked:
  - `dataset`
  - `feature_names`
  - `window_size`
  - `hidden_dim`
  - `latent_dim`
  - `batch_size`
  - `learning_rate`
  - `random_seed`

## Checkpoint Hashes

SHA256 hashes:

| Artifact | SHA256 |
| --- | --- |
| `train/classical_model.pt` | `F05705FF1A917F2FC9DB6FC763B8EFDC5C7ACF254373B3FEB23A468D14DE95E9` |
| `train/hybrid_model.pt` | `34ED1056CD1A314A3D9B92FC0E6AC52532E8876A4528F1DBA59A1ABF4ED5F07F` |
| `train/classical_model_metadata.json` | `E20BB7A84BF82EA55D5E6EEADDA5CF9E14FF3C9F7CD87778174B388B7C833803` |
| `train/hybrid_model_metadata.json` | `EE1EAD1AC9DDC741832D57F574FBBFFF9FC05B9D4B61D024A9FB08297EEB2EBA` |

## Reproduction Steps

1. Install training/evaluation dependencies.
2. Confirm frozen checkpoints exist at `train/classical_model.pt` and `train/hybrid_model.pt`.
3. Run `python evaluation/benchmark.py` for the final benchmark.
4. Run `python evaluation/statistical_validation.py` for multi-seed stability analysis.
5. Run `python evaluation/error_analysis.py` for per-sample error analysis.
6. Run `python evaluation/latent_analysis.py` for latent embedding extraction.
7. Run `python evaluation/publication_figures.py` for paper figure exports.
8. Generate paper tables from frozen outputs in `results/paper_tables/`.

## Notes

- This card describes the frozen experiments and paper-asset generation state.
- Paper figure/table generation did not modify models, benchmark logic, or checkpoints.
- The hybrid metadata JSON on disk predates the fully aligned metadata schema; benchmark compatibility fields for the hybrid model were verified using the frozen benchmark compatibility report and checkpoint metadata where needed.
