# Repository Structure

This repository is organized for public release of the AI telemetry anomaly detection benchmark and analysis workflow.

## Active Project Layout

```text
.
|-- README.md
|-- DOCUMENTATION.md
|-- QUICK_START.md
|-- docker-compose.yml
|-- Makefile
|-- data/
|   |-- train_FD00*.txt
|   |-- test_FD00*.txt
|   |-- RUL_FD00*.txt
|   |-- evaluation/
|   |   `-- evaluation.csv
|   `-- readme.txt
|-- train/
|   |-- train.py
|   |-- model.py
|   |-- quantum_layer.py
|   |-- dataset.py
|   |-- preprocess.py
|   |-- window.py
|   |-- requirements.txt
|   |-- classical_model.pt
|   |-- hybrid_model.pt
|   |-- classical_model_metadata.json
|   |-- hybrid_model_metadata.json
|   |-- norm_stats.json
|   |-- hybrid_global_stats.json
|   `-- global_stats.json
|-- evaluation/
|   |-- benchmark.py
|   |-- statistical_validation.py
|   |-- error_analysis.py
|   |-- latent_analysis.py
|   |-- metrics.py
|   `-- compute_metrics.py
|-- results/
|   |-- FinalResults/
|   |-- statistical_validation/
|   |-- error_analysis/
|   |-- latent_analysis/
|   `-- paper_figures/
|-- services/
|   |-- api/
|   |-- ingestion/
|   |-- model/
|   |-- simulator/
|   `-- threshold/
|-- scripts/
|-- specs/
|-- docker/
`-- archive/
```

## Active Components

### `train/`

Contains the frozen training and model implementation plus the official classical and hybrid checkpoints used in the paper benchmarks.

The public-release checkpoint artifacts are:

- `train/classical_model.pt`
- `train/hybrid_model.pt`
- `train/classical_model_metadata.json`
- `train/hybrid_model_metadata.json`
- `train/norm_stats.json`
- `train/hybrid_global_stats.json`
- `train/global_stats.json`

### `evaluation/`

Contains the active evaluation and analysis entry points:

- `benchmark.py`: final classical vs hybrid benchmark.
- `statistical_validation.py`: multi-seed benchmark reproducibility analysis.
- `error_analysis.py`: per-sample error analysis and model comparison.
- `latent_analysis.py`: latent embedding extraction for visualization.
- `metrics.py` and `compute_metrics.py`: reusable metric utilities.

Legacy exploratory scripts were moved to `archive/evaluation/`.

### `results/`

Contains release-ready benchmark and analysis artifacts:

- `FinalResults/`: frozen final benchmark outputs.
- `statistical_validation/`: multi-seed raw and aggregate statistics.
- `error_analysis/`: sample-level TP/TN/FP/FN analysis, score distributions, and summary tables.
- `latent_analysis/`: extracted latent embeddings and labels.
- `paper_figures/`: publication-ready figure bundle and figure manifest.

Older direct result files were moved to `archive/results/`.

### `data/`

Contains the NASA CMAPSS FD001-FD004 data files and the active evaluation CSV. Duplicate training data formerly under `train/data/` was archived.

### `services/`

Contains the Dockerized runtime services for API orchestration, ingestion, model serving, simulation, and thresholding.

### `archive/`

Contains files moved during cleanup without deletion. Paths are preserved under `archive/` to make provenance clear. Archived content includes:

- legacy exploratory evaluation scripts,
- old evaluation result directories,
- temporary CSV files,
- duplicate `train/data/`,
- smoke-test checkpoints,
- non-final model outputs,
- generated cache directories,
- superseded root-level plots and result files.

## Cleanup Policy

No files were deleted during cleanup. Obsolete or superseded files were moved into `archive/` so the active repository root, `train/`, `evaluation/`, and `results/` remain focused on the final benchmark workflow.
