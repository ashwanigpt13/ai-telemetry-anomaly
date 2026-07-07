# Project Audit

Audit date: 2026-07-08

This repository contains an AI telemetry anomaly detection project with two related layers:

- a frozen research benchmark comparing a classical LSTM VAE autoencoder against a hybrid quantum LSTM VAE autoencoder on NASA CMAPSS FD001;
- a Dockerized telemetry anomaly detection system with simulator, ingestion, model serving, adaptive thresholding, and API services.

The current publication-grade benchmark artifacts are concentrated in `train/`, `evaluation/`, `results/`, `BENCHMARK_REPRODUCIBILITY.md`, `repository_structure.md`, and `results/paper_assets/experiment_card.md`. The `archive/` directory preserves older exploratory scripts, duplicate data, caches, superseded plots, and non-final outputs.

## 1. Repository Overview

The active benchmark workflow is:

1. Load NASA CMAPSS data from `data/`.
2. Train or load frozen checkpoints from `train/`.
3. Evaluate classical and hybrid models using `evaluation/benchmark.py`.
4. Run secondary analyses using `evaluation/statistical_validation.py`, `evaluation/error_analysis.py`, and `evaluation/latent_analysis.py`.
5. Export publication figures and tables under `results/paper_figures/` and `results/paper_tables/`.
6. Record reproducibility details in `results/paper_assets/experiment_card.md` and `BENCHMARK_REPRODUCIBILITY.md`.

Frozen final benchmark summary:

| Model | ROC | PR | Precision | Recall | F1 | Balanced Accuracy | MCC | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classical | 0.943549 | 0.693213 | 0.260600 | 0.852400 | 0.399200 | 0.875530 | 0.436560 | 283 | 7120 | 803 | 49 |
| Hybrid Quantum | 0.728410 | 0.202050 | 0.041100 | 1.000000 | 0.079000 | 0.511233 | 0.030388 | 332 | 178 | 7745 | 0 |

The paper-facing benchmark is therefore a reproducible classical-vs-hybrid anomaly detection study, not the older drift-aware threshold-only narrative that appears in some top-level paper drafts.

## 2. Folder Purpose

| Path | Purpose |
| --- | --- |
| `.agents/` | Agent/local automation metadata. Not paper-facing. |
| `.git/` | Git repository metadata. Read-only for audit. |
| `archive/` | Preserved legacy and superseded material: older evaluation scripts, old result folders, smoke-test models, duplicate `train/data/`, caches, and temporary files. Useful for provenance, but not the active benchmark source of truth. |
| `data/` | NASA CMAPSS data files for FD001-FD004, `CMAPSSData.zip`, source PDF, and `data/evaluation/evaluation.csv`. FD001 is the frozen paper benchmark split. |
| `docker/` | Docker support configuration, currently `mosquitto.conf`. |
| `evaluation/` | Active evaluation and analysis scripts for final benchmark, statistical validation, error analysis, latent analysis, metrics, and publication figure generation. |
| `results/FinalResults/` | Frozen final benchmark outputs for seed 42, including metrics JSON, comparison tables, ROC/PR curves, confusion matrices, parameter comparison, training time, and efficiency metrics. |
| `results/error_analysis/` | Frozen per-sample error analysis, corrected-sample CSVs, confusion matrices, score distributions, and summary JSON/Markdown/CSV. |
| `results/latent_analysis/` | Frozen latent embeddings for classical and hybrid models plus labels and metadata. |
| `results/paper_assets/` | Publication reproducibility card. |
| `results/paper_figures/` | Publication-ready figure bundle in PNG/PDF/SVG where available, plus figure manifests and architecture/pipeline diagrams. |
| `results/paper_tables/` | Publication model-comparison table in CSV, Markdown, LaTeX, and XLSX. |
| `results/statistical_validation/` | Multi-seed validation outputs for seeds 42, 123, and 456 plus aggregate raw and summary tables. |
| `scripts/` | PowerShell helper scripts for starting, stopping, and testing the service stack. |
| `services/` | Runtime microservices: API gateway, ingestion, model service, simulator, and threshold service. |
| `specs/` | Design and training specs. These describe the system concept and older phase-0 training design. |
| `train/` | Active model implementation, training script, preprocessing/windowing utilities, quantum layer, frozen classical and hybrid checkpoints, metadata, normalization statistics, and threshold calibration artifacts. |

## 3. Important Files

### Model Implementations

| File | Role |
| --- | --- |
| `train/model.py` | Core LSTM variational autoencoder implementation: temporal attention, LSTM encoder/decoder, KL loss, model creation, save/load helpers. |
| `train/quantum_layer.py` | `QuantumLatentLayer`, the PennyLane/PyTorch quantum latent transformation used by the hybrid model. |
| `services/model/main.py` | Runtime inference-service model implementation for the microservice stack. This is separate from the frozen paper benchmark checkpoints. |
| `services/model/train_dummy_model.py` | Utility to create a dummy runtime model for service testing, not a paper benchmark model. |

### Training Scripts and Utilities

| File | Role |
| --- | --- |
| `train/train.py` | Main training entry point. Defines config, seeding, early stopping, VAE loss, contrastive loss, dataloaders, reconstruction-error statistics, checkpoint verification, metadata writing, and training history plots. |
| `train/dataset.py` | NASA CMAPSS loading and preparation utilities, including train/test loaders and healthy-data filtering. |
| `train/preprocess.py` | Feature normalization and dataframe preprocessing. |
| `train/window.py` | Sliding-window creation and splitting utilities. |
| `train/test_quantum_environment.py` | Quantum environment smoke test. |
| `train/test_quantum_learning.py` | Small quantum-learning sanity test. |
| `train/requirements.txt` | Training dependency floor: PyTorch, NumPy, pandas, scikit-learn, tqdm, PennyLane. |

### Evaluation Scripts

| File | Role |
| --- | --- |
| `evaluation/benchmark.py` | Final classical-vs-hybrid benchmark. Loads frozen checkpoints, verifies compatibility metadata, loads thresholds, scores windows, computes metrics, and writes final tables/figures. |
| `evaluation/statistical_validation.py` | Multi-seed reproducibility analysis for seeds 42, 123, and 456. |
| `evaluation/error_analysis.py` | Per-sample error classification, corrected-sample comparison, score distributions, confusion matrices, and error summaries. |
| `evaluation/latent_analysis.py` | Extracts classical and hybrid latent embeddings and labels. |
| `evaluation/publication_figures.py` | Exports publication figures in PNG/PDF/SVG and writes the figure manifest. |
| `evaluation/metrics.py` | Generic classification, per-entity, drift-recovery, and latency metric utilities. |
| `evaluation/compute_metrics.py` | Small metric computation entry point. |

### Reproducibility Documents

| File | Role |
| --- | --- |
| `results/paper_assets/experiment_card.md` | Primary frozen experiment card: dataset, hardware, software versions, repo state, seed, training/evaluation config, model config, thresholds, checkpoint hashes, and reproduction steps. |
| `BENCHMARK_REPRODUCIBILITY.md` | Threshold calibration policy and source-of-truth threshold values. |
| `repository_structure.md` | Public-release repository layout and cleanup policy. |
| `README.md` | Docker/microservice quick-start documentation. |
| `DOCUMENTATION.md` | Extended project documentation. |
| `QUICK_START.md` | Quick-start instructions. |
| `specs/training.md` | Phase-0 LSTM autoencoder training spec. Contains mojibake in headings/symbols. |
| `specs/mdf.md` | Machine design file for the streaming drift-aware system. Contains mojibake in headings/symbols. |

### Paper Drafts and Legacy Paper Docs

| File | Status |
| --- | --- |
| `PAPER.tex` | Existing LaTeX draft for an older drift-aware entity-threshold paper. It does not currently reflect the frozen classical-vs-hybrid quantum benchmark. |
| `RESEARCH_PAPER.md` | Older research-paper draft centered on drift-aware thresholds and microservices. |
| `RESEARCH_PAPER_ENHANCED.md` | Enhanced version of the older drift-aware paper narrative. |
| `PAPER_ANALYSIS_README.md` | Older analysis plan for threshold/baseline/scenario experiments. Some referenced files now live under `archive/` or are superseded. |
| `SUBMISSION_PACKAGE_SUMMARY.md` | Older submission-package summary for the drift-aware threshold work. |

## 4. Frozen Benchmark Artifacts

These artifacts define the frozen paper benchmark and should be treated as immutable unless intentionally creating a new benchmark version.

### Checkpoints and Metadata

| Artifact | Purpose |
| --- | --- |
| `train/classical_model.pt` | Frozen classical LSTM VAE checkpoint. |
| `train/hybrid_model.pt` | Frozen hybrid quantum LSTM VAE checkpoint. |
| `train/classical_model_metadata.json` | Classical model metadata. Fully records dataset, features, architecture, training config, best epoch/loss, timing, and parameter counts. |
| `train/hybrid_model_metadata.json` | Hybrid model metadata. Records best epoch/loss, timing, parameter counts, and quantum settings. It is less complete than the classical metadata; `evaluation/benchmark.py` fills compatibility fields from benchmark defaults/checkpoint where needed. |
| `train/norm_stats.json` | Training normalization statistics for benchmark models. |
| `norm_stats.json` | Root-level normalization statistics used by runtime or compatibility paths. |

Checkpoint hashes from `results/paper_assets/experiment_card.md`:

| Artifact | SHA256 |
| --- | --- |
| `train/classical_model.pt` | `F05705FF1A917F2FC9DB6FC763B8EFDC5C7ACF254373B3FEB23A468D14DE95E9` |
| `train/hybrid_model.pt` | `34ED1056CD1A314A3D9B92FC0E6AC52532E8876A4528F1DBA59A1ABF4ED5F07F` |
| `train/classical_model_metadata.json` | `E20BB7A84BF82EA55D5E6EEADDA5CF9E14FF3C9F7CD87778174B388B7C833803` |
| `train/hybrid_model_metadata.json` | `EE1EAD1AC9DDC741832D57F574FBBFFF9FC05B9D4B61D024A9FB08297EEB2EBA` |

### Threshold Calibration Artifacts

| Artifact | Frozen value |
| --- | ---: |
| `global_stats.json.percentiles.p95` | Classical threshold `0.6131619691848755` |
| `train/hybrid_global_stats.json.percentiles.p95` | Hybrid threshold `0.49388029724359506` |
| `train/global_stats.json.percentiles.p95` | Hybrid compatibility-path threshold `0.49388029724359506` |

`BENCHMARK_REPRODUCIBILITY.md` notes that final benchmark outputs record the historical hybrid compatibility path `train/global_stats.json`, while `train/hybrid_global_stats.json` is the clearer provenance artifact with the same numeric statistics.

### Final Benchmark Outputs

| Artifact | Purpose |
| --- | --- |
| `results/FinalResults/results.json` | Full frozen benchmark payload with metadata, metrics, curves, scores, and efficiency data. Large file because it includes curve arrays. |
| `results/FinalResults/comparison_table.csv` | Final benchmark comparison table. |
| `results/FinalResults/comparison_table.md` | Markdown comparison table. |
| `results/FinalResults/comparison_table.tex` | LaTeX comparison table. |
| `results/FinalResults/efficiency_metrics.json` | Training time, parameter count, model size, and inference time for both models. |
| `results/FinalResults/roc_curve.png` | Final ROC curve. |
| `results/FinalResults/pr_curve.png` | Final precision-recall curve. |
| `results/FinalResults/confusion_matrix_classical.png` | Classical confusion matrix. |
| `results/FinalResults/confusion_matrix_hybrid.png` | Hybrid confusion matrix. |
| `results/FinalResults/parameter_comparison.png` | Parameter comparison figure. |
| `results/FinalResults/training_time.png` | Training-time comparison figure. |

### Statistical Validation Outputs

| Artifact | Purpose |
| --- | --- |
| `results/statistical_validation/raw_results.csv` | Raw metrics across seeds. |
| `results/statistical_validation/summary.csv` | Aggregate mean/std metrics. |
| `results/statistical_validation/summary.md` | Markdown aggregate table. |
| `results/statistical_validation/summary.tex` | LaTeX aggregate table. |
| `results/statistical_validation/summary.json` | Structured aggregate statistics. |
| `results/statistical_validation/seed_42/` | Seed 42 benchmark output bundle. |
| `results/statistical_validation/seed_123/` | Seed 123 benchmark output bundle. |
| `results/statistical_validation/seed_456/` | Seed 456 benchmark output bundle. |

Aggregate validation summary:

| Metric | Classical Mean +/- Std | Hybrid Mean +/- Std |
| --- | ---: | ---: |
| ROC | 0.941643 +/- 0.002088 | 0.726842 +/- 0.003001 |
| PR | 0.682980 +/- 0.009197 | 0.202876 +/- 0.001060 |
| Precision | 0.255500 +/- 0.005100 | 0.041100 +/- 0.000000 |
| Recall | 0.844367 +/- 0.009220 | 1.000000 +/- 0.000000 |
| F1 | 0.392300 +/- 0.006421 | 0.078967 +/- 0.000058 |
| Inference Time | 0.302598 +/- 0.031152 | 7.476823 +/- 0.546800 |
| Training Time | 242.867065 +/- 0.000000 | 3854.864205 +/- 0.000000 |

### Error and Latent Analysis Outputs

| Artifact | Purpose |
| --- | --- |
| `results/error_analysis/error_summary.md` | Human-readable frozen error analysis. |
| `results/error_analysis/error_summary.json` | Structured error analysis summary. |
| `results/error_analysis/error_statistics.csv` | Error statistics table. |
| `results/error_analysis/classical_sample_errors.csv` | Classical per-sample errors. |
| `results/error_analysis/hybrid_sample_errors.csv` | Hybrid per-sample errors. |
| `results/error_analysis/samples_corrected_by_hybrid.csv` | Samples fixed by hybrid relative to classical. |
| `results/error_analysis/samples_corrected_by_classical.csv` | Samples fixed by classical relative to hybrid. |
| `results/error_analysis/score_distribution_classical.png` | Classical score distribution. |
| `results/error_analysis/score_distribution_hybrid.png` | Hybrid score distribution. |
| `results/latent_analysis/classical_latent.npy` | Classical latent embeddings. |
| `results/latent_analysis/hybrid_latent.npy` | Hybrid latent embeddings. |
| `results/latent_analysis/labels.npy` | Labels aligned to latent embeddings. |
| `results/latent_analysis/metadata.json` | Latent-analysis metadata. |

## 5. Files That Must Never Be Modified

Treat these as immutable for the paper unless intentionally starting a new benchmark version with a new experiment card, hashes, and results:

| Path | Reason |
| --- | --- |
| `train/classical_model.pt` | Frozen classical checkpoint. |
| `train/hybrid_model.pt` | Frozen hybrid checkpoint. |
| `train/classical_model_metadata.json` | Frozen classical checkpoint metadata and hash target. |
| `train/hybrid_model_metadata.json` | Frozen hybrid checkpoint metadata and hash target. |
| `train/norm_stats.json` | Frozen training normalization statistics. |
| `train/hybrid_global_stats.json` | Frozen hybrid threshold provenance. |
| `train/global_stats.json` | Frozen hybrid threshold compatibility path. |
| `global_stats.json` | Frozen classical threshold source. |
| `norm_stats.json` | Root normalization artifact used by compatibility/runtime flows. |
| `results/FinalResults/**` | Frozen final benchmark outputs. |
| `results/statistical_validation/**` | Frozen multi-seed validation outputs. |
| `results/error_analysis/**` | Frozen error-analysis outputs. |
| `results/latent_analysis/**` | Frozen latent embeddings and labels. |
| `results/paper_tables/**` | Publication tables generated from frozen outputs. |
| `results/paper_figures/**` | Publication figure bundle generated from frozen outputs. |
| `results/paper_assets/experiment_card.md` | Frozen experiment card and checkpoint hash record. |
| `BENCHMARK_REPRODUCIBILITY.md` | Frozen threshold policy and calibration record. |
| `data/train_FD001.txt`, `data/test_FD001.txt`, `data/RUL_FD001.txt` | Frozen FD001 benchmark data split. |
| `data/readme.txt`, `data/Damage Propagation Modeling.pdf`, `data/CMAPSSData.zip` | Dataset provenance/source material. |
| `archive/**` | Provenance-preserving legacy material. Avoid edits except explicit archival maintenance. |

Code files under `train/` and `evaluation/` should also remain unchanged while preparing the paper. If paper writing reveals a methodological issue, create a new benchmark version rather than silently modifying code beneath frozen artifacts.

## 6. Files That Will Be Referenced In The Paper

### Methods and Architecture

| File | Paper use |
| --- | --- |
| `train/model.py` | Classical LSTM VAE architecture, attention, encoder/decoder, latent bottleneck. |
| `train/quantum_layer.py` | Hybrid quantum latent layer implementation. |
| `train/train.py` | Training procedure, losses, early stopping, beta scheduling, contrastive term, metadata/statistics generation. |
| `train/dataset.py`, `train/preprocess.py`, `train/window.py` | Dataset loading, normalization, healthy-window selection, and sliding-window generation. |
| `evaluation/benchmark.py` | Final evaluation protocol, metadata compatibility checks, thresholds, scoring, and metrics. |
| `evaluation/statistical_validation.py` | Reproducibility and multi-seed stability protocol. |
| `evaluation/error_analysis.py` | Error analysis method. |
| `evaluation/latent_analysis.py` | Latent embedding extraction protocol. |
| `evaluation/publication_figures.py` | Figure export provenance. |

### Tables

| File | Paper use |
| --- | --- |
| `results/paper_tables/table_model_comparison.tex` | Primary LaTeX comparison table. |
| `results/paper_tables/table_model_comparison.md` | Human-readable comparison table. |
| `results/paper_tables/table_model_comparison.csv` | Machine-readable table source. |
| `results/statistical_validation/summary.tex` | Multi-seed validation table. |
| `results/FinalResults/comparison_table.tex` | Frozen final benchmark table. |

### Figures

| File | Paper use |
| --- | --- |
| `results/paper_figures/roc_curve.pdf` / `.png` / `.svg` | ROC comparison. |
| `results/paper_figures/pr_curve.pdf` / `.png` / `.svg` | Precision-recall comparison. |
| `results/paper_figures/training_curves.pdf` / `.png` / `.svg` | Training/validation curves. |
| `results/paper_figures/parameter_comparison.pdf` / `.png` / `.svg` | Parameter comparison. |
| `results/paper_figures/training_time.pdf` / `.png` / `.svg` | Training-time comparison. |
| `results/paper_figures/confusion_matrix_classical.pdf` / `.png` / `.svg` | Classical confusion matrix. |
| `results/paper_figures/confusion_matrix_hybrid.pdf` / `.png` / `.svg` | Hybrid confusion matrix. |
| `results/paper_figures/score_distribution_classical.pdf` / `.png` / `.svg` | Classical reconstruction-error distribution. |
| `results/paper_figures/score_distribution_hybrid.pdf` / `.png` / `.svg` | Hybrid reconstruction-error distribution. |
| `results/paper_figures/pca_classical.pdf` / `.png` / `.svg` | Classical PCA latent projection. |
| `results/paper_figures/pca_hybrid.pdf` / `.png` / `.svg` | Hybrid PCA latent projection. |
| `results/paper_figures/tsne_classical.pdf` / `.png` / `.svg` | Classical t-SNE latent projection. |
| `results/paper_figures/tsne_hybrid.pdf` / `.png` / `.svg` | Hybrid t-SNE latent projection. |
| `results/paper_figures/system_architecture.pdf` / `.svg` | System architecture diagram. |
| `results/paper_figures/hybrid_model_architecture.svg` | Hybrid model architecture diagram. |
| `results/paper_figures/training_pipeline.svg` | Training pipeline diagram. |
| `results/paper_figures/evaluation_pipeline.svg` | Evaluation pipeline diagram. |

### Reproducibility and Provenance

| File | Paper use |
| --- | --- |
| `results/paper_assets/experiment_card.md` | Canonical experiment configuration and reproduction card. |
| `BENCHMARK_REPRODUCIBILITY.md` | Threshold calibration policy. |
| `repository_structure.md` | Repository layout and cleanup provenance. |
| `results/paper_figures/figure_manifest.json` | Current structured figure manifest. |
| `results/paper_figures/figure_manifest.md` | Human-readable figure manifest, but see missing/inconsistency note below. |

## 7. Missing Assets Or Inconsistencies

No core frozen benchmark artifact appears missing: checkpoints, threshold stats, final benchmark outputs, statistical validation, error analysis, latent arrays, publication tables, and publication figures are present.

Issues to resolve before paper writing:

| Issue | Evidence | Recommended action |
| --- | --- | --- |
| Existing top-level paper drafts are not aligned with current frozen benchmark. | `PAPER.tex`, `RESEARCH_PAPER.md`, and `RESEARCH_PAPER_ENHANCED.md` describe a drift-aware entity-threshold microservices paper with F1 around 0.67, not the frozen classical-vs-hybrid quantum benchmark whose final F1 values are 0.3992 and 0.0790. | Treat existing drafts as legacy/reference only. Write the new paper from the frozen benchmark artifacts. |
| `results/paper_figures/figure_manifest.md` is stale. | It marks PCA/t-SNE figures as missing, but `results/paper_figures/` now contains `pca_classical`, `pca_hybrid`, `tsne_classical`, and `tsne_hybrid` in PNG/PDF/SVG formats. | Prefer `results/paper_figures/figure_manifest.json` or regenerate/update the Markdown manifest in a separate paper-assets pass. |
| Some architecture/pipeline diagrams do not have all export formats. | `system_architecture` has SVG/PDF; `hybrid_model_architecture`, `training_pipeline`, and `evaluation_pipeline` are SVG-only. | If the target venue requires PDF or PNG for every figure, export those SVG-only diagrams to PDF/PNG without changing their content. |
| Hybrid metadata is less complete than classical metadata. | `train/hybrid_model_metadata.json` lacks dataset, feature list, window size, hidden/latent dims, batch size, learning rate, and seed fields; benchmark compatibility fills them from defaults/checkpoint. | Do not edit the frozen metadata in place. Document this explicitly in the paper/reproducibility appendix or create a new metadata sidecar if needed. |
| Specs contain mojibake characters. | `specs/training.md` and `specs/mdf.md` show garbled emoji/arrows such as `ðŸ“˜` and `â†’`. | Avoid direct copy into the paper unless cleaned in a separate documentation-only pass. |
| Git working tree is dirty. | `git status --short` shows many deleted legacy files, new `archive/` content, new result folders, modified `evaluation/benchmark.py`, and new reproducibility assets. | Before final submission, commit or otherwise freeze the public-release cleanup state so the experiment card commit/provenance story is clear. |

## Audit Boundary

This audit did not modify code, checkpoints, data, benchmark results, figures, or existing documentation. The only intended write from this audit pass is this `PROJECT_AUDIT.md` file.
