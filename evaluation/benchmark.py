#!/usr/bin/env python3
"""Benchmark entry point for comparing trained anomaly detection models."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from evaluation.metrics import compute_classification_metrics, load_evaluation_data
except ImportError:  # pragma: no cover - allows direct execution from evaluation/
    from metrics import compute_classification_metrics, load_evaluation_data

try:
    from train.model import LSTMAutoencoder
    from train.dataset import clean_data, get_feature_columns, load_and_prepare_data, load_test_data
    from train.preprocess import FeatureNormalizer, preprocess_data
    from train.window import create_sliding_windows
except ImportError:  # pragma: no cover
    from model import LSTMAutoencoder
    from dataset import clean_data, get_feature_columns, load_and_prepare_data, load_test_data
    from preprocess import FeatureNormalizer, preprocess_data
    from window import create_sliding_windows


def load_model_from_checkpoint(checkpoint_path: Path, device: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a checkpoint with architecture chosen from checkpoint metadata and strict state-dict matching."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"[VERIFY] checkpoint keys: {list(checkpoint.keys())}")

    model_config = checkpoint.get("model_config") or {}
    if not isinstance(model_config, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} has invalid model_config metadata")

    checkpoint_use_quantum = checkpoint.get("use_quantum")
    model_config_use_quantum = model_config.get("use_quantum")
    if checkpoint_use_quantum is None and model_config_use_quantum is None:
        print("[VERIFY] use_quantum metadata missing")
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not declare use_quantum. "
            "Refusing to infer architecture from the filename."
        )

    use_quantum = bool(checkpoint_use_quantum if checkpoint_use_quantum is not None else model_config_use_quantum)
    print(f"[VERIFY] use_quantum from checkpoint: {use_quantum}")

    if "model_config" in checkpoint:
        print("[VERIFY] checkpoint contains model_config")
    else:
        print("[VERIFY] checkpoint does not contain model_config")

    if "num_qubits" in checkpoint:
        print(f"[VERIFY] num_qubits: {checkpoint['num_qubits']}")
    elif isinstance(model_config, dict) and "num_qubits" in model_config:
        print(f"[VERIFY] num_qubits from model_config: {model_config['num_qubits']}")
    else:
        print("[VERIFY] num_qubits not found in checkpoint or model_config")

    input_dim = int(checkpoint.get("input_dim", model_config.get("input_dim", 16)))
    hidden_dim = int(checkpoint.get("hidden_dim", model_config.get("hidden_dim", 64)))
    latent_dim = int(checkpoint.get("latent_dim", model_config.get("latent_dim", 32)))
    num_layers = int(checkpoint.get("num_layers", model_config.get("num_layers", 1)))
    dropout = float(checkpoint.get("dropout", model_config.get("dropout", 0.1)))

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain a model_state_dict mapping")

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_quantum=use_quantum,
    ).to(device)

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print(f"[VERIFY] checkpoint load failed: {exc}")
        raise

    print(f"[VERIFY] Missing keys : {len(missing)}")
    print(f"[VERIFY] Unexpected keys : {len(unexpected)}")
    if missing or unexpected:
        raise RuntimeError(f"Strict checkpoint loading failed for {checkpoint_path}: missing={missing}, unexpected={unexpected}")

    if use_quantum:
        print("[VERIFY] Loaded Hybrid Quantum Model")
        print("[VERIFY] Quantum Enabled : True")
    else:
        print("[VERIFY] Loaded Classical Model")
        print("[VERIFY] Quantum Enabled : False")

    print("[VERIFY] Checkpoint Loaded Successfully")
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark anomaly detection models")
    parser.add_argument(
        "--input",
        "-i",
        default=str(REPO_ROOT / "data" / "evaluation" / "evaluation.csv"),
        help="Path to evaluation CSV file",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        default=[
            str(REPO_ROOT / "train" / "classical_model.pt"),
            str(REPO_ROOT / "train" / "hybrid_model.pt"),
        ],
        help="One or more model checkpoint paths",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(REPO_ROOT / "results"),
        help="Directory for outputs",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (cpu/cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic benchmarking",
    )
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data"),
        help="Directory containing CMAPSS train/test/RUL files",
    )
    parser.add_argument(
        "--dataset",
        default="FD001",
        help="CMAPSS dataset split to evaluate",
    )
    parser.add_argument(
        "--healthy-ratio",
        type=float,
        default=0.7,
        help="Healthy training ratio used to reproduce training input statistics",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Sliding window size used during training",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride used during training",
    )
    parser.add_argument(
        "--rul-threshold",
        type=float,
        default=30.0,
        help="Remaining useful life threshold for anomaly labels",
    )
    return parser.parse_args()


def ensure_default_models(repo_root: Path) -> List[Path]:
    """Return expected default checkpoint paths without creating benchmark stand-ins."""
    train_dir = repo_root / "train"
    hybrid_model_path = train_dir / "hybrid_model.pt"
    classical_model_path = train_dir / "classical_model.pt"

    return [hybrid_model_path, classical_model_path]


def load_feature_names(repo_root: Path, checkpoint: Dict[str, Any], data_dir: str, dataset: str, healthy_ratio: float) -> List[str]:
    stats_path = repo_root / "norm_stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as handle:
            stats = json.load(handle)
        feature_names = stats.get("feature_names")
        if isinstance(feature_names, list) and feature_names:
            return [str(name) for name in feature_names]

    feature_names = checkpoint.get("feature_names")
    if isinstance(feature_names, list) and feature_names:
        return [str(name) for name in feature_names]

    train_df, feature_names = load_and_prepare_data(
        data_dir=data_dir,
        dataset=dataset,
        healthy_ratio=healthy_ratio,
        download=False,
    )
    return list(feature_names)


def build_checkpoint_normalizer(checkpoint: Dict[str, Any], feature_names: List[str]) -> FeatureNormalizer:
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")
    if feature_mean is None or feature_std is None:
        raise RuntimeError("Checkpoint does not contain feature_mean and feature_std normalization statistics")

    if len(feature_mean) != len(feature_names) or len(feature_std) != len(feature_names):
        raise RuntimeError(
            "Normalization statistics do not match feature names: "
            f"{len(feature_mean)} means, {len(feature_std)} stds, {len(feature_names)} names"
        )

    normalizer = FeatureNormalizer()
    normalizer.feature_names = feature_names
    normalizer.scaler.mean_ = np.asarray(feature_mean, dtype=np.float64)
    normalizer.scaler.scale_ = np.asarray(feature_std, dtype=np.float64)
    normalizer.scaler.var_ = normalizer.scaler.scale_ ** 2
    normalizer.scaler.n_features_in_ = len(feature_names)
    normalizer.is_fitted = True
    return normalizer


def verify_feature_alignment(feature_names: List[str], df_feature_names: List[str], checkpoint: Dict[str, Any]) -> None:
    input_dim = int(checkpoint.get("input_dim", checkpoint.get("model_config", {}).get("input_dim", len(feature_names))))
    if feature_names != df_feature_names:
        raise RuntimeError(
            "Evaluation feature order does not match training feature order. "
            f"training={feature_names}, evaluation={df_feature_names}"
        )
    if len(feature_names) != input_dim:
        raise RuntimeError(f"Feature count {len(feature_names)} does not match checkpoint input_dim {input_dim}")

    print("[ALIGNMENT] Feature names and order match training:")
    print(f"[ALIGNMENT] {feature_names}")
    print(f"[ALIGNMENT] Feature count: {len(feature_names)}")


def create_labeled_windows_by_engine(
    df,
    normalized_features: np.ndarray,
    rul_threshold: float,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    all_windows = []
    all_labels = []

    for engine_id in df["engine_id"].unique():
        mask = df["engine_id"] == engine_id
        engine_df = df.loc[mask]
        engine_features = normalized_features[mask]
        if len(engine_features) < window_size:
            continue

        windows = create_sliding_windows(engine_features, window_size=window_size, stride=stride)
        if len(windows) == 0:
            continue

        max_cycle = float(engine_df["cycle"].max())
        final_rul = float(engine_df["final_rul"].iloc[0])
        cycles = engine_df["cycle"].to_numpy(dtype=np.float64)
        rul = (max_cycle - cycles) + final_rul
        end_indices = np.arange(window_size - 1, len(engine_features), stride)
        labels = (rul[end_indices] <= rul_threshold).astype(np.int32)

        if len(labels) != len(windows):
            raise RuntimeError(f"Window/label mismatch for engine {engine_id}: {len(windows)} windows, {len(labels)} labels")

        all_windows.append(windows)
        all_labels.append(labels)

    if not all_windows:
        raise ValueError(f"No evaluation windows created. Check window_size={window_size} and stride={stride}.")

    return np.concatenate(all_windows, axis=0), np.concatenate(all_labels, axis=0)


def prepare_aligned_windows(args: argparse.Namespace, checkpoint: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    feature_names = load_feature_names(REPO_ROOT, checkpoint, args.data_dir, args.dataset, args.healthy_ratio)
    normalizer = build_checkpoint_normalizer(checkpoint, feature_names)

    train_df, train_feature_names = load_and_prepare_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        healthy_ratio=args.healthy_ratio,
        download=False,
    )
    verify_feature_alignment(feature_names, list(train_feature_names), checkpoint)
    train_sorted, train_normalized, _ = preprocess_data(train_df, feature_names, normalizer)
    train_windows, _ = create_labeled_windows_by_engine(
        train_sorted.assign(final_rul=0.0),
        train_normalized,
        rul_threshold=-1.0,
        window_size=args.window_size,
        stride=args.stride,
    )

    test_df, rul_df = load_test_data(data_dir=args.data_dir, dataset=args.dataset)
    test_df = clean_data(test_df)
    test_feature_names = get_feature_columns(test_df)
    verify_feature_alignment(feature_names, list(test_feature_names), checkpoint)

    final_rul_by_engine = {
        engine_id: float(rul)
        for engine_id, rul in zip(sorted(test_df["engine_id"].unique()), rul_df["rul"].to_numpy())
    }
    test_df = test_df.copy()
    test_df["final_rul"] = test_df["engine_id"].map(final_rul_by_engine)
    if test_df["final_rul"].isnull().any():
        raise RuntimeError("Missing final RUL values for one or more evaluation engines")

    eval_sorted, eval_normalized, _ = preprocess_data(test_df, feature_names, normalizer)
    eval_windows, labels = create_labeled_windows_by_engine(
        eval_sorted,
        eval_normalized,
        rul_threshold=args.rul_threshold,
        window_size=args.window_size,
        stride=args.stride,
    )

    stats = {
        "feature_names": feature_names,
        "train_mean": float(np.mean(train_windows)),
        "train_std": float(np.std(train_windows)),
        "eval_mean": float(np.mean(eval_windows)),
        "eval_std": float(np.std(eval_windows)),
        "train_shape": tuple(train_windows.shape),
        "eval_shape": tuple(eval_windows.shape),
        "label_counts": {
            "normal": int(np.sum(labels == 0)),
            "anomaly": int(np.sum(labels == 1)),
        },
    }

    print("[ALIGNMENT] Window size:", args.window_size)
    print("[ALIGNMENT] Training tensor shape:", stats["train_shape"])
    print("[ALIGNMENT] Evaluation tensor shape:", stats["eval_shape"])
    print(f"[ALIGNMENT] Training Input Mean: {stats['train_mean']:.6f}")
    print(f"[ALIGNMENT] Training Input Std: {stats['train_std']:.6f}")
    print(f"[ALIGNMENT] Evaluation Input Mean: {stats['eval_mean']:.6f}")
    print(f"[ALIGNMENT] Evaluation Input Std: {stats['eval_std']:.6f}")
    print(f"[ALIGNMENT] Evaluation labels: Normal : {stats['label_counts']['normal']}, Anomaly : {stats['label_counts']['anomaly']}")
    if abs(stats["train_mean"]) > 0.1 or not 0.5 <= stats["train_std"] <= 2.0:
        print(
            "[ALIGNMENT] WARNING: checkpoint normalization does not produce near-standardized "
            "training windows. This checkpoint may use stale feature statistics or feature order."
        )

    return eval_windows.astype(np.float32), labels.astype(np.int32), stats


def _print_window_diagnostics(
    model_name: str,
    sample_stats: List[Dict[str, float]],
    input_windows: List[np.ndarray],
    output_windows: List[np.ndarray],
) -> None:
    print(f"[DIAGNOSTIC] First {len(sample_stats)} evaluation samples for {model_name}:")
    for index, stats in enumerate(sample_stats, start=1):
        print(
            f"[DIAGNOSTIC] sample {index}: "
            f"input_mean={stats['input_mean']:.6f}, "
            f"input_std={stats['input_std']:.6f}, "
            f"output_mean={stats['output_mean']:.6f}, "
            f"output_std={stats['output_std']:.6f}, "
            f"reconstruction_error={stats['reconstruction_error']:.6f}"
        )

    if len(input_windows) > 1 and all(np.array_equal(input_windows[0], window) for window in input_windows[1:]):
        print("[DIAGNOSTIC] WARNING: first evaluation input windows are identical")
    if len(output_windows) > 1 and all(np.array_equal(output_windows[0], window) for window in output_windows[1:]):
        print("[DIAGNOSTIC] WARNING: first evaluation output windows are identical")


def build_scores(windows: np.ndarray, labels: np.ndarray, model: torch.nn.Module, model_name: str, device: str, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = []
    reconstruction_errors = []
    sample_stats = []
    diagnostic_inputs = []
    diagnostic_outputs = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch_windows = windows[start:start + batch_size]
            tensor = torch.from_numpy(batch_windows.astype(np.float32)).to(device)
            reconstructed, _, _, _ = model(tensor)
            batch_errors = torch.mean((reconstructed - tensor) ** 2, dim=(1, 2)).detach().cpu().numpy()
            reconstruction_errors.extend(batch_errors.tolist())
            scores.extend(batch_errors.tolist())

            for batch_index in range(min(len(batch_windows), max(0, 5 - len(sample_stats)))):
                input_window = tensor[batch_index].detach().cpu().numpy()
                output_window = reconstructed[batch_index].detach().cpu().numpy()
                diagnostic_inputs.append(input_window)
                diagnostic_outputs.append(output_window)
                sample_stats.append({
                    "input_mean": float(np.mean(input_window)),
                    "input_std": float(np.std(input_window)),
                    "output_mean": float(np.mean(output_window)),
                    "output_std": float(np.std(output_window)),
                    "reconstruction_error": float(batch_errors[batch_index]),
                })

    _print_window_diagnostics(model_name, sample_stats, diagnostic_inputs, diagnostic_outputs)

    return np.asarray(scores, dtype=np.float32), labels.astype(np.int32), np.asarray(reconstruction_errors, dtype=np.float32)


def compute_metrics(scores: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    metrics = compute_classification_metrics(
        [
            {"actual_anomaly": bool(label), "predicted_anomaly": bool(pred)}
            for label, pred in zip(labels, predictions)
        ]
    )

    return {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "true_positives": int(metrics["true_positives"]),
        "false_positives": int(metrics["false_positives"]),
        "true_negatives": int(metrics["true_negatives"]),
        "false_negatives": int(metrics["false_negatives"]),
        "total_samples": int(metrics["total_samples"]),
    }


def compute_efficiency_metrics(model_path: Path, model: torch.nn.Module, checkpoint: Dict[str, Any], device: str) -> Dict[str, Any]:
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum_parameters = int(getattr(model, "count_quantum_parameters", lambda: 0)())
    file_size_bytes = os.path.getsize(model_path)

    dummy_input = torch.randn(1, 1, int(checkpoint.get("input_dim", checkpoint.get("model_config", {}).get("input_dim", 16))), device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
        start = time.perf_counter()
        for _ in range(20):
            _ = model(dummy_input)
        elapsed = time.perf_counter() - start

    return {
        "model_path": str(model_path),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "quantum_parameters": int(quantum_parameters),
        "classical_parameters": int(trainable_parameters - quantum_parameters),
        "model_size_bytes": int(file_size_bytes),
        "model_size_mb": round(file_size_bytes / (1024 * 1024), 4),
        "average_inference_time_ms": round((elapsed / 20.0) * 1000.0, 4),
        "device": device,
        "use_quantum": bool(checkpoint.get("use_quantum", checkpoint.get("model_config", {}).get("use_quantum", True))),
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_comparison_tables(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path = output_dir / "comparison_table.csv"
    md_path = output_dir / "comparison_table.md"
    tex_path = output_dir / "comparison_table.tex"

    fields = [
        "model",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1_score",
        "total_parameters",
        "trainable_parameters",
        "quantum_parameters",
        "model_size_mb",
        "average_inference_time_ms",
    ]

    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(fields) + "\n")
        for row in rows:
            values = [str(row.get(field, "")) for field in fields]
            handle.write(",".join(values) + "\n")

    md_lines = ["| model | roc_auc | pr_auc | precision | recall | f1_score | total_parameters | trainable_parameters | quantum_parameters | model_size_mb | average_inference_time_ms |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        md_lines.append(
            f"| {row['model']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {row['total_parameters']:,} | {row['trainable_parameters']:,} | {row['quantum_parameters']:,} | {row['model_size_mb']:.4f} | {row['average_inference_time_ms']:.4f} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lrrrrrrrrrr}",
        "\\toprule",
        "Model & ROC-AUC & PR-AUC & Precision & Recall & F1 & Params & Trainable & Quantum & Size (MB) & Inf. (ms) \\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['model']} & {row['roc_auc']:.4f} & {row['pr_auc']:.4f} & {row['precision']:.4f} & {row['recall']:.4f} & {row['f1_score']:.4f} & {row['total_parameters']:,} & {row['trainable_parameters']:,} & {row['quantum_parameters']:,} & {row['model_size_mb']:.4f} & {row['average_inference_time_ms']:.4f} \\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def save_curves(output_dir: Path, rows: List[Dict[str, Any]], scores_by_model: Dict[str, np.ndarray], labels_by_model: Dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(5, 4))
    for row in rows:
        name = row["model"]
        scores = scores_by_model[name]
        labels = labels_by_model[name]
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 4))
    for row in rows:
        name = row["model"]
        scores = scores_by_model[name]
        labels = labels_by_model[name]
        precision, recall, _ = precision_recall_curve(labels, scores)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=200)
    plt.close()

    parameter_names = [row["model"] for row in rows]
    parameter_values = [row["total_parameters"] for row in rows]
    quantum_values = [row["quantum_parameters"] for row in rows]

    plt.figure(figsize=(5, 4))
    x = np.arange(len(parameter_names))
    width = 0.35
    plt.bar(x - width / 2, parameter_values, width, label="Total Parameters")
    plt.bar(x + width / 2, quantum_values, width, label="Quantum Parameters")
    plt.xticks(x, parameter_names, rotation=20)
    plt.ylabel("Parameter Count")
    plt.title("Parameter Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_comparison.png", dpi=200)
    plt.close()

    training_times = [row["training_time_seconds"] for row in rows]
    plt.figure(figsize=(5, 4))
    plt.bar(parameter_names, training_times, color="tab:orange")
    plt.ylabel("Seconds")
    plt.title("Training Time")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "training_time.png", dpi=200)
    plt.close()


def benchmark_models(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available() and args.device != "cpu":
        device = args.device
    else:
        device = "cpu"

    repo_root = REPO_ROOT
    ensure_default_models(repo_root)

    requested_paths = [Path(path).resolve() for path in args.models]
    resolved_models = []
    for path in requested_paths:
        if path.exists():
            resolved_models.append(path)
        elif path.name == "classical_model.pt" and (repo_root / "train" / "classical_model.pt").exists():
            resolved_models.append(repo_root / "train" / "classical_model.pt")
        elif path.name == "hybrid_model.pt" and (repo_root / "train" / "hybrid_model.pt").exists():
            resolved_models.append(repo_root / "train" / "hybrid_model.pt")
        else:
            resolved_models.append(path)

    print(f"Evaluation labels will be derived from {args.dataset} RUL with threshold <= {args.rul_threshold}")
    print("Benchmark configuration:")
    print(f"  Legacy evaluation CSV: {args.input}")
    print(f"  CMAPSS data directory: {args.data_dir}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Window size: {args.window_size}")
    print(f"  Stride: {args.stride}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Device: {device}")
    print("  Models:")
    for model_path in resolved_models:
        print(f"    - {model_path}")

    rows = []
    scores_by_model = {}
    labels_by_model = {}
    per_model_results = {}
    previous_predictions = None

    for model_path in resolved_models:
        print(f"\nEvaluating {model_path}...")
        print(f"[VERIFY] checkpoint path: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}. "
                "Train the model or pass the correct checkpoint with --models."
            )
        model, checkpoint = load_model_from_checkpoint(Path(model_path), device=device)

        use_quantum = bool(checkpoint.get("use_quantum", checkpoint.get("model_config", {}).get("use_quantum", True)))
        print(f"[VERIFY] use_quantum: {use_quantum}")

        model_name = model_path.stem.replace("_model", "")
        eval_windows, labels, alignment_stats = prepare_aligned_windows(args, checkpoint)
        scores, labels, reconstruction_errors = build_scores(eval_windows, labels, model, model_name, device)
        threshold = float(np.quantile(scores, 0.95)) if len(scores) else 0.0
        predictions = (scores > threshold).astype(int)
        print(f"[VERIFY] first 10 reconstruction errors: {np.array2string(reconstruction_errors[:10], precision=6)}")
        print(f"[VERIFY] threshold used: {threshold:.6f}")
        print(f"[VERIFY] first 10 anomaly scores: {np.array2string(scores[:10], precision=6)}")
        print(f"[DIAGNOSTIC] {model_name} reconstruction error min: {float(np.min(reconstruction_errors)):.6f}")
        print(f"[DIAGNOSTIC] {model_name} reconstruction error max: {float(np.max(reconstruction_errors)):.6f}")
        print(f"[DIAGNOSTIC] {model_name} reconstruction error mean: {float(np.mean(reconstruction_errors)):.6f}")
        print(f"[DIAGNOSTIC] {model_name} reconstruction error std: {float(np.std(reconstruction_errors)):.6f}")
        relative_error_std = float(np.std(reconstruction_errors) / max(abs(float(np.mean(reconstruction_errors))), 1e-12))
        print(f"[DIAGNOSTIC] {model_name} reconstruction error relative std: {relative_error_std:.12f}")
        unique_predictions, prediction_counts = np.unique(predictions, return_counts=True)
        prediction_count_map = {int(value): int(count) for value, count in zip(unique_predictions, prediction_counts)}
        normal_count = prediction_count_map.get(0, 0)
        anomaly_count = prediction_count_map.get(1, 0)
        print(f"[DIAGNOSTIC] {model_name} unique prediction values: {unique_predictions.tolist()}")
        print(f"[DIAGNOSTIC] {model_name} prediction counts: Normal : {normal_count}, Anomaly : {anomaly_count}")
        if np.array_equal(reconstruction_errors, np.full_like(reconstruction_errors, reconstruction_errors[0])):
            print(f"[DIAGNOSTIC] WARNING: {model_name} reconstruction errors are constant")
        elif relative_error_std < 1e-6:
            print(f"[DIAGNOSTIC] WARNING: {model_name} reconstruction errors are near-constant")
        if previous_predictions is None:
            print("[VERIFY] prediction arrays differ: no previous model to compare yet")
        else:
            identical = np.array_equal(predictions, previous_predictions)
            if identical:
                print("[VERIFY] WARNING: prediction arrays are identical to the previous model")
            else:
                print("[VERIFY] prediction arrays differ from the previous model")
        previous_predictions = predictions

        metrics = compute_metrics(scores, labels, predictions)
        efficiency = compute_efficiency_metrics(model_path, model, checkpoint, device)

        training_time_seconds = 0.0
        training_metadata_files = [
            repo_root / "results" / "training_metrics.json",
            repo_root / "train" / "hybrid_model_metadata.json",
            repo_root / "results" / "training_summary.json",
        ]
        for candidate in training_metadata_files:
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as handle:
                        metadata = json.load(handle)
                    if isinstance(metadata, dict):
                        training_time_seconds = float(metadata.get("training_time_seconds", metadata.get("total_duration_seconds", 0.0)))
                    break
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

        row = {
            "model": model_name,
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "total_parameters": efficiency["total_parameters"],
            "trainable_parameters": efficiency["trainable_parameters"],
            "quantum_parameters": efficiency["quantum_parameters"],
            "model_size_mb": efficiency["model_size_mb"],
            "average_inference_time_ms": efficiency["average_inference_time_ms"],
            "training_time_seconds": round(training_time_seconds, 4),
        }
        rows.append(row)
        scores_by_model[model_name] = scores
        labels_by_model[model_name] = labels

        per_model_results[model_name] = {
            "model_path": str(model_path),
            "metrics": metrics,
            "efficiency": efficiency,
            "threshold": threshold,
            "score_distribution": {
                "mean": round(float(np.mean(scores)), 6),
                "std": round(float(np.std(scores)), 6),
                "p95": round(float(np.quantile(scores, 0.95)), 6),
            },
            "input_alignment": alignment_stats,
        }

        independent_result_path = Path(args.output_dir) / f"{model_name}_results.json"
        save_json(independent_result_path, per_model_results[model_name])

    save_json(Path(args.output_dir) / "results.json", {
        "benchmark_metadata": {
            "input": str(Path(args.input).resolve()),
            "device": device,
            "seed": args.seed,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "models": per_model_results,
    })
    save_json(Path(args.output_dir) / "efficiency_metrics.json", {"models": per_model_results})
    save_comparison_tables(Path(args.output_dir), rows)
    save_curves(Path(args.output_dir), rows, scores_by_model, labels_by_model)

    print("\nSaved benchmark artifacts:")
    for artifact in [
        Path(args.output_dir) / "results.json",
        Path(args.output_dir) / "efficiency_metrics.json",
        Path(args.output_dir) / "comparison_table.csv",
        Path(args.output_dir) / "comparison_table.md",
        Path(args.output_dir) / "comparison_table.tex",
        Path(args.output_dir) / "roc_curve.png",
        Path(args.output_dir) / "pr_curve.png",
        Path(args.output_dir) / "parameter_comparison.png",
        Path(args.output_dir) / "training_time.png",
    ]:
        print(f"  - {artifact}")

    return {
        "output_dir": str(Path(args.output_dir).resolve()),
        "comparison_rows": rows,
    }


def main() -> int:
    args = parse_args()
    try:
        benchmark_models(args)
    except Exception as exc:  # pragma: no cover - keep CLI robust
        print(f"Benchmark failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
