#!/usr/bin/env python3
"""Final scientific benchmark for the classical and hybrid anomaly models."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from evaluation.metrics import compute_classification_metrics
except ImportError:  # pragma: no cover
    from metrics import compute_classification_metrics

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


OFFICIAL_MODELS = {
    "classical": REPO_ROOT / "train" / "classical_model.pt",
    "hybrid": REPO_ROOT / "train" / "hybrid_model.pt",
}

OFFICIAL_METADATA = {
    "classical": REPO_ROOT / "train" / "classical_model_metadata.json",
    "hybrid": REPO_ROOT / "train" / "hybrid_model_metadata.json",
}

COMPATIBILITY_FIELDS = [
    "dataset",
    "feature_names",
    "window_size",
    "hidden_dim",
    "latent_dim",
    "batch_size",
    "learning_rate",
    "random_seed",
]

DEFAULT_COMPATIBILITY_VALUES = {
    "batch_size": 8,
    "learning_rate": 0.001,
    "random_seed": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final classical vs hybrid scientific benchmark")
    parser.add_argument("--output-dir", "-o", default=str(REPO_ROOT / "results"), help="Directory for benchmark outputs")
    parser.add_argument("--device", default="cpu", help="Device for inference: cpu or cuda")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic benchmark seed")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"), help="CMAPSS data directory")
    parser.add_argument("--dataset", default="FD001", help="CMAPSS split")
    parser.add_argument("--healthy-ratio", type=float, default=0.7, help="Training healthy ratio used for feature discovery")
    parser.add_argument("--window-size", type=int, default=50, help="Evaluation window size")
    parser.add_argument("--stride", type=int, default=1, help="Evaluation window stride")
    parser.add_argument("--rul-threshold", type=float, default=30.0, help="RUL threshold for anomaly labels")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_model_from_checkpoint(checkpoint_path: Path, device: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a checkpoint and abort unless strict state-dict verification is exact."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Required checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config") or {}
    if not isinstance(model_config, dict):
        raise TypeError(f"Checkpoint {checkpoint_path} has invalid model_config metadata")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain model_state_dict")

    use_quantum = bool(checkpoint.get("use_quantum", model_config.get("use_quantum", False)))
    model = LSTMAutoencoder(
        input_dim=int(checkpoint.get("input_dim", model_config.get("input_dim", 16))),
        hidden_dim=int(checkpoint.get("hidden_dim", model_config.get("hidden_dim", 64))),
        latent_dim=int(checkpoint.get("latent_dim", model_config.get("latent_dim", 32))),
        num_layers=int(checkpoint.get("num_layers", model_config.get("num_layers", 1))),
        dropout=float(checkpoint.get("dropout", model_config.get("dropout", 0.1))),
        use_quantum=use_quantum,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"[VERIFY] {checkpoint_path}")
    print(f"[VERIFY] Missing keys : {len(missing)}")
    print(f"[VERIFY] Unexpected keys : {len(unexpected)}")
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint verification failed for {checkpoint_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )

    model.eval()
    return model, checkpoint


def load_training_feature_names(args: argparse.Namespace) -> List[str]:
    norm_candidates = [
        REPO_ROOT / "norm_stats.json",
        REPO_ROOT / "train" / "norm_stats.json",
    ]
    for path in norm_candidates:
        payload = read_json(path)
        feature_names = payload.get("feature_names")
        if isinstance(feature_names, list) and feature_names:
            return [str(name) for name in feature_names]

    _, feature_names = load_and_prepare_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        healthy_ratio=args.healthy_ratio,
        download=False,
    )
    return [str(name) for name in feature_names]


def checkpoint_value(checkpoint: Dict[str, Any], key: str) -> Any:
    model_config = checkpoint.get("model_config") or {}
    if key in checkpoint:
        return checkpoint[key]
    if isinstance(model_config, dict) and key in model_config:
        return model_config[key]
    return None


def build_effective_metadata(
    model_name: str,
    checkpoint: Dict[str, Any],
    args: argparse.Namespace,
    feature_names: List[str],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Build compatibility metadata from model JSON, checkpoint fields, and benchmark defaults."""
    raw_metadata = read_json(OFFICIAL_METADATA[model_name])
    effective: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    fallback_values = {
        "dataset": args.dataset,
        "feature_names": feature_names,
        "window_size": args.window_size,
        "hidden_dim": checkpoint_value(checkpoint, "hidden_dim"),
        "latent_dim": checkpoint_value(checkpoint, "latent_dim"),
        **DEFAULT_COMPATIBILITY_VALUES,
    }

    for field in COMPATIBILITY_FIELDS:
        if field in raw_metadata:
            effective[field] = raw_metadata[field]
            sources[field] = "metadata"
        elif checkpoint_value(checkpoint, field) is not None:
            effective[field] = checkpoint_value(checkpoint, field)
            sources[field] = "checkpoint"
        else:
            effective[field] = fallback_values[field]
            sources[field] = "benchmark_default"

    return effective, sources


def print_compatibility_report(metadata_by_model: Dict[str, Dict[str, Any]], sources_by_model: Dict[str, Dict[str, str]]) -> None:
    print("\nMetadata Compatibility Report")
    print("-" * 60)
    reference_name = "classical"
    reference = metadata_by_model[reference_name]
    compatible = True

    for field in COMPATIBILITY_FIELDS:
        classical_value = reference[field]
        hybrid_value = metadata_by_model["hybrid"][field]
        matches = classical_value == hybrid_value
        compatible = compatible and matches
        status = "OK" if matches else "MISMATCH"
        print(f"{field}: {status}")
        print(f"  classical ({sources_by_model['classical'][field]}): {classical_value}")
        print(f"  hybrid    ({sources_by_model['hybrid'][field]}): {hybrid_value}")

    if not compatible:
        raise RuntimeError("Metadata compatibility check failed. Benchmark aborted.")

    print("Compatibility: PASS")


def build_checkpoint_normalizer(checkpoint: Dict[str, Any], feature_names: List[str]) -> FeatureNormalizer:
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")
    if feature_mean is None or feature_std is None:
        raise RuntimeError("Checkpoint does not contain feature_mean and feature_std")
    if len(feature_mean) != len(feature_names) or len(feature_std) != len(feature_names):
        raise RuntimeError(
            "Checkpoint normalization statistics do not match feature names: "
            f"{len(feature_mean)} means, {len(feature_std)} stds, {len(feature_names)} features"
        )

    normalizer = FeatureNormalizer()
    normalizer.feature_names = feature_names
    normalizer.scaler.mean_ = np.asarray(feature_mean, dtype=np.float64)
    normalizer.scaler.scale_ = np.asarray(feature_std, dtype=np.float64)
    normalizer.scaler.var_ = normalizer.scaler.scale_ ** 2
    normalizer.scaler.n_features_in_ = len(feature_names)
    normalizer.is_fitted = True
    return normalizer


def create_labeled_windows_by_engine(
    df,
    normalized_features: np.ndarray,
    rul_threshold: float,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    windows_by_engine = []
    labels_by_engine = []

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

        windows_by_engine.append(windows)
        labels_by_engine.append(labels)

    if not windows_by_engine:
        raise ValueError(f"No evaluation windows created with window_size={window_size}, stride={stride}")

    return np.concatenate(windows_by_engine, axis=0), np.concatenate(labels_by_engine, axis=0)


def load_evaluation_frame(args: argparse.Namespace, feature_names: List[str]) -> Tuple[Any, np.ndarray]:
    train_df, train_feature_names = load_and_prepare_data(
        data_dir=args.data_dir,
        dataset=args.dataset,
        healthy_ratio=args.healthy_ratio,
        download=False,
    )
    if list(train_feature_names) != feature_names:
        raise RuntimeError(
            "Training feature order does not match metadata: "
            f"metadata={feature_names}, training={list(train_feature_names)}"
        )

    test_df, rul_df = load_test_data(data_dir=args.data_dir, dataset=args.dataset)
    test_df = clean_data(test_df)
    test_feature_names = get_feature_columns(test_df)
    if list(test_feature_names) != feature_names:
        raise RuntimeError(
            "Evaluation feature order does not match metadata: "
            f"metadata={feature_names}, evaluation={list(test_feature_names)}"
        )

    final_rul_by_engine = {
        engine_id: float(rul)
        for engine_id, rul in zip(sorted(test_df["engine_id"].unique()), rul_df["rul"].to_numpy())
    }
    test_df = test_df.copy()
    test_df["final_rul"] = test_df["engine_id"].map(final_rul_by_engine)
    if test_df["final_rul"].isnull().any():
        raise RuntimeError("Missing final RUL values for one or more evaluation engines")

    # Labels are independent of model normalization, so compute them once with a dummy normalization view.
    sorted_df = test_df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    dummy_features = sorted_df[feature_names].to_numpy(dtype=np.float32)
    _, labels = create_labeled_windows_by_engine(
        sorted_df,
        dummy_features,
        rul_threshold=args.rul_threshold,
        window_size=args.window_size,
        stride=args.stride,
    )
    return test_df, labels.astype(np.int32)


def create_model_windows(
    eval_df,
    labels: np.ndarray,
    checkpoint: Dict[str, Any],
    feature_names: List[str],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    normalizer = build_checkpoint_normalizer(checkpoint, feature_names)
    sorted_df, normalized_features, _ = preprocess_data(eval_df, feature_names, normalizer)
    windows, model_labels = create_labeled_windows_by_engine(
        sorted_df,
        normalized_features,
        rul_threshold=args.rul_threshold,
        window_size=args.window_size,
        stride=args.stride,
    )
    model_labels = model_labels.astype(np.int32)
    if not np.array_equal(labels, model_labels):
        raise RuntimeError("Evaluation labels changed across model preprocessing")
    return windows.astype(np.float32), model_labels


def load_threshold(model_name: str, checkpoint: Dict[str, Any]) -> Tuple[float, str]:
    checkpoint_stats = checkpoint.get("global_stats")
    if isinstance(checkpoint_stats, dict):
        percentiles = checkpoint_stats.get("percentiles") or {}
        if "p95" in percentiles:
            return float(percentiles["p95"]), "checkpoint.global_stats.percentiles.p95"
        if "global_mean" in checkpoint_stats and "global_std" in checkpoint_stats:
            return float(checkpoint_stats["global_mean"]) + 2.0 * float(checkpoint_stats["global_std"]), "checkpoint.global_stats.mean_plus_2std"

    # Frozen benchmark calibration policy:
    # Classical and hybrid models intentionally use separate training-error
    # calibration files. Do not recompute or merge these thresholds unless
    # creating a new benchmark version.
    candidates = {
        "classical": [
            REPO_ROOT / "global_stats.json",
            REPO_ROOT / "train" / "classical_global_stats.json",
            REPO_ROOT / "train" / "global_stats_classical.json",
        ],
        "hybrid": [
            # Compatibility path used by frozen benchmark outputs. The clearer
            # provenance copy is train/hybrid_global_stats.json with identical
            # numeric statistics and added metadata.
            REPO_ROOT / "train" / "global_stats.json",
            REPO_ROOT / "train" / "hybrid_global_stats.json",
            REPO_ROOT / "train" / "global_stats_hybrid.json",
        ],
    }[model_name]

    for path in candidates:
        stats = read_json(path)
        if not stats:
            continue
        percentiles = stats.get("percentiles") or {}
        if "p95" in percentiles:
            return float(percentiles["p95"]), f"{path}.percentiles.p95"
        if "global_mean" in stats and "global_std" in stats:
            return float(stats["global_mean"]) + 2.0 * float(stats["global_std"]), f"{path}.mean_plus_2std"

    raise RuntimeError(f"No training threshold source found for {model_name}")


def score_model(
    model: nn.Module,
    windows: np.ndarray,
    labels: np.ndarray,
    device: str,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    scores: List[float] = []
    model.eval()

    if len(windows) == 0:
        raise ValueError("Cannot score an empty evaluation set")

    with torch.no_grad():
        warmup_tensor = torch.from_numpy(windows[: min(batch_size, len(windows))]).to(device)
        torch.manual_seed(seed)
        _ = model(warmup_tensor)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        torch.manual_seed(seed)
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size]).to(device)
            reconstructed, _, _, _ = model(batch)
            errors = torch.mean((reconstructed - batch) ** 2, dim=(1, 2))
            scores.extend(errors.detach().cpu().numpy().tolist())
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

    average_inference_time_ms = (elapsed / float(len(windows))) * 1000.0
    return np.asarray(scores, dtype=np.float64), labels.astype(np.int32), average_inference_time_ms


def compute_metrics(scores: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    fpr, tpr, _ = roc_curve(labels, scores)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    classification = compute_classification_metrics(
        [
            {"actual_anomaly": bool(label), "predicted_anomaly": bool(prediction)}
            for label, prediction in zip(labels, predictions)
        ]
    )

    return {
        "roc_auc": float(auc(fpr, tpr)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "precision": float(classification["precision"]),
        "recall": float(classification["recall"]),
        "f1": float(classification["f1_score"]),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "matrix": matrix.astype(int).tolist(),
        },
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_curve": {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
        },
    }


def compute_model_stats(model_path: Path, model: nn.Module, metadata: Dict[str, Any], average_inference_time_ms: float) -> Dict[str, Any]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    quantum_parameters = int(getattr(model, "count_quantum_parameters", lambda: 0)())
    file_size_bytes = os.path.getsize(model_path)

    return {
        "training_time_seconds": float(metadata.get("training_time_seconds", 0.0)),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "quantum_parameters": int(quantum_parameters),
        "model_size_bytes": int(file_size_bytes),
        "model_size_mb": float(file_size_bytes / (1024 * 1024)),
        "average_inference_time_ms": float(average_inference_time_ms),
    }


def format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def save_comparison_tables(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "model",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "average_inference_time_ms",
        "training_time_seconds",
        "total_parameters",
        "quantum_parameters",
        "model_size_mb",
    ]

    with open(output_dir / "comparison_table.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})

    md_lines = [
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | TN | FP | FN | TP | Avg Inference (ms) | Training (s) | Total Params | Quantum Params | Size (MB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| {model} | {roc_auc:.6f} | {pr_auc:.6f} | {precision:.6f} | {recall:.6f} | {f1:.6f} | "
            "{true_negatives} | {false_positives} | {false_negatives} | {true_positives} | "
            "{average_inference_time_ms:.6f} | {training_time_seconds:.6f} | {total_parameters} | "
            "{quantum_parameters} | {model_size_mb:.6f} |".format(**row)
        )
    (output_dir / "comparison_table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lrrrrrrrrrrrrrr}",
        "\\toprule",
        "Model & ROC-AUC & PR-AUC & Precision & Recall & F1 & TN & FP & FN & TP & Inf. ms & Train s & Params & Q Params & MB \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(
            "{model} & {roc_auc:.6f} & {pr_auc:.6f} & {precision:.6f} & {recall:.6f} & {f1:.6f} & "
            "{true_negatives} & {false_positives} & {false_negatives} & {true_positives} & "
            "{average_inference_time_ms:.6f} & {training_time_seconds:.6f} & {total_parameters} & "
            "{quantum_parameters} & {model_size_mb:.6f} \\\\".format(**row)
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (output_dir / "comparison_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def save_curves(output_dir: Path, results: Dict[str, Dict[str, Any]]) -> None:
    plt.figure(figsize=(6, 4))
    for model_name, result in results.items():
        curve = result["metrics"]["roc_curve"]
        plt.plot(curve["fpr"], curve["tpr"], linewidth=2, label=f"{result['display_name']} ({result['metrics']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    for model_name, result in results.items():
        curve = result["metrics"]["pr_curve"]
        plt.plot(curve["recall"], curve["precision"], linewidth=2, label=f"{result['display_name']} ({result['metrics']['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=300)
    plt.close()

    names = [result["display_name"] for result in results.values()]
    total_parameters = [result["efficiency"]["total_parameters"] for result in results.values()]
    quantum_parameters = [result["efficiency"]["quantum_parameters"] for result in results.values()]
    x_positions = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x_positions - width / 2, total_parameters, width, label="Total")
    plt.bar(x_positions + width / 2, quantum_parameters, width, label="Quantum")
    plt.xticks(x_positions, names)
    plt.ylabel("Parameters")
    plt.title("Parameter Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_comparison.png", dpi=300)
    plt.close()

    training_times = [result["efficiency"]["training_time_seconds"] for result in results.values()]
    plt.figure(figsize=(6, 4))
    plt.bar(names, training_times, color=["tab:blue", "tab:orange"])
    plt.ylabel("Seconds")
    plt.title("Training Time")
    plt.tight_layout()
    plt.savefig(output_dir / "training_time.png", dpi=300)
    plt.close()

    for model_name, result in results.items():
        matrix = np.asarray(result["metrics"]["confusion_matrix"]["matrix"], dtype=int)
        plt.figure(figsize=(4, 4))
        plt.imshow(matrix, cmap="Blues")
        plt.title(f"Confusion Matrix: {result['display_name']}")
        plt.xticks([0, 1], ["Normal", "Anomaly"])
        plt.yticks([0, 1], ["Normal", "Anomaly"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        for row in range(2):
            for col in range(2):
                plt.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(output_dir / f"confusion_matrix_{model_name}.png", dpi=300)
        plt.close()


def print_benchmark_summary(results: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 30)
    for model_name in ["classical", "hybrid"]:
        result = results[model_name]
        metrics = result["metrics"]
        efficiency = result["efficiency"]
        print(result["display_name"])
        print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
        print(f"PR-AUC: {metrics['pr_auc']:.6f}")
        print(f"F1: {metrics['f1']:.6f}")
        print(f"Inference Time: {efficiency['average_inference_time_ms']:.6f} ms")
        print(f"Training Time: {efficiency['training_time_seconds']:.6f} s")
        print(f"Parameters: {efficiency['total_parameters']}")
        if model_name == "classical":
            print("-" * 30)
    print("=" * 30)


def print_comparison_summary(results: Dict[str, Dict[str, Any]]) -> None:
    classical = results["classical"]
    hybrid = results["hybrid"]

    roc_delta = hybrid["metrics"]["roc_auc"] - classical["metrics"]["roc_auc"]
    pr_delta = hybrid["metrics"]["pr_auc"] - classical["metrics"]["pr_auc"]
    training_delta = hybrid["efficiency"]["training_time_seconds"] - classical["efficiency"]["training_time_seconds"]
    inference_delta = hybrid["efficiency"]["average_inference_time_ms"] - classical["efficiency"]["average_inference_time_ms"]
    parameter_delta = hybrid["efficiency"]["total_parameters"] - classical["efficiency"]["total_parameters"]

    print("\nComparison Summary")
    print(f"Hybrid improves ROC by {roc_delta:.6f}")
    print(f"Hybrid improves PR by {pr_delta:.6f}")
    print(f"Training overhead {training_delta:.6f} s")
    print(f"Inference overhead {inference_delta:.6f} ms")
    print(f"Parameter increase {parameter_delta}")


def benchmark_models(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"

    print("Official benchmark checkpoints:")
    for model_name, path in OFFICIAL_MODELS.items():
        print(f"  {model_name}: {path}")

    models: Dict[str, nn.Module] = {}
    checkpoints: Dict[str, Dict[str, Any]] = {}
    metadata_json: Dict[str, Dict[str, Any]] = {}

    for model_name, path in OFFICIAL_MODELS.items():
        model, checkpoint = load_model_from_checkpoint(path, device)
        models[model_name] = model
        checkpoints[model_name] = checkpoint
        metadata_json[model_name] = read_json(OFFICIAL_METADATA[model_name])

    feature_names = load_training_feature_names(args)
    effective_metadata = {}
    metadata_sources = {}
    for model_name, checkpoint in checkpoints.items():
        effective_metadata[model_name], metadata_sources[model_name] = build_effective_metadata(
            model_name,
            checkpoint,
            args,
            feature_names,
        )
    print_compatibility_report(effective_metadata, metadata_sources)

    eval_df, labels = load_evaluation_frame(args, feature_names)
    print("\nEvaluation Dataset")
    print(f"Dataset: {args.dataset}")
    print(f"RUL anomaly threshold: <= {args.rul_threshold}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Labels: normal={int(np.sum(labels == 0))}, anomaly={int(np.sum(labels == 1))}")

    results: Dict[str, Dict[str, Any]] = {}
    predictions_by_model: Dict[str, np.ndarray] = {}
    display_names = {
        "classical": "Classical",
        "hybrid": "Hybrid Quantum",
    }

    for model_name in ["classical", "hybrid"]:
        print(f"\nEvaluating {display_names[model_name]}...")
        windows, model_labels = create_model_windows(eval_df, labels, checkpoints[model_name], feature_names, args)
        threshold, threshold_source = load_threshold(model_name, checkpoints[model_name])
        scores, model_labels, average_inference_time_ms = score_model(
            models[model_name],
            windows,
            model_labels,
            device,
            args.batch_size,
            args.seed,
        )
        predictions = (scores > threshold).astype(np.int32)
        metrics = compute_metrics(scores, model_labels, predictions)
        efficiency = compute_model_stats(
            OFFICIAL_MODELS[model_name],
            models[model_name],
            metadata_json[model_name],
            average_inference_time_ms,
        )

        predictions_by_model[model_name] = predictions
        results[model_name] = {
            "display_name": display_names[model_name],
            "model_path": str(OFFICIAL_MODELS[model_name]),
            "metadata_path": str(OFFICIAL_METADATA[model_name]),
            "metadata": effective_metadata[model_name],
            "metadata_sources": metadata_sources[model_name],
            "threshold": threshold,
            "threshold_source": threshold_source,
            "metrics": metrics,
            "efficiency": efficiency,
            "score_distribution": {
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            },
            "num_evaluation_windows": int(len(model_labels)),
        }

        print(f"Threshold: {threshold:.6f} ({threshold_source})")
        print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
        print(f"PR-AUC: {metrics['pr_auc']:.6f}")
        print(f"F1: {metrics['f1']:.6f}")

    if np.array_equal(predictions_by_model["classical"], predictions_by_model["hybrid"]):
        raise RuntimeError("Prediction arrays are identical. Benchmark aborted.")
    print("\nPrediction arrays differ: PASS")

    rows = []
    for model_name in ["classical", "hybrid"]:
        metrics = results[model_name]["metrics"]
        efficiency = results[model_name]["efficiency"]
        confusion = metrics["confusion_matrix"]
        rows.append({
            "model": results[model_name]["display_name"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "true_negatives": confusion["true_negatives"],
            "false_positives": confusion["false_positives"],
            "false_negatives": confusion["false_negatives"],
            "true_positives": confusion["true_positives"],
            "average_inference_time_ms": efficiency["average_inference_time_ms"],
            "training_time_seconds": efficiency["training_time_seconds"],
            "total_parameters": efficiency["total_parameters"],
            "quantum_parameters": efficiency["quantum_parameters"],
            "model_size_mb": efficiency["model_size_mb"],
        })

    benchmark_payload = {
        "benchmark_metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": device,
            "seed": args.seed,
            "dataset": args.dataset,
            "window_size": args.window_size,
            "stride": args.stride,
            "rul_threshold": args.rul_threshold,
            "checkpoints": {name: str(path) for name, path in OFFICIAL_MODELS.items()},
        },
        "metadata_compatibility": {
            "fields": COMPATIBILITY_FIELDS,
            "models": effective_metadata,
            "sources": metadata_sources,
        },
        "models": results,
    }
    save_json(output_dir / "results.json", benchmark_payload)
    save_json(output_dir / "efficiency_metrics.json", {
        "models": {
            model_name: results[model_name]["efficiency"]
            for model_name in ["classical", "hybrid"]
        }
    })
    save_comparison_tables(output_dir, rows)
    save_curves(output_dir, results)

    print("\nSaved benchmark artifacts:")
    for artifact in [
        "results.json",
        "comparison_table.csv",
        "comparison_table.md",
        "comparison_table.tex",
        "efficiency_metrics.json",
        "roc_curve.png",
        "pr_curve.png",
        "parameter_comparison.png",
        "training_time.png",
        "confusion_matrix_classical.png",
        "confusion_matrix_hybrid.png",
    ]:
        print(f"  - {output_dir / artifact}")

    print_benchmark_summary(results)
    print_comparison_summary(results)
    return benchmark_payload


def main() -> int:
    args = parse_args()
    try:
        benchmark_models(args)
    except Exception as exc:  # pragma: no cover
        print(f"Benchmark failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
