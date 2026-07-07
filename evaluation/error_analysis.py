#!/usr/bin/env python3
"""Error analysis artifacts for the frozen classical and hybrid benchmarks."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from evaluation import benchmark
except ImportError:  # pragma: no cover
    import benchmark


MODEL_ORDER = ["classical", "hybrid"]
DISPLAY_NAMES = {
    "classical": "Classical",
    "hybrid": "Hybrid",
}
SAMPLE_FIELDS = [
    "sample_id",
    "ground_truth",
    "prediction",
    "anomaly_score",
    "reconstruction_error",
    "classification",
]
COMPARISON_FIELDS = [
    "sample_id",
    "ground_truth",
    "classical_prediction",
    "hybrid_prediction",
    "classical_score",
    "hybrid_score",
    "classical_classification",
    "hybrid_classification",
]
STAT_FIELDS = [
    "model",
    "true_positives",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "false_positive_rate",
    "false_negative_rate",
    "balanced_accuracy",
    "mcc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate error analysis for frozen benchmark results")
    parser.add_argument("--benchmark-results", default=None, help="Path to frozen benchmark results.json")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "error_analysis"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def find_benchmark_results(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark results not found: {path}")
        return path

    candidates = [
        REPO_ROOT / "results" / "FinalResults" / "results.json",
        REPO_ROOT / "results" / "final_benchmark" / "results.json",
        REPO_ROOT / "results" / "results.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No frozen benchmark results.json found")


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def make_benchmark_args(payload: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    metadata = payload.get("benchmark_metadata") or {}
    return argparse.Namespace(
        output_dir=str(Path(args.output_dir) / "_unused"),
        device=args.device,
        seed=int(metadata.get("seed", 42)),
        data_dir=str(REPO_ROOT / "data"),
        dataset=str(metadata.get("dataset", "FD001")),
        healthy_ratio=0.7,
        window_size=int(metadata.get("window_size", 50)),
        stride=int(metadata.get("stride", 1)),
        rul_threshold=float(metadata.get("rul_threshold", 30.0)),
        batch_size=args.batch_size,
    )


def verify_frozen_configuration(payload: Dict[str, Any], run_args: argparse.Namespace) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    models = {}
    checkpoints = {}
    for model_name in MODEL_ORDER:
        model, checkpoint = benchmark.load_model_from_checkpoint(benchmark.OFFICIAL_MODELS[model_name], run_args.device)
        models[model_name] = model
        checkpoints[model_name] = checkpoint

    feature_names = benchmark.load_training_feature_names(run_args)
    metadata_by_model = {}
    sources_by_model = {}
    for model_name in MODEL_ORDER:
        metadata_by_model[model_name], sources_by_model[model_name] = benchmark.build_effective_metadata(
            model_name,
            checkpoints[model_name],
            run_args,
            feature_names,
        )

    benchmark.print_compatibility_report(metadata_by_model, sources_by_model)
    frozen_metadata = payload.get("metadata_compatibility", {}).get("models", {})
    for model_name in MODEL_ORDER:
        if frozen_metadata and metadata_by_model[model_name] != frozen_metadata.get(model_name):
            raise RuntimeError(f"Current {model_name} metadata differs from frozen benchmark results")

    reference_mean = np.asarray(checkpoints["classical"]["feature_mean"], dtype=np.float64)
    reference_std = np.asarray(checkpoints["classical"]["feature_std"], dtype=np.float64)
    for model_name in MODEL_ORDER:
        mean = np.asarray(checkpoints[model_name]["feature_mean"], dtype=np.float64)
        std = np.asarray(checkpoints[model_name]["feature_std"], dtype=np.float64)
        if len(mean) != len(feature_names) or len(std) != len(feature_names):
            raise RuntimeError(f"{model_name} normalization length does not match feature names")
        if not np.allclose(mean, reference_mean) or not np.allclose(std, reference_std):
            raise RuntimeError("Checkpoint normalization differs between models")

    return feature_names, models, checkpoints


def classify_sample(ground_truth: int, prediction: int) -> str:
    if ground_truth == 1 and prediction == 1:
        return "TP"
    if ground_truth == 0 and prediction == 0:
        return "TN"
    if ground_truth == 0 and prediction == 1:
        return "FP"
    return "FN"


def compute_error_statistics(sample_rows: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for row in sample_rows:
        counts[row["classification"]] += 1

    tp = counts["TP"]
    tn = counts["TN"]
    fp = counts["FP"]
    fn = counts["FN"]
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0

    return {
        "model": model_name,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }


def build_sample_rows(labels: np.ndarray, scores: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
    predictions = (scores > threshold).astype(np.int32)
    rows = []
    for sample_id, (ground_truth, prediction, score) in enumerate(zip(labels, predictions, scores)):
        ground_truth_int = int(ground_truth)
        prediction_int = int(prediction)
        score_float = float(score)
        rows.append({
            "sample_id": sample_id,
            "ground_truth": ground_truth_int,
            "prediction": prediction_int,
            "anomaly_score": score_float,
            "reconstruction_error": score_float,
            "classification": classify_sample(ground_truth_int, prediction_int),
        })
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def compare_models(classical_rows: List[Dict[str, Any]], hybrid_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    corrected_by_hybrid = []
    corrected_by_classical = []
    for classical, hybrid in zip(classical_rows, hybrid_rows):
        classical_correct = classical["ground_truth"] == classical["prediction"]
        hybrid_correct = hybrid["ground_truth"] == hybrid["prediction"]
        comparison = {
            "sample_id": classical["sample_id"],
            "ground_truth": classical["ground_truth"],
            "classical_prediction": classical["prediction"],
            "hybrid_prediction": hybrid["prediction"],
            "classical_score": classical["anomaly_score"],
            "hybrid_score": hybrid["anomaly_score"],
            "classical_classification": classical["classification"],
            "hybrid_classification": hybrid["classification"],
        }
        if hybrid_correct and not classical_correct:
            corrected_by_hybrid.append(comparison)
        elif classical_correct and not hybrid_correct:
            corrected_by_classical.append(comparison)
    return corrected_by_hybrid, corrected_by_classical


def save_score_distribution(output_dir: Path, model_name: str, rows: List[Dict[str, Any]]) -> None:
    normal_scores = [row["anomaly_score"] for row in rows if row["ground_truth"] == 0]
    anomaly_scores = [row["anomaly_score"] for row in rows if row["ground_truth"] == 1]
    plt.figure(figsize=(6, 4))
    plt.hist(normal_scores, bins=50, alpha=0.65, label="Normal", density=True)
    plt.hist(anomaly_scores, bins=50, alpha=0.65, label="Anomaly", density=True)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.title(f"Score Distribution: {DISPLAY_NAMES[model_name]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"score_distribution_{model_name}.png", dpi=300)
    plt.close()


def save_confusion_matrix(output_dir: Path, model_name: str, stats: Dict[str, Any]) -> None:
    matrix = np.asarray(
        [
            [stats["true_negatives"], stats["false_positives"]],
            [stats["false_negatives"], stats["true_positives"]],
        ],
        dtype=int,
    )
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"Confusion Matrix: {DISPLAY_NAMES[model_name]}")
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


def write_markdown_summary(
    output_dir: Path,
    benchmark_path: Path,
    statistics_rows: List[Dict[str, Any]],
    corrected_by_hybrid: List[Dict[str, Any]],
    corrected_by_classical: List[Dict[str, Any]],
) -> None:
    stats_by_model = {row["model"]: row for row in statistics_rows}
    lines = [
        "# Error Analysis Summary",
        "",
        f"Frozen benchmark results: `{benchmark_path}`",
        "",
        "## Error Counts",
        "",
        "| Model | TP | TN | FP | FN | FPR | FNR | Balanced Accuracy | MCC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in MODEL_ORDER:
        display = DISPLAY_NAMES[model_name]
        stats = stats_by_model[display]
        lines.append(
            f"| {display} | {stats['true_positives']} | {stats['true_negatives']} | "
            f"{stats['false_positives']} | {stats['false_negatives']} | "
            f"{stats['false_positive_rate']:.6f} | {stats['false_negative_rate']:.6f} | "
            f"{stats['balanced_accuracy']:.6f} | {stats['mcc']:.6f} |"
        )

    lines.extend([
        "",
        "## Model Comparison",
        "",
        f"Samples corrected by hybrid: {len(corrected_by_hybrid)}",
        f"Samples corrected by classical: {len(corrected_by_classical)}",
        "",
        "Measured observations:",
    ])
    classical = stats_by_model["Classical"]
    hybrid = stats_by_model["Hybrid"]
    lines.extend([
        f"- Classical false positives: {classical['false_positives']}; hybrid false positives: {hybrid['false_positives']}.",
        f"- Classical false negatives: {classical['false_negatives']}; hybrid false negatives: {hybrid['false_negatives']}.",
        f"- Classical balanced accuracy: {classical['balanced_accuracy']:.6f}; hybrid balanced accuracy: {hybrid['balanced_accuracy']:.6f}.",
        f"- Classical MCC: {classical['mcc']:.6f}; hybrid MCC: {hybrid['mcc']:.6f}.",
    ])
    (output_dir / "error_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_error_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    benchmark_path = find_benchmark_results(args.benchmark_results)
    payload = read_json(benchmark_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_args = make_benchmark_args(payload, args)

    print(f"Loaded frozen benchmark results: {benchmark_path}")
    feature_names, models, checkpoints = verify_frozen_configuration(payload, run_args)
    eval_df, labels = benchmark.load_evaluation_frame(run_args, feature_names)

    sample_rows_by_model = {}
    statistics_rows = []
    for model_name in MODEL_ORDER:
        windows, model_labels = benchmark.create_model_windows(eval_df, labels, checkpoints[model_name], feature_names, run_args)
        scores, model_labels, _ = benchmark.score_model(
            models[model_name],
            windows,
            model_labels,
            run_args.device,
            run_args.batch_size,
            run_args.seed,
        )
        threshold = float(payload["models"][model_name]["threshold"])
        sample_rows = build_sample_rows(model_labels, scores, threshold)
        sample_rows_by_model[model_name] = sample_rows
        stats = compute_error_statistics(sample_rows, DISPLAY_NAMES[model_name])
        statistics_rows.append(stats)

        write_csv(output_dir / f"{model_name}_sample_errors.csv", SAMPLE_FIELDS, sample_rows)
        save_score_distribution(output_dir, model_name, sample_rows)
        save_confusion_matrix(output_dir, model_name, stats)

    corrected_by_hybrid, corrected_by_classical = compare_models(
        sample_rows_by_model["classical"],
        sample_rows_by_model["hybrid"],
    )
    write_csv(output_dir / "samples_corrected_by_hybrid.csv", COMPARISON_FIELDS, corrected_by_hybrid)
    write_csv(output_dir / "samples_corrected_by_classical.csv", COMPARISON_FIELDS, corrected_by_classical)
    write_csv(output_dir / "error_statistics.csv", STAT_FIELDS, statistics_rows)

    summary = {
        "benchmark_results": str(benchmark_path),
        "statistics": {
            row["model"]: {key: value for key, value in row.items() if key != "model"}
            for row in statistics_rows
        },
        "comparison": {
            "samples_corrected_by_hybrid": len(corrected_by_hybrid),
            "samples_corrected_by_classical": len(corrected_by_classical),
        },
        "outputs": {
            "classical_sample_errors": str(output_dir / "classical_sample_errors.csv"),
            "hybrid_sample_errors": str(output_dir / "hybrid_sample_errors.csv"),
            "samples_corrected_by_hybrid": str(output_dir / "samples_corrected_by_hybrid.csv"),
            "samples_corrected_by_classical": str(output_dir / "samples_corrected_by_classical.csv"),
            "error_statistics": str(output_dir / "error_statistics.csv"),
        },
    }
    save_json(output_dir / "error_summary.json", summary)
    write_markdown_summary(output_dir, benchmark_path, statistics_rows, corrected_by_hybrid, corrected_by_classical)

    print("\nError Analysis Summary")
    for row in statistics_rows:
        print(
            f"{row['model']}: TP={row['true_positives']} TN={row['true_negatives']} "
            f"FP={row['false_positives']} FN={row['false_negatives']} "
            f"FPR={row['false_positive_rate']:.6f} FNR={row['false_negative_rate']:.6f} "
            f"BalancedAcc={row['balanced_accuracy']:.6f} MCC={row['mcc']:.6f}"
        )
    print(f"Samples corrected by hybrid: {len(corrected_by_hybrid)}")
    print(f"Samples corrected by classical: {len(corrected_by_classical)}")
    print("\nSaved error analysis artifacts:")
    for artifact in [
        "classical_sample_errors.csv",
        "hybrid_sample_errors.csv",
        "samples_corrected_by_hybrid.csv",
        "samples_corrected_by_classical.csv",
        "score_distribution_classical.png",
        "score_distribution_hybrid.png",
        "confusion_matrix_classical.png",
        "confusion_matrix_hybrid.png",
        "error_summary.md",
        "error_summary.json",
        "error_statistics.csv",
    ]:
        print(f"  - {output_dir / artifact}")

    return summary


def main() -> int:
    args = parse_args()
    try:
        run_error_analysis(args)
    except Exception as exc:  # pragma: no cover
        print(f"Error analysis failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
