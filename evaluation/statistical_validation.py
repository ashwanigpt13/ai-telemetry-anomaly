#!/usr/bin/env python3
"""Multi-seed statistical validation for the frozen final benchmark."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from evaluation import benchmark
except ImportError:  # pragma: no cover
    import benchmark


SEEDS = [42, 123, 456]
MODEL_ORDER = ["classical", "hybrid"]
DISPLAY_NAMES = {
    "classical": "Classical",
    "hybrid": "Hybrid",
}
AGGREGATE_METRICS = [
    ("roc_auc", "ROC"),
    ("pr_auc", "PR"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("average_inference_time_ms", "Inference Time"),
    ("training_time_seconds", "Training Time"),
]
RAW_FIELDS = [
    "seed",
    "model",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "training_time_seconds",
    "average_inference_time_ms",
    "total_parameters",
    "quantum_parameters",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed statistical validation")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "statistical_validation"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--healthy-ratio", type=float, default=0.7)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rul-threshold", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def make_benchmark_args(args: argparse.Namespace, seed: int, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(output_dir),
        device=args.device,
        seed=seed,
        data_dir=args.data_dir,
        dataset=args.dataset,
        healthy_ratio=args.healthy_ratio,
        window_size=args.window_size,
        stride=args.stride,
        rul_threshold=args.rul_threshold,
        batch_size=args.batch_size,
    )


def normalize_metadata_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def metadata_signature(metadata_by_model: Dict[str, Dict[str, Any]]) -> Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], ...]:
    signature = []
    for model_name in MODEL_ORDER:
        fields = tuple(
            (field, normalize_metadata_value(metadata_by_model[model_name][field]))
            for field in benchmark.COMPATIBILITY_FIELDS
        )
        signature.append((model_name, fields))
    return tuple(signature)


def normalization_signature(checkpoints: Dict[str, Dict[str, Any]]) -> Tuple[Tuple[str, Tuple[float, ...], Tuple[float, ...]], ...]:
    signature = []
    for model_name in MODEL_ORDER:
        checkpoint = checkpoints[model_name]
        feature_mean = checkpoint.get("feature_mean")
        feature_std = checkpoint.get("feature_std")
        if feature_mean is None or feature_std is None:
            raise RuntimeError(f"{model_name} checkpoint is missing normalization statistics")
        signature.append((
            model_name,
            tuple(round(float(value), 12) for value in feature_mean),
            tuple(round(float(value), 12) for value in feature_std),
        ))
    return tuple(signature)


def verify_normalization(checkpoints: Dict[str, Dict[str, Any]], feature_names: List[str]) -> None:
    reference_mean = np.asarray(checkpoints["classical"].get("feature_mean"), dtype=np.float64)
    reference_std = np.asarray(checkpoints["classical"].get("feature_std"), dtype=np.float64)
    if len(reference_mean) != len(feature_names) or len(reference_std) != len(feature_names):
        raise RuntimeError("Classical normalization length does not match feature names")

    for model_name in MODEL_ORDER:
        mean = np.asarray(checkpoints[model_name].get("feature_mean"), dtype=np.float64)
        std = np.asarray(checkpoints[model_name].get("feature_std"), dtype=np.float64)
        if len(mean) != len(feature_names) or len(std) != len(feature_names):
            raise RuntimeError(f"{model_name} normalization length does not match feature names")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise RuntimeError(f"{model_name} normalization contains non-finite values")
        if np.any(std <= 0):
            raise RuntimeError(f"{model_name} normalization contains non-positive standard deviations")
        if not np.allclose(mean, reference_mean) or not np.allclose(std, reference_std):
            raise RuntimeError("Checkpoint normalization differs between models")


def verify_seed_metadata(
    run_args: argparse.Namespace,
    seed: int,
    expected_metadata_signature: Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], ...] | None,
    expected_normalization_signature: Tuple[Tuple[str, Tuple[float, ...], Tuple[float, ...]], ...] | None,
) -> Tuple[
    Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], ...],
    Tuple[Tuple[str, Tuple[float, ...], Tuple[float, ...]], ...],
]:
    print(f"\nPreflight verification for seed {seed}")
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
    if metadata_by_model["classical"]["dataset"] != run_args.dataset:
        raise RuntimeError("Metadata dataset does not match requested dataset")
    if metadata_by_model["classical"]["window_size"] != run_args.window_size:
        raise RuntimeError("Metadata window size does not match requested window size")

    verify_normalization(checkpoints, feature_names)
    current_metadata_signature = metadata_signature(metadata_by_model)
    current_normalization_signature = normalization_signature(checkpoints)

    if expected_metadata_signature is not None and current_metadata_signature != expected_metadata_signature:
        raise RuntimeError(f"Metadata changed before seed {seed}; aborting")
    if expected_normalization_signature is not None and current_normalization_signature != expected_normalization_signature:
        raise RuntimeError(f"Normalization changed before seed {seed}; aborting")

    print("Normalization: PASS")
    print("Checkpoint compatibility: PASS")
    return current_metadata_signature, current_normalization_signature


def extract_raw_rows(seed: int, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for model_name in MODEL_ORDER:
        model_result = payload["models"][model_name]
        metrics = model_result["metrics"]
        efficiency = model_result["efficiency"]
        rows.append({
            "seed": seed,
            "model": DISPLAY_NAMES[model_name],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "training_time_seconds": efficiency["training_time_seconds"],
            "average_inference_time_ms": efficiency["average_inference_time_ms"],
            "total_parameters": efficiency["total_parameters"],
            "quantum_parameters": efficiency["quantum_parameters"],
        })
    return rows


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    value_list = [float(value) for value in values]
    if not value_list:
        raise ValueError("Cannot aggregate an empty metric list")
    mean = statistics.mean(value_list)
    std = statistics.stdev(value_list) if len(value_list) > 1 else 0.0
    return float(mean), float(std)


def aggregate_results(raw_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name in DISPLAY_NAMES.values():
        model_rows = [row for row in raw_rows if row["model"] == model_name]
        summary[model_name] = {}
        for metric_key, _ in AGGREGATE_METRICS:
            mean, std = mean_std(row[metric_key] for row in model_rows)
            summary[model_name][metric_key] = {
                "mean": mean,
                "std": std,
            }
    return summary


def write_raw_results(output_dir: Path, raw_rows: List[Dict[str, Any]]) -> None:
    with open(output_dir / "raw_results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({field: row[field] for field in RAW_FIELDS})


def write_summary_csv(output_dir: Path, summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    fields = ["model", "metric", "mean", "std"]
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name in DISPLAY_NAMES.values():
            for metric_key, metric_label in AGGREGATE_METRICS:
                stats = summary[model_name][metric_key]
                writer.writerow({
                    "model": model_name,
                    "metric": metric_label,
                    "mean": stats["mean"],
                    "std": stats["std"],
                })


def mean_pm_std(stats: Dict[str, float]) -> str:
    return f"{stats['mean']:.6f} +/- {stats['std']:.6f}"


def write_publication_tables(output_dir: Path, summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    md_lines = [
        "| Metric | Classical Mean +/- Std | Hybrid Mean +/- Std |",
        "| --- | ---: | ---: |",
    ]
    tex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Metric & Classical Mean $\\pm$ Std & Hybrid Mean $\\pm$ Std \\\\",
        "\\midrule",
    ]

    for metric_key, metric_label in AGGREGATE_METRICS:
        classical = mean_pm_std(summary["Classical"][metric_key])
        hybrid = mean_pm_std(summary["Hybrid"][metric_key])
        md_lines.append(f"| {metric_label} | {classical} | {hybrid} |")
        tex_lines.append(
            f"{metric_label} & {classical.replace('+/-', '$\\pm$')} & {hybrid.replace('+/-', '$\\pm$')} \\\\"
        )

    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (output_dir / "summary.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def write_summary_json(
    output_dir: Path,
    raw_rows: List[Dict[str, Any]],
    summary: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    payload = {
        "seeds": SEEDS,
        "raw_results": raw_rows,
        "summary": summary,
        "parameter_counts": {
            model_name: [
                {
                    "seed": row["seed"],
                    "total_parameters": row["total_parameters"],
                    "quantum_parameters": row["quantum_parameters"],
                }
                for row in raw_rows
                if row["model"] == model_name
            ]
            for model_name in DISPLAY_NAMES.values()
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def print_console_summary(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    print("\n" + "=" * 36)
    print("Statistical Validation")
    for seed in SEEDS:
        print(f"Seed {seed}")
    print("-" * 36)

    for model_name in DISPLAY_NAMES.values():
        print(model_name)
        for metric_key, metric_label in AGGREGATE_METRICS:
            stats = summary[model_name][metric_key]
            print(f"Mean {metric_label}: {stats['mean']:.6f}")
            print(f"Std {metric_label}: {stats['std']:.6f}")
        if model_name != "Hybrid":
            print("-" * 36)
    print("=" * 36)


def run_statistical_validation(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, Any]] = []
    expected_metadata_signature = None
    expected_normalization_signature = None

    print("=" * 36)
    print("Statistical Validation")
    print("Seeds:", ", ".join(str(seed) for seed in SEEDS))
    print("=" * 36)

    for seed in SEEDS:
        print(f"\nSeed {seed}")
        seed_output_dir = output_dir / f"seed_{seed}"
        run_args = make_benchmark_args(args, seed, seed_output_dir)
        expected_metadata_signature, expected_normalization_signature = verify_seed_metadata(
            run_args,
            seed,
            expected_metadata_signature,
            expected_normalization_signature,
        )
        payload = benchmark.benchmark_models(run_args)
        raw_rows.extend(extract_raw_rows(seed, payload))

    summary = aggregate_results(raw_rows)
    write_raw_results(output_dir, raw_rows)
    write_summary_csv(output_dir, summary)
    write_publication_tables(output_dir, summary)
    write_summary_json(output_dir, raw_rows, summary)
    print_console_summary(summary)

    print("\nSaved statistical validation artifacts:")
    for artifact in ["raw_results.csv", "summary.csv", "summary.md", "summary.tex", "summary.json"]:
        print(f"  - {output_dir / artifact}")

    return {
        "raw_results": raw_rows,
        "summary": summary,
    }


def main() -> int:
    args = parse_args()
    try:
        run_statistical_validation(args)
    except Exception as exc:  # pragma: no cover
        print(f"Statistical validation failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
