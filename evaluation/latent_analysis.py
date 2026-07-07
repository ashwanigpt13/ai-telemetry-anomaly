#!/usr/bin/env python3
"""Extract latent embeddings from the frozen classical and hybrid models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from evaluation import benchmark
except ImportError:  # pragma: no cover
    import benchmark


MODEL_ORDER = ["classical", "hybrid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract latent embeddings from frozen benchmark models")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "latent_analysis"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--healthy-ratio", type=float, default=0.7)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rul-threshold", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_device(requested_device: str) -> str:
    if requested_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested_device


def load_frozen_models(device: str) -> Tuple[Dict[str, torch.nn.Module], Dict[str, Dict[str, Any]]]:
    models: Dict[str, torch.nn.Module] = {}
    checkpoints: Dict[str, Dict[str, Any]] = {}
    for model_name in MODEL_ORDER:
        model, checkpoint = benchmark.load_model_from_checkpoint(benchmark.OFFICIAL_MODELS[model_name], device)
        models[model_name] = model
        checkpoints[model_name] = checkpoint
    return models, checkpoints


def verify_metadata(args: argparse.Namespace, checkpoints: Dict[str, Dict[str, Any]], feature_names: list) -> Dict[str, Dict[str, Any]]:
    metadata_by_model = {}
    sources_by_model = {}
    for model_name in MODEL_ORDER:
        metadata_by_model[model_name], sources_by_model[model_name] = benchmark.build_effective_metadata(
            model_name,
            checkpoints[model_name],
            args,
            feature_names,
        )
    benchmark.print_compatibility_report(metadata_by_model, sources_by_model)
    return metadata_by_model


def extract_latent_embeddings(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start:start + batch_size].astype(np.float32)).to(device)
            latent = model.encode(batch, sample=False)
            embeddings.append(latent.detach().cpu().numpy())
    if not embeddings:
        raise ValueError("No latent embeddings were extracted")
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def run_latent_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models, checkpoints = load_frozen_models(device)
    feature_names = benchmark.load_training_feature_names(args)
    metadata_by_model = verify_metadata(args, checkpoints, feature_names)
    eval_df, labels = benchmark.load_evaluation_frame(args, feature_names)

    latent_by_model: Dict[str, np.ndarray] = {}
    for model_name in MODEL_ORDER:
        windows, model_labels = benchmark.create_model_windows(
            eval_df,
            labels,
            checkpoints[model_name],
            feature_names,
            args,
        )
        if not np.array_equal(labels, model_labels):
            raise RuntimeError(f"Labels changed while preparing {model_name} windows")
        latent_by_model[model_name] = extract_latent_embeddings(
            models[model_name],
            windows,
            device,
            args.batch_size,
        )

    if latent_by_model["classical"].shape[0] != latent_by_model["hybrid"].shape[0]:
        raise RuntimeError("Classical and hybrid latent sample counts differ")
    if latent_by_model["classical"].shape[0] != len(labels):
        raise RuntimeError("Latent sample count does not match labels")

    np.save(output_dir / "classical_latent.npy", latent_by_model["classical"])
    np.save(output_dir / "hybrid_latent.npy", latent_by_model["hybrid"])
    np.save(output_dir / "labels.npy", labels.astype(np.int32))

    metadata = {
        "dataset": args.dataset,
        "window_size": args.window_size,
        "stride": args.stride,
        "rul_threshold": args.rul_threshold,
        "device": device,
        "seed": args.seed,
        "num_samples": int(len(labels)),
        "num_anomaly_samples": int(np.sum(labels == 1)),
        "num_normal_samples": int(np.sum(labels == 0)),
        "latent_dimensions": {
            model_name: int(latent.shape[1])
            for model_name, latent in latent_by_model.items()
        },
        "latent_shapes": {
            model_name: list(latent.shape)
            for model_name, latent in latent_by_model.items()
        },
        "checkpoints": {
            model_name: str(benchmark.OFFICIAL_MODELS[model_name])
            for model_name in MODEL_ORDER
        },
        "metadata_compatibility": metadata_by_model,
        "outputs": {
            "classical_latent": str(output_dir / "classical_latent.npy"),
            "hybrid_latent": str(output_dir / "hybrid_latent.npy"),
            "labels": str(output_dir / "labels.npy"),
        },
    }
    save_json(output_dir / "metadata.json", metadata)

    print("\nLatent Embedding Extraction")
    print(f"Number of samples: {metadata['num_samples']}")
    print(f"Classical latent dimension: {metadata['latent_dimensions']['classical']}")
    print(f"Hybrid latent dimension: {metadata['latent_dimensions']['hybrid']}")
    print(f"Number of anomaly samples: {metadata['num_anomaly_samples']}")
    print(f"Number of normal samples: {metadata['num_normal_samples']}")
    print("\nSaved latent artifacts:")
    for artifact in ["classical_latent.npy", "hybrid_latent.npy", "labels.npy", "metadata.json"]:
        print(f"  - {output_dir / artifact}")

    return metadata


def main() -> int:
    args = parse_args()
    try:
        run_latent_analysis(args)
    except Exception as exc:  # pragma: no cover
        print(f"Latent analysis failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
