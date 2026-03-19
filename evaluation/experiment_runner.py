"""
Experiment Runner for Anomaly Detection Evaluation

Runs experiments across different threshold modes, collects evaluation data,
computes metrics, and saves results for comparison.

Usage:
    python experiment_runner.py --duration 300 --output results/
    python experiment_runner.py --modes global_static,global_dynamic --duration 600
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Import metrics module
try:
    from metrics import (
        load_evaluation_data,
        compute_classification_metrics,
        compute_per_entity_metrics,
        compute_drift_recovery_time,
        compute_latency_metrics
    )
except ImportError:
    # Handle running from different directory
    sys.path.insert(0, str(Path(__file__).parent))
    from metrics import (
        load_evaluation_data,
        compute_classification_metrics,
        compute_per_entity_metrics,
        compute_drift_recovery_time,
        compute_latency_metrics
    )


# Available threshold modes
THRESHOLD_MODES = [
    "global_static",
    "global_dynamic", 
    "entity_dynamic",
    "entity_drift"
]

# Default experiment settings
DEFAULT_DURATION_SECONDS = 300  # 5 minutes
DEFAULT_WARMUP_SECONDS = 30     # Wait for services to stabilize
DEFAULT_COOLDOWN_SECONDS = 10   # Wait for final data flush


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def ensure_results_dir(output_dir: Path) -> None:
    """Ensure the results directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)


def get_evaluation_log_path() -> Path:
    """Get the path to evaluation log file."""
    return get_project_root() / "data" / "evaluation" / "evaluation.csv"


def copy_evaluation_from_container() -> bool:
    """Copy evaluation log from Docker container to local directory."""
    local_path = get_evaluation_log_path()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy from ingestion container
    result = subprocess.run(
        ["docker", "cp", "ingestion-service:/data/evaluation.csv", str(local_path)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"  Warning: Failed to copy evaluation data: {result.stderr}")
        return False
    
    print(f"  Copied evaluation data to {local_path}")
    return True


def clear_evaluation_log() -> None:
    """Clear the evaluation log for fresh experiment."""
    # Clear local file
    log_path = get_evaluation_log_path()
    if log_path.exists():
        log_path.unlink()
        print(f"  Cleared local evaluation log: {log_path}")
    
    # Clear in container (will be created fresh by ingestion service)
    subprocess.run(
        ["docker", "exec", "ingestion-service", "rm", "-f", "/data/evaluation.csv"],
        capture_output=True
    )
    print(f"  Cleared container evaluation log")


def run_docker_compose_command(command: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
    """Run a docker-compose command."""
    if cwd is None:
        cwd = get_project_root()
    
    full_command = ["docker-compose"] + command
    return subprocess.run(
        full_command,
        cwd=cwd,
        capture_output=True,
        text=True
    )


def set_threshold_mode(mode: str) -> None:
    """Update the threshold mode in docker-compose environment."""
    # We'll use docker-compose with environment variable override
    print(f"  Setting threshold mode to: {mode}")


def restart_services_with_mode(mode: str) -> bool:
    """Restart services with a specific threshold mode."""
    project_root = get_project_root()
    
    # Create environment override
    env = os.environ.copy()
    env["THRESHOLD_MODE"] = mode
    env["EVALUATION_LOG_ENABLED"] = "true"
    env["PREDICTION_LOG_ENABLED"] = "true"
    
    print(f"  Stopping services...")
    result = subprocess.run(
        ["docker-compose", "down"],
        cwd=project_root,
        capture_output=True,
        text=True,
        env=env
    )
    
    if result.returncode != 0:
        print(f"  Warning: docker-compose down returned {result.returncode}")
        print(f"  stderr: {result.stderr}")
    
    # Clear evaluation log for fresh data
    clear_evaluation_log()
    
    print(f"  Starting services with THRESHOLD_MODE={mode}...")
    result = subprocess.run(
        ["docker-compose", "up", "-d"],
        cwd=project_root,
        capture_output=True,
        text=True,
        env=env
    )
    
    if result.returncode != 0:
        print(f"  Error starting services: {result.stderr}")
        return False
    
    return True


def wait_for_services(timeout: int = 60) -> bool:
    """Wait for all services to be healthy."""
    print(f"  Waiting for services to be healthy (timeout: {timeout}s)...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = run_docker_compose_command(["ps", "--format", "json"])
        if result.returncode == 0:
            # Check if services are running
            try:
                services = json.loads(f"[{result.stdout.replace('}{', '},{')}]") if result.stdout else []
                all_running = all(
                    s.get("State") == "running" or s.get("Status", "").startswith("Up")
                    for s in services
                )
                if all_running and len(services) > 0:
                    print(f"  All {len(services)} services running")
                    return True
            except json.JSONDecodeError:
                pass
        
        time.sleep(2)
    
    print("  Timeout waiting for services")
    return False


def collect_data(duration: int, warmup: int = DEFAULT_WARMUP_SECONDS) -> None:
    """Collect evaluation data for specified duration."""
    print(f"  Warmup period: {warmup}s...")
    time.sleep(warmup)
    
    print(f"  Collecting data for {duration}s...")
    
    # Progress indicator
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = int(time.time() - start_time)
        remaining = duration - elapsed
        print(f"\r  Progress: {elapsed}/{duration}s (remaining: {remaining}s)  ", end="", flush=True)
        time.sleep(5)
    
    print(f"\n  Data collection complete")


def flush_services(cooldown: int = DEFAULT_COOLDOWN_SECONDS) -> None:
    """Wait for services to flush remaining data."""
    print(f"  Cooldown period: {cooldown}s (flushing buffers)...")
    time.sleep(cooldown)


def compute_experiment_metrics(mode: str) -> Dict[str, Any]:
    """Compute metrics from collected evaluation data."""
    # First copy data from container
    copy_evaluation_from_container()
    
    log_path = get_evaluation_log_path()
    
    if not log_path.exists():
        print(f"  Warning: No evaluation data found at {log_path}")
        return {"mode": mode, "error": "No evaluation data"}
    
    print(f"  Loading evaluation data from {log_path}...")
    records = load_evaluation_data(str(log_path))
    
    if not records:
        print(f"  Warning: No records in evaluation data")
        return {"mode": mode, "error": "No records"}
    
    print(f"  Computing metrics for {len(records)} records...")
    
    # Compute all metrics
    classification = compute_classification_metrics(records)
    per_entity = compute_per_entity_metrics(records)
    drift_recovery = compute_drift_recovery_time(records)
    latency = compute_latency_metrics(records)
    
    return {
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "total_records": len(records),
        "classification": classification,
        "per_entity": per_entity,
        "drift_recovery": drift_recovery,
        "latency": latency
    }


def save_experiment_results(results: Dict[str, Any], output_dir: Path) -> Path:
    """Save experiment results to JSON file."""
    mode = results["mode"]
    output_file = output_dir / f"{mode}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  Results saved to: {output_file}")
    return output_file


def copy_evaluation_log(mode: str, output_dir: Path) -> None:
    """Copy evaluation log for this experiment."""
    log_path = get_evaluation_log_path()
    if log_path.exists():
        dest = output_dir / f"{mode}_evaluation.csv"
        shutil.copy(log_path, dest)
        print(f"  Evaluation log copied to: {dest}")


def run_single_experiment(
    mode: str,
    duration: int,
    output_dir: Path,
    warmup: int = DEFAULT_WARMUP_SECONDS,
    cooldown: int = DEFAULT_COOLDOWN_SECONDS
) -> Dict[str, Any]:
    """Run a single experiment for one threshold mode."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {mode}")
    print(f"{'='*60}")
    
    # Restart services with new mode
    if not restart_services_with_mode(mode):
        return {"mode": mode, "error": "Failed to start services"}
    
    # Wait for services to be healthy
    if not wait_for_services():
        return {"mode": mode, "error": "Services not healthy"}
    
    # Collect data
    collect_data(duration, warmup)
    
    # Allow buffers to flush
    flush_services(cooldown)
    
    # Compute metrics
    results = compute_experiment_metrics(mode)
    
    # Save results
    save_experiment_results(results, output_dir)
    
    # Copy evaluation log
    copy_evaluation_log(mode, output_dir)
    
    return results


def run_all_experiments(
    modes: List[str],
    duration: int,
    output_dir: Path,
    warmup: int = DEFAULT_WARMUP_SECONDS,
    cooldown: int = DEFAULT_COOLDOWN_SECONDS
) -> List[Dict[str, Any]]:
    """Run experiments for all specified modes."""
    all_results = []
    
    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT RUNNER")
    print(f"# Modes: {', '.join(modes)}")
    print(f"# Duration per mode: {duration}s")
    print(f"# Output directory: {output_dir}")
    print(f"{'#'*60}")
    
    ensure_results_dir(output_dir)
    
    # Save experiment configuration
    config = {
        "timestamp": datetime.now().isoformat(),
        "modes": modes,
        "duration_seconds": duration,
        "warmup_seconds": warmup,
        "cooldown_seconds": cooldown
    }
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run each experiment
    for i, mode in enumerate(modes, 1):
        print(f"\n[{i}/{len(modes)}] Running experiment for mode: {mode}")
        results = run_single_experiment(mode, duration, output_dir, warmup, cooldown)
        all_results.append(results)
    
    # Save combined results
    combined_output = output_dir / "all_results.json"
    with open(combined_output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to: {combined_output}")
    
    return all_results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a summary of all experiment results."""
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    header = f"{'Mode':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>6} {'FN':>6} {'Latency p95':>12}"
    print(header)
    print("-" * len(header))
    
    for r in results:
        mode = r.get("mode", "unknown")
        
        if "error" in r:
            print(f"{mode:<20} {'ERROR: ' + r['error']}")
            continue
        
        classification = r.get("classification", {})
        latency = r.get("latency", {})
        
        precision = classification.get("precision", 0)
        recall = classification.get("recall", 0)
        f1 = classification.get("f1_score", 0)
        fp = classification.get("false_positives", 0)
        fn = classification.get("false_negatives", 0)
        lat_p95 = latency.get("p95_latency_ms", 0)
        
        print(f"{mode:<20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {fp:>6} {fn:>6} {lat_p95:>12.2f}ms")
    
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Run anomaly detection experiments across threshold modes"
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(THRESHOLD_MODES),
        help=f"Comma-separated list of threshold modes to test. Available: {', '.join(THRESHOLD_MODES)}"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help=f"Duration in seconds for each experiment (default: {DEFAULT_DURATION_SECONDS})"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_SECONDS,
        help=f"Warmup period in seconds (default: {DEFAULT_WARMUP_SECONDS})"
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=DEFAULT_COOLDOWN_SECONDS,
        help=f"Cooldown period in seconds (default: {DEFAULT_COOLDOWN_SECONDS})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results (default: results/)"
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Run single experiment for specified mode without restarting services"
    )
    
    args = parser.parse_args()
    
    # Parse modes
    modes = [m.strip() for m in args.modes.split(",")]
    invalid_modes = [m for m in modes if m not in THRESHOLD_MODES]
    if invalid_modes:
        print(f"Error: Invalid threshold modes: {invalid_modes}")
        print(f"Available modes: {THRESHOLD_MODES}")
        sys.exit(1)
    
    output_dir = get_project_root() / args.output
    
    # Single mode without restart (for testing)
    if args.single:
        if args.single not in THRESHOLD_MODES:
            print(f"Error: Invalid mode: {args.single}")
            sys.exit(1)
        
        print(f"Running single experiment (no restart): {args.single}")
        ensure_results_dir(output_dir)
        collect_data(args.duration, args.warmup)
        flush_services(args.cooldown)
        results = compute_experiment_metrics(args.single)
        save_experiment_results(results, output_dir)
        print_summary([results])
        return
    
    # Run all experiments
    results = run_all_experiments(
        modes=modes,
        duration=args.duration,
        output_dir=output_dir,
        warmup=args.warmup,
        cooldown=args.cooldown
    )
    
    # Print summary
    print_summary(results)
    
    # Cleanup - stop services at the end
    print("\nStopping services...")
    run_docker_compose_command(["down"])
    print("Done!")


if __name__ == "__main__":
    main()
