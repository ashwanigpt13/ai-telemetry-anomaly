"""
Result Table Generator for Paper Publication

Generates comparison tables from experiment results for academic paper inclusion.
Produces formatted console output, CSV, LaTeX, and Markdown tables.

Usage:
    python generate_results_table.py --input results/
    python generate_results_table.py --input results/ --format latex --output table.tex
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


# Column configuration for the comparison table
TABLE_COLUMNS = [
    ("Method", "mode", None),
    ("Precision", "classification.precision", ".4f"),
    ("Recall", "classification.recall", ".4f"),
    ("F1", "classification.f1_score", ".4f"),
    ("FP", "classification.false_positives", "d"),
    ("FN", "classification.false_negatives", "d"),
    ("Drift Recovery (s)", "_computed_drift_recovery", ".2f"),
    ("Latency p95 (ms)", "latency.p95_latency_ms", ".2f"),
]

# Method display names for paper
METHOD_DISPLAY_NAMES = {
    "global_static": "Static Threshold",
    "global_dynamic": "Global Dynamic",
    "entity_dynamic": "Per-Entity Dynamic",
    "entity_drift": "Entity + Drift Detection",
}


def get_nested_value(data: Dict, key_path: str, default: Any = None) -> Any:
    """Get a value from nested dictionary using dot notation."""
    keys = key_path.split(".")
    value = data
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def compute_aggregate_drift_recovery(result: Dict) -> Optional[float]:
    """Compute aggregate drift recovery time from per-entity data (in seconds)."""
    drift_recovery = result.get("drift_recovery", {})
    
    if not drift_recovery:
        return None
    
    # Collect all avg recovery times
    recovery_times = []
    for entity_data in drift_recovery.values():
        if isinstance(entity_data, dict) and "avg_recovery_time_ms" in entity_data:
            recovery_times.append(entity_data["avg_recovery_time_ms"])
    
    if recovery_times:
        # Return average in seconds
        return sum(recovery_times) / len(recovery_times) / 1000.0
    
    return None


def load_experiment_results(input_dir: Path) -> List[Dict[str, Any]]:
    """Load all experiment result JSON files from directory."""
    results = []
    
    # Try to load combined results first
    combined_file = input_dir / "all_results.json"
    if combined_file.exists():
        with open(combined_file) as f:
            return json.load(f)
    
    # Otherwise load individual files
    for json_file in sorted(input_dir.glob("*.json")):
        if json_file.name in ["experiment_config.json", "all_results.json"]:
            continue
        
        with open(json_file) as f:
            data = json.load(f)
            if "mode" in data:
                results.append(data)
    
    return results


def format_value(value: Any, fmt: Optional[str]) -> str:
    """Format a value for display."""
    if value is None:
        return "N/A"
    
    if fmt is None:
        return str(value)
    
    try:
        if fmt == "d":
            return str(int(value))
        else:
            return format(float(value), fmt)
    except (ValueError, TypeError):
        return str(value)


def build_table_data(results: List[Dict[str, Any]], use_display_names: bool = True) -> List[List[str]]:
    """Build table data from results."""
    rows = []
    
    for result in results:
        if "error" in result:
            continue
        
        row = []
        for col_name, key_path, fmt in TABLE_COLUMNS:
            if key_path == "mode":
                mode = result.get("mode", "unknown")
                if use_display_names:
                    value = METHOD_DISPLAY_NAMES.get(mode, mode)
                else:
                    value = mode
            elif key_path == "_computed_drift_recovery":
                value = compute_aggregate_drift_recovery(result)
            else:
                value = get_nested_value(result, key_path)
            
            row.append(format_value(value, fmt))
        
        rows.append(row)
    
    return rows


def get_headers() -> List[str]:
    """Get column headers."""
    return [col[0] for col in TABLE_COLUMNS]


def print_ascii_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print a formatted ASCII table."""
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    
    # Build format string
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    separator = "-+-".join("-" * w for w in col_widths)
    
    # Print table
    print()
    print(fmt.format(*headers))
    print(separator)
    for row in rows:
        print(fmt.format(*row))
    print()


def generate_csv(headers: List[str], rows: List[List[str]], output_path: Path) -> None:
    """Generate CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    
    print(f"CSV saved to: {output_path}")


def generate_latex(headers: List[str], rows: List[List[str]], output_path: Path) -> None:
    """Generate LaTeX table."""
    # Column alignment
    alignment = "l" + "r" * (len(headers) - 1)
    
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison of Anomaly Detection Methods}",
        "\\label{tab:comparison}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]
    
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    content = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(content)
    
    print(f"LaTeX saved to: {output_path}")
    print("\nLaTeX preview:")
    print(content)


def generate_markdown(headers: List[str], rows: List[List[str]], output_path: Path) -> None:
    """Generate Markdown table."""
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    
    lines = []
    
    # Header row
    header_cells = [h.ljust(col_widths[i]) for i, h in enumerate(headers)]
    lines.append("| " + " | ".join(header_cells) + " |")
    
    # Separator row (right-align numeric columns)
    separators = []
    for i, w in enumerate(col_widths):
        if i == 0:
            separators.append("-" * w)
        else:
            separators.append("-" * (w - 1) + ":")
    lines.append("| " + " | ".join(separators) + " |")
    
    # Data rows
    for row in rows:
        cells = [cell.ljust(col_widths[i]) for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(cells) + " |")
    
    content = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(content)
    
    print(f"Markdown saved to: {output_path}")
    print("\nMarkdown preview:")
    print(content)


def generate_summary_stats(results: List[Dict[str, Any]]) -> None:
    """Print summary statistics across all methods."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    # Find best performing method for each metric
    # Format: (label, key_path_or_callable, higher_is_better)
    metrics = [
        ("Highest Precision", "classification.precision", True),
        ("Highest Recall", "classification.recall", True),
        ("Highest F1", "classification.f1_score", True),
        ("Lowest False Positives", "classification.false_positives", False),
        ("Lowest False Negatives", "classification.false_negatives", False),
        ("Fastest Drift Recovery", "_computed_drift_recovery", False),
        ("Lowest Latency p95", "latency.p95_latency_ms", False),
    ]
    
    for label, key_path, higher_better in metrics:
        best_mode = None
        best_value = None
        
        for result in results:
            if "error" in result:
                continue
            
            # Handle computed metrics
            if key_path == "_computed_drift_recovery":
                value = compute_aggregate_drift_recovery(result)
            else:
                value = get_nested_value(result, key_path)
            
            if value is None:
                continue
            
            if best_value is None:
                best_value = value
                best_mode = result.get("mode")
            elif higher_better and value > best_value:
                best_value = value
                best_mode = result.get("mode")
            elif not higher_better and value < best_value:
                best_value = value
                best_mode = result.get("mode")
        
        if best_mode:
            display_name = METHOD_DISPLAY_NAMES.get(best_mode, best_mode)
            print(f"{label}: {display_name} ({best_value})")
    
    print("=" * 60)


def generate_per_entity_breakdown(results: List[Dict[str, Any]], output_path: Optional[Path] = None) -> None:
    """Generate per-entity breakdown table."""
    print("\n" + "=" * 60)
    print("PER-ENTITY BREAKDOWN")
    print("=" * 60)
    
    all_data = []
    
    for result in results:
        if "error" in result:
            continue
        
        mode = result.get("mode", "unknown")
        display_name = METHOD_DISPLAY_NAMES.get(mode, mode)
        per_entity = result.get("per_entity", {})
        
        print(f"\n{display_name}:")
        print("-" * 50)
        
        for entity_id, metrics in per_entity.items():
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)
            f1 = metrics.get("f1_score", 0)
            total = metrics.get("total", 0)
            
            print(f"  {entity_id}: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f} (n={total})")
            
            all_data.append({
                "method": mode,
                "entity_id": entity_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "total": total
            })
    
    # Save per-entity CSV if output path provided
    if output_path:
        entity_csv = output_path.parent / "per_entity_results.csv"
        with open(entity_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "entity_id", "precision", "recall", "f1", "total"])
            writer.writeheader()
            writer.writerows(all_data)
        print(f"\nPer-entity CSV saved to: {entity_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison tables from experiment results"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results",
        help="Input directory containing experiment results (default: results/)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (auto-named based on format if not specified)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "latex", "markdown", "all"],
        default="all",
        help="Output format (default: all)"
    )
    parser.add_argument(
        "--no-display-names",
        action="store_true",
        help="Use raw mode names instead of display names"
    )
    parser.add_argument(
        "--per-entity",
        action="store_true",
        help="Include per-entity breakdown"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=True,
        help="Include summary statistics (default: True)"
    )
    
    args = parser.parse_args()
    
    # Find input directory
    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        # Try relative to script location first
        script_dir = Path(__file__).parent.parent
        if (script_dir / args.input).exists():
            input_dir = script_dir / args.input
        elif not input_dir.exists():
            print(f"Error: Input directory not found: {args.input}")
            sys.exit(1)
    
    # Load results
    results = load_experiment_results(input_dir)
    
    if not results:
        print(f"Error: No experiment results found in {input_dir}")
        sys.exit(1)
    
    print(f"Loaded {len(results)} experiment results from {input_dir}")
    
    # Build table data
    use_display_names = not args.no_display_names
    headers = get_headers()
    rows = build_table_data(results, use_display_names)
    
    # Always print ASCII table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON TABLE")
    print("=" * 80)
    print_ascii_table(headers, rows)
    
    # Generate requested formats
    output_path = Path(args.output) if args.output else input_dir / "comparison_table"
    
    if args.format == "csv" or args.format == "all":
        csv_path = output_path.with_suffix(".csv") if args.output else input_dir / "comparison_table.csv"
        generate_csv(headers, rows, csv_path)
    
    if args.format == "latex" or args.format == "all":
        latex_path = output_path.with_suffix(".tex") if args.output else input_dir / "comparison_table.tex"
        generate_latex(headers, rows, latex_path)
    
    if args.format == "markdown" or args.format == "all":
        md_path = output_path.with_suffix(".md") if args.output else input_dir / "comparison_table.md"
        generate_markdown(headers, rows, md_path)
    
    # Summary statistics
    if args.summary:
        generate_summary_stats(results)
    
    # Per-entity breakdown
    if args.per_entity:
        generate_per_entity_breakdown(results, output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
