"""Draw the main paper figures from saved run records."""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from scipy import stats

from .figures import (
    plot_label_control_bars,
    plot_mode_forest,
    plot_performance_bars,
)
from .mode_comparison import select_pairs
from .reporting import aggregate_runs, load_runs

_PAPER_TASKS = (
    "surface_tension", "beta_sheet_formation", "self_assembly",
    "cancer_breast", "cancer_lung",
)


def _check_cover(
    rows: list[dict[str, Any]], expected_models: int, label: str,
) -> None:
    """Check that a figure selection covers every model and task."""
    models = {(str(row["model_type"]), str(row["model"])) for row in rows}
    datasets = {str(row["dataset"]) for row in rows}
    if len(models) != expected_models or datasets != set(_PAPER_TASKS):
        raise ValueError(
            f"{label} covers {len(models)} models and {len(datasets)} tasks."
        )


def _complete_groups(
    rows: list[dict[str, Any]], variant: str, mode: str,
) -> list[dict[str, Any]]:
    """Return the complete groups of one variant and inference mode."""
    selected = [
        row for row in rows
        if row.get("complete")
        and str(row["variant"]) == variant
        and str(row["mode"]) == mode
    ]
    if not selected:
        raise ValueError(f"No complete {variant} groups with mode {mode}.")
    return selected


def _majority_class_baselines(
    records: list[dict[str, Any]], tasks: Sequence[str],
) -> dict[str, float]:
    """Return the majority-class macro-F1 for each task.

    The baseline predicts the most frequent test label. Its macro-F1 uses the
    per-class test supports in the completed run records. A task needs one
    consistent label order and support count across all of its records.

    Args:
        records: Saved run records.
        tasks: Tasks that need a baseline.

    Returns:
        The majority-class macro-F1 for each task.

    Raises:
        ValueError: A task has no class supports, inconsistent supports, or
            no test rows.
    """
    observed: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {}
    for record in records:
        if record.get("status") != "completed":
            continue
        dataset = str(record.get("dataset", ""))
        if dataset not in tasks:
            continue
        metrics = record.get("metrics") or {}
        per_class = metrics.get("per_class") or {}
        labels = tuple(str(label) for label in metrics.get("labels") or ())
        if not per_class or not labels:
            continue
        supports = tuple(
            int(per_class[label]["support"]) for label in labels
        )
        signature = (labels, supports)
        previous = observed.get(dataset)
        if previous is not None and previous != signature:
            raise ValueError(
                f"Inconsistent test class supports for {dataset}."
            )
        observed[dataset] = signature
    missing = [dataset for dataset in tasks if dataset not in observed]
    if missing:
        raise ValueError(f"Missing test class supports for: {missing}.")
    baselines = {}
    for dataset in tasks:
        labels, supports = observed[dataset]
        total = sum(supports)
        if total <= 0:
            raise ValueError(f"{dataset} has no test rows.")
        majority = max(supports)
        baselines[dataset] = (
            2.0 * majority / (total + majority) / len(labels)
        )
    return baselines


def _mode_strata(
    records: list[dict[str, Any]], expected_replicates: int,
) -> list[dict[str, Any]]:
    """Return one paired difference and its 95% t interval per mode stratum.

    The interval uses the sample standard deviation of the replicate
    differences and a two-sided t critical value with n - 1 degrees of
    freedom. A stratum with fewer than two replicates or with zero variance
    has a zero-width interval.

    Args:
        records: Saved run records.
        expected_replicates: Replicate count of the mode comparison.

    Returns:
        One entry per task, model, and reference set. Each entry has
        ``dataset``, ``model``, ``model_type``, ``variant``, ``delta``,
        ``ci_low``, and ``ci_high``.
    """
    pairs, _ = select_pairs(records, expected_replicates)
    if not pairs:
        raise ValueError("No complete paired mode groups exist.")
    strata = []
    for pair in pairs:
        deltas = [
            one - all_at_once
            for one, all_at_once in zip(
                pair.one_by_one, pair.all_at_once, strict=True,
            )
        ]
        mean_delta = fmean(deltas)
        spread = stdev(deltas) if len(deltas) > 1 else 0.0
        if spread > 0.0:
            half = float(stats.t.ppf(0.975, len(deltas) - 1))
            half *= spread / math.sqrt(len(deltas))
            ci_low, ci_high = mean_delta - half, mean_delta + half
        else:
            ci_low = ci_high = mean_delta
        strata.append({
            "dataset": pair.dataset,
            "model": pair.model,
            "model_type": pair.model_type,
            "variant": pair.variant,
            "delta": mean_delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    return strata


def render_paper_figures(
    results_dir: Path, output_dir: Path, expected_replicates: int = 5,
) -> dict[str, Any]:
    """Draw Figures 2 to 4 and return the artifact paths.

    Figure 2 compares inference modes. Figures 3 and 4 use both the full and
    the balanced reference sets. The zero-shot results are common to both
    sets.

    Args:
        results_dir: Directory with saved run records.
        output_dir: Destination for the figure files.
        expected_replicates: Replicate count of the mode comparison.

    Returns:
        The written figure paths, the number of paired strata, and the
        majority-class macro-F1 for each task.
    """
    records = load_runs(results_dir)
    rows = aggregate_runs(records)
    full_endpoint = _complete_groups(rows, "full", "one_by_one")
    full_local = _complete_groups(rows, "full", "local")
    balanced_endpoint = _complete_groups(rows, "balanced", "one_by_one")
    balanced_local = _complete_groups(rows, "balanced", "local")
    zero_shot = _complete_groups(rows, "zero_shot", "one_by_one")
    corrupted = _complete_groups(rows, "corrupted_full", "one_by_one")
    corrupted_balanced = _complete_groups(
        rows, "corrupted_balanced", "one_by_one",
    )
    majority_baseline = _majority_class_baselines(records, _PAPER_TASKS)
    strata = _mode_strata(records, expected_replicates)
    if len(strata) != 20:
        raise ValueError(
            f"Expected 20 paired mode strata, found {len(strata)}."
        )
    performance_rows = [*full_endpoint, *full_local]
    balanced_performance_rows = [*balanced_endpoint, *balanced_local]
    _check_cover(performance_rows, 8, "The performance figure")
    _check_cover(full_endpoint, 5, "The normal control panel")
    _check_cover(
        balanced_performance_rows, 8, "The balanced performance figure",
    )
    _check_cover(balanced_endpoint, 5, "The balanced control panel")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    artifacts.extend(plot_mode_forest(strata, output_dir))
    artifacts.extend(plot_performance_bars(
        performance_rows, output_dir, tasks=list(_PAPER_TASKS),
    ))
    artifacts.extend(plot_performance_bars(
        balanced_performance_rows,
        output_dir,
        name="figure_3_performance_balanced",
        tasks=list(_PAPER_TASKS),
    ))
    artifacts.extend(plot_label_control_bars(
        full_endpoint, zero_shot, corrupted, output_dir,
        majority_baseline,
        tasks=list(_PAPER_TASKS),
    ))
    artifacts.extend(plot_label_control_bars(
        balanced_endpoint, zero_shot, corrupted_balanced, output_dir,
        majority_baseline,
        name="figure_4_controls_balanced",
        tasks=list(_PAPER_TASKS),
    ))
    return {
        "figures": artifacts,
        "strata": len(strata),
        "majority_baseline": majority_baseline,
    }


def _parser() -> argparse.ArgumentParser:
    """Define the paper-figure command arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/paper"),
        help="Directory with saved run records.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/paper"),
        help="Destination for the figure files.",
    )
    parser.add_argument(
        "--expected-replicates", type=int, default=5,
        help="Replicate count of the mode comparison.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Write the main paper figures and print their paths."""
    args = _parser().parse_args(argv)
    try:
        result = render_paper_figures(
            args.results_dir, args.output_dir, args.expected_replicates,
        )
    except Exception as exc:  # Report failures at the CLI boundary.
        print(f"paper figures: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
