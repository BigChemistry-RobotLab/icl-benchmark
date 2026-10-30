"""Compare one-by-one and all-at-once endpoint runs with paired tests."""

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from scipy import stats

from .reporting import load_runs

_MODES = ("one_by_one", "all_at_once")
_IDENTITY_FIELDS = ("dataset", "variant", "model_type", "model", "split_id")
_DATASET_ORDER = {
    "surface_tension": 0,
    "beta_sheet_formation": 1,
    "self_assembly": 2,
    "cancer_breast": 3,
    "cancer_lung": 4,
}
_SUMMARY_FIELDS = (
    "group", "strata", "mean_delta", "ci_low", "ci_high", "t_p", "wilcoxon_p",
)


@dataclass(frozen=True)
class ModePair:
    """One dataset and model type with both inference modes."""

    dataset: str
    model_type: str
    one_by_one: tuple[float, ...]
    all_at_once: tuple[float, ...]
    model: str = ""
    variant: str = ""

    @property
    def delta(self) -> float:
        """Return the mean one-by-one score minus the mean all-at-once score."""
        return fmean(self.one_by_one) - fmean(self.all_at_once)


@dataclass(frozen=True)
class ModeSummary:
    """Paired test result for one group of strata."""

    group: str
    strata: int
    mean_delta: float
    ci_low: float
    ci_high: float
    t_p: float | None
    wilcoxon_p: float | None


def _identity(record: dict[str, Any]) -> tuple[str, ...]:
    """Return the fields that identify one mode-comparison stratum."""
    return tuple(str(record.get(field, "")) for field in _IDENTITY_FIELDS)


def _identity_label(identity: tuple[str, ...]) -> str:
    """Return a readable label for one mode-pair identity."""
    return ", ".join(
        f"{field}={value}"
        for field, value in zip(_IDENTITY_FIELDS, identity, strict=True)
    )


def _macro_f1(record: dict[str, Any]) -> float:
    """Return the macro-F1 score of a completed run record."""
    return float(record["metrics"]["macro_f1"])


def select_pairs(
    records: Sequence[dict[str, Any]], expected_replicates: int
) -> tuple[list[ModePair], list[str]]:
    """Pair complete one-by-one and all-at-once runs.

    The function selects records whose replicate identifiers are exactly
    0 to expected_replicates - 1. The per-record expected_replicates field
    is not used for selection, because the replicate count is a resume
    target and not part of the protocol identity.

    Args:
        records: Saved run records.
        expected_replicates: Number of replicate identifiers of the target protocol.

    Returns:
        The paired strata and one message for each skipped group.

    Raises:
        ValueError: The replicate count is invalid or a group repeats a replicate.
    """
    if expected_replicates < 1:
        raise ValueError("Expected replicates must be at least 1.")
    grouped: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        if record.get("status") != "completed":
            continue
        mode = str(record.get("mode", ""))
        if mode not in _MODES:
            continue
        metrics = record.get("metrics") or {}
        if "macro_f1" not in metrics:
            continue
        identity = _identity(record)
        grouped.setdefault(identity, {}).setdefault(mode, []).append(record)

    pairs = []
    skipped = []
    for identity, by_mode in grouped.items():
        label = _identity_label(identity)
        missing = [mode for mode in _MODES if mode not in by_mode]
        if missing:
            skipped.append(f"{label} (missing {', '.join(missing)})")
            continue
        one_by_one = by_mode["one_by_one"]
        all_at_once = by_mode["all_at_once"]
        replicates_one = sorted(int(run["replicate"]) for run in one_by_one)
        replicates_all = sorted(int(run["replicate"]) for run in all_at_once)
        if len(set(replicates_one)) != len(replicates_one):
            raise ValueError(f"Mode group {label} repeats a replicate identifier.")
        if len(set(replicates_all)) != len(replicates_all):
            raise ValueError(f"Mode group {label} repeats a replicate identifier.")
        if (
            replicates_one != replicates_all
            or len(replicates_one) != expected_replicates
        ):
            skipped.append(
                f"{label} (replicates {replicates_one} and {replicates_all}; "
                f"expected {expected_replicates})"
            )
            continue
        one_by_one = sorted(one_by_one, key=lambda run: int(run["replicate"]))
        all_at_once = sorted(all_at_once, key=lambda run: int(run["replicate"]))
        pairs.append(
            ModePair(
                dataset=identity[0],
                model_type=identity[2],
                one_by_one=tuple(_macro_f1(run) for run in one_by_one),
                all_at_once=tuple(_macro_f1(run) for run in all_at_once),
                model=identity[3],
                variant=identity[1],
            )
        )
    return pairs, skipped


def _summarize(group: str, pairs: Sequence[ModePair]) -> ModeSummary | None:
    """Return one paired summary, or None for an empty group."""
    if not pairs:
        return None
    deltas = [pair.delta for pair in pairs]
    mean_delta = fmean(deltas)
    if len(deltas) < 2:
        return ModeSummary(group, 1, mean_delta, math.nan, math.nan, None, None)
    spread = stdev(deltas)
    if spread == 0:
        if all(delta == 0.0 for delta in deltas):
            return ModeSummary(group, len(deltas), mean_delta, 0.0, 0.0, 1.0, 1.0)
        return ModeSummary(
            group, len(deltas), mean_delta, mean_delta, mean_delta, None, None
        )
    scale = spread / math.sqrt(len(deltas))
    interval = stats.t.interval(
        0.95, len(deltas) - 1, loc=mean_delta, scale=scale
    )
    return ModeSummary(
        group=group,
        strata=len(deltas),
        mean_delta=mean_delta,
        ci_low=float(interval[0]),
        ci_high=float(interval[1]),
        t_p=float(stats.ttest_1samp(deltas, 0.0).pvalue),
        wilcoxon_p=float(stats.wilcoxon(deltas).pvalue),
    )


def summarize(pairs: Sequence[ModePair]) -> list[ModeSummary]:
    """Return paired summaries for the full set, model families, and datasets.

    Args:
        pairs: Paired mode-comparison strata.

    Returns:
        One row for all strata, one row for each endpoint family, and one row
        for each dataset in the paper order.
    """
    groups: list[tuple[str, list[ModePair]]] = [
        ("All endpoint models", list(pairs)),
        ("LLMs", [pair for pair in pairs if pair.model_type == "llm"]),
        ("Decision models", [pair for pair in pairs if pair.model_type == "decision"]),
    ]
    datasets = sorted(
        {pair.dataset for pair in pairs},
        key=lambda name: (_DATASET_ORDER.get(name, len(_DATASET_ORDER)), name),
    )
    groups.extend(
        (dataset, [pair for pair in pairs if pair.dataset == dataset])
        for dataset in datasets
    )
    rows = []
    for group, selected in groups:
        row = _summarize(group, selected)
        if row is not None:
            rows.append(row)
    return rows


def _format_p(value: float | None) -> str:
    """Return a formatted p-value or a placeholder."""
    return "n/a" if value is None else f"{value:.4f}"


def format_table(
    rows: Sequence[ModeSummary],
    results_dir: Path,
    expected_replicates: int,
    pair_count: int,
) -> str:
    """Return the Markdown table and its method notes.

    Args:
        rows: Paired summaries.
        results_dir: Directory that supplied the saved records.
        expected_replicates: Replicate count of the selected protocol.
        pair_count: Number of paired strata.

    Returns:
        A Markdown document with one summary table.
    """
    lines = [
        "Mode comparison: one-by-one vs all-at-once",
        f"Results directory: {results_dir}",
        f"Expected replicates: {expected_replicates}",
        f"Paired strata: {pair_count}",
        f"Runs: {pair_count * 2 * expected_replicates}",
        "",
        (
            "Delta is the mean macro-F1 of one-by-one minus all-at-once. "
            "Each stratum averages its replicate scores. "
            "The paired tests are two-sided across strata."
        ),
        "",
        (
            "| Group | Strata (n) | Mean delta macro-F1 "
            "(one-by-one - all-at-once) | 95% CI | Paired t p | Wilcoxon p |"
        ),
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        if math.isnan(row.ci_low) or math.isnan(row.ci_high):
            interval = "n/a"
        else:
            interval = f"{row.ci_low:+.4f} to {row.ci_high:+.4f}"
        lines.append(
            f"| {row.group} | {row.strata} | {row.mean_delta:+.4f} | "
            f"{interval} | {_format_p(row.t_p)} | {_format_p(row.wilcoxon_p)} |"
        )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[ModeSummary]) -> None:
    """Write the summary rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _parser() -> argparse.ArgumentParser:
    """Define the mode-comparison command arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/paper"),
        help="Directory with saved run records.",
    )
    parser.add_argument(
        "--expected-replicates",
        type=int,
        default=5,
        help="Replicate count of the mode comparison.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path for the summary CSV.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Print the paired mode-comparison table."""
    args = _parser().parse_args(argv)
    try:
        records = load_runs(args.results_dir)
        pairs, skipped = select_pairs(records, args.expected_replicates)
        if not pairs:
            raise ValueError(
                "No complete mode pairs found for "
                f"{args.expected_replicates} replicates."
            )
        rows = summarize(pairs)
        if args.csv is not None:
            _write_csv(args.csv, rows)
        print(
            format_table(
                rows, args.results_dir, args.expected_replicates, len(pairs)
            )
        )
        for reason in skipped:
            print(f"experiments: skipped {reason}", file=sys.stderr)
    except Exception as exc:  # Report command failures at the CLI boundary.
        print(f"experiments: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
