"""Load saved run records and write metrics and resource tables."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .figures import plot_performance, plot_run_scores

_GROUP_FIELDS = (
    "dataset", "variant", "model_type", "model", "mode", "split_id", "protocol_id",
)
_RECORD_FIELDS = ("run_id", *_GROUP_FIELDS, "replicate")
_REQUIRED_RECORD_FIELDS = (*_RECORD_FIELDS, "expected_replicates", "status")
_SUMMARY_FIELDS = (
    *_GROUP_FIELDS, "expected_replicates", "completed_replicates", "recorded_replicates",
    "failed_replicates", "complete", "macro_f1_mean", "macro_f1_std",
    "positive_f1_mean", "positive_f1_std",
)
_RESOURCE_FIELDS = (
    *_RECORD_FIELDS, "status", "error", "latency_seconds", "cost", "input_tokens",
    "output_tokens", "total_tokens", "device", "profile_platform", "profile_wall_seconds",
    "profile_process_peak_rss_bytes", "profile_cuda_peak_allocated_bytes", "record_path",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    """Write a table with the supplied column order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _normalize_record(record: Any, path: Path) -> dict[str, Any] | None:
    """Convert one saved record to the current identity fields."""
    if not isinstance(record, dict):
        return None
    if "family" in record:
        if record.get("phase") == "pilot":
            return None
        condition = str(record.get("condition") or "default")
        control = record.get("control")
        if control == "zero_shot":
            variant = "zero_shot"
        elif control == "corrupted":
            variant = f"corrupted_{condition}"
        else:
            variant = condition
        normalized = dict(record)
        normalized.update(
            model_type=str(record["family"]),
            variant=variant,
            protocol_id=record.get("configuration_hash") or record.get("protocol_id") or "",
            mode=str(record.get("mode") or "local"),
        )
    elif "model_type" in record and "variant" in record:
        normalized = dict(record)
    else:
        return None
    missing = [field for field in _REQUIRED_RECORD_FIELDS if field not in normalized]
    if missing:
        raise ValueError(f"Experiment record {path} is missing fields: {missing}")
    normalized["record_path"] = str(path)
    return normalized


def load_runs(results_dir: Path) -> list[dict[str, Any]]:
    """Load every parseable record and reject repeated run identifiers."""
    if not results_dir.is_dir():
        raise ValueError(f"Results directory is missing: {results_dir}")
    records = []
    seen: set[str] = set()
    for path in sorted(results_dir.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read result file {path}: {error}") from error
        normalized = _normalize_record(record, path)
        if normalized is None:
            continue
        run_id = str(normalized.get("run_id", path))
        if run_id in seen:
            raise ValueError(f"Repeated run identifier: {run_id}")
        seen.add(run_id)
        records.append(normalized)
    if not records:
        raise ValueError(f"No experiment records found in {results_dir}")
    return records


def aggregate_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group runs with the same split, protocol, and mode settings.

    The replicate count is a resume target and not part of the protocol
    identity. Thus one group can hold records that declare different counts.
    The group target is the largest declared count.
    """
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(str(record.get(field, "")) for field in _GROUP_FIELDS)
        groups[key].append(record)
    summary = []
    for key, runs in sorted(groups.items()):
        row: dict[str, Any] = dict(zip(_GROUP_FIELDS, key, strict=True))
        replicates = [run["replicate"] for run in runs]
        if len(set(replicates)) != len(replicates):
            raise ValueError("A run group has repeated replicate identifiers.")
        valid = [
            run for run in runs
            if run.get("status") == "completed"
            and "macro_f1" in (run.get("metrics") or {})
        ]
        expected = max(int(run["expected_replicates"]) for run in runs)
        row.update({
            "expected_replicates": expected,
            "completed_replicates": len(valid),
            "recorded_replicates": len(runs),
            "failed_replicates": len(runs) - len(valid),
            "complete": len(valid) == expected,
        })
        for metric in ("macro_f1", "positive_f1"):
            values = [
                float(run["metrics"][metric]) for run in valid if metric in run["metrics"]
            ]
            row[f"{metric}_mean"] = mean(values) if values else None
            row[f"{metric}_std"] = stdev(values) if len(values) > 1 else None
        summary.append(row)
    return summary


def _identity(record: dict[str, Any]) -> dict[str, Any]:
    """Return the record identity columns used by every table."""
    return {field: record.get(field) for field in _RECORD_FIELDS}


def _profile(record: dict[str, Any]) -> dict[str, Any]:
    """Return measured resource fields of a local model record."""
    metadata = record.get("metadata") or {}
    profile = metadata.get("profile") if isinstance(metadata, dict) else None
    profile = profile if isinstance(profile, dict) else {}
    return {
        "device": metadata.get("device") if isinstance(metadata, dict) else None,
        "profile_platform": profile.get("platform"),
        "profile_wall_seconds": profile.get("wall_seconds"),
        "profile_process_peak_rss_bytes": profile.get("process_peak_rss_bytes"),
        "profile_cuda_peak_allocated_bytes": profile.get("cuda_peak_allocated_bytes"),
    }


def _write_tables(
    records: list[dict[str, Any]], rows: list[dict[str, Any]], output_dir: Path,
) -> None:
    """Write the metrics, per-class, confusion, run-score, and resource tables."""
    _write_csv(output_dir / "metrics_summary.csv", rows, _SUMMARY_FIELDS)
    per_class_rows = []
    confusion_rows = []
    run_score_rows = []
    resource_rows = []
    for record in records:
        identity = _identity(record)
        usage = record.get("usage") or {}
        resource_rows.append({
            **identity,
            "status": record.get("status"),
            "error": record.get("error"),
            "latency_seconds": record.get("latency_seconds"),
            "cost": record.get("cost"),
            "input_tokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
            "output_tokens": usage.get("output_tokens") if isinstance(usage, dict) else None,
            "total_tokens": usage.get("total_tokens") if isinstance(usage, dict) else None,
            **_profile(record),
            "record_path": record.get("record_path"),
        })
        if record.get("status") != "completed":
            continue
        metrics = record.get("metrics") or {}
        run_score_rows.append({
            **identity,
            "macro_f1": metrics.get("macro_f1"),
            "positive_f1": metrics.get("positive_f1"),
        })
        for label, scores in (metrics.get("per_class") or {}).items():
            per_class_rows.append({**identity, "label": label, **scores})
        labels = metrics.get("labels") or []
        for true_label, counts in zip(labels, metrics.get("confusion_matrix") or [], strict=False):
            for predicted_label, count in zip(labels, counts, strict=False):
                confusion_rows.append({
                    **identity, "true_label": true_label,
                    "predicted_label": predicted_label, "count": count,
                })
    _write_csv(
        output_dir / "per_class_metrics.csv", per_class_rows,
        (*_RECORD_FIELDS, "label", "precision", "recall", "f1", "support"),
    )
    _write_csv(
        output_dir / "confusion_matrices.csv", confusion_rows,
        (*_RECORD_FIELDS, "true_label", "predicted_label", "count"),
    )
    _write_csv(
        output_dir / "run_f1_scores.csv", run_score_rows,
        (*_RECORD_FIELDS, "macro_f1", "positive_f1"),
    )
    _write_csv(output_dir / "resources.csv", resource_rows, _RESOURCE_FIELDS)


def write_reports(results_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Write result tables and figures without executing models.

    Args:
        results_dir: Directory with saved run records.
        output_dir: Destination for tables and figures.

    Returns:
        Aggregate groups, artifact paths, and run counts.

    Raises:
        ValueError: No parseable records exist or the record set is ambiguous.
    """
    records = load_runs(results_dir)
    rows = aggregate_runs(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(records, rows, output_dir)
    complete = {
        tuple(str(row[field]) for field in _GROUP_FIELDS) for row in rows if row["complete"]
    }
    selected = [
        record for record in records
        if record.get("status") == "completed"
        and "macro_f1" in (record.get("metrics") or {})
        and tuple(str(record.get(field, "")) for field in _GROUP_FIELDS) in complete
    ]
    figures = plot_performance(rows, output_dir)
    figures.extend(plot_run_scores(selected, output_dir))
    result = {
        "groups": rows,
        "runs": len(records),
        "figures": figures,
        "failed_runs": sum(record.get("status") != "completed" for record in records),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Saved experiment results\n\n"
        "Only complete replicate groups enter figures. Tables retain incomplete groups "
        "and failures. Blank standard deviations indicate fewer than two valid "
        "replicates. Variability is conditional on each frozen split. Groups with "
        "different split, protocol, variant, or mode settings remain separate. "
        "Run F1 boxplots show one macro-F1 score per evaluation within its dataset "
        "panel. One row holds one model and both inference modes share the row; "
        "the colour is the model type and the fill is the mode. Figures are "
        "written as PNG, SVG, and PDF. Zero-shot and corrupted variants use "
        "separate figures. Blank costs indicate that endpoint pricing was "
        "unavailable.\n",
        encoding="utf-8",
    )
    return result
