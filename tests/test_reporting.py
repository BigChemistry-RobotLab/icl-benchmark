"""Check report tables, figures, and saved-record normalization."""

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.analysis.metrics import compute_metrics
from experiments.analysis.reporting import write_reports

LABELS = ("false", "true")


@pytest.fixture(autouse=True)
def _matplotlib_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the Matplotlib cache to a writable test folder."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))


def _metrics() -> dict[str, Any]:
    """Return a complete metric payload for the two-label test task."""
    return compute_metrics(list(LABELS), list(LABELS), list(LABELS))


def _new_record(replicate: int, **changes: Any) -> dict[str, Any]:
    """Build a schema-version-2 run record for the reporting tests."""
    record: dict[str, Any] = {
        "schema_version": 2,
        "run_id": f"run-{replicate}",
        "protocol_id": "protocol",
        "split_id": "split",
        "dataset": "surface_tension",
        "variant": "full",
        "model_type": "llm",
        "model": "llm-a",
        "mode": "one_by_one",
        "replicate": replicate,
        "seed": 42,
        "expected_replicates": 3,
        "status": "completed",
        "labels": list(LABELS),
        "metrics": _metrics(),
        "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
        "cost": None,
        "latency_seconds": 1.5,
        "error": None,
        "requests": [],
        "checkpoint_path": "checkpoints/request.json",
        "metadata": {},
        "max_length": 3,
    }
    record.update(changes)
    return record


def _legacy_record(replicate: int, **changes: Any) -> dict[str, Any]:
    """Build a legacy run record with the removed phase and control fields."""
    record: dict[str, Any] = {
        "run_id": f"legacy-{replicate}",
        "replicate": replicate,
        "phase": "main",
        "dataset": "surface_tension",
        "family": "llm",
        "model": "llm-a",
        "condition": "full",
        "control": "baseline",
        "mode": "one_by_one",
        "split_id": "split",
        "protocol_id": "stale-protocol",
        "configuration_hash": "configuration",
        "expected_replicates": 3,
        "status": "completed",
        "metrics": _metrics(),
        "usage": {},
        "cost": None,
        "latency_seconds": 1.0,
    }
    record.update(changes)
    return record


def _save(source: Path, records: list[dict[str, Any]]) -> None:
    """Write records as separate JSON files."""
    source.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        (source / f"record-{index:04d}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one written report table."""
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _capture_figures(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Record figure names, row labels, and legend entries without images."""
    captured: dict[str, dict[str, Any]] = {}

    def capture(figure: Any, output_dir: Path, name: str) -> list[str]:
        captured[name] = {
            "labels": [
                tick.get_text()
                for axis in figure.axes
                for tick in axis.get_xticklabels()
            ],
            "rows": [
                tick.get_text()
                for axis in figure.axes
                for tick in axis.get_yticklabels()
            ],
            "legend": [
                text.get_text()
                for legend in figure.legends
                for text in legend.get_texts()
            ],
        }
        return []

    monkeypatch.setattr("experiments.analysis.figures.save_figure", capture)
    return captured


def test_binary_metrics_and_label_order() -> None:
    """Use the last allowed label as the positive binary class."""
    metrics = compute_metrics(
        ["false", "false", "true", "true"],
        ["false", "true", "true", "true"],
        ["false", "true"],
    )
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["positive_f1"] == pytest.approx(0.8)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["per_class"]["true"]["support"] == 2


def test_missing_class_counts_as_zero() -> None:
    """Include every allowed class in macro F1."""
    metrics = compute_metrics(["a", "a"], ["a", "a"], ["a", "b", "c"])
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert "positive_f1" not in metrics


@pytest.mark.parametrize("true,predicted,labels", [
    ([], [], ["a"]), (["a"], [], ["a"]),
    (["a"], ["b"], ["a"]), (["a"], ["a"], ["a", "a"]),
])
def test_invalid_metric_inputs(true: list[str], predicted: list[str],
                               labels: list[str]) -> None:
    """Reject inputs that would produce misleading metrics."""
    with pytest.raises(ValueError):
        compute_metrics(true, predicted, labels)


def test_write_reports_uses_recorded_expected_replicates(tmp_path: Path) -> None:
    """Read replicate counts from records and write every report artifact."""
    source = tmp_path / "results"
    records = []
    for replicate in range(3):
        records.append(_new_record(replicate, run_id=f"llm-full-{replicate}"))
        records.append(_new_record(
            replicate, run_id=f"llm-zero-{replicate}",
            variant="zero_shot", protocol_id="protocol-zero",
        ))
        records.append(_new_record(
            replicate, run_id=f"llm-corrupted-{replicate}",
            variant="corrupted_balanced", protocol_id="protocol-corrupted",
        ))
    for replicate in range(100):
        records.append(_new_record(
            replicate, run_id=f"xgboost-{replicate}", model_type="xgboost",
            model="xgboost", mode="local", expected_replicates=100,
            protocol_id="protocol-xgboost",
        ))
    _save(source, records)
    output = tmp_path / "report"
    result = write_reports(source, output)
    assert result["runs"] == 109
    assert result["failed_runs"] == 0
    rows = {(row["variant"], row["model_type"]): row for row in result["groups"]}
    assert set(rows) == {
        ("full", "llm"),
        ("zero_shot", "llm"),
        ("corrupted_balanced", "llm"),
        ("full", "xgboost"),
    }
    endpoint = rows[("full", "llm")]
    assert endpoint["expected_replicates"] == 3
    assert endpoint["recorded_replicates"] == 3
    assert endpoint["completed_replicates"] == 3
    assert endpoint["complete"] is True
    local = rows[("full", "xgboost")]
    assert local["expected_replicates"] == 100
    assert local["recorded_replicates"] == 100
    assert local["complete"] is True
    assert local["mode"] == "local"
    summary = _read_csv(output / "metrics_summary.csv")
    assert len(summary) == 4
    summary_counts = {
        (row["variant"], row["model_type"]): row["expected_replicates"]
        for row in summary
    }
    assert summary_counts[("full", "llm")] == "3"
    assert summary_counts[("full", "xgboost")] == "100"
    resources = _read_csv(output / "resources.csv")
    assert len(resources) == 109
    assert len(_read_csv(output / "run_f1_scores.csv")) == 109
    first = next(row for row in resources if row["model_type"] == "llm")
    assert first["input_tokens"] == "4"
    assert first["total_tokens"] == "6"
    assert first["record_path"].endswith(".json")
    for variant in ("full", "zero_shot", "corrupted_balanced"):
        for extension in ("png", "svg", "pdf"):
            assert (output / f"performance_{variant}.{extension}").is_file()
            assert (output / f"run_f1_boxplot_{variant}.{extension}").is_file()
    assert len(result["figures"]) == 18
    assert (output / "summary.json").is_file()


def test_figures_group_modes_and_sanitize_variant_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group both modes under one model row and use a safe file name."""
    source = tmp_path / "results"
    records = []
    for replicate in range(2):
        records.append(_new_record(
            replicate, run_id=f"endpoint-{replicate}", variant="full v2",
            expected_replicates=2,
        ))
        records.append(_new_record(
            replicate, run_id=f"local-{replicate}", variant="balanced",
            model_type="xgboost", model="xgboost", mode="local",
            expected_replicates=2, protocol_id="protocol-local",
        ))
    _save(source, records)
    captured = _capture_figures(monkeypatch)
    result = write_reports(source, tmp_path / "report")
    assert set(captured) == {
        "performance_full_v2", "performance_balanced",
        "run_f1_boxplot_full_v2", "run_f1_boxplot_balanced",
    }
    endpoint = captured["performance_full_v2"]
    assert "one-by-one" in endpoint["legend"]
    assert "all-at-once" in endpoint["legend"]
    assert "LLM" in endpoint["legend"]
    assert any("Llm A" in row for row in endpoint["rows"])
    assert not any("/ local" in row for row in endpoint["rows"])
    local = captured["performance_balanced"]
    assert any("XGBoost" in row for row in local["rows"])
    assert not any("/ local" in row for row in local["rows"])
    assert {row["variant"] for row in result["groups"]} == {"full v2", "balanced"}
    summary = _read_csv(tmp_path / "report" / "metrics_summary.csv")
    assert {row["variant"] for row in summary} == {"full v2", "balanced"}


def test_legacy_records_are_normalized_and_pilot_is_skipped(tmp_path: Path) -> None:
    """Map the removed fields and keep only the main legacy records."""
    source = tmp_path / "results"
    records = []
    for replicate in range(3):
        records.append(_legacy_record(replicate, run_id=f"legacy-full-{replicate}"))
        records.append(_legacy_record(
            replicate, run_id=f"legacy-zero-{replicate}", condition="common",
            control="zero_shot", mode="all_at_once",
            configuration_hash="configuration-zero",
        ))
        records.append(_legacy_record(
            replicate, run_id=f"legacy-corrupted-{replicate}", condition="balanced",
            control="corrupted", mode="all_at_once",
            configuration_hash="configuration-corrupted",
        ))
        records.append(_legacy_record(
            replicate, run_id=f"legacy-xgboost-{replicate}", family="xgboost",
            model="xgboost", mode="local", condition="balanced",
            expected_replicates=10, configuration_hash="configuration-xgboost",
        ))
    records.append(_legacy_record(
        replicate=0, run_id="legacy-pilot", phase="pilot", mode="all_at_once",
        configuration_hash="configuration-pilot",
    ))
    _save(source, records)
    output = tmp_path / "report"
    result = write_reports(source, output)
    assert result["runs"] == 12
    rows = {(row["variant"], row["model_type"]): row for row in result["groups"]}
    assert set(rows) == {
        ("full", "llm"),
        ("zero_shot", "llm"),
        ("corrupted_balanced", "llm"),
        ("balanced", "xgboost"),
    }
    full = rows[("full", "llm")]
    assert full["mode"] == "one_by_one"
    assert full["protocol_id"] == "configuration"
    assert full["complete"] is True
    zero_shot = rows[("zero_shot", "llm")]
    assert zero_shot["mode"] == "all_at_once"
    assert zero_shot["protocol_id"] == "configuration-zero"
    corrupted = rows[("corrupted_balanced", "llm")]
    assert corrupted["protocol_id"] == "configuration-corrupted"
    local = rows[("balanced", "xgboost")]
    assert local["mode"] == "local"
    assert local["protocol_id"] == "configuration-xgboost"
    assert local["expected_replicates"] == 10
    assert local["recorded_replicates"] == 3
    assert local["complete"] is False
    assert all(
        row["variant"] != "full" or row["mode"] != "all_at_once"
        for row in result["groups"]
    )
    resources = _read_csv(output / "resources.csv")
    assert len(resources) == 12
    assert "legacy-pilot" not in {row["run_id"] for row in resources}
    for variant in ("full", "zero_shot", "corrupted_balanced"):
        for extension in ("png", "svg", "pdf"):
            assert (output / f"performance_{variant}.{extension}").is_file()
            assert (output / f"run_f1_boxplot_{variant}.{extension}").is_file()
    assert not (output / "performance_balanced.png").exists()


def test_groups_keep_split_protocol_and_mode_separate(tmp_path: Path) -> None:
    """Group runs only when every identity field matches."""
    source = tmp_path / "results"
    groups = [
        ("one_by_one", "split-a", "protocol-a"),
        ("all_at_once", "split-a", "protocol-a"),
        ("one_by_one", "split-b", "protocol-a"),
        ("one_by_one", "split-a", "protocol-b"),
    ]
    records = []
    for group, (mode, split_id, protocol_id) in enumerate(groups):
        for replicate in range(2):
            records.append(_new_record(
                replicate, run_id=f"group-{group}-{replicate}",
                expected_replicates=2, mode=mode, split_id=split_id,
                protocol_id=protocol_id,
            ))
    _save(source, records)
    result = write_reports(source, tmp_path / "report")
    assert len(result["groups"]) == 4
    assert {
        (row["mode"], row["split_id"], row["protocol_id"])
        for row in result["groups"]
    } == set(groups)
    assert len(_read_csv(tmp_path / "report" / "metrics_summary.csv")) == 4


def test_failed_records_stay_in_tables_but_not_in_figures(tmp_path: Path) -> None:
    """Retain failures in tables and keep them out of figures."""
    source = tmp_path / "results"
    records = []
    for replicate in range(2):
        records.append(_new_record(replicate, run_id=f"full-{replicate}"))
    records.append(_new_record(
        2, run_id="full-failed", status="failed", metrics={},
        error="context limit",
    ))
    for replicate in range(3):
        records.append(_new_record(
            replicate, run_id=f"balanced-{replicate}", variant="balanced",
            protocol_id="protocol-balanced",
        ))
    _save(source, records)
    output = tmp_path / "report"
    result = write_reports(source, output)
    assert result["runs"] == 6
    assert result["failed_runs"] == 1
    rows = {row["variant"]: row for row in result["groups"]}
    assert rows["full"]["completed_replicates"] == 2
    assert rows["full"]["recorded_replicates"] == 3
    assert rows["full"]["failed_replicates"] == 1
    assert rows["full"]["complete"] is False
    assert rows["balanced"]["complete"] is True
    resources = _read_csv(output / "resources.csv")
    assert len(resources) == 6
    assert any(row["error"] == "context limit" for row in resources)
    assert len(_read_csv(output / "run_f1_scores.csv")) == 5
    assert len(_read_csv(output / "per_class_metrics.csv")) == 10
    assert len(_read_csv(output / "confusion_matrices.csv")) == 20
    assert (output / "performance_balanced.png").is_file()
    assert not (output / "performance_full.png").exists()
    assert not (output / "run_f1_boxplot_full.svg").exists()


def test_empty_or_unreadable_results_fail_loudly(tmp_path: Path) -> None:
    """Reject missing folders, empty folders, and broken JSON files."""
    with pytest.raises(ValueError, match="missing"):
        write_reports(tmp_path / "missing", tmp_path / "report")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No experiment records"):
        write_reports(empty, tmp_path / "report")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "checkpoint.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="No experiment records"):
        write_reports(unrelated, tmp_path / "report")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "record.json").write_text("{not JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read result file"):
        write_reports(broken, tmp_path / "report")


def test_repeated_identifiers_are_rejected(
    tmp_path: Path,
) -> None:
    """Reject repeated run identifiers and repeated replicates."""
    first = _new_record(0, run_id="same")
    source = tmp_path / "run-id"
    _save(source, [first, dict(first)])
    with pytest.raises(ValueError, match="Repeated run identifier"):
        write_reports(source, tmp_path / "report")
    source = tmp_path / "replicate"
    _save(source, [first, _new_record(0, run_id="other")])
    with pytest.raises(ValueError, match="repeated replicate"):
        write_reports(source, tmp_path / "report")


def test_aggregate_uses_largest_expected_replicates(tmp_path: Path) -> None:
    """Use the largest declared replicate count as the group target.

    Args:
        tmp_path: Temporary folder for the records and reports.
    """
    source = tmp_path / "largest"
    records = [_new_record(replicate) for replicate in range(3)]
    records.extend(
        _new_record(replicate, expected_replicates=5) for replicate in range(3, 5)
    )
    _save(source, records)
    result = write_reports(source, tmp_path / "report")
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["expected_replicates"] == 5
    assert group["completed_replicates"] == 5
    assert group["complete"] is True

    partial = tmp_path / "partial"
    partial_records = [_new_record(replicate) for replicate in range(3)]
    partial_records.append(_new_record(3, expected_replicates=5))
    _save(partial, partial_records)
    result = write_reports(partial, tmp_path / "partial-report")
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["expected_replicates"] == 5
    assert group["completed_replicates"] == 4
    assert group["complete"] is False


def test_record_missing_required_fields_is_rejected(tmp_path: Path) -> None:
    """Reject a run record without the identity fields of the report."""
    source = tmp_path / "results"
    record = _new_record(0)
    del record["model"]
    _save(source, [record])
    with pytest.raises(ValueError, match="missing fields"):
        write_reports(source, tmp_path / "report")
