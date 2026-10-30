"""Check the paired mode-comparison table and its record selection."""

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.analysis.mode_comparison import (
    format_table,
    main,
    select_pairs,
    summarize,
)
from experiments.analysis.reporting import load_runs


def _record(
    mode: str,
    replicate: int,
    score: float,
    *,
    model: str = "llm-a",
    model_type: str = "llm",
    dataset: str = "surface_tension",
    variant: str = "full",
    split_id: str = "split-a",
    expected: int = 3,
    protocol: str = "protocol-a",
) -> dict[str, Any]:
    """Build one completed endpoint run record."""
    return {
        "schema_version": 2,
        "run_id": (
            f"{protocol}-{model}-{dataset}-{variant}-{mode}-{replicate}-{expected}"
        ),
        "protocol_id": protocol,
        "split_id": split_id,
        "dataset": dataset,
        "variant": variant,
        "model_type": model_type,
        "model": model,
        "mode": mode,
        "replicate": replicate,
        "expected_replicates": expected,
        "status": "completed",
        "metrics": {"macro_f1": score},
    }


def _paired_scores(
    one_by_one: list[float], all_at_once: list[float], **changes: Any
) -> list[dict[str, Any]]:
    """Build both modes for one model, dataset, variant, and split."""
    records = []
    for replicate, (one, all_at_once_score) in enumerate(
        zip(one_by_one, all_at_once, strict=True)
    ):
        records.append(_record("one_by_one", replicate, one, **changes))
        records.append(
            _record("all_at_once", replicate, all_at_once_score, **changes)
        )
    return records


def _save(source: Path, records: list[dict[str, Any]]) -> None:
    """Write records as separate JSON files."""
    (source / "runs").mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        path = source / "runs" / f"record-{index:04d}.json"
        path.write_text(json.dumps(record), encoding="utf-8")


def _comparison_records() -> list[dict[str, Any]]:
    """Return two models over two datasets with varied mode differences."""
    return [
        *_paired_scores([0.5, 0.6, 0.7], [0.4, 0.45, 0.55]),
        *_paired_scores(
            [0.8, 0.75, 0.7], [0.7, 0.7, 0.65], dataset="self_assembly"
        ),
        *_paired_scores(
            [0.9, 0.85, 0.8],
            [0.85, 0.8, 0.75],
            model="decision-a",
            model_type="decision",
        ),
        *_paired_scores(
            [0.6, 0.65, 0.7],
            [0.62, 0.66, 0.7],
            model="decision-a",
            model_type="decision",
            dataset="self_assembly",
        ),
    ]


def test_select_pairs_and_summarize(tmp_path: Path) -> None:
    """Pair modes by identity and average the replicate scores."""
    source = tmp_path / "results"
    _save(source, _comparison_records())
    pairs, skipped = select_pairs(load_runs(source), 3)
    assert len(pairs) == 4
    assert skipped == []
    assert [pair.delta for pair in pairs] == pytest.approx(
        [0.1333333, 0.0666667, 0.05, -0.01]
    )

    rows = summarize(pairs)
    assert [row.group for row in rows] == [
        "All endpoint models",
        "LLMs",
        "Decision models",
        "surface_tension",
        "self_assembly",
    ]
    assert [row.strata for row in rows] == [4, 2, 2, 2, 2]
    assert [row.mean_delta for row in rows] == pytest.approx(
        [0.06, 0.1, 0.02, 0.0916667, 0.0283333], abs=1e-6
    )
    assert all(row.ci_low < row.ci_high for row in rows)
    assert all(row.t_p is not None and row.wilcoxon_p is not None for row in rows)


def test_select_pairs_skips_incomplete_groups() -> None:
    """Skip missing modes and replicate sets that differ from the target."""
    records = [
        *[
            _record("one_by_one", replicate, 0.5, model="llm-b")
            for replicate in range(3)
        ],
        *[
            _record("one_by_one", replicate, 0.5, model="llm-c")
            for replicate in range(3)
        ],
        *[
            _record("all_at_once", replicate, 0.4, model="llm-c")
            for replicate in range(2)
        ],
        *[
            _record("one_by_one", replicate, 0.5, model="llm-d", expected=5)
            for replicate in range(5)
        ],
        *[
            _record("all_at_once", replicate, 0.4, model="llm-d", expected=5)
            for replicate in range(5)
        ],
    ]
    pairs, skipped = select_pairs(records, 3)
    assert pairs == []
    assert len(skipped) == 3
    assert any("llm-b" in message for message in skipped)
    assert any("llm-c" in message for message in skipped)
    assert any("llm-d" in message for message in skipped)


def test_select_pairs_accepts_mixed_expected_counts() -> None:
    """Accept a group that mixes two declared replicate counts.

    Args:
        None.
    """
    records = [
        *[
            _record("one_by_one", replicate, 0.5)
            for replicate in range(3)
        ],
        *[
            _record("one_by_one", replicate, 0.5, expected=5)
            for replicate in range(3, 5)
        ],
        *[
            _record("all_at_once", replicate, 0.4)
            for replicate in range(3)
        ],
        *[
            _record("all_at_once", replicate, 0.4, expected=5)
            for replicate in range(3, 5)
        ],
    ]
    pairs, skipped = select_pairs(records, 5)
    assert skipped == []
    assert len(pairs) == 1
    assert len(pairs[0].one_by_one) == 5
    assert len(pairs[0].all_at_once) == 5

    pairs, skipped = select_pairs(records, 3)
    assert pairs == []
    assert len(skipped) == 1


def test_select_pairs_rejects_repeated_replicates() -> None:
    """Reject two protocols that write the same replicate identifier."""
    records = [
        _record("one_by_one", 0, 0.5, protocol="protocol-one"),
        _record("one_by_one", 0, 0.5, protocol="protocol-two"),
        *[
            _record("one_by_one", replicate, 0.5)
            for replicate in range(1, 3)
        ],
        *[
            _record("all_at_once", replicate, 0.4)
            for replicate in range(3)
        ],
    ]
    with pytest.raises(ValueError, match="repeats a replicate identifier"):
        select_pairs(records, 3)


def test_format_table_uses_paper_dataset_order() -> None:
    """Print the summary rows with the two datasets in paper order."""
    pairs, _ = select_pairs(_comparison_records(), 3)
    text = format_table(summarize(pairs), Path("results/paper"), 3, len(pairs))
    assert "| All endpoint models | 4 | +0.0600 |" in text
    assert text.index("surface_tension") < text.index("self_assembly")
    assert "The paired tests are two-sided across strata." in text


def test_main_prints_table_and_writes_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the table and write the optional summary CSV."""
    source = tmp_path / "results"
    _save(source, _comparison_records())
    csv_path = tmp_path / "report" / "mode_comparison.csv"
    main([
        "--results-dir", str(source),
        "--expected-replicates", "3",
        "--csv", str(csv_path),
    ])
    captured = capsys.readouterr()
    assert "| All endpoint models | 4 |" in captured.out
    assert captured.err == ""
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert rows[0]["group"] == "All endpoint models"
    assert float(rows[0]["mean_delta"]) == pytest.approx(0.06)


def test_main_fails_without_complete_pairs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report an error when no complete mode pair exists."""
    source = tmp_path / "results"
    _save(source, [
        _record("one_by_one", replicate, 0.5) for replicate in range(3)
    ])
    with pytest.raises(SystemExit) as error:
        main(["--results-dir", str(source)])
    assert error.value.code == 1
    assert "No complete mode pairs found" in capsys.readouterr().err
