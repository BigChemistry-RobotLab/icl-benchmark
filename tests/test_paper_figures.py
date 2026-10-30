"""Check the paper figures and their data selection."""

import json
import math
import statistics
from pathlib import Path
from typing import Any

import pytest
from scipy import stats

from experiments.analysis.figures import (
    _control_matrices,
    _heatmap_matrix,
    plot_label_control_bars,
    plot_mode_forest,
)
from experiments.analysis.paper_figures import (
    _majority_class_baselines,
    _mode_strata,
    render_paper_figures,
)

_DATASETS = (
    "surface_tension", "beta_sheet_formation", "self_assembly",
    "cancer_breast", "cancer_lung",
)
_TEST_LABELS = {
    "surface_tension": ("false", "true"),
    "beta_sheet_formation": ("false", "true"),
    "self_assembly": ("false", "true"),
    "cancer_breast": (
        "experimentally_inactive", "moderately_active", "very_active",
    ),
    "cancer_lung": (
        "experimentally_inactive", "moderately_active", "very_active",
    ),
}
_TEST_SUPPORTS = {
    "surface_tension": (51, 30),
    "beta_sheet_formation": (36, 18),
    "self_assembly": (24, 50),
    "cancer_breast": (17, 20, 3),
    "cancer_lung": (11, 15, 5),
}
_PILOT = ("surface_tension", "self_assembly")
_ENDPOINTS = (
    ("llm", "openai/gpt-6.1-sol"),
    ("llm", "deepseek/deepseek-v4.1-flash-1017"),
    ("llm", "google/gemma-4-31b-it"),
    ("decision", "typesafe/jev-1.14"),
    ("decision", "openai/gpt-6-luna-decisions"),
)
_LOCALS = (
    ("tabpfn", "tabpfn"), ("tabfm", "tabfm"), ("xgboost", "xgboost"),
)


def _record(
    dataset: str, variant: str, model_type: str, model: str, mode: str,
    replicate: int, score: float,
) -> dict[str, Any]:
    """Build one completed run record."""
    identity = f"{variant}-{model_type}-{model}-{dataset}-{mode}-{replicate}"
    return {
        "schema_version": 2,
        "run_id": identity,
        "protocol_id": f"protocol-{variant}-{mode}",
        "split_id": f"split-{dataset}",
        "dataset": dataset,
        "variant": variant,
        "model_type": model_type,
        "model": model,
        "mode": mode,
        "replicate": replicate,
        "expected_replicates": 5,
        "status": "completed",
        "metrics": {
            "macro_f1": score,
            "labels": list(_TEST_LABELS[dataset]),
            "per_class": {
                label: {"support": support}
                for label, support in zip(
                    _TEST_LABELS[dataset],
                    _TEST_SUPPORTS[dataset],
                    strict=True,
                )
            },
        },
    }


def _normal_score(model_index: int, dataset_index: int) -> float:
    """Return a distinct normal macro-F1 for one model and task."""
    return 0.50 + 0.02 * model_index + 0.01 * dataset_index


def _expected_baseline(dataset: str) -> float:
    """Return the majority-class macro-F1 of one test split."""
    supports = _TEST_SUPPORTS[dataset]
    total = sum(supports)
    majority = max(supports)
    return 2.0 * majority / (total + majority) / len(supports)


def _paper_records() -> list[dict[str, Any]]:
    """Return records that cover the four paper figures."""
    records = []
    for model_index, (model_type, model) in enumerate(_ENDPOINTS):
        for dataset_index, dataset in enumerate(_DATASETS):
            normal = _normal_score(model_index, dataset_index)
            balanced = normal - 0.05
            for replicate in range(5):
                records.append(_record(
                    dataset, "full", model_type, model, "one_by_one",
                    replicate, normal,
                ))
                records.append(_record(
                    dataset, "balanced", model_type, model, "one_by_one",
                    replicate, balanced,
                ))
                records.append(_record(
                    dataset, "zero_shot", model_type, model, "one_by_one",
                    replicate, normal - 0.2,
                ))
                records.append(_record(
                    dataset, "corrupted_full", model_type, model, "one_by_one",
                    replicate, normal - 0.3,
                ))
                records.append(_record(
                    dataset, "corrupted_balanced", model_type, model,
                    "one_by_one", replicate, balanced - 0.3,
                ))
        for dataset in _PILOT:
            dataset_index = _DATASETS.index(dataset)
            normal = _normal_score(model_index, dataset_index)
            balanced = normal - 0.05
            for variant, score in (
                ("full", normal - 0.01), ("balanced", balanced - 0.01),
            ):
                for replicate in range(5):
                    records.append(_record(
                        dataset, variant, model_type, model, "all_at_once",
                        replicate, score,
                    ))
    for model_index, (model_type, model) in enumerate(_LOCALS):
        for dataset_index, dataset in enumerate(_DATASETS):
            for replicate in range(5):
                records.append(_record(
                    dataset, "full", model_type, model, "local", replicate,
                    0.40 + 0.02 * model_index + 0.01 * dataset_index,
                ))
                records.append(_record(
                    dataset, "balanced", model_type, model, "local", replicate,
                    0.35 + 0.02 * model_index + 0.01 * dataset_index,
                ))
    return records


def _mode_pair_records(deltas: list[float]) -> list[dict[str, Any]]:
    """Return paired mode records with the given replicate differences."""
    records = []
    for replicate, delta in enumerate(deltas):
        records.append(_record(
            "surface_tension", "full", "llm", "model-a", "one_by_one",
            replicate, 0.5 + delta,
        ))
        records.append(_record(
            "surface_tension", "full", "llm", "model-a", "all_at_once",
            replicate, 0.5,
        ))
    return records


def _save(source: Path, records: list[dict[str, Any]]) -> None:
    """Write each record to its own JSON file."""
    source.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        path = source / f"record-{index:04d}.json"
        path.write_text(json.dumps(record), encoding="utf-8")


def _capture_figures(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Record the figure names and their tick and legend labels."""
    captured: dict[str, dict[str, Any]] = {}

    def capture(
        figure: Any, output_dir: Path, name: str, **_: Any,
    ) -> list[str]:
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
                if tick.get_text()
            ],
            "legend": [
                text.get_text()
                for legend in figure.legends
                for text in legend.get_texts()
            ],
            "texts": [
                text.get_text()
                for axis in figure.axes
                for text in axis.texts
                if text.get_text()
            ],
            "titles": [
                axis.get_title(loc="left") or axis.get_title()
                for axis in figure.axes
                if axis.get_title(loc="left") or axis.get_title()
            ],
            "xlabels": [
                axis.get_xlabel() for axis in figure.axes if axis.get_xlabel()
            ],
            "patches": [len(axis.patches) for axis in figure.axes],
            "errorbars": [
                sum(
                    hasattr(container, "lines")
                    for container in axis.containers
                )
                for axis in figure.axes
            ],
            "heights": [
                [patch.get_height() for patch in axis.patches]
                for axis in figure.axes
            ],
            "positions": [
                tuple(axis.get_position().bounds) for axis in figure.axes
            ],
            "yticklabels": [
                [tick.get_text() for tick in axis.get_yticklabels()]
                for axis in figure.axes
            ],
            "ytick_visible": [
                [tick.get_visible() for tick in axis.get_yticklabels()]
                for axis in figure.axes
            ],
            "line_values": [
                [float(value) for value in line.get_ydata()]
                for axis in figure.axes
                for line in axis.lines
                if len(line.get_ydata()) > 1
            ],
            "text_anchors": [
                [
                    {
                        "text": text.get_text(),
                        "xy": tuple(
                            getattr(text, "xy", text.get_position())
                        ),
                        "offset": tuple(getattr(text, "xyann", (0.0, 0.0))),
                        "ha": text.get_ha(),
                        "va": text.get_va(),
                    }
                    for text in axis.texts
                    if text.get_text()
                ]
                for axis in figure.axes
            ],
        }
        return []

    monkeypatch.setattr("experiments.analysis.figures.save_figure", capture)
    return captured


def test_render_paper_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draw all five figures from complete records."""
    source = tmp_path / "results"
    _save(source, _paper_records())
    captured = _capture_figures(monkeypatch)
    result = render_paper_figures(source, tmp_path / "figures")
    assert set(captured) == {
        "figure_2_mode_forest",
        "figure_3_performance",
        "figure_3_performance_balanced",
        "figure_4_controls",
        "figure_4_controls_balanced",
    }
    assert result["strata"] == 20
    assert result["majority_baseline"] == pytest.approx({
        dataset: _expected_baseline(dataset) for dataset in _DATASETS
    })
    forest = captured["figure_2_mode_forest"]
    assert "Overall mean" in forest["rows"]
    assert "All strata" not in forest["rows"]
    assert any("GPT-6.1 Sol" in row for row in forest["rows"])
    assert "Overall mean (95% CI)" in forest["legend"]
    assert "95% CI excludes 0" in forest["legend"]
    assert "95% CI includes 0" in forest["legend"]
    assert {
        "(a) Full reference set",
        "(b) Class-balanced reference set",
    } <= set(forest["titles"])
    assert "Surface activity" in forest["texts"]
    assert "Self-assembly" in forest["texts"]
    performance = captured["figure_3_performance"]
    assert performance["patches"] == [40]
    assert performance["titles"] == []
    assert performance["labels"] == []
    task_names = [
        "Surface activity", "\u03b2-sheet formation", "Self-assembly",
        "Breast-cancer activity", "Lung-cancer activity",
    ]
    short_labels = {
        "GPT-6.1 Sol", "DS4.1 Flash", "Gemma-4", "Jev", "GPT-6 Luna",
        "TabPFN", "TabFM", "XGBoost",
    }
    texts = performance["texts"]
    assert set(task_names) <= set(texts)
    assert short_labels <= set(texts)
    assert all(
        text in task_names or text in short_labels
        or text.replace(".", "", 1).isdigit()
        for text in texts
    )
    heights = performance["heights"][0]
    assert len(heights) == 40
    for task_index in range(5):
        expected = [
            *[0.50 + 0.02 * index + 0.01 * task_index for index in range(5)],
            *[0.40 + 0.02 * index + 0.01 * task_index for index in range(3)],
        ]
        group = heights[task_index * 8:(task_index + 1) * 8]
        assert group == pytest.approx(expected)
    assert performance["legend"][:2] == [
        "LLM (n=5)", "Decision model (n=5)",
    ]
    assert performance["legend"][2].startswith("Tabular foundation model (n=")
    assert performance["legend"][3].startswith("Classical baseline (n=")
    balanced_performance = captured["figure_3_performance_balanced"]
    assert balanced_performance["patches"] == [40]
    assert balanced_performance["titles"] == []
    assert balanced_performance["labels"] == []
    assert set(task_names) <= set(balanced_performance["texts"])
    assert short_labels <= set(balanced_performance["texts"])
    assert balanced_performance["legend"] == performance["legend"]
    balanced_heights = balanced_performance["heights"][0]
    assert len(balanced_heights) == 40
    for task_index in range(5):
        expected = [
            *[
                0.45 + 0.02 * index + 0.01 * task_index
                for index in range(5)
            ],
            *[
                0.35 + 0.02 * index + 0.01 * task_index
                for index in range(3)
            ],
        ]
        group = balanced_heights[task_index * 8:(task_index + 1) * 8]
        assert group == pytest.approx(expected)
    controls = captured["figure_4_controls"]
    assert controls["legend"] == [
        "Standard ICL", "Zero-shot", "Corrupted labels",
        "Majority-class baseline",
    ]
    assert controls["patches"] == [15, 15, 15, 15, 15]
    assert controls["errorbars"] == [15, 15, 15, 15, 15]
    assert controls["titles"] == task_names
    expected_baseline_texts = [
        f"{_expected_baseline(dataset):.3f}" for dataset in _DATASETS
    ]
    assert controls["texts"] == expected_baseline_texts
    expected_ticks = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    assert controls["yticklabels"] == [expected_ticks] * 5
    assert all(all(visible) for visible in controls["ytick_visible"])
    assert controls["line_values"] == [
        [pytest.approx(_expected_baseline(dataset))] * 2
        for dataset in _DATASETS
    ]
    for anchors, dataset in zip(
        controls["text_anchors"], _DATASETS, strict=True,
    ):
        assert len(anchors) == 1
        anchor = anchors[0]
        assert anchor["xy"][0] == pytest.approx(0.985)
        assert anchor["xy"][1] == pytest.approx(_expected_baseline(dataset))
        assert anchor["offset"][1] > 0
        assert anchor["ha"] == "right"
        assert anchor["va"] == "bottom"
    assert controls["xlabels"] == []
    endpoints = [
        "GPT-6.1 Sol", "DeepSeek-V4.1 Flash", "Gemma-4 31B", "Jev 1.14",
        "GPT-6 Luna",
    ]
    assert controls["labels"][:5] == endpoints
    balanced_controls = captured["figure_4_controls_balanced"]
    assert balanced_controls["legend"] == controls["legend"]
    assert balanced_controls["patches"] == [15, 15, 15, 15, 15]
    assert balanced_controls["errorbars"] == [15, 15, 15, 15, 15]
    assert balanced_controls["titles"] == task_names
    assert balanced_controls["texts"] == expected_baseline_texts
    assert balanced_controls["yticklabels"] == [expected_ticks] * 5
    assert all(
        all(visible) for visible in balanced_controls["ytick_visible"]
    )
    assert balanced_controls["line_values"] == controls["line_values"]
    assert balanced_controls["text_anchors"] == controls["text_anchors"]
    assert balanced_controls["xlabels"] == []
    assert balanced_controls["labels"][:5] == endpoints
    for task_index, heights in enumerate(balanced_controls["heights"]):
        for model_index in range(5):
            standard = 0.45 + 0.02 * model_index + 0.01 * task_index
            assert heights[3 * model_index:3 * model_index + 3] == (
                pytest.approx([standard, standard - 0.15, standard - 0.3])
            )
    positions = controls["positions"]
    assert len(positions) == 5
    top, bottom = positions[:3], positions[3:]
    assert bottom[0][0] > top[0][0]
    assert bottom[1][0] + bottom[1][2] < top[2][0] + top[2][2]
    top_center = (top[0][0] + top[2][0] + top[2][2]) / 2.0
    bottom_center = (bottom[0][0] + bottom[1][0] + bottom[1][2]) / 2.0
    assert bottom_center == pytest.approx(top_center)


def test_majority_class_baselines_use_test_supports() -> None:
    """Calculate each baseline from the per-class test supports."""
    records = [
        _record(dataset, "full", "llm", "model-a", "one_by_one", 0, 0.5)
        for dataset in _DATASETS
    ]
    baselines = _majority_class_baselines(records, _DATASETS)
    assert baselines == pytest.approx({
        dataset: _expected_baseline(dataset) for dataset in _DATASETS
    })
    conflicting = [
        _record(
            "surface_tension", "full", "llm", "model-a", "one_by_one",
            replicate, 0.5,
        )
        for replicate in range(2)
    ]
    conflicting[1]["metrics"]["per_class"]["true"]["support"] += 1
    with pytest.raises(ValueError, match="Inconsistent test class supports"):
        _majority_class_baselines(conflicting, _DATASETS)


def test_mode_strata_ci_uses_paired_t_interval() -> None:
    """Calculate each interval from the five paired differences."""
    deltas = [0.01, 0.02, 0.03, 0.04, 0.05]
    strata = _mode_strata(_mode_pair_records(deltas), 5)
    assert len(strata) == 1
    mean_delta = statistics.fmean(deltas)
    half = stats.t.ppf(0.975, 4) * statistics.stdev(deltas) / math.sqrt(5)
    assert strata[0]["delta"] == pytest.approx(mean_delta)
    assert strata[0]["ci_low"] == pytest.approx(mean_delta - half)
    assert strata[0]["ci_high"] == pytest.approx(mean_delta + half)


def test_mode_strata_zero_width_intervals() -> None:
    """Give zero-width intervals to fixed runs and a single replicate."""
    fixed = _mode_strata(_mode_pair_records([0.02] * 5), 5)
    assert fixed[0]["ci_low"] == fixed[0]["ci_high"] == pytest.approx(0.02)
    single = _mode_strata(_mode_pair_records([0.02]), 1)
    assert single[0]["ci_low"] == single[0]["ci_high"] == pytest.approx(0.02)


def test_mode_forest_draws_each_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draw one interval per stratum and include every interval end."""
    captured: dict[str, Any] = {}

    def capture(figure: Any, output_dir: Path, name: str) -> list[str]:
        captured["figure"] = figure
        return []

    monkeypatch.setattr("experiments.analysis.figures.save_figure", capture)
    strata = [
        {
            "dataset": "surface_tension", "model": "model-a",
            "model_type": "llm", "variant": "full", "delta": 0.010,
            "ci_low": -0.010, "ci_high": 0.030,
        },
        {
            "dataset": "surface_tension", "model": "model-a",
            "model_type": "llm", "variant": "balanced", "delta": 0.020,
            "ci_low": 0.020, "ci_high": 0.020,
        },
    ]
    plot_mode_forest(strata, Path("unused"))
    axes = captured["figure"].axes
    assert len(axes) == 2
    assert all(len(axis.containers) == 1 for axis in axes)
    assert all(
        len(container[1]) == 2
        for axis in axes for container in axis.containers
    )
    for axis in axes:
        low, high = axis.get_xlim()
        assert low <= -0.010
        assert high >= 0.030


def test_control_matrices_use_normal_minus_control() -> None:
    """Calculate each control loss from the matching normal cell."""
    rows: list[dict[str, Any]] = []
    for index, dataset in enumerate(_DATASETS):
        rows.append({
            "model_type": "llm", "model": "model-a", "dataset": dataset,
            "complete": True, "macro_f1_mean": 0.6 + 0.01 * index,
        })
    models, tasks, baseline, zero_loss, corrupted_loss = _control_matrices(
        rows,
        [{**row, "macro_f1_mean": row["macro_f1_mean"] - 0.1} for row in rows],
        [{**row, "macro_f1_mean": row["macro_f1_mean"] - 0.2} for row in rows],
    )
    assert models == [("llm", "model-a")]
    assert tasks == list(_DATASETS)
    assert baseline[0][0] == pytest.approx(0.6)
    assert zero_loss[0][0] == pytest.approx(0.1)
    assert corrupted_loss[0][0] == pytest.approx(0.2)


def test_control_matrices_allow_a_control_above_normal() -> None:
    """Keep a negative difference when zero-shot beats normal ICL."""
    rows: list[dict[str, Any]] = []
    for index, dataset in enumerate(_DATASETS):
        rows.append({
            "model_type": "llm", "model": "model-a", "dataset": dataset,
            "complete": True, "macro_f1_mean": 0.6 + 0.01 * index,
        })
    zero = [{**row, "macro_f1_mean": row["macro_f1_mean"] + 0.05} for row in rows]
    corrupted = [
        {**row, "macro_f1_mean": row["macro_f1_mean"] - 0.2} for row in rows
    ]
    _, _, _, zero_loss, _ = _control_matrices(rows, zero, corrupted)
    assert zero_loss[0][0] == pytest.approx(-0.05)


def test_control_bars_use_condition_sd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Draw the replicate SD on each condition bar."""
    captured: dict[str, Any] = {}

    def capture(
        figure: Any, output_dir: Path, name: str, **_: Any,
    ) -> list[str]:
        captured["figure"] = figure
        return []

    monkeypatch.setattr("experiments.analysis.figures.save_figure", capture)
    means = (0.7, 0.3, 0.2)
    spreads = (0.02, 0.03, 0.04)
    conditions = [
        [
            {
                "model_type": "llm", "model": "model-a",
                "dataset": dataset, "complete": True,
                "macro_f1_mean": mean, "macro_f1_std": spread,
            }
            for dataset in _DATASETS
        ]
        for mean, spread in zip(means, spreads, strict=True)
    ]
    plot_label_control_bars(
        conditions[0], conditions[1], conditions[2], Path("unused"),
        {dataset: 0.25 for dataset in _DATASETS},
    )
    axis = captured["figure"].axes[0]
    errorbars = [
        container
        for container in axis.containers
        if hasattr(container, "lines")
    ]
    assert len(errorbars) == 3
    for container, mean, spread in zip(
        errorbars, means, spreads, strict=True,
    ):
        segment = container.lines[2][0].get_segments()[0]
        assert segment[0][1] == pytest.approx(mean - spread)
        assert segment[1][1] == pytest.approx(mean + spread)


def test_heatmap_rejects_a_missing_cell() -> None:
    """Fail when one model and task combination is absent."""
    rows = [
        {
            "model_type": "llm", "model": "model-a", "dataset": dataset,
            "complete": True, "macro_f1_mean": 0.5,
        }
        for dataset in _DATASETS[:-1]
    ]
    with pytest.raises(ValueError, match="Missing heatmap cells"):
        _heatmap_matrix(rows, list(_DATASETS))
