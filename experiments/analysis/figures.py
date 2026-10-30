"""Draw publication figures from aggregated groups and saved run records."""

import itertools
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

_COHORT_FIELDS = ("model_type", "model", "mode", "split_id", "protocol_id")
_FORMATS = ("png", "svg", "pdf")
_ROW_GAP = 1.0
_FAMILY_GAP = 0.55
_PANEL_MARGIN = 0.6
_FIGURE_WIDTH = 7.16  # Double-column width in inches.
_UNIT_HEIGHT = 0.225  # Height in inches of one row unit.
_FOOTER_HEIGHT = 1.0

_FAMILY_BY_TYPE = {
    "llm": "LLM",
    "decision": "Decision model",
    "tabpfn": "Tabular foundation model",
    "tabfm": "Tabular foundation model",
    "xgboost": "Classical baseline",
}
_FAMILY_ORDER = {
    "LLM": 0,
    "Decision model": 1,
    "Tabular foundation model": 2,
    "Classical baseline": 3,
}
_FAMILY_COLORS = {
    "LLM": "#0072B2",
    "Decision model": "#56B4E9",
    "Tabular foundation model": "#E69F00",
    "Classical baseline": "#009E73",
}
_PAPER_TASK_NAMES = {
    "surface_tension": "Surface activity",
    "beta_sheet_formation": "\u03b2-sheet formation",
    "self_assembly": "Self-assembly",
    "cancer_breast": "Breast-cancer activity",
    "cancer_lung": "Lung-cancer activity",
}
_FAMILY_SHORT_NAMES = {
    "llm": "LLM",
    "decision": "TDM",
    "tabpfn": "TFM",
    "tabfm": "TFM",
    "xgboost": "Baseline",
}
_MODEL_SHORT_LABELS = {
    "openai/gpt-6.1-sol": "GPT-6.1 Sol",
    "deepseek/deepseek-v4.1-flash-1017": "DS4.1 Flash",
    "google/gemma-4-31b-it": "Gemma-4",
    "typesafe/jev-1.14": "Jev",
    "openai/gpt-6-luna-decisions": "GPT-6 Luna",
    "tabpfn": "TabPFN",
    "tabfm": "TabFM",
    "xgboost": "XGBoost",
}
_PERFORMANCE_BAR_MODEL_ORDER = (
    "openai/gpt-6.1-sol",
    "deepseek/deepseek-v4.1-flash-1017",
    "google/gemma-4-31b-it",
    "typesafe/jev-1.14",
    "openai/gpt-6-luna-decisions",
    "tabpfn",
    "tabfm",
    "xgboost",
)
_MODEL_NAMES = {
    "openai/gpt-6.1-sol": "GPT-6.1 Sol",
    "deepseek/deepseek-v4.1-flash-1017": "DeepSeek-V4.1 Flash",
    "google/gemma-4-31b-it": "Gemma-4 31B",
    "typesafe/jev-1.14": "Jev 1.14",
    "openai/gpt-6-luna-decisions": "GPT-6 Luna",
    "tabpfn": "TabPFN v2",
    "tabfm": "TabFM",
    "xgboost": "XGBoost",
}
_MODEL_ORDER = {
    "openai/gpt-6.1-sol": 0,
    "deepseek/deepseek-v4.1-flash-1017": 1,
    "google/gemma-4-31b-it": 2,
    "typesafe/jev-1.14": 0,
    "openai/gpt-6-luna-decisions": 1,
    "tabpfn": 0,
    "tabfm": 1,
    "xgboost": 0,
}
_MODE_NAMES = {
    "one_by_one": "one-by-one",
    "all_at_once": "all-at-once",
    "local": "local",
}
_MODE_ORDER = {"one_by_one": 0, "all_at_once": 1, "local": 0}
_MODE_TASK_NAMES = {"surface_tension": "Surface activity"}
_DATASET_NAMES = {
    "surface_tension": "Surface tension",
    "beta_sheet_formation": "\u03b2-sheet formation",
    "self_assembly": "Self-assembly",
    "cancer_breast": "Breast-cancer activity",
    "cancer_lung": "Lung-cancer activity",
}
_DATASET_ORDER = tuple(_DATASET_NAMES)
_DATASET_SHORT_NAMES = {
    "surface_tension": "Surface\nactivity",
    "beta_sheet_formation": "\u03b2-sheet\nformation",
    "self_assembly": "Self-\nassembly",
    "cancer_breast": "Breast-cancer\nactivity",
    "cancer_lung": "Lung-cancer\nactivity",
}
_DATASET_ONE_LINE_NAMES = {
    "surface_tension": "Surface",
    "beta_sheet_formation": "\u03b2-sheet",
    "self_assembly": "Self-assembly",
    "cancer_breast": "Breast",
    "cancer_lung": "Lung",
}
_VARIANT_ORDER = {"full": 0, "balanced": 1}
_VARIANT_LABELS = {
    "full": "Full reference set",
    "balanced": "Class-balanced reference set",
}
_ENDPOINT_TYPES = ("llm", "decision")
_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "xtick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.width": 0.0,
    "ytick.major.size": 0.0,
    "grid.color": "#B3B3B3",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "hatch.color": "edge",
    "hatch.linewidth": 0.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def _sanitize(name: str) -> str:
    """Return a filename-safe form of a variant name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "variant"


def pyplot() -> Any:
    """Return a non-interactive pyplot module."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(
    figure: Any,
    output_dir: Path,
    name: str,
    pad_inches: float = 0.02,
) -> list[str]:
    """Save a figure as raster, vector, and portable-document formats.

    Args:
        figure: Figure to save.
        output_dir: Destination directory.
        name: File name without extension.
        pad_inches: White space around the tight crop, in inches.

    Returns:
        Paths of the written figure files.
    """
    paths = []
    for extension in _FORMATS:
        path = output_dir / f"{name}.{extension}"
        figure.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=pad_inches,
        )
        paths.append(str(path))
    return paths


def _family(model_type: str) -> str:
    """Return the model family shown by the box colour."""
    return _FAMILY_BY_TYPE.get(model_type, "Other")


def _family_label(model_type: str) -> str:
    """Return the short model-family label used beside a heatmap strip."""
    return _FAMILY_SHORT_NAMES.get(model_type, _family(model_type))


def _model_name(model: str) -> str:
    """Return the short model name used on the axis."""
    if model in _MODEL_NAMES:
        return _MODEL_NAMES[model]
    return str(model).rsplit("/", 1)[-1].replace("_", " ").replace("-", " ").title()


def _paper_task_name(dataset: str) -> str:
    """Return the one-line task name used by the paper bar figures."""
    return _PAPER_TASK_NAMES.get(dataset, _dataset_name(dataset))


def _model_short_label(model: str) -> str:
    """Return the compact model label used inside a bar."""
    return _MODEL_SHORT_LABELS.get(model, _model_name(model).split()[0])


def _mode_name(mode: str) -> str:
    """Return the readable mode name used in legends and labels."""
    return _MODE_NAMES.get(mode, str(mode).replace("_", "-"))


def _dataset_name(dataset: str) -> str:
    """Return the publication title of one dataset panel."""
    if dataset in _DATASET_NAMES:
        return _DATASET_NAMES[dataset]
    return str(dataset).replace("_", " ").capitalize()


def _ordered_datasets(names: set[str]) -> list[str]:
    """Keep the paper dataset order and append unknown datasets."""
    known = [name for name in _DATASET_ORDER if name in names]
    return known + sorted(names - set(_DATASET_ORDER))


def _row_sort_key(model_type: str, model: str) -> tuple[int, int, str]:
    """Order rows by paper family, model, and display name."""
    family = _family(model_type)
    return (
        _FAMILY_ORDER.get(family, len(_FAMILY_ORDER)),
        _MODEL_ORDER.get(model, len(_MODEL_ORDER)),
        _model_name(model).lower(),
    )


def _cohort_sort_key(cohort: dict[str, Any]) -> tuple[int, str, str]:
    """Order the cohorts inside one model row by mode and identity."""
    return (
        _MODE_ORDER.get(str(cohort["mode"]), len(_MODE_ORDER)),
        str(cohort.get("split_id") or ""),
        str(cohort.get("protocol_id") or ""),
    )


def _offsets(count: int) -> list[float]:
    """Spread grouped cohorts above and below the row centre."""
    if count == 1:
        return [0.0]
    spread = 0.21 if count == 2 else 0.24
    return [spread - 2.0 * spread * index / (count - 1) for index in range(count)]


def _layout_panel(cohorts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group cohorts into one row per model and place the rows.

    Args:
        cohorts: Cohort entries of one dataset panel.

    Returns:
        The rows with their vertical positions. The first row is at zero and
        later rows have decreasing positions.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cohort in cohorts:
        groups[(str(cohort["model_type"]), str(cohort["model"]))].append(cohort)
    rows = []
    position = 0.0
    previous_family = None
    for key in sorted(groups, key=lambda key: _row_sort_key(key[0], key[1])):
        family = _family(key[0])
        if previous_family is not None and family != previous_family:
            position -= _FAMILY_GAP
        members = sorted(groups[key], key=_cohort_sort_key)
        repeated = len({str(member["mode"]) for member in members}) != len(members)
        if repeated:
            for member in members:
                suffix = f"[{str(member.get('split_id') or '')[:6]}/"
                suffix += f"{str(member.get('protocol_id') or '')[:6]}]"
                rows.append(
                    {
                        "label": f"{_model_name(key[1])} \u00b7 "
                        f"{_mode_name(member['mode'])} {suffix}",
                        "family": family,
                        "y": position,
                        "items": [(member, 0.0)],
                    }
                )
                position -= _ROW_GAP
        else:
            rows.append(
                {
                    "label": _model_name(key[1]),
                    "family": family,
                    "y": position,
                    "items": list(zip(members, _offsets(len(members)), strict=True)),
                }
            )
            position -= _ROW_GAP
        previous_family = family
    return rows


def _panel_span(rows: list[dict[str, Any]]) -> float:
    """Return the vertical extent of one panel in row units."""
    return rows[0]["y"] - rows[-1]["y"] + 2.0 * _PANEL_MARGIN


def _create_panels(
    plt: Any,
    panels: list[dict[str, Any]],
    columns: int,
) -> tuple[Any, list[Any]]:
    """Create an adaptive panel grid and centre an incomplete last row.

    Args:
        plt: Non-interactive pyplot module.
        panels: Dataset panels with their laid-out rows.
        columns: Number of panels per full row.

    Returns:
        The figure and one axis per panel.
    """
    rows = -(-len(panels) // columns)
    ratios = [1.0] * rows
    for index, panel in enumerate(panels):
        row = index // columns
        ratios[row] = max(ratios[row], _panel_span(panel["rows"]) + 0.9)
    left, right, top, bottom = 0.17, 0.995, 0.975, 0.135
    width = (right - left) / (columns + 0.95 * (columns - 1))
    horizontal_gap = 0.95 * width
    vertical_gap = 0.10
    height = (top - bottom - (rows - 1) * vertical_gap) / sum(ratios)
    figure = plt.figure(
        figsize=(_FIGURE_WIDTH, _FOOTER_HEIGHT + _UNIT_HEIGHT * sum(ratios))
    )
    axes = []
    for index, panel in enumerate(panels):
        row = index // columns
        column = index % columns
        in_row = min(columns, len(panels) - row * columns)
        if in_row == columns:
            x = left + column * (width + horizontal_gap)
        else:
            used = in_row * width + (in_row - 1) * horizontal_gap
            x = left + (right - left - used) / 2 + column * (width + horizontal_gap)
        y_top = top - sum(ratios[:row]) * height - row * vertical_gap
        axes.append(
            figure.add_axes(
                (x, y_top - ratios[row] * height, width, ratios[row] * height),
                sharex=axes[0] if axes else None,
            )
        )
    return figure, axes


def _draw_family_dividers(axis: Any, rows: list[dict[str, Any]]) -> None:
    """Draw a light divider where the model family changes."""
    for upper, lower in itertools.pairwise(rows):
        if upper["family"] != lower["family"]:
            axis.axhline(
                (upper["y"] + lower["y"]) / 2.0,
                color="#CCCCCC",
                linewidth=0.5,
                zorder=1,
            )


def _draw_row_labels(axis: Any, rows: list[dict[str, Any]]) -> None:
    """Place one model label per row and remove the y-axis ticks."""
    axis.set_yticks([row["y"] for row in rows])
    axis.set_yticklabels([row["label"] for row in rows])
    axis.set_ylim(
        rows[-1]["y"] - _PANEL_MARGIN,
        rows[0]["y"] + _PANEL_MARGIN,
    )
    axis.tick_params(axis="y", length=0, pad=3)


_ALL_AT_ONCE_HATCH = "////"


def _box_style(family: str, mode: str, striped: bool) -> dict[str, Any]:
    """Return the fill, edge, and hatch that encode family and mode.

    Args:
        family: Model family of the cohort.
        mode: Mode of the cohort.
        striped: Add stripes to the all-at-once style. Bars are wide enough to
            show the stripes. Thin boxes are not.

    Returns:
        Style keys for a bar or box patch. The all-at-once style is pale and,
        when requested, striped.
    """
    from matplotlib.colors import to_rgba

    color = _FAMILY_COLORS.get(family, "#777777")
    if mode == "all_at_once":
        return {
            "facecolor": to_rgba(color, 0.30),
            "edgecolor": color,
            "linewidth": 0.8,
            "hatch": _ALL_AT_ONCE_HATCH if striped else None,
        }
    return {
        "facecolor": color,
        "edgecolor": "#333333",
        "linewidth": 0.5,
        "hatch": None,
    }


def _draw_box(
    axis: Any,
    values: list[float],
    position: float,
    width: float,
    style: dict[str, Any],
) -> None:
    """Draw one horizontal box with median, quartiles, and 1.5-IQR whiskers."""
    from matplotlib.patches import Rectangle

    ordered = sorted(values)
    if len(ordered) > 1:
        first, _, third = statistics.quantiles(
            ordered,
            n=4,
            method="inclusive",
        )
    else:
        first = third = ordered[0]
    low = min(value for value in ordered if value >= first - 1.5 * (third - first))
    high = max(value for value in ordered if value <= third + 1.5 * (third - first))
    half = width / 2.0
    axis.add_patch(
        Rectangle(
            (first, position - half),
            third - first,
            width,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            linewidth=style["linewidth"],
            zorder=2,
        )
    )
    axis.vlines(
        statistics.median(ordered),
        position - half,
        position + half,
        color="black",
        linewidth=0.9,
        zorder=3,
    )
    axis.hlines(position, low, high, color="#4D4D4D", linewidth=0.6, zorder=2)
    axis.vlines(
        [low, high],
        position - half / 2.0,
        position + half / 2.0,
        color="#4D4D4D",
        linewidth=0.6,
        zorder=2,
    )


def _jitter_seed(cohort: dict[str, Any]) -> str:
    """Return a stable seed for the run-point jitter of one cohort."""
    return "|".join(str(cohort.get(field, "")) for field in _COHORT_FIELDS)


def _draw_footer(
    figure: Any,
    rows: list[dict[str, Any]],
    xlabel: str,
    show_mean: bool,
    striped: bool,
) -> None:
    """Draw the shared x label and the legend.

    Args:
        figure: Figure that receives the footer.
        rows: Panel rows that define the model families in the legend.
        xlabel: Label of the shared horizontal axis.
        show_mean: Add the mean marker to the legend.
        striped: Show the all-at-once legend entry with stripes.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    families = list(dict.fromkeys(row["family"] for row in rows))
    handles: list[Any] = [
        Patch(
            facecolor=_FAMILY_COLORS.get(family, "#777777"),
            edgecolor="#333333",
            linewidth=0.5,
            label=family,
        )
        for family in families
    ]
    handles.extend(
        [
            Patch(
                facecolor="#9E9E9E",
                edgecolor="#333333",
                linewidth=0.5,
                label="one-by-one",
            ),
            Patch(
                facecolor="#D9D9D9",
                edgecolor="#555555",
                linewidth=0.5,
                hatch=_ALL_AT_ONCE_HATCH if striped else None,
                label="all-at-once",
            ),
        ]
    )
    if show_mean:
        handles.append(
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                color="none",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.6,
                markersize=4,
                label="mean",
            )
        )
    figure.text(
        0.5,
        0.065,
        xlabel,
        ha="center",
        va="bottom",
        fontsize=7.0,
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=min(len(handles), 7),
        handlelength=1.8,
        handleheight=1.5,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def _finish_panel_grid(
    figure: Any,
    axes: list[Any],
    rows: list[dict[str, Any]],
    xlabel: str,
    show_mean: bool,
    striped: bool,
) -> None:
    """Apply the shared axes, x label, and legend to a panel grid."""
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(axis="x", linestyle="--")
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axis.tick_params(axis="x", labelbottom=True)
    _draw_footer(figure, rows, xlabel, show_mean, striped)


def _panel_title(index: int, dataset: str) -> str:
    """Return the lettered panel title."""
    return f"({chr(ord('a') + index)}) {_dataset_name(dataset)}"


def plot_performance(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    """Draw complete variant groups as horizontal mean macro-F1 bars."""
    plt = pyplot()
    artifacts = []
    variants = sorted({str(row["variant"]) for row in rows if row["complete"]})
    with plt.rc_context(_STYLE):
        for variant in variants:
            selected = [
                row
                for row in rows
                if row["complete"] and str(row["variant"]) == variant
            ]
            datasets = _ordered_datasets({str(row["dataset"]) for row in selected})
            panels: list[dict[str, Any]] = []
            for dataset in datasets:
                cohorts = [
                    {
                        "model_type": row["model_type"],
                        "model": row["model"],
                        "mode": row["mode"],
                        "split_id": row["split_id"],
                        "protocol_id": row["protocol_id"],
                        "mean": float(row["macro_f1_mean"] or 0.0),
                        "std": float(row["macro_f1_std"] or 0.0),
                    }
                    for row in selected
                    if str(row["dataset"]) == dataset
                ]
                panels.append({"dataset": dataset, "rows": _layout_panel(cohorts)})
            figure, axes = _create_panels(plt, panels, columns=3)
            all_rows: list[dict[str, Any]] = []
            for index, (axis, panel) in enumerate(zip(axes, panels, strict=True)):
                _draw_family_dividers(axis, panel["rows"])
                for row in panel["rows"]:
                    for cohort, offset in row["items"]:
                        style = _box_style(
                            row["family"],
                            str(cohort["mode"]),
                            striped=True,
                        )
                        axis.barh(
                            [row["y"] + offset],
                            [cohort["mean"]],
                            height=0.30 if len(row["items"]) > 1 else 0.40,
                            color=style["facecolor"],
                            edgecolor=style["edgecolor"],
                            linewidth=style["linewidth"],
                            hatch=style["hatch"],
                            xerr=[cohort["std"]],
                            error_kw={
                                "ecolor": "#333333",
                                "elinewidth": 0.6,
                                "capsize": 1.5,
                            },
                            zorder=2,
                        )
                _draw_row_labels(axis, panel["rows"])
                axis.set_title(_panel_title(index, panel["dataset"]), loc="left", pad=4)
                all_rows.extend(panel["rows"])
            _finish_panel_grid(
                figure,
                axes,
                all_rows,
                "Macro-F1",
                show_mean=False,
                striped=True,
            )
            artifacts.extend(
                save_figure(figure, output_dir, f"performance_{_sanitize(variant)}")
            )
            plt.close(figure)
    return artifacts


def plot_run_scores(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    """Draw one macro-F1 point per evaluation within each dataset and variant."""
    plt = pyplot()
    artifacts = []
    variants = sorted({str(record["variant"]) for record in records})
    with plt.rc_context(_STYLE):
        for variant in variants:
            runs = [record for record in records if str(record["variant"]) == variant]
            datasets = _ordered_datasets({str(record["dataset"]) for record in runs})
            panels: list[dict[str, Any]] = []
            for dataset in datasets:
                cohorts: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
                for record in runs:
                    if str(record["dataset"]) != dataset:
                        continue
                    key = tuple(str(record.get(field, "")) for field in _COHORT_FIELDS)
                    cohorts[key].append(record)
                entries = []
                for key, evaluations in cohorts.items():
                    evaluations.sort(key=lambda item: int(item["replicate"]))
                    entries.append(
                        {
                            "model_type": key[0],
                            "model": key[1],
                            "mode": key[2],
                            "split_id": key[3],
                            "protocol_id": key[4],
                            "values": [
                                float(item["metrics"]["macro_f1"])
                                for item in evaluations
                            ],
                        }
                    )
                panels.append({"dataset": dataset, "rows": _layout_panel(entries)})
            figure, axes = _create_panels(plt, panels, columns=3)
            all_rows: list[dict[str, Any]] = []
            for index, (axis, panel) in enumerate(zip(axes, panels, strict=True)):
                _draw_family_dividers(axis, panel["rows"])
                for row in panel["rows"]:
                    for cohort, offset in row["items"]:
                        position = row["y"] + offset
                        style = _box_style(
                            row["family"],
                            str(cohort["mode"]),
                            striped=False,
                        )
                        width = 0.30 if len(row["items"]) > 1 else 0.40
                        _draw_box(axis, cohort["values"], position, width, style)
                        rng = random.Random(_jitter_seed(cohort))
                        jitter = [
                            rng.uniform(-0.42, 0.42) * width for _ in cohort["values"]
                        ]
                        axis.scatter(
                            cohort["values"],
                            [position + shift for shift in jitter],
                            s=1.8,
                            color="#1A1A1A",
                            alpha=0.35,
                            linewidths=0,
                            zorder=3,
                        )
                        axis.scatter(
                            [sum(cohort["values"]) / len(cohort["values"])],
                            [position],
                            marker="D",
                            s=10,
                            facecolor="white",
                            edgecolor="#1A1A1A",
                            linewidth=0.6,
                            zorder=4,
                        )
                _draw_row_labels(axis, panel["rows"])
                axis.set_title(_panel_title(index, panel["dataset"]), loc="left", pad=4)
                all_rows.extend(panel["rows"])
            _finish_panel_grid(
                figure,
                axes,
                all_rows,
                "Macro-F1 per evaluation",
                show_mean=True,
                striped=False,
            )
            artifacts.extend(
                save_figure(figure, output_dir, f"run_f1_boxplot_{_sanitize(variant)}")
            )
            plt.close(figure)
    return artifacts


def _short_dataset_name(dataset: str) -> str:
    """Return the wrapped task name used below a heatmap column."""
    if dataset in _DATASET_SHORT_NAMES:
        return _DATASET_SHORT_NAMES[dataset]
    return _dataset_name(dataset).replace(" ", "\n", 1)


def _one_line_dataset_name(dataset: str) -> str:
    """Return a compact one-line task name."""
    if dataset in _DATASET_ONE_LINE_NAMES:
        return _DATASET_ONE_LINE_NAMES[dataset]
    return _dataset_name(dataset).split(" ")[0]


def _mean_interval(values: list[float]) -> tuple[float, float, float]:
    """Return the sample mean and its 95% t interval.

    Args:
        values: Independent estimates.

    Returns:
        The mean, the lower interval bound, and the upper interval bound. A
        single value or a zero-variance sample gives a zero-width interval.
    """
    from scipy import stats

    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, mean, mean
    spread = statistics.stdev(values)
    if spread == 0.0:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(values) - 1))
    half *= spread / math.sqrt(len(values))
    return mean, mean - half, mean + half


def plot_mode_forest(
    strata: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    """Draw paired one-by-one minus all-at-once differences as two panels.

    Each reference set gets its own panel, with the full set first and the
    class-balanced set second. Each model appears once in each panel, so the
    paired estimates do not share a row. A final row in each panel gives the
    overall mean across tasks and models for that reference set. Points with
    a 95% interval that includes zero are drawn in light grey.

    Args:
        strata: One entry per task, model, and reference set. Each entry has
            ``dataset``, ``model``, ``model_type``, ``variant``, ``delta``,
            ``ci_low``, and ``ci_high``.
        output_dir: Destination for the figure files.

    Returns:
        Paths of the written figure files.

    Raises:
        ValueError: The input is empty, a reference set is unknown, or a
            model and task repeat a reference set.
    """
    from matplotlib.ticker import MaxNLocator

    if not strata:
        raise ValueError("The mode forest plot needs at least one stratum.")
    unknown = {str(item["variant"]) for item in strata} - set(_VARIANT_ORDER)
    if unknown:
        raise ValueError(f"Unknown reference sets: {sorted(unknown)}.")
    variants = [
        variant
        for variant in _VARIANT_ORDER
        if any(str(item["variant"]) == variant for item in strata)
    ]
    datasets = _ordered_datasets({str(item["dataset"]) for item in strata})
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in strata:
        by_dataset[str(item["dataset"])].append(item)

    rows: list[dict[str, Any]] = []
    task_headers: list[tuple[str, float]] = []
    position = 0.0
    for dataset in datasets:
        models = sorted(
            {
                (str(item["model_type"]), str(item["model"]))
                for item in by_dataset[dataset]
            },
            key=lambda key: _row_sort_key(*key),
        )
        task_headers.append((dataset, position + 0.75))
        for key in models:
            members = [
                item
                for item in by_dataset[dataset]
                if (str(item["model_type"]), str(item["model"])) == key
            ]
            estimates: dict[str, dict[str, Any]] = {}
            for item in members:
                variant = str(item["variant"])
                if variant in estimates:
                    raise ValueError(
                        f"Repeated mode stratum for {_model_name(key[1])} on {dataset}."
                    )
                estimates[variant] = item
            rows.append(
                {
                    "y": position,
                    "label": _model_name(key[1]),
                    "items": estimates,
                }
            )
            position -= 1.0
        position -= 0.9
    summary_y = position + 0.05

    summaries = {
        variant: _mean_interval(
            [float(item["delta"]) for item in strata if str(item["variant"]) == variant]
        )
        for variant in variants
    }
    interval_lows = [float(item["ci_low"]) for item in strata]
    interval_highs = [float(item["ci_high"]) for item in strata]
    low = min(interval_lows + [bound[1] for bound in summaries.values()])
    high = max(interval_highs + [bound[2] for bound in summaries.values()])
    padding = max(0.005, 0.06 * (high - low))

    plt = pyplot()
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(
            1,
            len(variants),
            figsize=(_FIGURE_WIDTH, 4.15),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axes = axes[0]
        figure.subplots_adjust(
            left=0.205,
            right=0.99,
            top=0.895,
            bottom=0.195,
            wspace=0.10,
        )
        for index, (axis, variant) in enumerate(zip(axes, variants, strict=True)):
            axis.set_xlim(low - padding, high + padding)
            axis.set_ylim(summary_y - 0.75, task_headers[0][1] + 0.55)
            axis.set_axisbelow(True)
            axis.grid(axis="x", linestyle="--")
            for dataset, header_y in task_headers:
                axis.axhspan(
                    header_y - 0.28,
                    header_y + 0.28,
                    facecolor="#F2F2F2",
                    edgecolor="none",
                    zorder=0,
                )
                axis.text(
                    0.012,
                    header_y,
                    _MODE_TASK_NAMES.get(dataset, _dataset_name(dataset)),
                    transform=axis.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=6.3,
                    color="#4D4D4D",
                )
            axis.axvline(0.0, color="#8C8C8C", linewidth=0.7, zorder=1)
            for row in rows:
                item = row["items"].get(variant)
                if item is None:
                    continue
                delta = float(item["delta"])
                ci_low = float(item["ci_low"])
                ci_high = float(item["ci_high"])
                excludes_zero = ci_low > 0.0 or ci_high < 0.0
                color = "#1A1A1A" if excludes_zero else "#A6A6A6"
                axis.errorbar(
                    [delta],
                    [row["y"]],
                    xerr=[[delta - ci_low], [ci_high - delta]],
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.55,
                    capsize=2.6,
                    capthick=0.55,
                    zorder=2,
                )
                axis.scatter(
                    [delta],
                    [row["y"]],
                    marker="o",
                    s=13.0,
                    facecolor=color if excludes_zero else "#BFBFBF",
                    edgecolor=color,
                    linewidth=0.55,
                    zorder=3,
                )
            mean_delta, mean_low, mean_high = summaries[variant]
            axis.axhline(
                summary_y + 0.52,
                color="#B3B3B3",
                linewidth=0.55,
                zorder=1,
            )
            axis.hlines(
                summary_y,
                mean_low,
                mean_high,
                color="#333333",
                linewidth=1.0,
                zorder=2,
            )
            axis.scatter(
                [mean_delta],
                [summary_y],
                marker="D",
                s=20.0,
                facecolor="#333333",
                edgecolor="#333333",
                linewidth=0.4,
                zorder=4,
            )
            axis.set_title(
                f"({chr(ord('a') + index)}) {_VARIANT_LABELS[variant]}",
                loc="left",
                pad=5,
            )

        axes[0].set_yticks([row["y"] for row in rows] + [summary_y])
        axes[0].set_yticklabels(
            [row["label"] for row in rows] + ["Overall mean"],
        )
        axes[0].tick_params(axis="y", length=0, pad=3)
        axes[0].get_yticklabels()[-1].set_fontweight("bold")
        for axis in axes[1:]:
            axis.tick_params(axis="y", which="both", left=False, labelleft=False)
        for axis in axes:
            axis.spines["bottom"].set_visible(True)
            axis.tick_params(axis="x", length=2.5, width=0.6)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))

        figure.text(
            0.6,
            0.08,
            ("\u2190 all-at-once better" + 30 * " ") + 
            ("\u0394 macro-F1 (one-by-one \u2212 all-at-once)" + 30 * " ") +
            "one-by-one better \u2192",
            ha="center",
            va="center",
            fontsize=7.0,
        )

        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="#1A1A1A",
                markeredgecolor="#1A1A1A",
                markeredgewidth=0.55,
                markersize=4.2,
                label="95% CI excludes 0",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="#BFBFBF",
                markeredgecolor="#A6A6A6",
                markeredgewidth=0.55,
                markersize=4.2,
                label="95% CI includes 0",
            ),
            Line2D(
                [],
                [],
                marker="D",
                linestyle="none",
                color="#333333",
                markersize=4.2,
                label="Overall mean (95% CI)",
            ),
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.6, -0.03),
            ncol=3,
            columnspacing=1.5,
            handletextpad=0.5,
            handlelength=1.8,
        )
        artifacts = save_figure(figure, output_dir, "figure_2_mode_forest")
        plt.close(figure)
    return artifacts


def _value_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    """Return one complete macro-F1 mean for each model and task."""
    lookup = {}
    for row in rows:
        if not row.get("complete"):
            raise ValueError(
                f"Incomplete group for {_model_name(str(row['model']))} "
                f"on {row['dataset']}."
            )
        value = row.get("macro_f1_mean")
        if value is None:
            raise ValueError(
                f"Missing macro-F1 for {_model_name(str(row['model']))} "
                f"on {row['dataset']}."
            )
        key = (str(row["model_type"]), str(row["model"]), str(row["dataset"]))
        if key in lookup:
            raise ValueError(f"Repeated group for {_model_name(key[1])} on {key[2]}.")
        lookup[key] = float(value)
    return lookup


def _spread_lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], float]:
    """Return the replicate macro-F1 SD for each model and task."""
    return {
        (str(row["model_type"]), str(row["model"]), str(row["dataset"])):
            float(row.get("macro_f1_std") or 0.0)
        for row in rows
    }


def _heatmap_matrix(
    rows: list[dict[str, Any]],
    tasks: list[str] | None = None,
) -> tuple[list[tuple[str, str]], list[str], list[list[float]]]:
    """Return model rows, task columns, and the macro-F1 values.

    Args:
        rows: One complete group per model and task.
        tasks: Expected task order. The default uses the tasks in the rows.

    Returns:
        The model identities, task names, and the value matrix.

    Raises:
        ValueError: A group is incomplete, missing, or repeated.
    """
    lookup = _value_lookup(rows)
    models = sorted(
        {(model_type, model) for model_type, model, _ in lookup},
        key=lambda key: _row_sort_key(*key),
    )
    task_order = (
        list(tasks)
        if tasks is not None
        else _ordered_datasets({dataset for _, _, dataset in lookup})
    )
    missing = [
        (_model_name(model), _dataset_name(dataset))
        for model_type, model in models
        for dataset in task_order
        if (model_type, model, dataset) not in lookup
    ]
    if missing:
        raise ValueError(f"Missing heatmap cells: {missing}.")
    matrix = [
        [lookup[(model_type, model, dataset)] for dataset in task_order]
        for model_type, model in models
    ]
    return models, task_order, matrix


def _control_matrices(
    normal: list[dict[str, Any]],
    zero_shot: list[dict[str, Any]],
    corrupted: list[dict[str, Any]],
    tasks: list[str] | None = None,
) -> tuple[
    list[tuple[str, str]],
    list[str],
    list[list[float]],
    list[list[float]],
    list[list[float]],
]:
    """Return normal macro-F1 and the loss under both label controls.

    A control can score above the normal condition. Such a cell has a negative
    loss, and the plot shows the control value above the normal bar.
    """
    models, task_order, baseline = _heatmap_matrix(normal, tasks)
    losses = []
    for rows, label in ((zero_shot, "zero-shot"), (corrupted, "corrupted-label")):
        lookup = _value_lookup(rows)
        missing = [
            (_model_name(model), _dataset_name(dataset))
            for model_type, model in models
            for dataset in task_order
            if (model_type, model, dataset) not in lookup
        ]
        if missing:
            raise ValueError(f"Missing {label} cells: {missing}.")
        loss = [
            [
                baseline[row_index][column_index] - lookup[(model_type, model, dataset)]
                for column_index, dataset in enumerate(task_order)
            ]
            for row_index, (model_type, model) in enumerate(models)
        ]
        losses.append(loss)
    return models, task_order, baseline, losses[0], losses[1]


def _performance_bar_data(
    rows: list[dict[str, Any]],
    tasks: list[str] | None = None,
) -> tuple[
    list[tuple[str, str]],
    list[str],
    list[list[float]],
    list[list[float]],
]:
    """Return the model order, task order, means, and replicate SDs.

    Models keep the fixed publication order and stay grouped by family. The
    bars do not change position between tasks.

    Args:
        rows: One complete group per model and task.
        tasks: Expected task order. The default uses the tasks in the rows.

    Returns:
        The model identities, task names, mean macro-F1 values, and replicate
        standard deviations. Both matrices use the model order first.

    Raises:
        ValueError: A group is incomplete, missing, or repeated.
    """
    models, task_order, values = _heatmap_matrix(rows, tasks)
    order = {model: index for index, model in enumerate(_PERFORMANCE_BAR_MODEL_ORDER)}
    ranking = sorted(
        range(len(models)),
        key=lambda index: (
            _FAMILY_ORDER.get(
                _family(models[index][0]),
                len(_FAMILY_ORDER),
            ),
            order.get(models[index][1], len(order)),
            _model_name(models[index][1]).lower(),
        ),
    )
    models = [models[index] for index in ranking]
    values = [
        [values[index][task_index] for task_index in range(len(task_order))]
        for index in ranking
    ]
    lookup = {
        (str(row["model_type"]), str(row["model"]), str(row["dataset"])): row
        for row in rows
    }
    spread = [
        [
            float(lookup[(model_type, model, dataset)].get("macro_f1_std") or 0.0)
            for dataset in task_order
        ]
        for model_type, model in models
    ]
    return models, task_order, values, spread


def plot_performance_bars(
    rows: list[dict[str, Any]],
    output_dir: Path,
    name: str = "figure_3_performance",
    tasks: list[str] | None = None,
) -> list[str]:
    """Draw grouped bars of mean macro-F1 for each task.

    Bars keep a fixed model order across all tasks. Short model names sit
    inside the bars, and the best value in each task sits above its bar.
    Family colour identifies the model family. Bars start at zero and
    carry the replicate standard deviation. The figure has no title or
    explanatory text. A family legend sits below the axis.

    Args:
        rows: One complete group per model and task. Endpoint groups use
            one-by-one inference, and local groups use local inference.
        output_dir: Destination for the figure files.
        name: File name without extension.
        tasks: Expected task order. The default uses the tasks in the rows.

    Returns:
        Paths of the written figure files.
    """
    from matplotlib.patches import Patch

    models, task_order, values, spread = _performance_bar_data(rows, tasks)
    count = len(models)
    group_width = count + 1.2
    centers = [index * group_width for index in range(len(task_order))]
    positions = [
        center + (slot - (count - 1) / 2.0)
        for center in centers
        for slot in range(count)
    ]
    families = list(dict.fromkeys(_family(model_type) for model_type, _ in models))
    styles = {
        "LLM": ("#0072B2", None),
        "Decision model": ("#56B4E9", None),
        "Tabular foundation model": ("#E69F00", None),
        "Classical baseline": ("#009E73", None),
    }
    replicates: dict[str, int] = {}
    for row in rows:
        family = _family(str(row["model_type"]))
        replicates.setdefault(family, int(row.get("expected_replicates") or 0))

    plt = pyplot()
    with plt.rc_context(_STYLE):
        figure, axis = plt.subplots(figsize=(_FIGURE_WIDTH, 3.75))
        figure.subplots_adjust(left=0.060, right=0.990, top=0.965, bottom=0.220)
        for task_index in range(len(task_order)):
            best_index = max(
                range(count),
                key=lambda index: values[index][task_index],
            )
            for model_index in range(count):
                family = _family(models[model_index][0])
                color, hatch = styles.get(family, ("#777777", None))
                position = positions[task_index * count + model_index]
                value = values[model_index][task_index]
                spread_value = spread[model_index][task_index]
                axis.bar(
                    [position],
                    [value],
                    width=0.82,
                    color=color,
                    edgecolor="#333333",
                    linewidth=0.35,
                    hatch=hatch,
                    yerr=[spread_value],
                    error_kw={
                        "ecolor": "#4D4D4D",
                        "elinewidth": 0.55,
                        "capsize": 1.4,
                        "capthick": 0.55,
                    },
                    zorder=2,
                )
                axis.text(
                    position + 0.06,
                    0.015,
                    _model_short_label(models[model_index][1]),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6.0,
                    color="white"
                    if family in ["LLM", "Classical baseline"]
                    else "#1A1A1A",
                    zorder=3,
                )
                if model_index == best_index:
                    axis.text(
                        position,
                        value + spread_value + 0.020,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        fontweight="bold",
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "none",
                            "pad": 0.15,
                        },
                    )
        axis.set_xticks([])
        axis.set_xlim(positions[0] - 1.2, positions[-1] + 0.7)
        axis.set_ylim(0.0, 1.0)
        axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axis.set_ylabel("Macro-F1", fontsize=7.0)
        axis.set_axisbelow(True)
        axis.grid(axis="y", linestyle="--")
        axis.spines["left"].set_visible(True)
        axis.spines["bottom"].set_visible(True)
        axis.tick_params(axis="x", length=0, pad=2)
        axis.tick_params(axis="y", length=2.0, width=0.6)
        for center, dataset in zip(centers, task_order, strict=True):
            axis.text(
                center,
                -0.055,
                _paper_task_name(dataset),
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.8,
                clip_on=False,
            )

        handles = [
            Patch(
                facecolor=styles.get(family, ("#777777", None))[0],
                edgecolor="#333333",
                linewidth=0.35,
                hatch=styles.get(family, ("#777777", None))[1],
                label=f"{family} (n={replicates.get(family, 0)})",
            )
            for family in families
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.040),
            ncol=len(handles),
            columnspacing=1.4,
            handletextpad=0.5,
            handlelength=1.6,
        )
        artifacts = save_figure(
            figure,
            output_dir,
            name,
            pad_inches=0.060,
        )
        plt.close(figure)
    return artifacts


def plot_label_control_bars(
    normal: list[dict[str, Any]],
    zero_shot: list[dict[str, Any]],
    corrupted: list[dict[str, Any]],
    output_dir: Path,
    majority_baseline: dict[str, float],
    name: str = "figure_4_controls",
    tasks: list[str] | None = None,
) -> list[str]:
    """Draw grouped condition bars for the five label-control panels.

    The five task panels use two rows. The three panels of the first row span
    the full width. The two panels of the second row are centered. One shared
    legend for the three conditions sits below all panels. Each bar carries
    the replicate standard deviation. A dashed horizontal line gives the
    majority-class macro-F1 in each panel. Its value sits above the right end
    of the line, and the legend identifies the line. Every panel shows the
    y-axis tick labels.

    Args:
        normal: Complete groups with the correct reference labels.
        zero_shot: Complete groups without reference rows.
        corrupted: Complete groups with corrupted reference labels.
        output_dir: Destination for the figure files.
        majority_baseline: Majority-class macro-F1 for each task.
        name: File name without extension.
        tasks: Expected task order. The default uses the tasks in the rows.

    Returns:
        Paths of the written figure files.

    Raises:
        ValueError: The task selection does not contain five tasks, or a task
            has no majority-class baseline.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    models, task_order, baseline, zero_loss, corrupted_loss = _control_matrices(
        normal,
        zero_shot,
        corrupted,
        tasks,
    )
    normal_spread = _spread_lookup(normal)
    zero_spread = _spread_lookup(zero_shot)
    corrupted_spread = _spread_lookup(corrupted)
    if len(task_order) != 5:
        raise ValueError("The label-control bar figure needs five tasks.")
    missing_baselines = [
        dataset for dataset in task_order if dataset not in majority_baseline
    ]
    if missing_baselines:
        raise ValueError(
            f"Missing majority-class baselines: {missing_baselines}."
        )
    zero = [
        [
            baseline[row][column] - zero_loss[row][column]
            for column in range(len(task_order))
        ]
        for row in range(len(models))
    ]
    corrupted_values = [
        [
            baseline[row][column] - corrupted_loss[row][column]
            for column in range(len(task_order))
        ]
        for row in range(len(models))
    ]
    positions = []
    position = 0.0
    previous_family = None
    for model_type, _ in models:
        family = _family(model_type)
        if previous_family is not None and family != previous_family:
            position += 0.4
        positions.append(position)
        position += 1.0
        previous_family = family
    width = 0.24
    standard_color = "#1F4E79"
    zero_color = "#5B9BD5"
    corrupted_color = "#D9D9D9"
    error_style = {
        "ecolor": "#4D4D4D",
        "elinewidth": 0.5,
        "capsize": 1.0,
        "capthick": 0.5,
    }
    plt = pyplot()
    with plt.rc_context(_STYLE):
        figure = plt.figure(figsize=(_FIGURE_WIDTH, 4.35))
        grid = figure.add_gridspec(
            2,
            6,
            left=0.065,
            right=0.990,
            top=0.975,
            bottom=0.250,
            wspace=0.55,
            hspace=0.90,
        )
        axes = [figure.add_subplot(grid[0, 0:2])]
        for bounds in (
            grid[0, 2:4],
            grid[0, 4:6],
            grid[1, 1:3],
            grid[1, 3:5],
        ):
            axes.append(figure.add_subplot(bounds, sharey=axes[0]))
        for task_index, (axis, dataset) in enumerate(
            zip(axes, task_order, strict=True)
        ):
            for model_index, position in enumerate(positions):
                key = (*models[model_index], dataset)
                axis.bar(
                    position - width,
                    baseline[model_index][task_index],
                    width=width,
                    color=standard_color,
                    edgecolor="#333333",
                    linewidth=0.35,
                    yerr=[normal_spread[key]],
                    error_kw=error_style,
                    zorder=2,
                )
                axis.bar(
                    position,
                    zero[model_index][task_index],
                    width=width,
                    color=zero_color,
                    edgecolor="#333333",
                    linewidth=0.35,
                    yerr=[zero_spread[key]],
                    error_kw=error_style,
                    zorder=2,
                )
                axis.bar(
                    position + width,
                    corrupted_values[model_index][task_index],
                    width=width,
                    color=corrupted_color,
                    edgecolor="#555555",
                    linewidth=0.35,
                    hatch="////",
                    yerr=[corrupted_spread[key]],
                    error_kw=error_style,
                    zorder=2,
                )
            baseline_value = float(majority_baseline[dataset])
            axis.axhline(
                baseline_value,
                color="#8B1A1A",
                linewidth=0.6,
                linestyle=(0, (2.0, 1.5)),
                zorder=1,
            )
            axis.annotate(
                f"{baseline_value:.3f}",
                xy=(0.985, baseline_value),
                xycoords=axis.get_yaxis_transform(),
                xytext=(0.0, 2.0),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=5.4,
                color="#8B1A1A",
                zorder=5,
            )
            axis.set_xlim(positions[0] - 0.55, positions[-1] + 0.95)
            axis.set_ylim(0.0, 1.0)
            axis.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            axis.set_xticks(positions)
            axis.set_xticklabels(
                [_model_name(model) for _, model in models],
                rotation=45,
                ha="right",
                rotation_mode="anchor",
                fontsize=5.6,
            )
            axis.set_title(
                _paper_task_name(dataset),
                fontsize=6.8,
                # labelpad=1,
            )
            axis.set_axisbelow(True)
            axis.grid(axis="y", linestyle="--")
            axis.spines["left"].set_visible(True)
            axis.spines["bottom"].set_visible(True)
            axis.tick_params(axis="x", length=0, pad=2)
            axis.tick_params(
                axis="y",
                length=2.0,
                width=0.6,
                labelleft=True,
            )
        axes[0].set_ylabel("Macro-F1", fontsize=7.0)

        handles = [
            Patch(
                facecolor=standard_color,
                edgecolor="#333333",
                linewidth=0.35,
                label="Standard ICL",
            ),
            Patch(
                facecolor=zero_color,
                edgecolor="#333333",
                linewidth=0.35,
                label="Zero-shot",
            ),
            Patch(
                facecolor=corrupted_color,
                edgecolor="#555555",
                linewidth=0.35,
                hatch="////",
                label="Corrupted labels",
            ),
            Line2D(
                [0],
                [0],
                color="#8B1A1A",
                linewidth=0.6,
                linestyle=(0, (2.0, 1.5)),
                label="Majority-class baseline",
            ),
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=4,
            columnspacing=1.4,
            handletextpad=0.5,
            handlelength=1.4,
        )
        artifacts = save_figure(figure, output_dir, name)
        plt.close(figure)
    return artifacts
