"""Read text rows and write selected columns as CSV."""

import csv
import io
from collections.abc import Mapping, Sequence
from pathlib import Path

type Row = dict[str, str]
type DataSource = str | Path | Sequence[Row]


def _check_columns(columns: Sequence[str], required: Sequence[str], label: str) -> None:
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"{label}: missing required columns {missing!r}")


def load_rows(source: DataSource, required: Sequence[str], label: str) -> list[Row]:
    """Load rows and check all cells before prediction starts.

    Args:
        source: CSV path or sequence of text dictionaries.
        required: Required column names.
        label: Dataset name for error messages.

    Returns:
        Rows with only the required columns.

    Raises:
        ValueError: A header, row, or cell is invalid.
        TypeError: The source or a row has an invalid type.
    """
    rows: list[Row] = []
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            try:
                headers = next(reader, None)
                if not headers:
                    raise ValueError(f"{label}: CSV requires a header")
                if len(headers) != len(set(headers)):
                    raise ValueError(f"{label}: CSV headers must be unique")
                _check_columns(headers, required, label)
                index = 0
                while True:
                    try:
                        cells = next(reader)
                    except StopIteration:
                        break
                    except csv.Error as exc:
                        raise ValueError(f"{label} row {index}: {exc}") from exc
                    if len(cells) != len(headers):
                        raise ValueError(
                            f"{label} row {index}: expected {len(headers)} cells, "
                            f"got {len(cells)}"
                        )
                    row = dict(zip(headers, cells, strict=True))
                    rows.append({name: row[name] for name in required})
                    index += 1
            except csv.Error as exc:
                raise ValueError(f"{label}: invalid CSV header: {exc}") from exc
    else:
        if not isinstance(source, Sequence):
            raise TypeError(f"{label}: expected a CSV path or sequence of rows")
        for index, row in enumerate(source):
            location = f"{label} row {index}"
            if not isinstance(row, Mapping):
                raise TypeError(f"{location}: expected a row dictionary")
            if any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in row.items()
            ):
                raise ValueError(f"{location}: column names and cells must be strings")
            _check_columns(list(row), required, location)
            rows.append({name: row[name] for name in required})
    return rows


def rows_to_csv(rows: Sequence[Row], columns: Sequence[str]) -> str:
    """Write selected cells without changing their text."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(columns)
    writer.writerows([row[name] for name in columns] for row in rows)
    return stream.getvalue()
