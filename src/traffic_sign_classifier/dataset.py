"""Read and validate traffic-sign dataset annotations."""

import csv
from pathlib import Path

REQUIRED_COLUMNS = frozenset(
    {
        "Width",
        "Height",
        "Roi.X1",
        "Roi.Y1",
        "Roi.X2",
        "Roi.Y2",
        "ClassId",
        "Path",
    }
)


def validate_annotation_header(csv_path: Path) -> None:
    """Raise ValueError if the annotation header is invalid."""
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)

    if not header:
        raise ValueError(f"{csv_path}: missing CSV header")

    if len(header) != len(set(header)):
        raise ValueError(f"{csv_path}: duplicate column names")

    missing = REQUIRED_COLUMNS.difference(header)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{csv_path}: missing required columns: {names}")
