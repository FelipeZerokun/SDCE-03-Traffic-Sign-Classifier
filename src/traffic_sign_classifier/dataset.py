"""Read and validate traffic-sign dataset annotations."""

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

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


@dataclass(frozen=True)
class Annotation:
    """Metadata describing one traffic-sign image."""

    image_path: Path
    class_id: int
    width: int
    height: int
    roi_x1: int
    roi_y1: int
    roi_x2: int
    roi_y2: int


def validate_annotation(annotation: Annotation) -> None:
    """Validate image dimensions and the GTSRB class ID."""
    if annotation.width <= 0 or annotation.height <= 0:
        raise ValueError("Image width and height must be positive")

    if not 0 <= annotation.class_id < 43:
        raise ValueError("Class ID must be between 0 and 42")


def parse_annotation(row: Mapping[str, str]) -> Annotation:
    """Convert a CSV row into an annotation."""
    try:
        image_path = row["Path"]
        if not image_path.strip():
            raise ValueError("Path must not be empty")

        annotation = Annotation(
            image_path=Path(image_path),
            class_id=int(row["ClassId"]),
            width=int(row["Width"]),
            height=int(row["Height"]),
            roi_x1=int(row["Roi.X1"]),
            roi_y1=int(row["Roi.Y1"]),
            roi_x2=int(row["Roi.X2"]),
            roi_y2=int(row["Roi.Y2"]),
        )

        validate_annotation(annotation)
        return annotation

    except KeyError as error:
        raise ValueError(f"Missing annotation field: {error.args[0]}") from error
    except ValueError as error:
        raise ValueError(f"Invalid annotation: {error}") from error


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


def read_annotations(csv_path: Path) -> list[Annotation]:
    """Read and validate all annotation records from a CSV file."""
    validate_annotation_header(csv_path)
    annotations: list[Annotation] = []

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"{csv_path}: line {reader.line_num}: "
                    "row length does not match the header"
                )

            try:
                annotation = parse_annotation(row)
            except ValueError as error:
                raise ValueError(
                    f"{csv_path}: line {reader.line_num}: {error}"
                ) from error

            annotations.append(annotation)

    if not annotations:
        raise ValueError(f"{csv_path}: no annotation records")

    return annotations


def inspect_image(annotation: Annotation, data_root: Path) -> str:
    """Check image readability and dimensions, returning its image mode."""
    image_path = data_root / annotation.image_path

    try:
        with Image.open(image_path) as image:
            image.load()
            actual_size = image.size
            mode = image.mode
    except OSError as error:
        raise ValueError(f"{image_path}: cannot read image") from error

    expected_size = (annotation.width, annotation.height)
    if actual_size != expected_size:
        raise ValueError(
            f"{image_path}: expected size {expected_size}, found {actual_size}"
        )

    return mode


def training_track_key(annotation: Annotation) -> tuple[int, int]:
    """Extract a class-and-track key from a Kaggle training filename."""
    match = re.fullmatch(
        r"([0-9]{5})_([0-9]{5})_([0-9]{5})\.png",
        annotation.image_path.name,
    )
    if match is None:
        raise ValueError(f"Unexpected training filename: {annotation.image_path.name}")

    filename_class = int(match.group(1))
    track_id = int(match.group(2))

    if filename_class != annotation.class_id:
        raise ValueError("Filename class does not match annotation class")

    return annotation.class_id, track_id
